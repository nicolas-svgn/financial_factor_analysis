import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional # Added Optional

logger = logging.getLogger(__name__)

def calculate_correlation_matrix(returns_df: pd.DataFrame, method: str = 'pearson') -> Optional[pd.DataFrame]: # Use Optional
    """
    Calculates the pairwise static correlation matrix for the given returns data.

    Args:
        returns_df (pd.DataFrame): DataFrame with returns (dates as index, tickers as columns).
        method (str): Method of correlation ('pearson', 'kendall', 'spearman').

    Returns:
        Optional[pd.DataFrame]: The calculated correlation matrix, or None on error.
    """
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        logger.error("Invalid or empty DataFrame provided for static correlation calculation.")
        return None
    # Check for NaNs before calculation
    if returns_df.isnull().values.any():
         logger.warning("Input DataFrame contains NaNs. Static correlation might produce NaNs.")
         # Consider adding handling here if needed, e.g., returns_df.dropna() or fillna(0)
         # For now, pandas corr handles NaNs by pairwise deletion by default.

    try:
        logger.info(f"Calculating static {method} correlation matrix...")
        corr_matrix = returns_df.corr(method=method)
        logger.info(f"Static correlation matrix calculated ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
        # Check if result is all NaNs
        if corr_matrix.isnull().all().all():
             logger.error("Static correlation matrix calculation resulted in all NaNs.")
             return None
        return corr_matrix
    except Exception as e:
        logger.error(f"Error calculating static correlation matrix: {e}", exc_info=True)
        return None

def calculate_ewma_correlation_matrix(returns_df: pd.DataFrame, span: int = 60) -> Optional[pd.DataFrame]: # Use Optional
    """
    Compute the EWMA correlation matrix for the last date in the DataFrame.

    Args:
        returns_df (pd.DataFrame): DataFrame with returns (dates as index, tickers as columns).
                                   Should not contain NaNs.
        span (int): EWMA span parameter.

    Returns:
        Optional[pd.DataFrame]: DataFrame with the EWMA correlation matrix as of the last date,
                                indexed by stock tickers, or None on error.
    """
    if not isinstance(returns_df, pd.DataFrame):
        logger.error("Input 'returns_df' must be a pandas DataFrame for EWMA correlation.")
        return None
    if returns_df.empty:
        logger.error("Input DataFrame is empty for EWMA correlation.")
        return None
    if returns_df.isnull().values.any():
        logger.error("Input DataFrame contains NaN values. Please handle missing data before EWMA.")
        # Consider adding ffill/dropna here, but it's better handled upstream (e.g., in PCAnalyzer prep)
        return None
    if not np.all(returns_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        logger.error("Input DataFrame must contain only numeric data for EWMA correlation.")
        return None
    if span <= 1:
        logger.error(f"EWMA span must be greater than 1, got {span}")
        return None

    logger.info(f"Calculating EWMA correlation matrix with span={span}...")
    try:
        # Compute EWMA covariance matrix - adjust=False is common for financial series
        ewma_cov = returns_df.ewm(span=span, adjust=False).cov(pairwise=True) # pairwise=True is default but explicit

        # Get the last day's covariance matrix
        last_date = returns_df.index[-1]
        # Accessing the last date's matrix from the MultiIndex result
        cov_matrix_last = ewma_cov.loc[pd.IndexSlice[last_date, :], :]

        # Check if extraction worked
        if cov_matrix_last.empty:
             logger.error(f"Could not extract EWMA covariance matrix for the last date {last_date}. Check data/index.")
             return None
        # Ensure cov_matrix_last is a DataFrame
        if not isinstance(cov_matrix_last, pd.DataFrame):
             logger.error(f"Extracted EWMA covariance for {last_date} is not a DataFrame (type: {type(cov_matrix_last)}).")
             return None

        # Compute correlation from covariance
        variances = np.diag(cov_matrix_last)
        # Check for non-positive variances which make correlation calculation invalid
        if np.any(variances <= 1e-12): # Use a small threshold instead of exact zero
            zero_var_tickers = cov_matrix_last.index[variances <= 1e-12].tolist()
            logger.warning(f"EWMA covariance matrix contains near-zero or negative variances for tickers: {zero_var_tickers}. "
                           "Correlation may be unstable or NaN for these.")
            variances[variances <= 1e-12] = 1e-12 # Replace with small positive number to avoid division by zero

        std_dev = np.sqrt(variances)
        # Use np.outer for element-wise division
        corr_matrix = cov_matrix_last.values / np.outer(std_dev, std_dev)

        # Clip to handle potential numerical inaccuracies slightly outside [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        # Restore DataFrame structure
        corr_matrix_df = pd.DataFrame(corr_matrix, index=cov_matrix_last.index, columns=cov_matrix_last.columns)

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix_df.values, 1.0)

        logger.info(f"EWMA correlation matrix calculated for date {last_date} ({corr_matrix_df.shape[0]}x{corr_matrix_df.shape[1]})")
        return corr_matrix_df

    except KeyError:
        logger.error(f"KeyError: Could not extract EWMA covariance matrix for date {last_date}. "
                     "Likely issue with MultiIndex access or data length relative to span.")
        return None
    except Exception as e:
        logger.error(f"Error calculating EWMA correlation matrix: {e}", exc_info=True)
        return None


def calculate_sector_correlations(corr_matrix: pd.DataFrame,
                                  metadata_df: pd.DataFrame,
                                  sector_col: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]: # Use Optional
    """
    Calculates average within-sector and across-sector correlations using a provided correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): The pre-computed correlation matrix (tickers x tickers). Can be static or EWMA.
        metadata_df (pd.DataFrame): Metadata DataFrame, indexed by ticker, containing the sector column.
        sector_col (str): The name of the column in metadata_df containing sector information.

    Returns:
        Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
            - Series of average within-sector correlations (indexed by sector).
            - DataFrame of average across-sector correlations (sector x sector).
            Returns (None, None) if inputs are invalid or metadata is missing.
    """
    if not isinstance(corr_matrix, pd.DataFrame) or corr_matrix.empty:
        logger.error("Invalid correlation matrix provided for sector analysis.")
        return None, None
    if not isinstance(metadata_df, pd.DataFrame) or metadata_df.empty:
        logger.warning("Metadata not provided or empty. Cannot calculate sector correlations.")
        return None, None
    if sector_col not in metadata_df.columns:
        logger.error(f"Sector column '{sector_col}' not found in metadata.")
        return None, None

    logger.info(f"Calculating within-sector and across-sector correlations using column '{sector_col}'...")

    # Ensure we only work with tickers present in both returns (corr_matrix) and metadata
    common_tickers = corr_matrix.index.intersection(metadata_df.index)
    if len(common_tickers) == 0:
        logger.error("No common tickers found between correlation matrix and metadata.")
        return None, None
    if len(common_tickers) < len(corr_matrix.index):
        logger.warning(f"Metadata missing for {len(corr_matrix.index) - len(common_tickers)} tickers in correlation matrix. "
                       f"Using {len(common_tickers)} common tickers for sector analysis.")

    # Filter correlation matrix and metadata
    corr_matrix_filtered = corr_matrix.loc[common_tickers, common_tickers]
    metadata_filtered = metadata_df.loc[common_tickers]

    # Get sector mapping for common tickers, dropping tickers with NaN sectors
    sectors = metadata_filtered[sector_col].dropna().to_dict()
    if not sectors:
         logger.error(f"No valid sector information found in column '{sector_col}' for common tickers.")
         return None, None
    unique_sectors = sorted(list(set(sectors.values()))) # Get unique sectors from the filtered dict

    within_sector_corrs = {}
    across_sector_corrs = pd.DataFrame(index=unique_sectors, columns=unique_sectors, dtype=float)

    for i, sector1 in enumerate(unique_sectors):
        tickers1 = [t for t, s in sectors.items() if s == sector1]

        # --- Within-sector ---
        if len(tickers1) > 1:
            sector_corr = corr_matrix_filtered.loc[tickers1, tickers1]
            # Use numpy to efficiently get upper triangle values (excluding diagonal k=1)
            mask = np.triu(np.ones(sector_corr.shape, dtype=bool), k=1)
            sector_values = sector_corr.values[mask]
            if sector_values.size > 0:
                # Ignore NaNs that might arise from specific pairwise calculations
                within_sector_corrs[sector1] = np.nanmean(sector_values)
            else:
                 within_sector_corrs[sector1] = np.nan
        else:
            within_sector_corrs[sector1] = np.nan # Correlation requires at least 2 stocks

        # --- Across-sector ---
        for j, sector2 in enumerate(unique_sectors):
            # Diagonal is within-sector average (calculated above)
            if i == j:
                 across_sector_corrs.loc[sector1, sector2] = within_sector_corrs.get(sector1, np.nan)
                 continue
            # Matrix is symmetric, calculate only upper triangle (j > i)
            if j < i:
                continue

            tickers2 = [t for t, s in sectors.items() if s == sector2]

            if tickers1 and tickers2: # Check if both sectors have stocks
                # Extract the cross-correlation block
                cross_corr_block = corr_matrix_filtered.loc[tickers1, tickers2]
                if cross_corr_block.size > 0:
                    # Calculate the mean, ignoring potential NaNs within the block
                    mean_cross_corr = np.nanmean(cross_corr_block.values)
                    across_sector_corrs.loc[sector1, sector2] = mean_cross_corr
                    across_sector_corrs.loc[sector2, sector1] = mean_cross_corr # Fill symmetric part
                else:
                    across_sector_corrs.loc[sector1, sector2] = np.nan
                    across_sector_corrs.loc[sector2, sector1] = np.nan
            else:
                # If one sector has no stocks (shouldn't happen with this logic, but safe)
                across_sector_corrs.loc[sector1, sector2] = np.nan
                across_sector_corrs.loc[sector2, sector1] = np.nan

    within_sector_series = pd.Series(within_sector_corrs, name='Average Within-Sector Correlation')
    logger.info("Sector correlation calculations complete.")
    # Return None if Series/DataFrame are empty or all NaN
    return (within_sector_series if not within_sector_series.dropna().empty else None,
            across_sector_corrs if not across_sector_corrs.dropna(how='all').empty else None)


def calculate_rolling_correlations(returns_df: pd.DataFrame, window: int) -> Optional[pd.Series]: # Use Optional
    """
    Calculates the average pairwise rolling correlation over time.

    Args:
        returns_df (pd.DataFrame): DataFrame with returns (dates as index, tickers as columns).
                                   Should handle NaNs appropriately before calling.
        window (int): The rolling window size in periods (days).

    Returns:
        Optional[pd.Series]: A Series containing the average correlation for each time point
                             in the rolling window, or None on error.
    """
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        logger.error("Invalid or empty DataFrame provided for rolling correlation.")
        return None
    if window <= 1 or window > len(returns_df):
         logger.error(f"Invalid window size {window} for data length {len(returns_df)}.")
         return None
    # It's better if NaNs are handled before this function, but add a check/warning
    if returns_df.isnull().values.any():
         logger.warning(f"Input DataFrame for rolling correlation contains NaNs. Results may be affected or NaN.")
         # Option: returns_df = returns_df.fillna(0) or ffill() if appropriate for the use case

    logger.info(f"Calculating rolling correlations with window={window}...")
    try:
        # Calculate rolling correlations - yields MultiIndex DataFrame (date, ticker1, ticker2)
        # This can be memory intensive!
        rolling_corr = returns_df.rolling(window=window, min_periods=max(2, window // 2)).corr(pairwise=True) # Use min_periods

        if rolling_corr.empty:
            logger.warning(f"Rolling correlation calculation with window {window} yielded no results (check min_periods?).")
            return None

        # Efficiently calculate the mean of the upper triangle for each date's matrix
        def mean_upper_triangle(matrix_df):
            if not isinstance(matrix_df, pd.DataFrame) or matrix_df.shape[0] < 2:
                return np.nan
            # Get underlying numpy array for speed
            matrix_values = matrix_df.values
            # Create mask for upper triangle (k=1 excludes diagonal)
            mask = np.triu(np.ones(matrix_values.shape, dtype=bool), k=1)
            # Extract values using the mask
            upper_triangle_values = matrix_values[mask]
            # Calculate mean, ignoring NaNs if any exist
            if upper_triangle_values.size > 0:
                return np.nanmean(upper_triangle_values)
            else:
                return np.nan # Should not happen if shape[0] >= 2

        # Group by date (level 0) and apply the function
        # This can still be slow for very large datasets/many dates
        avg_rolling_corr = rolling_corr.groupby(level=0).apply(mean_upper_triangle)

        # Drop initial NaNs resulting from the rolling window min_periods
        avg_rolling_corr = avg_rolling_corr.dropna()

        if avg_rolling_corr.empty:
             logger.warning(f"Average rolling correlation series is empty after dropping NaNs for window={window}.")
             return None

        logger.info(f"Rolling average correlation calculation complete for window={window}.")
        return avg_rolling_corr

    except MemoryError:
         logger.error(f"MemoryError calculating rolling correlations with window={window}. "
                      "Consider reducing the number of stocks, increasing memory, or using an alternative method.")
         return None
    except Exception as e:
        logger.error(f"Error calculating rolling correlations: {e}", exc_info=True)
        return None

