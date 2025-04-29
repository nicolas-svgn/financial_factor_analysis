import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# --- Helper function to convert Covariance to Correlation ---
def _cov_to_corr(cov_matrix: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Converts a covariance matrix DataFrame to a correlation matrix DataFrame."""
    if not isinstance(cov_matrix, pd.DataFrame) or cov_matrix.empty:
        logger.debug("Input to _cov_to_corr is not a valid DataFrame or is empty.")
        return None
    try:
        variances = np.diag(cov_matrix)
        # Handle non-positive variances
        if np.any(variances <= 1e-12):
            # Find tickers with issues for potential logging, but avoid logging excessively in loops
            zero_var_tickers = cov_matrix.index[variances <= 1e-12].tolist()
            logger.debug(f"Cov matrix contains near-zero/negative variances for: {zero_var_tickers}")
            variances[variances <= 1e-12] = 1e-12 # Replace for calculation stability

        std_dev = np.sqrt(variances)
        # Check for division by zero possibility if std_dev is zero
        if np.any(std_dev < 1e-12):
             logger.warning("Near-zero standard deviation encountered during cov_to_corr.")
             std_dev[std_dev < 1e-12] = 1e-12 # Prevent division by zero

        # Calculate correlation
        # Use np.errstate to temporarily ignore invalid division warnings if needed,
        # as we handle near-zero std_dev manually.
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix_vals = cov_matrix.values / np.outer(std_dev, std_dev)
            # Replace NaNs resulting from 0/0 division (should be rare after std_dev adjustment)
            corr_matrix_vals = np.nan_to_num(corr_matrix_vals, nan=0.0) # Replace NaN with 0 correlation

        # Clip and restore DataFrame structure
        corr_matrix_vals = np.clip(corr_matrix_vals, -1.0, 1.0)
        corr_matrix_df = pd.DataFrame(corr_matrix_vals, index=cov_matrix.index, columns=cov_matrix.columns)
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix_df.values, 1.0)
        return corr_matrix_df
    except Exception as e:
        logger.error(f"Error converting covariance to correlation: {e}", exc_info=True)
        return None

# --- Helper function to calculate mean upper triangle correlation ---
def _mean_upper_triangle(matrix_df: pd.DataFrame) -> float:
    """Calculates the mean of the upper triangle (excluding diagonal) of a matrix."""
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.shape[0] < 2:
        return np.nan
    matrix_values = matrix_df.values
    mask = np.triu(np.ones(matrix_values.shape, dtype=bool), k=1)
    upper_triangle_values = matrix_values[mask]
    if upper_triangle_values.size > 0:
        # Ignore NaNs that might exist in the correlation matrix itself
        return np.nanmean(upper_triangle_values)
    else:
        return np.nan


# --- Static Correlation (Pearson, Spearman, etc.) ---
def calculate_correlation_matrix(returns_df: pd.DataFrame, method: str = 'pearson') -> Optional[pd.DataFrame]:
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
    if returns_df.isnull().values.any():
         logger.warning("Input DataFrame contains NaNs. Static correlation might produce NaNs.")

    try:
        logger.info(f"Calculating static {method} correlation matrix...")
        corr_matrix = returns_df.corr(method=method)
        logger.info(f"Static correlation matrix calculated ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
        if corr_matrix.isnull().all().all():
             logger.error("Static correlation matrix calculation resulted in all NaNs.")
             return None
        return corr_matrix
    except Exception as e:
        logger.error(f"Error calculating static correlation matrix: {e}", exc_info=True)
        return None

# --- EWMA Correlation (Point-in-time for last date) ---
def calculate_ewma_correlation_matrix(returns_df: pd.DataFrame, span: int = 60) -> Optional[pd.DataFrame]:
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
        return None
    if not np.all(returns_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        logger.error("Input DataFrame must contain only numeric data for EWMA correlation.")
        return None
    if span <= 1:
        logger.error(f"EWMA span must be greater than 1, got {span}")
        return None

    logger.info(f"Calculating EWMA correlation matrix with span={span} (for last date)...")
    try:
        # Compute EWMA covariance matrix time series
        ewma_cov_ts = returns_df.ewm(span=span, adjust=False, min_periods=max(2, span//2)).cov(pairwise=True)

        # Get the last day's covariance matrix
        last_date = returns_df.index[-1]
        # Use .xs to select from MultiIndex level 0 (date) - potentially more robust
        cov_matrix_last = ewma_cov_ts.xs(last_date, level=0, drop_level=True)

        if cov_matrix_last.empty:
             logger.error(f"Could not extract EWMA covariance matrix for the last date {last_date}.")
             return None
        if not isinstance(cov_matrix_last, pd.DataFrame):
             logger.error(f"Extracted EWMA covariance for {last_date} is not a DataFrame (type: {type(cov_matrix_last)}).")
             return None

        # Convert the last covariance matrix to correlation
        corr_matrix_df = _cov_to_corr(cov_matrix_last)

        if corr_matrix_df is None:
             logger.error(f"Failed to convert last date's EWMA covariance to correlation.")
             return None

        logger.info(f"EWMA correlation matrix calculated for date {last_date} ({corr_matrix_df.shape[0]}x{corr_matrix_df.shape[1]})")
        return corr_matrix_df

    except KeyError:
        logger.error(f"KeyError: Could not extract EWMA covariance matrix for date {last_date}. "
                     "Likely issue with MultiIndex access or data length relative to span/min_periods.")
        return None
    except Exception as e:
        logger.error(f"Error calculating point-in-time EWMA correlation matrix: {e}", exc_info=True)
        return None

# --- Sector Correlations (Uses a pre-computed correlation matrix) ---
def calculate_sector_correlations(corr_matrix: pd.DataFrame,
                                  metadata_df: pd.DataFrame,
                                  sector_col: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
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

    # Align indices
    common_tickers = corr_matrix.index.intersection(metadata_df.index)
    if len(common_tickers) == 0:
        logger.error("No common tickers found between correlation matrix and metadata.")
        return None, None
    if len(common_tickers) < len(corr_matrix.index):
        logger.warning(f"Metadata missing for {len(corr_matrix.index) - len(common_tickers)} tickers in correlation matrix. "
                       f"Using {len(common_tickers)} common tickers for sector analysis.")

    corr_matrix_filtered = corr_matrix.loc[common_tickers, common_tickers]
    metadata_filtered = metadata_df.loc[common_tickers]

    sectors = metadata_filtered[sector_col].dropna().to_dict()
    if not sectors:
         logger.error(f"No valid sector information found in column '{sector_col}' for common tickers.")
         return None, None
    unique_sectors = sorted(list(set(sectors.values())))

    within_sector_corrs = {}
    across_sector_corrs = pd.DataFrame(index=unique_sectors, columns=unique_sectors, dtype=float)

    for i, sector1 in enumerate(unique_sectors):
        tickers1 = [t for t, s in sectors.items() if s == sector1]
        if len(tickers1) > 1:
            sector_corr = corr_matrix_filtered.loc[tickers1, tickers1]
            within_sector_corrs[sector1] = _mean_upper_triangle(sector_corr)
        else:
            within_sector_corrs[sector1] = np.nan

        for j, sector2 in enumerate(unique_sectors):
            if i == j:
                 across_sector_corrs.loc[sector1, sector2] = within_sector_corrs.get(sector1, np.nan)
                 continue
            if j < i: continue # Symmetric matrix

            tickers2 = [t for t, s in sectors.items() if s == sector2]
            if tickers1 and tickers2:
                cross_corr_block = corr_matrix_filtered.loc[tickers1, tickers2]
                if cross_corr_block.size > 0:
                    mean_cross_corr = np.nanmean(cross_corr_block.values)
                    across_sector_corrs.loc[sector1, sector2] = mean_cross_corr
                    across_sector_corrs.loc[sector2, sector1] = mean_cross_corr
                else:
                    across_sector_corrs.loc[sector1, sector2] = np.nan
                    across_sector_corrs.loc[sector2, sector1] = np.nan
            else:
                across_sector_corrs.loc[sector1, sector2] = np.nan
                across_sector_corrs.loc[sector2, sector1] = np.nan

    within_sector_series = pd.Series(within_sector_corrs, name='Average Within-Sector Correlation')
    logger.info("Sector correlation calculations complete.")
    return (within_sector_series if not within_sector_series.dropna().empty else None,
            across_sector_corrs if not across_sector_corrs.dropna(how='all').empty else None)


# --- Dynamic Correlation (Time Series of Average Rolling Window EWMA Correlation) ---
def calculate_rolling_window_ewma_correlation(returns_df: pd.DataFrame,
                                              window: int,
                                              span: int) -> Optional[pd.Series]:
    """
    Calculates the time series of average correlation based on EWMA
    applied *only* within a rolling window.

    Args:
        returns_df (pd.DataFrame): DataFrame with returns (dates as index, tickers as columns).
                                   Should handle NaNs appropriately before calling.
        window (int): The size of the rolling window used to slice the data.
        span (int): The span parameter for the EWMA calculation within each window.

    Returns:
        Optional[pd.Series]: A Series containing the average correlation for each time point,
                             calculated using EWMA on the preceding window, or None on error.
    """
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        logger.error("Invalid or empty DataFrame provided for rolling window EWMA correlation.")
        return None
    if window <= 1 or window > len(returns_df):
         logger.error(f"Invalid window size {window} for data length {len(returns_df)}.")
         return None
    if span <= 1:
         logger.error(f"Invalid span {span}. Must be > 1.")
         return None
    if returns_df.isnull().values.any():
         logger.warning(f"Input DataFrame for rolling window EWMA correlation contains NaNs. Results may be affected or NaN.")
         # Consider returns_df = returns_df.ffill().fillna(0) if appropriate

    logger.info(f"Calculating rolling window average EWMA correlation with window={window}, span={span}...")

    results = {} # Dictionary to store date: avg_corr pairs
    min_ewm_periods = max(2, span // 2) # Minimum periods for stable EWMA

    # Loop through the data using a rolling window approach
    # Start calculation only when we have a full window
    for i in range(window, len(returns_df) + 1):
        # Define the slice for the current window
        start_index = i - window
        window_slice_df = returns_df.iloc[start_index:i]
        current_date = window_slice_df.index[-1] # Date for which we calculate the metric

        # Ensure slice is long enough for EWMA calculation
        if window_slice_df.shape[0] < min_ewm_periods:
             logger.debug(f"Skipping date {current_date}: slice too short ({window_slice_df.shape[0]}) for span {span} min_periods {min_ewm_periods}.")
             continue

        try:
            # 1. Calculate EWMA covariance ON THE SLICE
            ewma_cov_slice_ts = window_slice_df.ewm(span=span, adjust=False, min_periods=min_ewm_periods).cov(pairwise=True)

            if ewma_cov_slice_ts.empty:
                 logger.debug(f"EWMA covariance calculation empty for slice ending {current_date}.")
                 continue

            # 2. Extract the covariance matrix for the last date OF THE SLICE
            # Use .xs for potentially more robust MultiIndex selection
            cov_matrix_slice_last = ewma_cov_slice_ts.xs(current_date, level=0, drop_level=True)

            if cov_matrix_slice_last.empty or not isinstance(cov_matrix_slice_last, pd.DataFrame):
                 logger.debug(f"Could not extract valid covariance matrix for slice ending {current_date}.")
                 continue

            # 3. Convert to correlation
            corr_matrix_slice = _cov_to_corr(cov_matrix_slice_last)

            if corr_matrix_slice is None:
                 logger.debug(f"Could not convert covariance to correlation for slice ending {current_date}.")
                 continue

            # 4. Calculate average correlation for this window's end date
            avg_corr = _mean_upper_triangle(corr_matrix_slice)
            if not np.isnan(avg_corr): # Only store valid results
                results[current_date] = avg_corr
            else:
                 logger.debug(f"NaN average correlation calculated for slice ending {current_date}.")


        except KeyError:
            # This might happen if the last date isn't in the ewma_cov index for the slice
            logger.debug(f"KeyError accessing EWMA result for slice ending {current_date}.")
            continue
        except Exception as e:
            # Log less severe warnings in the loop to avoid flooding logs
            logger.warning(f"Error calculating rolling window EWMA for slice ending {current_date}: {type(e).__name__}", exc_info=False)
            continue

    if not results:
        logger.warning(f"Rolling window EWMA correlation calculation produced no valid results for window={window}, span={span}.")
        return None

    # Convert results dictionary to a pandas Series
    avg_rolling_ewma_corr = pd.Series(results).sort_index() # Sort by date

    if avg_rolling_ewma_corr.empty:
         logger.warning(f"Rolling window EWMA correlation series is empty after processing for window={window}, span={span}.")
         return None

    logger.info(f"Rolling window average EWMA correlation calculation complete (window={window}, span={span}).")
    return avg_rolling_ewma_corr
