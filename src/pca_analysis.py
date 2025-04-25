import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from typing import Dict, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)

class PCAnalyzer:
    """
    Performs Standard and Hierarchical Principal Component Analysis on financial returns data.
    """

    def __init__(self,
                 returns_df: pd.DataFrame,
                 config: Dict,
                 metadata_df: Optional[pd.DataFrame] = None):
        """
        Initializes the PCAnalyzer.

        Args:
            returns_df (pd.DataFrame): DataFrame of stock returns (dates as index, tickers as columns).
            config (Dict): Configuration dictionary (loaded from config.py).
            metadata_df (Optional[pd.DataFrame]): DataFrame of stock metadata (indexed by ticker).
                                                   Required for hierarchical PCA and interpretation.
        """
        if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
            raise ValueError("Valid returns_df DataFrame is required.")

        self.raw_returns_df = returns_df.copy() # Keep original for reference
        self.config = config
        self.metadata_df = metadata_df.copy() if metadata_df is not None else None
        self.sector_col = config['DATASETS'][config['CURRENT_DATASET_KEY']]['metadata_sector_col'] \
                          if metadata_df is not None and config.get('CURRENT_DATASET_KEY') else None


        self.cleaned_returns_df: Optional[pd.DataFrame] = None
        self.scaled_data: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None # Stores column names after cleaning

        self._standard_pca_results: Optional[Dict] = None
        self._hierarchical_pca_results: Optional[Dict] = None

        self._validate_and_prepare_data()

    def _validate_and_prepare_data(self):
        """
        Validates, cleans, and standardizes the input returns data.
        Handles NaNs and removes low-variance columns.
        """
        logger.info("Validating and preparing returns data...")
        df = self.raw_returns_df.copy()

        # 1. Handle NaNs
        initial_nan_count = df.isnull().sum().sum()
        if initial_nan_count > 0:
            logger.warning(f"Input returns data contains {initial_nan_count} NaN values. Applying forward fill.")
            df = df.ffill()
            # Check for NaNs remaining at the beginning
            remaining_nan = df.isnull().sum().sum()
            if remaining_nan > 0:
                logger.warning(f"{remaining_nan} NaNs remain after ffill (likely at start). Filling with 0.")
                df = df.fillna(0) # Or use df.bfill().ffill() for potentially better handling

        # 2. Remove low-variance columns (potential for near-zero std dev)
        min_variance = 1e-10 # Define a threshold for near-zero variance
        variances = df.var()
        low_variance_cols = variances[variances < min_variance].index.tolist()
        if low_variance_cols:
            logger.warning(f"Removing {len(low_variance_cols)} columns with variance < {min_variance}: "
                           f"{', '.join(low_variance_cols)}")
            df = df.drop(columns=low_variance_cols)

        if df.empty or df.shape[1] == 0:
             logger.error("No valid data remaining after cleaning.")
             raise ValueError("No valid data remaining after cleaning.")

        self.cleaned_returns_df = df
        self.feature_names = df.columns.tolist() # Store the final column names

        # 3. Standardize the data (mean=0, std=1)
        logger.info("Standardizing cleaned returns data...")
        self.scaler = StandardScaler()
        try:
            self.scaled_data = self.scaler.fit_transform(self.cleaned_returns_df)
            logger.info(f"Data standardized. Shape: {self.scaled_data.shape}")
        except Exception as e:
            logger.error(f"Error during data scaling: {e}", exc_info=True)
            raise

    def _run_pca_instance(self, data: np.ndarray, n_components: Optional[int] = None) -> Tuple[PCA, int]:
        """Helper function to run PCA and determine optimal components."""
        # Fit PCA initially to determine optimal number of components
        pca_full = PCA(n_components=None, random_state=self.config.get('PCA_RANDOM_STATE'))
        pca_full.fit(data)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        variance_threshold = self.config.get('PCA_VARIANCE_THRESHOLD', 0.95)
        max_components_config = self.config.get('PCA_MAX_COMPONENTS') # Can be None

        # Find the number of components needed to reach the threshold
        optimal_n = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Apply max_components limit if set
        if max_components_config is not None:
            optimal_n = min(optimal_n, max_components_config)

        # Ensure optimal_n is not more than available features
        optimal_n = min(optimal_n, data.shape[1])

        logger.debug(f"Variance threshold ({variance_threshold:.1%}) reached with {optimal_n} components.")

        # Re-run PCA with the determined optimal number of components
        pca_optimal = PCA(n_components=optimal_n, random_state=self.config.get('PCA_RANDOM_STATE'))
        pca_optimal.fit(data)

        return pca_optimal, optimal_n


    def perform_standard_pca(self) -> Optional[Dict]:
        """
        Performs standard PCA on the entire dataset.

        Returns:
            Optional[Dict]: A dictionary containing PCA results, or None if an error occurs.
                            Keys include: 'explained_variance_ratio', 'cumulative_variance_ratio',
                            'components' (eigenvectors), 'loadings', 'transformed_data',
                            'optimal_n_components', 'pca_object', 'scaler_object'.
        """
        if self.scaled_data is None or self.feature_names is None or self.cleaned_returns_df is None:
            logger.error("Data not prepared. Run _validate_and_prepare_data first.")
            return None

        logger.info("Performing Standard PCA...")
        try:
            pca, optimal_n = self._run_pca_instance(self.scaled_data)

            # Calculate loadings: Correlation between original variables and principal components
            # Loadings = Eigenvectors * sqrt(Eigenvalues)
            # Note: pca.explained_variance_ are the eigenvalues
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            loadings_df = pd.DataFrame(
                loadings,
                index=self.feature_names,
                columns=[f'PC{i+1}' for i in range(optimal_n)]
            )

            transformed_data = pca.transform(self.scaled_data)
            transformed_df = pd.DataFrame(
                transformed_data,
                index=self.cleaned_returns_df.index,
                columns=[f'PC{i+1}' for i in range(optimal_n)]
            )

            self._standard_pca_results = {
                'explained_variance': pca.explained_variance_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_, # Eigenvectors (rows are components)
                'loadings': loadings_df,
                'transformed_data': transformed_df,
                'optimal_n_components': optimal_n,
                'pca_object': pca,
                'scaler_object': self.scaler # Store the scaler used
            }
            logger.info(f"Standard PCA completed. Optimal components: {optimal_n}. "
                        f"Variance explained: {self._standard_pca_results['cumulative_variance_ratio'][-1]:.2%}")
            return self._standard_pca_results

        except Exception as e:
            logger.error(f"Error during Standard PCA: {e}", exc_info=True)
            self._standard_pca_results = None
            return None

    def perform_hierarchical_pca(self) -> Optional[Dict]:
        """
        Performs hierarchical PCA, first within sectors, then on sector factors.

        Requires metadata with sector information to be provided during initialization.

        Returns:
            Optional[Dict]: Dictionary containing 'sector_results' and 'global_results',
                            or None if metadata is missing or an error occurs.
        """
        if self.metadata_df is None or self.sector_col is None or self.sector_col not in self.metadata_df.columns:
            logger.error("Metadata with a valid sector column is required for Hierarchical PCA.")
            return None
        if self.cleaned_returns_df is None:
             logger.error("Cleaned returns data not available for Hierarchical PCA.")
             return None

        logger.info("Performing Hierarchical PCA...")

        # Ensure metadata index aligns with cleaned returns columns
        common_tickers = self.cleaned_returns_df.columns.intersection(self.metadata_df.index)
        if len(common_tickers) < len(self.cleaned_returns_df.columns):
             logger.warning(f"Metadata missing for {len(self.cleaned_returns_df.columns) - len(common_tickers)} tickers. "
                           f"Using {len(common_tickers)} common tickers for hierarchical analysis.")
        metadata_filtered = self.metadata_df.loc[common_tickers]
        returns_filtered = self.cleaned_returns_df[common_tickers]

        # Group tickers by sector
        sectors_map = metadata_filtered.groupby(self.sector_col).groups
        min_sector_size = self.config.get('HIERARCHICAL_PCA_MIN_SECTOR_SIZE', 5)

        sector_results = {}
        sector_factors_list = [] # List to collect transformed data from each sector PCA

        # --- Stage 1: PCA within each sector ---
        for sector, tickers in sectors_map.items():
            tickers = list(tickers) # Ensure it's a list
            if len(tickers) < min_sector_size:
                logger.warning(f"Skipping sector '{sector}': Only {len(tickers)} stocks "
                               f"(minimum required: {min_sector_size}).")
                continue

            logger.info(f"Performing PCA for sector '{sector}' ({len(tickers)} stocks)...")
            sector_returns = returns_filtered[tickers]

            # Standardize sector data
            sector_scaler = StandardScaler()
            try:
                scaled_sector_data = sector_scaler.fit_transform(sector_returns)

                # Run PCA for the sector
                pca_sector, optimal_n_sector = self._run_pca_instance(scaled_sector_data)

                # Calculate sector loadings
                sector_loadings = pca_sector.components_.T * np.sqrt(pca_sector.explained_variance_)
                sector_loadings_df = pd.DataFrame(
                    sector_loadings,
                    index=tickers,
                    columns=[f'{sector}_PC{i+1}' for i in range(optimal_n_sector)]
                )

                # Get transformed data (sector factors)
                transformed_sector_data = pca_sector.transform(scaled_sector_data)
                transformed_sector_df = pd.DataFrame(
                    transformed_sector_data,
                    index=sector_returns.index,
                    columns=[f'{sector}_PC{i+1}' for i in range(optimal_n_sector)]
                )

                sector_results[sector] = {
                    'explained_variance_ratio': pca_sector.explained_variance_ratio_,
                    'cumulative_variance_ratio': np.cumsum(pca_sector.explained_variance_ratio_),
                    'components': pca_sector.components_,
                    'loadings': sector_loadings_df,
                    'transformed_data': transformed_sector_df,
                    'optimal_n_components': optimal_n_sector,
                    'pca_object': pca_sector,
                    'scaler_object': sector_scaler,
                    'tickers': tickers
                }
                # Append the transformed data (sector factors) for the next stage
                sector_factors_list.append(transformed_sector_df)
                logger.info(f"Sector '{sector}' PCA complete. Optimal components: {optimal_n_sector}. "
                            f"Variance explained: {sector_results[sector]['cumulative_variance_ratio'][-1]:.2%}")

            except Exception as e:
                 logger.error(f"Error during PCA for sector '{sector}': {e}", exc_info=True)
                 continue # Skip to next sector on error

        if not sector_factors_list:
             logger.error("No sectors had sufficient stocks for Stage 1 Hierarchical PCA.")
             return None

        # --- Stage 2: PCA on combined sector factors ---
        logger.info("Performing Stage 2 Hierarchical PCA on combined sector factors...")
        try:
            # Combine factors, ensuring alignment by date index
            combined_sector_factors = pd.concat(sector_factors_list, axis=1)
            # Drop rows with any NaNs that might arise from concatenation if indices aren't perfectly aligned
            combined_sector_factors = combined_sector_factors.dropna()

            if combined_sector_factors.empty or combined_sector_factors.shape[1] < 2:
                logger.error("Not enough combined sector factors for Stage 2 PCA.")
                return None

            # Standardize the combined factors
            global_scaler = StandardScaler()
            scaled_global_factors = global_scaler.fit_transform(combined_sector_factors)

            # Run PCA on the scaled factors
            pca_global, optimal_n_global = self._run_pca_instance(scaled_global_factors)

            # Calculate global loadings (loadings on the sector factors)
            global_loadings = pca_global.components_.T * np.sqrt(pca_global.explained_variance_)
            global_loadings_df = pd.DataFrame(
                global_loadings,
                index=combined_sector_factors.columns, # Index are the sector_PC names
                columns=[f'Global_PC{i+1}' for i in range(optimal_n_global)]
            )

            # Get transformed global factors
            transformed_global_data = pca_global.transform(scaled_global_factors)
            transformed_global_df = pd.DataFrame(
                transformed_global_data,
                index=combined_sector_factors.index,
                columns=[f'Global_PC{i+1}' for i in range(optimal_n_global)]
            )

            global_results = {
                'explained_variance_ratio': pca_global.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca_global.explained_variance_ratio_),
                'components': pca_global.components_,
                'loadings': global_loadings_df, # Loadings on sector PCs
                'transformed_data': transformed_global_df, # The final global factors
                'optimal_n_components': optimal_n_global,
                'pca_object': pca_global,
                'scaler_object': global_scaler
            }

            self._hierarchical_pca_results = {
                'sector_results': sector_results,
                'global_results': global_results
            }
            logger.info(f"Hierarchical PCA Stage 2 complete. Optimal global components: {optimal_n_global}. "
                        f"Variance explained: {global_results['cumulative_variance_ratio'][-1]:.2%}")
            return self._hierarchical_pca_results

        except Exception as e:
            logger.error(f"Error during Stage 2 Hierarchical PCA: {e}", exc_info=True)
            self._hierarchical_pca_results = None
            return None


    def get_standard_pca_results(self) -> Optional[Dict]:
        """Returns the results of the standard PCA."""
        if self._standard_pca_results is None:
            logger.warning("Standard PCA has not been run or failed. Returning None.")
        return self._standard_pca_results

    def get_hierarchical_pca_results(self) -> Optional[Dict]:
        """Returns the results of the hierarchical PCA."""
        if self._hierarchical_pca_results is None:
            logger.warning("Hierarchical PCA has not been run or failed. Returning None.")
        return self._hierarchical_pca_results

