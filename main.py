import logging
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys # Import sys for potential early exit
# Import necessary types for hinting
from typing import Optional, Dict

# Import configuration and utility functions
try:
    import config
    from src import utils
    from src import data_loader
    from src import correlation_analysis
    from src import pca_analysis
    from src import visualization
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running this script from the project root directory ('financial_factor_analysis/')")
    print("and that the 'src' directory and all required modules exist.")
    sys.exit(1) # Exit if core modules can't be imported

# --- Setup Logging ---
# Determine log file path relative to the configured output directory
# Handle potential error during config loading or directory creation
try:
    log_file_path = config.OUTPUT_DIR / config.LOG_FILENAME if config.LOG_TO_FILE else None
    utils.setup_logging(log_level=config.LOG_LEVEL,
                        log_format=config.LOG_FORMAT,
                        log_file=log_file_path)
except AttributeError as e:
     print(f"Error accessing configuration for logging: {e}. Is config.py set up correctly?")
     sys.exit(1)
except Exception as e:
     print(f"Error setting up logging: {e}")
     # Continue without file logging if setup fails but allow console logging if possible
     utils.setup_logging(log_level='INFO', log_format=config.LOG_FORMAT, log_file=None)


logger = logging.getLogger(__name__) # Get logger instance

# --- Pipeline Functions ---

def run_correlation_pipeline(returns_df: pd.DataFrame,
                             metadata_df: Optional[pd.DataFrame], # Use Optional from typing
                             dataset_key: str,
                             cfg: dict,
                             dirs: Dict[str, Path]): # Use Dict from typing
    """
    Runs the full correlation analysis pipeline for a single dataset,
    using EWMA for the primary matrix analysis.

    Args:
        returns_df (pd.DataFrame): DataFrame of returns. Should be cleaned (no NaNs).
        metadata_df (Optional[pd.DataFrame]): DataFrame of metadata (can be None).
        dataset_key (str): Identifier for the dataset (e.g., "NIFTY100").
        cfg (dict): The configuration dictionary.
        dirs (Dict[str, Path]): Dictionary containing output directory paths for the dataset.
    """
    logger.info(f"--- Starting Correlation Analysis for {dataset_key} ---")
    analysis_successful = True

    # --- Use EWMA Correlation as the primary matrix ---
    ewma_span = cfg.get('EWMA_SPAN')
    if ewma_span is None:
        logger.error(f"EWMA_SPAN not defined in config for {dataset_key}. Skipping EWMA analysis.")
        return False # Indicate critical failure

    # Ensure returns_df has no NaNs before EWMA calculation
    if returns_df.isnull().values.any():
         logger.warning(f"NaNs detected in returns data for {dataset_key} before EWMA. Attempting ffill/fillna(0).")
         # Apply cleaning similar to PCAnalyzer's prep step if needed
         returns_df_cleaned = returns_df.ffill().fillna(0)
         if returns_df_cleaned.isnull().values.any():
              logger.error(f"Could not clean NaNs from returns data for {dataset_key}. Skipping EWMA.")
              return False
    else:
         returns_df_cleaned = returns_df # Use original if already clean

    ewma_corr_matrix = correlation_analysis.calculate_ewma_correlation_matrix(
        returns_df=returns_df_cleaned, # Use cleaned data
        span=ewma_span
    )

    if ewma_corr_matrix is None:
        logger.error(f"EWMA correlation calculation failed for {dataset_key} (span={ewma_span}). Skipping dependent steps.")
        return False # Indicate failure

    # Save and Plot EWMA Matrix
    ewma_params = {"span": ewma_span}
    if cfg.get('SAVE_NUMERICAL_RESULTS', True):
        fname = utils.generate_filename(f"{dataset_key}_ewma_correlation_matrix", params=ewma_params, extension="csv")
        utils.save_dataframe(ewma_corr_matrix, dirs['corr_num'] / fname)
    if cfg.get('VIZ_SAVE_PLOTS', True):
        fname = utils.generate_filename(f"{dataset_key}_ewma_correlation_heatmap", params=ewma_params, extension=cfg.get('VIZ_PLOT_FORMAT', 'png'))
        visualization.plot_correlation_heatmap(
            corr_matrix=ewma_corr_matrix,
            title=f"{dataset_key} - EWMA Correlation Matrix (Span={ewma_span})",
            output_path=dirs['corr_plots'] / fname,
            figsize=cfg.get('VIZ_FIGSIZE_HEATMAP', (18, 15)),
            save_plot=True
        )

    # --- Sector Correlations (using the EWMA matrix) ---
    if metadata_df is not None:
        sector_col = cfg['DATASETS'][dataset_key].get('metadata_sector_col')
        if sector_col:
            logger.info("Calculating sector correlations using EWMA matrix...")
            # Align metadata index with correlation matrix index/columns
            common_tickers = ewma_corr_matrix.index.intersection(metadata_df.index)
            if len(common_tickers) > 0:
                metadata_aligned = metadata_df.loc[common_tickers]
                corr_aligned = ewma_corr_matrix.loc[common_tickers, common_tickers] # Use EWMA matrix

                within_sector_corr, across_sector_corr = correlation_analysis.calculate_sector_correlations(
                    corr_matrix=corr_aligned, # Pass the EWMA matrix
                    metadata_df=metadata_aligned,
                    sector_col=sector_col
                )

                if cfg.get('SAVE_NUMERICAL_RESULTS', True):
                    if within_sector_corr is not None and not within_sector_corr.empty:
                        fname_within = utils.generate_filename(f"{dataset_key}_within_sector_correlation_ewma", params=ewma_params, extension="csv")
                        utils.save_dataframe(within_sector_corr.to_frame(), dirs['corr_num'] / fname_within)
                    if across_sector_corr is not None and not across_sector_corr.empty:
                        fname_across = utils.generate_filename(f"{dataset_key}_across_sector_correlation_ewma", params=ewma_params, extension="csv")
                        utils.save_dataframe(across_sector_corr, dirs['corr_num'] / fname_across)

                if cfg.get('VIZ_SAVE_PLOTS', True):
                     # Reuse the same plotting function, titles will indicate EWMA implicitly
                     visualization.plot_sector_correlation_summary(
                         within_sector_series=within_sector_corr,
                         across_sector_matrix=across_sector_corr,
                         title_prefix=f"{dataset_key} (EWMA Span {ewma_span})", # Update title prefix
                         output_dir=dirs['corr_plots'],
                         figsize_heatmap=cfg.get('VIZ_FIGSIZE_SECTOR_HEATMAP', (10, 8)),
                         save_plot=True
                     )
            else:
                 logger.warning(f"No common tickers found between EWMA correlation matrix and metadata for {dataset_key}. Skipping sector analysis.")
        else:
            logger.warning(f"Sector column not defined for {dataset_key} in config. Skipping sector correlation analysis.")
    else:
        logger.warning(f"Metadata not loaded for {dataset_key}. Skipping sector correlation analysis.")

    # --- Rolling Correlations (Independent of EWMA/Static choice above) ---
    for window in cfg.get('ROLLING_WINDOW_SIZES', []):
        logger.info(f"Calculating rolling correlation for window={window}...")
        # Use the original (or cleaned) returns_df for rolling calculation
        avg_rolling_corr = correlation_analysis.calculate_rolling_correlations(
            returns_df=returns_df_cleaned, window=window # Use cleaned data here too
        )
        if avg_rolling_corr is not None:
            params = {"window": window}
            if cfg.get('SAVE_NUMERICAL_RESULTS', True):
                fname = utils.generate_filename(f"{dataset_key}_rolling_correlation_avg", params=params, extension="csv")
                utils.save_dataframe(avg_rolling_corr.to_frame(), dirs['corr_num'] / fname)
            if cfg.get('VIZ_SAVE_PLOTS', True):
                 fname_plot = utils.generate_filename(f"{dataset_key}_rolling_correlation_avg_plot", params=params, extension=cfg.get('VIZ_PLOT_FORMAT', 'png'))
                 visualization.plot_rolling_correlation(
                     rolling_corr_series=avg_rolling_corr,
                     window=window,
                     title=f"{dataset_key} - Average Rolling Correlation ({window}-Day Window)",
                     output_path=dirs['corr_plots'] / fname_plot,
                     figsize=cfg.get('VIZ_FIGSIZE_GENERAL', (12, 6)),
                     save_plot=True
                 )
        else:
            logger.warning(f"Rolling correlation calculation failed for window={window}.")
            # Don't necessarily mark overall as failed, but log it
            analysis_successful = False if analysis_successful else False # Keep existing status if already failed

    # --- Optional: Calculate and save static Pearson matrix separately ---
    # You might still want the long-term average correlation for comparison
    # if cfg.get('CALCULATE_STATIC_PEARSON_TOO', False): # Add this flag to config if needed
    #     static_corr_matrix = correlation_analysis.calculate_correlation_matrix(
    #         returns_df, method='pearson'
    #     )
    #     if static_corr_matrix is not None:
    #         fname_static = utils.generate_filename(f"{dataset_key}_static_pearson_correlation_matrix", extension="csv")
    #         utils.save_dataframe(static_corr_matrix, dirs['corr_num'] / fname_static)
            # Optionally plot it too

    logger.info(f"--- Correlation Analysis for {dataset_key} Finished ---")
    return analysis_successful


# =============================================================================
#  PCA Pipeline Function (run_pca_pipeline) - No changes needed here for EWMA
# =============================================================================
def run_pca_pipeline(returns_df: pd.DataFrame,
                     metadata_df: Optional[pd.DataFrame], # Use Optional
                     dataset_key: str,
                     cfg: dict,
                     dirs: Dict[str, Path]): # Use Dict
    """
    Runs the full PCA analysis pipeline for a single dataset.
    PCA uses standardized returns, independent of the correlation matrix choice.

    Args:
        returns_df (pd.DataFrame): DataFrame of returns.
        metadata_df (Optional[pd.DataFrame]): DataFrame of metadata (can be None).
        dataset_key (str): Identifier for the dataset (e.g., "NIFTY100").
        cfg (dict): The configuration dictionary.
        dirs (Dict[str, Path]): Dictionary containing output directory paths for the dataset.
    """
    logger.info(f"--- Starting PCA Analysis for {dataset_key} ---")
    analysis_successful = True

    # Add the current dataset key to config temporarily for PCAnalyzer init
    # This allows PCAnalyzer to access dataset-specific config like sector column name
    cfg['CURRENT_DATASET_KEY'] = dataset_key

    try:
        # PCAnalyzer handles its own data cleaning (ffill, drop low var) and standardization
        analyzer = pca_analysis.PCAnalyzer(
            returns_df=returns_df, # Pass original returns, analyzer cleans it
            config=cfg,
            metadata_df=metadata_df
        )
    except ValueError as e:
         logger.error(f"Failed to initialize PCAnalyzer for {dataset_key}: {e}. Skipping PCA.")
         analysis_successful = False
    except Exception as e:
        logger.error(f"Unexpected error initializing PCAnalyzer for {dataset_key}: {e}", exc_info=True)
        analysis_successful = False

    if not analysis_successful:
         if 'CURRENT_DATASET_KEY' in cfg: del cfg['CURRENT_DATASET_KEY'] # Clean up temp key
         return False # Indicate failure

    # --- 1. Standard PCA ---
    logger.info("Running Standard PCA...")
    std_pca_results = analyzer.perform_standard_pca()
    if std_pca_results:
        if cfg.get('SAVE_NUMERICAL_RESULTS', True):
            logger.debug("Saving Standard PCA numerical results...")
            fname_load = utils.generate_filename(f"{dataset_key}_standard_pca_loadings", extension="csv")
            utils.save_dataframe(std_pca_results['loadings'], dirs['pca_num'] / fname_load)
            fname_trans = utils.generate_filename(f"{dataset_key}_standard_pca_transformed", extension="csv")
            utils.save_dataframe(std_pca_results['transformed_data'], dirs['pca_num'] / fname_trans)
            variance_info = {
                'explained_variance': std_pca_results['explained_variance'],
                'explained_variance_ratio': std_pca_results['explained_variance_ratio'],
                'cumulative_variance_ratio': std_pca_results['cumulative_variance_ratio'],
                'optimal_n_components': std_pca_results['optimal_n_components']
            }
            fname_var = utils.generate_filename(f"{dataset_key}_standard_pca_variance", extension="json")
            utils.save_dict_to_json(variance_info, dirs['pca_num'] / fname_var)

        if cfg.get('VIZ_SAVE_PLOTS', True):
            logger.debug("Generating Standard PCA plots...")
            fname_scree = utils.generate_filename(f"{dataset_key}_standard_pca_scree_plot", extension=cfg.get('VIZ_PLOT_FORMAT', 'png'))
            visualization.plot_scree(
                explained_variance_ratio=std_pca_results['explained_variance_ratio'],
                cumulative_variance_ratio=std_pca_results['cumulative_variance_ratio'],
                variance_threshold=cfg.get('PCA_VARIANCE_THRESHOLD', 0.95),
                title=f"{dataset_key} - Standard PCA Scree Plot",
                output_path=dirs['pca_plots'] / fname_scree,
                figsize=cfg.get('VIZ_FIGSIZE_SCREE', (12, 6)),
                save_plot=True
            )
            n_comp_plot = min(std_pca_results['optimal_n_components'], cfg.get('VIZ_TOP_N_COMPONENTS', 10))
            for i in range(1, n_comp_plot + 1):
                 params = {"component": i}
                 fname_load_plot = utils.generate_filename(f"{dataset_key}_standard_pca_loadings_pc{i}", params=params, extension=cfg.get('VIZ_PLOT_FORMAT', 'png'))
                 visualization.plot_component_loadings(
                     loadings_df=std_pca_results['loadings'],
                     component_number=i,
                     top_n_stocks=cfg.get('VIZ_TOP_N_STOCKS_LOADINGS', 20),
                     title=f"{dataset_key} - Standard PCA Loadings - Component {i}",
                     output_path=dirs['pca_plots'] / fname_load_plot,
                     figsize=cfg.get('VIZ_FIGSIZE_LOADINGS', (15, 8)),
                     save_plot=True
                 )
    else:
        logger.error(f"Standard PCA failed for {dataset_key}.")
        analysis_successful = False

    # --- 2. Hierarchical PCA ---
    if metadata_df is not None: # Only run if metadata is available
        logger.info("Running Hierarchical PCA...")
        hier_pca_results = analyzer.perform_hierarchical_pca()
        if hier_pca_results:
             if cfg.get('SAVE_NUMERICAL_RESULTS', True):
                 logger.debug("Saving Hierarchical PCA numerical results...")
                 # Save global results
                 global_res = hier_pca_results['global_results']
                 fname_glob_load = utils.generate_filename(f"{dataset_key}_hierarchical_pca_global_loadings", extension="csv")
                 utils.save_dataframe(global_res['loadings'], dirs['pca_num'] / fname_glob_load)
                 fname_glob_trans = utils.generate_filename(f"{dataset_key}_hierarchical_pca_global_transformed", extension="csv")
                 utils.save_dataframe(global_res['transformed_data'], dirs['pca_num'] / fname_glob_trans)
                 global_var_info = {
                     'explained_variance_ratio': global_res['explained_variance_ratio'],
                     'cumulative_variance_ratio': global_res['cumulative_variance_ratio'],
                     'optimal_n_components': global_res['optimal_n_components']
                 }
                 fname_glob_var = utils.generate_filename(f"{dataset_key}_hierarchical_pca_global_variance", extension="json")
                 utils.save_dict_to_json(global_var_info, dirs['pca_num'] / fname_glob_var)

                 # Save sector results
                 for sector, sector_res in hier_pca_results['sector_results'].items():
                     safe_sector_name = sector.replace('/', '_').replace(' ', '_') # Make filename safe
                     params = {"sector": safe_sector_name}
                     fname_sec_load = utils.generate_filename(f"{dataset_key}_hierarchical_pca_sector_loadings", params=params, extension="csv")
                     utils.save_dataframe(sector_res['loadings'], dirs['pca_num'] / fname_sec_load)
                     fname_sec_trans = utils.generate_filename(f"{dataset_key}_hierarchical_pca_sector_transformed", params=params, extension="csv")
                     utils.save_dataframe(sector_res['transformed_data'], dirs['pca_num'] / fname_sec_trans)
                     sector_var_info = {
                         'explained_variance_ratio': sector_res['explained_variance_ratio'],
                         'cumulative_variance_ratio': sector_res['cumulative_variance_ratio'],
                         'optimal_n_components': sector_res['optimal_n_components']
                     }
                     fname_sec_var = utils.generate_filename(f"{dataset_key}_hierarchical_pca_sector_variance", params=params, extension="json")
                     utils.save_dict_to_json(sector_var_info, dirs['pca_num'] / fname_sec_var)

             if cfg.get('VIZ_SAVE_PLOTS', True):
                 logger.warning("Hierarchical PCA plotting functions need implementation in visualization.py")
                 # Add calls to hierarchical plotting functions here when implemented
                 # e.g., visualization.plot_hierarchical_global_loadings(...)
                 # e.g., visualization.plot_hierarchical_sector_summary(...)
        else:
            logger.error(f"Hierarchical PCA failed for {dataset_key}.")
            # Don't mark overall analysis as failed just because hierarchical failed,
            # standard PCA might have succeeded. But log the error.
    else:
        logger.warning(f"Metadata not available for {dataset_key}. Skipping Hierarchical PCA.")

    # Clean up temporary config key
    if 'CURRENT_DATASET_KEY' in cfg: del cfg['CURRENT_DATASET_KEY']
    logger.info(f"--- PCA Analysis for {dataset_key} Finished ---")
    return analysis_successful


# =============================================================================
#  Main Execution Block
# =============================================================================
def main():
    """Main function to execute the analysis pipelines for all configured datasets."""
    script_start_time = time.time()
    logger.info("=" * 60)
    logger.info("=== Starting Financial Factor Analysis Script ===")
    logger.info("=" * 60)

    # --- Load Configuration ---
    try:
        # Load config variables into a dictionary for easier access
        cfg = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load configuration from config.py: {e}", exc_info=True)
        sys.exit(1)

    # --- Process Datasets ---
    datasets_to_process = cfg.get('DATASETS', {})
    if not datasets_to_process:
        logger.warning("No datasets defined in config.DATASETS. Exiting.")
        sys.exit(0)

    overall_status = "COMPLETED_SUCCESS" # Assume success unless errors occur

    for dataset_key in datasets_to_process.keys():
        dataset_start_time = time.time()
        logger.info(f"--- Processing dataset: {dataset_key} ---")

        # Create output directories for this dataset
        try:
            output_dirs = utils.create_output_dirs(Path(cfg['OUTPUT_DIR']), dataset_key)
        except Exception as e:
            logger.error(f"Failed to create output directories for {dataset_key}: {e}. Skipping dataset.", exc_info=True)
            overall_status = "FAILED_DIR_CREATION"
            continue # Skip to next dataset

        # --- Load Data ---
        logger.info("Loading data...")
        returns_df, metadata_df = data_loader.load_dataset(cfg, dataset_key)

        # --- Initialize Summary ---
        analysis_summary = {
            "dataset": dataset_key,
            "run_timestamp": datetime.now().isoformat(),
            "config_snapshot": {k:v for k,v in cfg.items() if k != 'DATASETS'}, # Save relevant config
            "data_shape_returns": None,
            "data_shape_metadata": None,
            "time_period": None,
            "status": "STARTED",
            "duration_seconds": None
        }

        # Check if data loading was successful
        if returns_df is None:
            logger.error(f"Failed to load returns data for {dataset_key}. Analysis cannot proceed.")
            analysis_summary["status"] = "FAILED_LOAD_RETURNS"
            overall_status = "FAILED" # Mark overall script as failed if any dataset fails critically
        else:
            # Store data info in summary
            analysis_summary["data_shape_returns"] = returns_df.shape
            analysis_summary["time_period"] = {
                "start": returns_df.index.min().strftime('%Y-%m-%d'),
                "end": returns_df.index.max().strftime('%Y-%m-%d')
            }
            if metadata_df is not None:
                 analysis_summary["data_shape_metadata"] = metadata_df.shape
            else:
                 # Log if metadata was expected but not loaded
                 if cfg['DATASETS'][dataset_key].get('metadata_file'):
                     logger.warning(f"Metadata DataFrame is None for {dataset_key}. Analyses requiring metadata will be skipped.")

            # --- Run Analysis Pipelines ---
            try:
                logger.info("Starting analysis pipelines...")
                # Pass the original returns_df, cleaning is handled within pipelines if needed
                corr_success = run_correlation_pipeline(returns_df, metadata_df, dataset_key, cfg, output_dirs)
                pca_success = run_pca_pipeline(returns_df, metadata_df, dataset_key, cfg, output_dirs)

                if corr_success and pca_success:
                    analysis_summary["status"] = "COMPLETED_SUCCESS"
                else:
                    # Be more specific about partial success if possible
                    if not corr_success and not pca_success:
                         analysis_summary["status"] = "FAILED_BOTH_PIPELINES"
                    elif not corr_success:
                         analysis_summary["status"] = "COMPLETED_PCA_ONLY"
                    elif not pca_success:
                         analysis_summary["status"] = "COMPLETED_CORR_ONLY"
                    else: # Should not happen based on logic, but fallback
                         analysis_summary["status"] = "COMPLETED_WITH_ERRORS"

                    if overall_status == "COMPLETED_SUCCESS": # Only downgrade if not already failed
                        overall_status = "COMPLETED_WITH_ERRORS"

            except Exception as e:
                 logger.critical(f"Unhandled exception during analysis pipeline for {dataset_key}: {e}", exc_info=True)
                 analysis_summary["status"] = f"FAILED_UNHANDLED_EXCEPTION: {type(e).__name__}"
                 overall_status = "FAILED" # Mark overall script as failed

        # --- Finalize Summary ---
        dataset_end_time = time.time()
        analysis_summary["duration_seconds"] = round(dataset_end_time - dataset_start_time, 2)
        fname_summary = f"summary_report_{dataset_key.lower()}.json"
        utils.save_dict_to_json(analysis_summary, output_dirs['base'] / fname_summary)
        logger.info(f"Summary report saved for {dataset_key}. Status: {analysis_summary['status']}")
        logger.info(f"--- Finished processing dataset: {dataset_key} in {analysis_summary['duration_seconds']:.2f} seconds ---")


    # --- Script Completion ---
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    logger.info("=" * 60)
    logger.info(f"=== Financial Factor Analysis Script Finished ===")
    logger.info(f"=== Overall Status: {overall_status} ===")
    logger.info(f"=== Total Duration: {total_duration:.2f} seconds ===")
    logger.info("=" * 60)


if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    main()
