import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, List, Tuple, Any

# Get the logger instance from the main script 
logger = logging.getLogger(__name__)

def load_returns_data(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads returns data from a CSV file.

    Args:
        file_path (Path): The absolute path to the returns CSV file.

    Returns:
        pd.DataFrame | None: DataFrame with returns (dates as index, tickers as columns),
                             or None if loading fails.
    """
    if not file_path.is_file():
        logger.error(f"Returns file not found: {file_path}")
        return None
    try:
        returns_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if not isinstance(returns_df.index, pd.DatetimeIndex):
            logger.warning(f"Index of {file_path} is not DatetimeIndex. Attempting conversion.")
            # Attempt conversion or raise error 
            returns_df.index = pd.to_datetime(returns_df.index)
        logger.info(f"Successfully loaded returns data from: {file_path} "
                    f"({returns_df.shape[0]} rows, {returns_df.shape[1]} columns)")
        return returns_df
    except Exception as e:
        logger.error(f"Failed to load returns data from {file_path}: {e}", exc_info=True)
        return None

def load_metadata(file_path: Path, ticker_col: str) -> Optional[pd.DataFrame]:
    """
    Loads metadata from a CSV file.

    Args:
        file_path (Path): The absolute path to the metadata CSV file.
        ticker_col (str): The name of the column containing the stock tickers/symbols.

    Returns:
        pd.DataFrame | None: DataFrame with metadata, indexed by ticker,
                             or None if loading fails.
    """
    if not file_path.is_file():
        logger.error(f"Metadata file not found: {file_path}")
        return None
    try:
        metadata_df = pd.read_csv(file_path)
        # Check if ticker column exists
        if ticker_col not in metadata_df.columns:
            logger.error(f"Ticker column '{ticker_col}' not found in metadata file: {file_path}")
            return None
        # Set ticker column as index
        metadata_df = metadata_df.set_index(ticker_col)
        logger.info(f"Successfully loaded metadata from: {file_path} "
                    f"({metadata_df.shape[0]} entries)")
        return metadata_df
    except Exception as e:
        logger.error(f"Failed to load metadata from {file_path}: {e}", exc_info=True)
        return None

def get_dataset_paths(config: dict, dataset_key: str) -> tuple[Optional[Path], Optional[Path]]:
    """
    Constructs the full paths for a dataset's returns and metadata files.

    Args:
        config (dict): The configuration dictionary (e.g., from config.py).
        dataset_key (str): The key identifying the dataset (e.g., "NIFTY100").

    Returns:
        tuple[Path | None, Path | None]: A tuple containing the returns file path
                                         and metadata file path, or None if not configured.
    """
    if dataset_key not in config['DATASETS']:
        logger.error(f"Dataset key '{dataset_key}' not found in configuration.")
        return None, None

    dataset_config = config['DATASETS'][dataset_key]
    input_dir = Path(config['INPUT_DATA_DIR']) / dataset_key.lower() # Assumes subfolder per dataset

    returns_file = input_dir / dataset_config['returns_file']
    metadata_file = input_dir / dataset_config['metadata_file']

    return returns_file, metadata_file

def load_dataset(config: dict, dataset_key: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads returns and metadata for a specific dataset defined in the config.

    Args:
        config (dict): The configuration dictionary.
        dataset_key (str): The key identifying the dataset.

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: Returns DataFrame and Metadata DataFrame.
                                                        Either can be None if loading fails or
                                                        file is not specified/found.
    """
    logger.info(f"--- Loading dataset: {dataset_key} ---")
    returns_path, metadata_path = get_dataset_paths(config, dataset_key)

    returns_df = load_returns_data(returns_path) if returns_path else None

    metadata_df = None
    if metadata_path:
        ticker_col = config['DATASETS'][dataset_key].get('metadata_ticker_col')
        if ticker_col:
            metadata_df = load_metadata(metadata_path, ticker_col)
        else:
            logger.error(f"Metadata ticker column not specified for dataset '{dataset_key}' in config.")

    return returns_df, metadata_df

