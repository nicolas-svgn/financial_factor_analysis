import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Dict, Optional, List, Tuple, Any

# --- Logging Setup ---
def setup_logging(log_level: str = 'INFO',
                  log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                  log_file: Path | None = None):
    """
    Configures logging for the application.

    Args:
        log_level (str): The minimum logging level (e.g., 'DEBUG', 'INFO').
        log_format (str): The format string for log messages.
        log_file (Path | None): Optional path to a file for logging. If None, logs only to console.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)] # Always log to console

    if log_file:
        # Ensure the directory for the log file exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode

    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    logging.info(f"Logging configured. Level: {log_level}. File: {log_file if log_file else 'Console only'}")

# --- Directory and File Handling ---
def create_output_dirs(base_output_dir: Path, dataset_key: str) -> Dict[str, Path]:
    """
    Creates standard output subdirectories for a given dataset.

    Args:
        base_output_dir (Path): The main output directory (from config).
        dataset_key (str): The identifier for the dataset (e.g., "NIFTY100").

    Returns:
        Dict[str, Path]: A dictionary mapping directory types ('base', 'corr_plots',
                         'corr_num', 'pca_plots', 'pca_num') to their Path objects.
    """
    dataset_dir = base_output_dir / dataset_key.lower()
    dirs = {
        'base': dataset_dir,
        'corr_plots': dataset_dir / "correlation" / "plots",
        'corr_num': dataset_dir / "correlation" / "numerical",
        'pca_plots': dataset_dir / "pca" / "plots",
        'pca_num': dataset_dir / "pca" / "numerical",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Ensured output directories exist for {dataset_key} under {base_output_dir}")
    return dirs

def generate_filename(base_name: str,
                      params: Optional[Dict] = None,
                      timestamp: bool = False,
                      extension: str = "png") -> str:
    """
    Generates a detailed filename.

    Args:
        base_name (str): The base identifier for the file (e.g., "correlation_matrix").
        params (Optional[Dict]): Dictionary of parameters to include (e.g., {"window": 126}).
        timestamp (bool): Whether to prepend a timestamp.
        extension (str): The file extension (without dot).

    Returns:
        str: The formatted filename.
    """
    parts = [base_name]
    if params:
        for key, value in params.items():
            parts.append(f"{key}_{str(value).lower()}")
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.insert(0, ts)

    return "_".join(parts) + f".{extension}"

# --- Data Saving ---
def save_dataframe(df: pd.DataFrame, output_path: Path):
    """Saves a pandas DataFrame to a CSV file."""
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Attempted to save non-DataFrame object to {output_path}")
        return
    try:
        df.to_csv(output_path)
        logging.info(f"DataFrame saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {output_path}: {e}", exc_info=True)

def convert_to_serializable(obj):
    """Converts numpy types and other non-serializable objects for JSON."""
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (set, tuple)):
         return list(obj) # Convert sets/tuples to lists
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    # Add more type conversions if needed
    try:
        # Attempt default serialization for unknown types
        json.dumps(obj)
        return obj
    except TypeError:
        logging.warning(f"Object of type {type(obj)} could not be made JSON serializable, converting to string.")
        return str(obj) # Fallback to string representation

def save_dict_to_json(data_dict: dict, output_path: Path, indent: int = 2):
    """Saves a dictionary to a JSON file, handling non-serializable types."""
    if not isinstance(data_dict, dict):
        logging.error(f"Attempted to save non-dict object to {output_path}")
        return
    try:
        serializable_dict = convert_to_serializable(data_dict)
        with open(output_path, 'w') as f:
            json.dump(serializable_dict, f, indent=indent)
        logging.info(f"Dictionary saved to JSON: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save dictionary to JSON {output_path}: {e}", exc_info=True)

def save_plot(fig: plt.Figure, output_path: Path, dpi: int = 300):
    """Saves a matplotlib Figure to a file."""
    if not isinstance(fig, plt.Figure):
         logging.error(f"Attempted to save non-Figure object to {output_path}")
         return
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logging.info(f"Plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {output_path}: {e}", exc_info=True)

