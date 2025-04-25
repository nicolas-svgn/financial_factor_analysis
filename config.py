import os
from pathlib import Path
import logging # Import logging to use its constants like logging.INFO

# --- Core Paths ---
# Determine the project root directory dynamically.
# This assumes config.py is in the project's root directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Input Data Directory: Assumes a 'data' folder in the project root.
# *** USER ACTION REQUIRED: Verify this points to your actual data location ***
# If your data is elsewhere, provide the absolute path: Path("/path/to/your/data")
INPUT_DATA_DIR = PROJECT_ROOT / "data"

# Output Directory: Assumes an 'output' folder in the project root.
# All generated results will be saved here.
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- Dataset Configurations ---
# Define the datasets to be analyzed.
# Keys (e.g., "NIFTY100") are used as identifiers and for output folder names.
DATASETS = {
    "NIFTY100": {
        # --- File Names (relative to INPUT_DATA_DIR / dataset_key.lower()) ---
        "returns_file": "NIFTY100_returns_processed.csv",
        "metadata_file": "NIFTY100_metadata.csv",

        # --- Metadata Column Names (MUST match your CSV headers) ---
        "metadata_ticker_col": "ticker",      # Column containing stock symbols/tickers
        "metadata_sector_col": "sector",      # Column containing sector classification (e.g., GICS)
    },
    "SP500": {
        # --- File Names (relative to INPUT_DATA_DIR / dataset_key.lower()) ---
        "returns_file": "SP500_returns_processed.csv", # Example filename, adjust as needed
        "metadata_file": "SP500_metadata.csv",       # Example filename, adjust as needed

        # --- Metadata Column Names (MUST match your CSV headers) ---
        # *** USER ACTION REQUIRED: Verify these column names for your S&P500 metadata file ***
        "metadata_ticker_col": "Symbol",      # Common name, but verify
        "metadata_sector_col": "GICS Sector", # Common name, but verify
    }
    # Add more dataset configurations here following the same structure
}

# --- Correlation Analysis Parameters ---
CORRELATION_METHOD = 'pearson'          # Method for static correlation ('pearson', 'spearman', 'kendall') - Still useful if calculating separately
EWMA_SPAN = 60                          # Span for Exponentially Weighted Moving Average correlation (e.g., 60 trading days)
ROLLING_WINDOW_SIZES = [126, 252]       # Rolling window periods (trading days) for dynamic analysis
                                        # (Approx. 6 months, 1 year)

# --- PCA Analysis Parameters ---
PCA_METHOD = 'STANDARD'                 # Currently only 'STANDARD' (using sklearn PCA) is implemented
PCA_VARIANCE_THRESHOLD = 0.95           # Target cumulative variance explained by selected components
PCA_MAX_COMPONENTS = None               # Optional: Integer limit for the number of components. None means no limit other than features count.
PCA_RANDOM_STATE = 42                   # Seed for reproducibility in PCA algorithms if applicable

# --- Hierarchical PCA Parameters ---
HIERARCHICAL_PCA_MIN_SECTOR_SIZE = 5    # Minimum number of stocks needed in a sector to perform sector-level PCA

# --- Visualization Parameters ---
VIZ_SAVE_PLOTS = True                   # Master switch: Set to False to disable saving all plots
VIZ_PLOT_FORMAT = 'png'                 # Output format for saved plots ('png', 'pdf', 'svg', etc.)
VIZ_PLOT_DPI = 300                      # Resolution (dots per inch) for saved plots

# Specific plot settings
VIZ_TOP_N_COMPONENTS = 10               # Number of top PCA components for which to plot detailed loadings
VIZ_TOP_N_STOCKS_LOADINGS = 20          # Number of top stocks (positive/negative) to display in loading plots
VIZ_FIGSIZE_GENERAL = (12, 8)           # Default figure size for general plots (e.g., rolling corr)
VIZ_FIGSIZE_HEATMAP = (18, 15)          # Figure size for large correlation heatmaps
VIZ_FIGSIZE_SECTOR_HEATMAP = (10, 8)    # Figure size for across-sector correlation heatmap
VIZ_FIGSIZE_LOADINGS = (15, 8)          # Figure size for individual component loading plots
VIZ_FIGSIZE_SCREE = (12, 6)             # Figure size for scree plots

# --- Output Control ---
SAVE_NUMERICAL_RESULTS = True           # Master switch: Set to False to disable saving numerical results (CSV, JSON)
DETAILED_FILENAMING = True              # If True, include parameters in output filenames for clarity

# --- Logging Configuration ---
LOG_LEVEL = 'INFO'                      # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Log message format
LOG_TO_FILE = True                      # Set to False to log only to the console
LOG_FILENAME = 'financial_analysis.log' # Name of the log file (will be saved in OUTPUT_DIR)

# --- Utility: Ensure Output Directory Exists ---
# This runs when the config module is imported
try:
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")
except Exception as e:
    print(f"Error creating output directory {OUTPUT_DIR}: {e}")
    # Depending on requirements, you might want to exit here if the output dir is critical

