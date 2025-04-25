# Financial Market Factor Analysis using PCA and Correlation

## Overview

This project provides a framework for analyzing the underlying structure of financial markets using stock return data. It focuses on two primary techniques: **Correlation Analysis** and **Principal Component Analysis (PCA)**, applied to datasets like the S&P 500 and NIFTY 100.

The goal is to uncover key insights into:
* How stocks move together (correlation).
* How these relationships change over time (dynamic correlation).
* The main underlying factors driving market variance (PCA).
* Sector-specific vs. market-wide dynamics (Hierarchical PCA).

The analyses performed serve as valuable inputs for quantitative strategies, risk management, and portfolio construction techniques like eigenportfolios.

## Key Analyses Performed

1.  **Correlation Analysis:**
    * **Static Correlation:** Computes the standard Pearson correlation matrix across all stocks for the entire period. Visualized as a heatmap.
    * **Sector Correlations:** Calculates the average correlation *within* each GICS sector and the average correlation *between* different sectors. Helps understand sector cohesion and inter-sector relationships.
    * **Dynamic Correlation:** Implements rolling window correlations (e.g., 6-month and 1-year windows) to track how the average pairwise correlation evolves, providing insights into market regime changes and correlation stability.

2.  **Principal Component Analysis (PCA):**
    * **Standard PCA:** Applies PCA to the standardized returns of the entire stock universe.
        * Determines the number of principal components (PCs) required to explain a specified variance threshold (e.g., 95%).
        * Extracts eigenvalues (variance explained by each PC) and eigenvectors (the PCs themselves).
        * Analyzes the loadings of the most significant PCs to understand which stocks contribute most to these underlying factors.
    * **Hierarchical PCA:** A two-stage approach:
        * *Stage 1:* Performs PCA separately *within* each GICS sector.
        * *Stage 2:* Performs PCA on the principal components derived from the sectors to identify broader, cross-sector factors. Compares these factors to the standard market-wide PCs.

## Project Structure

financial_factor_analysis/│├── .gitignore              # Git ignore file├── README.md               # This file├── requirements.txt        # Python dependencies├── config.py               # Configuration settings (paths, parameters)│├── data/                   # Input data directory│   ├── nifty100/│   │   ├── NIFTY100_returns_processed.csv│   │   └── NIFTY100_metadata.csv│   └── sp500/│       ├── SP500_returns_processed.csv│       └── SP500_metadata.csv│├── notebooks/              # Optional: Jupyter notebooks for exploration│   └── exploratory_analysis.ipynb│├── output/                 # Generated results directory│   ├── nifty100/           # Results for NIFTY100 dataset│   │   ├── correlation/│   │   │   ├── plots/      # Correlation plots│   │   │   └── numerical/  # Correlation numerical data (CSV)│   │   ├── pca/│   │   │   ├── plots/      # PCA plots│   │   │   └── numerical/  # PCA numerical data (CSV, JSON)│   │   └── summary_report_nifty100.json # Summary of the analysis run│   └── sp500/              # Results for S&P 500 dataset│       ├── ... (similar structure) ...│       └── summary_report_sp500.json│├── src/                    # Source code│   ├── init.py│   ├── data_loader.py      # Data loading functions│   ├── correlation_analysis.py # Correlation calculation functions│   ├── pca_analysis.py       # PCA calculation class/functions│   ├── visualization.py    # Plotting functions│   └── utils.py            # Utility functions (saving, logging, etc.)│└── main.py                 # Main execution script
## Prerequisites

* Python 3.8 or higher
* `pip` and `venv` (or your preferred package/environment manager)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd financial_factor_analysis
    ```

2.  **Set up Virtual Environment:**
    It's highly recommended to use a virtual environment:
    ```bash
    # Create the environment (named 'venv')
    python -m venv venv
    # Activate the environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows (Command Prompt/PowerShell):
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    * Obtain the necessary CSV files for returns and metadata (e.g., `NIFTY100_returns_processed.csv`, `NIFTY100_metadata.csv`, etc.).
    * Place these files into the correct subdirectories within the `data/` folder (e.g., `data/nifty100/`, `data/sp500/`).

5.  **Configure Analysis:**
    * Open the `config.py` file.
    * **Crucially, verify and update `INPUT_DATA_DIR` and `OUTPUT_DIR`** if your data or desired output location differs from the default project structure.
    * Adjust filenames within the `DATASETS` dictionary if yours are different.
    * Review and modify other parameters (e.g., `ROLLING_WINDOW_SIZES`, `PCA_VARIANCE_THRESHOLD`, visualization settings) as needed for your specific analysis goals.

## Running the Analysis

Execute the main script from the root directory of the project (`financial_factor_analysis/`):

```bash
python main.py
The script will perform the analyses configured in config.py for each specified dataset. Progress will be logged to the console (and optionally a file, based on config.py).OutputThe analysis generates several outputs saved in the output/ directory, organized by dataset:Correlation Plots: Heatmaps of static correlation, rolling average correlation plots, sector correlation summaries.Correlation Numerical Data: CSV files containing correlation matrices, sector averages.PCA Plots: Scree plots, component loading plots.PCA Numerical Data: CSV/JSON files containing eigenvalues, eigenvectors (components), loadings, transformed data.Summary Report: A JSON file (summary_report_<dataset>.json) summarizing the configuration and key details of each analysis run.ContributingContributions