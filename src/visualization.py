import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union # Ensure necessary types are imported

# Use relative import for utils within the same package (src)
try:
    from . import utils
except ImportError:
    # Fallback for scenarios where the script might be run differently
    # (e.g., directly for testing, though not recommended for the project structure)
    import utils
    print("Warning: Using fallback import for utils. Ensure running from project root.")

logger = logging.getLogger(__name__)

# --- Plotting Style Configuration ---
# Central place to set style defaults if desired, or rely on external settings
plt.style.use('seaborn-v0_8-whitegrid') # Example style

# --- Correlation Plots ---

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                             title: str,
                             output_path: Path,
                             figsize: tuple = (18, 15),
                             cmap: str = 'coolwarm',
                             vmin: float = -1.0,
                             vmax: float = 1.0,
                             annot: bool = False,
                             save_plot: bool = True,
                             fmt: str = ".2f",
                             **kwargs): # Allow passing extra kwargs to heatmap
    """
    Plots and saves a heatmap of a correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to plot.
        title (str): The title for the plot.
        output_path (Path): The full path (including filename) to save the plot.
        figsize (tuple): Figure size.
        cmap (str): Colormap for the heatmap.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        annot (bool): Whether to annotate cells with values (use False for large matrices).
        save_plot (bool): Whether to save the plot to file.
        fmt (str): String format for annotations if annot is True.
        **kwargs: Additional keyword arguments passed to sns.heatmap.
    """
    if not isinstance(corr_matrix, pd.DataFrame) or corr_matrix.empty:
        logger.error("Cannot plot heatmap: Invalid or empty correlation matrix provided.")
        return

    logger.info(f"Generating heatmap: {title}")
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.heatmap(corr_matrix,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    square=False, # Allow non-square aspect ratio
                    annot=annot,
                    fmt=fmt,
                    linewidths=0.5,
                    linecolor='lightgrey', # Faint lines between cells
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                    ax=ax,
                    **kwargs) # Pass extra arguments
        ax.set_title(title, fontsize=16, pad=20)

        # Adjust tick labels for readability based on matrix size
        num_ticks = len(corr_matrix)
        if num_ticks > 50:
            ax.tick_params(axis='x', labelsize=8, rotation=90)
            ax.tick_params(axis='y', labelsize=8)
        elif num_ticks > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        else: # Keep default for small matrices
            ax.tick_params(axis='x', rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)


        plt.tight_layout(pad=1.5) # Add padding

        if save_plot:
            utils.save_plot(fig, output_path) # utils handles directory creation
        else:
            plt.show() # Show plot if not saving

    except Exception as e:
        logger.error(f"Error generating heatmap '{title}': {e}", exc_info=True)
    finally:
        plt.close(fig) # Ensure figure is closed


def plot_rolling_correlation(rolling_corr_series: pd.Series,
                             window: int,
                             title: str,
                             output_path: Path,
                             figsize: tuple = (12, 6),
                             save_plot: bool = True):
    """
    Plots the average rolling correlation over time.

    Args:
        rolling_corr_series (pd.Series): Series with average rolling correlation (index=date).
        window (int): Rolling window size used for context in the title.
        title (str): Plot title.
        output_path (Path): Full path to save the plot.
        figsize (tuple): Figure size.
        save_plot (bool): Whether to save the plot.
    """
    if not isinstance(rolling_corr_series, pd.Series) or rolling_corr_series.empty:
        logger.error("Cannot plot rolling correlation: Invalid or empty Series provided.")
        return

    logger.info(f"Generating rolling correlation plot (window={window}): {title}")
    fig, ax = plt.subplots(figsize=figsize)
    try:
        rolling_corr_series.plot(ax=ax,
                                 xlabel="Date",
                                 ylabel="Average Pairwise Correlation",
                                 alpha=0.9)
        ax.set_title(title, fontsize=14, pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_plot:
            utils.save_plot(fig, output_path)
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error generating rolling correlation plot '{title}': {e}", exc_info=True)
    finally:
        plt.close(fig)


def plot_sector_correlation_summary(within_sector_series: Optional[pd.Series],
                                    across_sector_matrix: Optional[pd.DataFrame],
                                    title_prefix: str,
                                    output_dir: Path,
                                    figsize_bar: tuple = (12, 7),
                                    figsize_heatmap: tuple = (10, 8),
                                    save_plot: bool = True):
    """
    Plots summaries of within and across sector correlations.

    Generates two plots:
    1. Bar chart of average within-sector correlations.
    2. Heatmap of average across-sector correlations.

    Args:
        within_sector_series (Optional[pd.Series]): Average within-sector correlations.
        across_sector_matrix (Optional[pd.DataFrame]): Average across-sector correlations.
        title_prefix (str): Prefix for plot titles (e.g., "NIFTY100").
        output_dir (Path): Directory to save the plots (filenames are fixed within this dir).
        figsize_bar (tuple): Figure size for the bar chart.
        figsize_heatmap (tuple): Figure size for the heatmap.
        save_plot (bool): Whether to save the plots.
    """
    # --- Within-Sector Bar Plot ---
    if isinstance(within_sector_series, pd.Series) and not within_sector_series.dropna().empty:
        logger.info("Generating within-sector correlation summary plot...")
        fig_bar, ax_bar = plt.subplots(figsize=figsize_bar)
        try:
            plot_data = within_sector_series.dropna().sort_values(ascending=False)
            plot_data.plot(
                kind='bar',
                ax=ax_bar,
                ylabel="Average Correlation",
                xlabel="Sector",
                alpha=0.8,
                width=0.8
            )
            ax_bar.set_title(f"{title_prefix} - Average Within-Sector Correlation", fontsize=14, pad=15)
            # Correct way to set rotation and alignment for bar plots using the axes object
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right')
            ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

            plt.tight_layout()
            if save_plot:
                output_path = output_dir / "within_sector_correlation_summary.png"
                utils.save_plot(fig_bar, output_path)
            else:
                plt.show()
        except Exception as e:
             logger.error(f"Error generating within-sector correlation plot for {title_prefix}: {e}", exc_info=True)
        finally:
             plt.close(fig_bar) # Close figure even if plotting failed
    else:
        logger.warning(f"Skipping within-sector correlation plot for {title_prefix}: No valid data.")

    # --- Across-Sector Heatmap ---
    if isinstance(across_sector_matrix, pd.DataFrame) and not across_sector_matrix.empty:
        logger.info("Generating across-sector correlation heatmap...")
        # Fill diagonal NaNs for visualization
        plot_matrix = across_sector_matrix.copy()
        if isinstance(within_sector_series, pd.Series):
             diag_values = within_sector_series.reindex(plot_matrix.index).values
             np.fill_diagonal(plot_matrix.values, diag_values) # Use within-sector avg on diag
        else:
             np.fill_diagonal(plot_matrix.values, 1.0) # Fallback if within-sector is missing

        # Drop rows/cols that might be all NaN if a sector had no cross-correlations
        plot_matrix = plot_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

        if plot_matrix.empty:
             logger.warning(f"Skipping across-sector correlation heatmap for {title_prefix}: Matrix became empty after dropping NaNs.")
             return # Exit the function

        fig_heat, ax_heat = plt.subplots(figsize=figsize_heatmap)
        try:
            sns.heatmap(plot_matrix,
                        cmap='coolwarm',
                        center=0,
                        vmin=-1, # Adjust vmin/vmax based on typical ranges if needed
                        vmax=1,
                        square=True,
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        linecolor='lightgrey',
                        cbar_kws={"shrink": 0.8, "label": "Average Correlation"},
                        ax=ax_heat)
            ax_heat.set_title(f"{title_prefix} - Average Across-Sector Correlation", fontsize=14, pad=20)
            ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=10)

            plt.tight_layout(pad=1.5)
            if save_plot:
                output_path = output_dir / "across_sector_correlation_heatmap.png"
                utils.save_plot(fig_heat, output_path)
            else:
                plt.show()
        except Exception as e:
             logger.error(f"Error generating across-sector correlation heatmap for {title_prefix}: {e}", exc_info=True)
        finally:
            plt.close(fig_heat) # Close figure even if plotting failed
    else:
        logger.warning(f"Skipping across-sector correlation heatmap for {title_prefix}: No valid data.")


# --- PCA Plots ---

def plot_scree(explained_variance_ratio: np.ndarray,
               cumulative_variance_ratio: np.ndarray,
               variance_threshold: float,
               title: str,
               output_path: Path,
               figsize: tuple = (12, 6),
               save_plot: bool = True):
    """
    Plots the PCA scree plot (individual and cumulative variance explained).

    Args:
        explained_variance_ratio (np.ndarray): Variance explained by each component.
        cumulative_variance_ratio (np.ndarray): Cumulative variance explained.
        variance_threshold (float): The target variance threshold to indicate on the plot.
        title (str): Plot title.
        output_path (Path): Full path to save the plot.
        figsize (tuple): Figure size.
        save_plot (bool): Whether to save the plot.
    """
    if not isinstance(explained_variance_ratio, np.ndarray) or explained_variance_ratio.size == 0:
        logger.error("Cannot plot scree plot: Invalid explained variance data.")
        return
    if not isinstance(cumulative_variance_ratio, np.ndarray) or cumulative_variance_ratio.size == 0:
         logger.error("Cannot plot scree plot: Invalid cumulative variance data.")
         return

    logger.info(f"Generating scree plot: {title}")
    n_components = len(explained_variance_ratio)
    component_numbers = np.arange(1, n_components + 1) # Use numpy arange for component numbers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    try:
        # Individual explained variance
        ax1.bar(component_numbers, explained_variance_ratio, color='steelblue', alpha=0.8, width=0.8)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Variance Explained')
        ax1.set_xticks(component_numbers) # Ensure integer ticks for components
        ax1.tick_params(axis='x', rotation=90, labelsize=8)
        ax1.margins(x=0.02) # Add small horizontal margin
        ax1.grid(axis='y', linestyle='--', alpha=0.6)

        # Cumulative explained variance
        ax2.plot(component_numbers, cumulative_variance_ratio, marker='o', linestyle='-', color='darkorange')
        ax2.axhline(y=variance_threshold, color='r', linestyle='--',
                    label=f'{variance_threshold:.1%} Variance Threshold')
        # Mark the optimal number of components (first point >= threshold)
        optimal_indices = np.where(cumulative_variance_ratio >= variance_threshold)[0]
        if optimal_indices.size > 0:
            optimal_n = optimal_indices[0] + 1
            ax2.axvline(x=optimal_n, color='g', linestyle=':',
                        label=f'{optimal_n} Components ≥ Threshold')

        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Variance Explained')
        ax2.set_xticks(component_numbers)
        ax2.tick_params(axis='x', rotation=90, labelsize=8)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.margins(x=0.02)

        plt.suptitle(title, fontsize=16, y=1.03) # Use suptitle for overall title
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent suptitle overlap

        if save_plot:
            utils.save_plot(fig, output_path)
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error generating scree plot '{title}': {e}", exc_info=True)
    finally:
        plt.close(fig)


def plot_component_loadings(loadings_df: pd.DataFrame,
                            component_number: int,
                            top_n_stocks: int,
                            title: str,
                            output_path: Path,
                            figsize: tuple = (15, 8),
                            save_plot: bool = True):
    """
    Plots the loadings for a single principal component, showing top positive/negative contributors.

    Args:
        loadings_df (pd.DataFrame): DataFrame of loadings (index=tickers, columns=PCs).
        component_number (int): The component number (1-based index) to plot.
        top_n_stocks (int): Number of top positive AND top negative loading stocks to show.
        title (str): Plot title.
        output_path (Path): Full path to save the plot.
        figsize (tuple): Figure size.
        save_plot (bool): Whether to save the plot.
    """
    pc_col = f'PC{component_number}'
    if pc_col not in loadings_df.columns:
        logger.error(f"Cannot plot loadings: Column '{pc_col}' not found in loadings DataFrame.")
        return

    logger.info(f"Generating loadings plot for {pc_col}: {title}")

    # Get loadings for the component and sort by absolute value
    pc_loadings = loadings_df[pc_col].sort_values(key=abs, ascending=False)

    # Select top N positive and top N negative (or fewer if not enough)
    top_positive = pc_loadings[pc_loadings > 0].head(top_n_stocks)
    top_negative = pc_loadings[pc_loadings <= 0].tail(top_n_stocks) # tail for smallest (most negative)

    # Combine and sort for plotting (e.g., positive first, then negative)
    # Sorting by value makes the bar chart flow better
    top_combined = pd.concat([top_positive, top_negative]).sort_values(ascending=False)

    if top_combined.empty:
        logger.warning(f"No loadings found to plot for {pc_col}.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    try:
        colors = ['g' if x > 0 else 'r' for x in top_combined.values]
        top_combined.plot(kind='bar', ax=ax, color=colors, alpha=0.8, width=0.8)

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_ylabel('Loading Value')
        ax.set_xlabel('Stock Ticker')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.tick_params(axis='x', rotation=75, labelsize=9) # Rotate more if needed
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_plot:
            utils.save_plot(fig, output_path)
        else:
            plt.show()

    except Exception as e:
         logger.error(f"Error generating loadings plot '{title}': {e}", exc_info=True)
    finally:
        plt.close(fig)

# --- Placeholder Functions for Hierarchical PCA Plots ---

def plot_hierarchical_global_loadings(global_loadings_df: pd.DataFrame,
                                      title: str,
                                      output_path: Path,
                                      figsize: tuple = (15, 10),
                                      save_plot: bool = True):
    """
    Placeholder: Plots a heatmap of the loadings of global factors on sector factors.
    """
    logger.warning(f"Plotting function 'plot_hierarchical_global_loadings' is not fully implemented.")
    # Implementation similar to plot_correlation_heatmap
    # Use global_loadings_df as input
    # Adjust cmap, vmin, vmax, title as needed
    # Example call:
    # plot_correlation_heatmap(
    #     corr_matrix=global_loadings_df, # Pass the loadings here
    #     title=title,
    #     output_path=output_path,
    #     figsize=figsize,
    #     cmap='viridis', # Or another suitable map
    #     vmin=None, # Auto-scale or set appropriate range
    #     vmax=None,
    #     annot=True, # Usually feasible for fewer global factors
    #     save_plot=save_plot
    # )
    pass # Remove pass when implemented


def plot_hierarchical_sector_summary(sector_results: dict,
                                     title_prefix: str,
                                     output_dir: Path,
                                     figsize: tuple = (12, 7),
                                     save_plot: bool = True):
    """
    Placeholder: Plots a summary of the hierarchical PCA sector results,
                 e.g., variance explained by the first PC in each sector.
    """
    logger.warning(f"Plotting function 'plot_hierarchical_sector_summary' is not fully implemented.")
    # Example: Plot variance explained by PC1 for each sector
    # variance_pc1 = {}
    # for sector, results in sector_results.items():
    #     if 'explained_variance_ratio' in results and len(results['explained_variance_ratio']) > 0:
    #         variance_pc1[sector] = results['explained_variance_ratio'][0]
    # if variance_pc1:
    #     variance_series = pd.Series(variance_pc1).sort_values(ascending=False)
    #     fig, ax = plt.subplots(figsize=figsize)
    #     variance_series.plot(kind='bar', ax=ax, title=f"{title_prefix} - Variance Explained by Sector PC1")
    #     plt.tight_layout()
    #     if save_plot:
    #         utils.save_plot(fig, output_dir / "hierarchical_sector_pc1_variance.png")
    #     plt.close(fig)
    pass # Remove pass when implemented

