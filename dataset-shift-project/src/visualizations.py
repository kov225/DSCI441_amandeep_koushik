"""
Functions for plotting result visualizations like curves and heatmaps.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_performance_curves(results_df, shift_type, metric):
    """
    Shows how a metric degrades as shift intensity increases for each model.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    # Standard 10x6 plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot model lines
    sns.lineplot(
        data=df_filtered, 
        x="Intensity", 
        y=metric, 
        hue="Model", 
        marker="o",
        ax=ax,
        linewidth=2.5,
        markersize=8
    )
    
    ax.set_title(f"{metric} Degradation under {shift_type}", fontsize=14, pad=15)
    ax.set_xlabel("Shift Intensity", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Legend on the side
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return fig

def plot_model_comparison_heatmap(results_df, shift_type, metric):
    """
    Heatmap to compare all models at different intensities for a specific shift.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    # Matrix for the heatmap
    pivot_df = df_filtered.pivot(index="Model", columns="Intensity", values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pick a color scale (inverted for Brier score where lower is better)
    if metric == "Brier_Score":
        cmap = "YlOrRd"
    else:
        cmap = "YlGnBu"
        
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        ax=ax,
        cbar_kws={'label': metric},
        linewidths=0.5
    )
    
    ax.set_title(f"Model Comparison: {metric} under {shift_type}", fontsize=14, pad=15)
    plt.tight_layout()
    
    return fig
