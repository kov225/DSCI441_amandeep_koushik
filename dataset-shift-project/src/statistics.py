"""
Statistical Analysis and Distribution Shift Measurement Module

This module provides tools for quantifying the statistical divergence between 
data distributions. It specifically implements the Kolmogorov-Smirnov (KS) test 
to measure feature-level drift during dataset shift simulations, providing a 
mathematical basis for intensity measurements.
"""

import numpy as np
from scipy.stats import ks_2samp

def calculate_feature_drift(X_baseline, X_shifted, feature_indices):
    """
    Quantifies the distribution shift for specific features using the KS test.

    The Kolmogorov-Smirnov test is computed for each specified feature between 
     the baseline and the shifted datasets. This measures the maximum distance 
    between the empirical cumulative distribution functions, providing a value 
    between 0 (identical) and 1 (completely divergent).

    Args:
        X_baseline (np.ndarray): The original, uncorrupted feature matrix.
        X_shifted (np.ndarray): The feature matrix after shift simulation.
        feature_indices (list): Indices of features to analyze (typically continuous).

    Returns:
        dict: A dictionary containing the average KS statistic and individual 
              feature p-values for granular analysis.
    """
    if X_baseline is None or X_shifted is None:
        return {"avg_ks": 0.0, "max_ks": 0.0}
        
    ks_stats = []
    
    for idx in feature_indices:
        # We compare the same feature across the two samples
        stat, p_val = ks_2samp(X_baseline[:, idx], X_shifted[:, idx])
        ks_stats.append(stat)
        
    avg_ks = np.mean(ks_stats) if ks_stats else 0.0
    max_ks = np.max(ks_stats) if ks_stats else 0.0
    
    return {
        "avg_ks": avg_ks,
        "max_ks": max_ks
    }
