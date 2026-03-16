"""
Functions to simulate different types of dataset shift.
1. Covariate (feature noise)
2. Prior (class balance change)
3. Concept-Adjacent (key feature corruption)
"""

import numpy as np
import pandas as pd
from copy import deepcopy

def apply_covariate_shift(X, continuous_indices, intensity=0.0):
    """
    Adds Gaussian noise and scales continuous features to simulate covariate shift.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    
    num_samples = X_shifted.shape[0]
    num_continuous = len(continuous_indices)
    
    # Generate noise based on the intensity
    noise = np.random.normal(loc=0.0, scale=intensity, size=(num_samples, num_continuous))
    
    # Apply the noise and a slight drift to the selected columns
    for i, col_idx in enumerate(continuous_indices):
        X_shifted[:, col_idx] = X_shifted[:, col_idx] * (1.0 + (intensity * 0.1)) + noise[:, i]
        
    return X_shifted

def apply_prior_shift(X, y, intensity=0.0):
    """
    Subsamples the test set to change the class balance (Prior Probability Shift).
    It increases the proportion of the majority class by dropping minority samples.
    """
    if intensity == 0.0:
        return X, y
        
    # Get class counts
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y 
        
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    
    maj_indices = np.where(y == majority_class)[0]
    min_indices = np.where(y == minority_class)[0]
    
    # Drop minority samples based on the shift intensity
    # We drop up to 95% of them at max intensity
    drop_fraction = min(0.95, intensity) 
    num_keep_min = int(len(min_indices) * (1.0 - drop_fraction))
    
    if num_keep_min == 0:
        num_keep_min = 1 
        
    # Pick which ones to keep randomly
    keep_min_indices = np.random.choice(min_indices, size=num_keep_min, replace=False)
    
    # Rebuild the dataset
    new_indices = np.concatenate([maj_indices, keep_min_indices])
    np.random.shuffle(new_indices)
    
    return X[new_indices], y[new_indices]

def apply_concept_adjacent_shift(X, top_n_indices, intensity=0.0):
    """
    Corrupts the most informative features to simulate concept shift.
    Useful for seeing how models handle the loss of their best predictors.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    num_samples = X_shifted.shape[0]
    
    # Determine how many samples to mess with
    num_corrupt = int(num_samples * min(1.0, intensity))
    
    for col_idx in top_n_indices:
        # Match the original range with noise
        col_min = np.min(X[:, col_idx])
        col_max = np.max(X[:, col_idx])
        
        # Pick random rows to swap out
        corrupt_indices = np.random.choice(num_samples, size=num_corrupt, replace=False)
        
        # Inject random values
        random_noise = np.random.uniform(low=col_min, high=col_max, size=num_corrupt)
        X_shifted[corrupt_indices, col_idx] = random_noise
        
    return X_shifted
