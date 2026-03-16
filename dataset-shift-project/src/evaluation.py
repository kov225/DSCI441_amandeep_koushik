"""
Model Evaluation and Feature Importance Module

This module provides tools for quantifying model performance across multiple 
statistical dimensions. It includes a unified evaluation framework for 
calculating classification metrics and a utility for identifying informative 
features to guide concept-adjacent shift simulations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def evaluate_models(trained_models, X_test, y_test, shift_type="None", intensity=0.0):
    """
    Evaluates a collection of trained models on a specified test dataset.

    This function iterates through a dictionary of models, generating predictions 
    and probability estimates. It calculates a standardized suite of metrics 
    including Accuracy, Weighted F1-Score, ROC-AUC, and Brier Score. Logic is 
    included to handle edge cases such as single-class test sets and models 
    losing probabilistic output capability.

    Args:
        trained_models (dict): Mapping of model names (str) to fitted model instances.
        X_test (np.ndarray): The feature matrix for evaluation.
        y_test (np.ndarray): The ground truth labels.
        shift_type (str): The name of the dataset shift applied (for recording).
        intensity (float): The magnitude of the shift applied (for recording).

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a model's performance 
                      metrics under the specified experiment conditions.
    """
    results = []
    
    # ROC AUC calculation requires at least one sample from each class (0 and 1)
    has_multiple_classes = len(np.unique(y_test)) > 1
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        # Probabilistic metrics require class probabilities; we fallback to decision 
        # functions or hard predictions if probability mapping is unavailable.
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Simple sigmoid transformation serves as a probability approximation
            decision = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision)) 
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        # Weighted F1 accounts for class imbalance prevalent in the Adult dataset
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        if has_multiple_classes:
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
            
        try:
            brier = brier_score_loss(y_test, y_prob)
        except ValueError:
            brier = np.nan
            
        results.append({
            "Model": name,
            "Shift_Type": shift_type,
            "Intensity": intensity,
            "Accuracy": acc,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "Brier_Score": brier
        })
        
    return pd.DataFrame(results)

def get_top_n_features(X_train, y_train, n=5):
    """
    Identifies the top N most informative features using a Random Forest heuristic.

    By training a shallow ensemble on the training data, we can extract Gini 
    importance scores. These indices are used to selectively corrupt the 
    most critical dimensions during 'Concept-Adjacent' shift simulations.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        n (int): The number of top feature indices to retrieve.

    Returns:
        list: A list of integer indices corresponding to the most importantes features.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Shallow depth keeps computation fast while still surfacing strong linear/non-linear signals
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Argsort provides indices that sort the array; we slice the tail for highest values
    top_n_indices = np.argsort(importances)[::-1][:n].tolist()
    
    return top_n_indices
