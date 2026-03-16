"""
Helper functions for calculating model performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def evaluate_models(trained_models, X_test, y_test, shift_type="None", intensity=0.0):
    """
    Runs evaluation for a set of models and returns a DataFrame with the results.
    """
    results = []
    
    # Can't do ROC AUC if there is only one class in the test set
    has_multiple_classes = len(np.unique(y_test)) > 1
    
    for name, model in trained_models.items():
        # Get standard predictions
        y_pred = model.predict(X_test)
        
        # Try to get class probabilities for ROC AUC and Brier scoring
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Approximate probabilities using the sigmoid of the decision function
            decision = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision)) 
        else:
            y_prob = y_pred
            
        # Basic accuracy and F1 (weighted for imbalance)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Only calculate ROC AUC if classes permit
        if has_multiple_classes:
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
            
        # Standard Brier score
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
    Uses a quick Random Forest to find the top N most important features.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Simple RF to get feature importances quickly
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Grab the indices of the most informative features
    top_n_indices = np.argsort(importances)[::-1][:n].tolist()
    
    return top_n_indices
