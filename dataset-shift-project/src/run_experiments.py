"""
Script to run the full dataset shift experiment pipeline.
It loads data, trains models, applies shifts, and saves the final results.
"""

import os
import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_data
from models import get_models, train_models
from shift_simulators import apply_covariate_shift, apply_prior_shift, apply_concept_adjacent_shift
from evaluation import evaluate_models, get_top_n_features

def main():
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Prep output folders
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.csv")
    
    print("="*50)
    print("Starting Dataset Shift Experiments")
    print("="*50)
    
    # Load and clean the data
    X_train, X_test, y_train, y_test, continuous_indices, preprocessor = load_and_preprocess_data()
    print(f"Data loaded: Train size={X_train.shape}, Test size={X_test.shape}")
    
    # Get and train our 7 models
    models_dict = get_models()
    trained_models = train_models(models_dict, X_train, y_train)
    
    all_results = pd.DataFrame()
    
    # Baseline run with no shift
    print("\nEvaluating Baseline (No Shift)...")
    baseline_res = evaluate_models(trained_models, X_test, y_test, shift_type="Baseline", intensity=0.0)
    all_results = pd.concat([all_results, baseline_res], ignore_index=True)
    
    # Experiment settings
    intensities = [0.1, 0.25, 0.5, 0.75, 1.0] 
    
    # Identify key features for concept shift
    top_n_indices = get_top_n_features(X_train, y_train, n=3)
    
    # Run the different shift types
    
    # 1. Covariate Shift (adding noise)
    print("\nRunning Covariate Shift Experiments...")
    for intensity in intensities:
        # Scale intensity to standard deviation units
        cov_intensity = intensity * 2.0 
        X_test_shifted = apply_covariate_shift(X_test, continuous_indices, cov_intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Covariate Shift", intensity=cov_intensity)
        all_results = pd.concat([all_results, res], ignore_index=True)
        print(f"  Completed intensity: {cov_intensity:.2f}")
        
    # 2. Prior Probability Shift (changing class balance)
    print("\nRunning Prior Probability Shift Experiments...")
    for intensity in intensities:
        X_test_shifted, y_test_shifted = apply_prior_shift(X_test, y_test, intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test_shifted, shift_type="Prior Probability Shift", intensity=intensity)
        all_results = pd.concat([all_results, res], ignore_index=True)
        print(f"  Completed intensity: {intensity:.2f}")
        
    # 3. Concept-Adjacent Shift (corrupting key features)
    print("\nRunning Concept-Adjacent Shift Experiments...")
    for intensity in intensities:
        X_test_shifted = apply_concept_adjacent_shift(X_test, top_n_indices, intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Concept-Adjacent Shift", intensity=intensity)
        all_results = pd.concat([all_results, res], ignore_index=True)
        print(f"  Completed intensity: {intensity:.2f}")
        
    # Save everything to CSV
    all_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nExperiments finished! Results are in '{RESULTS_FILE}'")

if __name__ == "__main__":
    main()
