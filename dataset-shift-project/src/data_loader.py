"""
Loads and prepares the UCI Adult Income data. 
Has a synthetic fallback if the online download fails.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Fetches the Adult dataset and applies standard scaling and encoding.
    """
    try:
        print("Grabbing UCI data from OpenML...")
        adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        X = adult.data
        # Label is 1 for >50K, 0 otherwise
        y = (adult.target == '>50K').astype(int) 
        
        # Sort features into categories
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        continuous_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
    except Exception as e:
        warnings.warn(f"OpenML fetch failed ({e}). Falling back to synthetic data.")
        print("Generating synthetic dataset...")
        X, y = make_classification(
            n_samples=10000, 
            n_features=14, 
            n_informative=10, 
            n_redundant=4,
            random_state=random_state,
            weights=[0.76, 0.24] 
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(14)])
        y = pd.Series(y)
        categorical_cols = []
        continuous_cols = X.columns.tolist()

    # Numeric pipeline: fill missing and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: fill missing and encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Standard split with stratification to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Process everything
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Track continuous column indices (they appear first after transformation)
    num_continuous = len(continuous_cols)
    continuous_indices = list(range(num_continuous))

    return X_train_processed, X_test_processed, y_train.values, y_test.values, continuous_indices, preprocessor
