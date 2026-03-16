"""
Defines the 7 ML models and the training logic.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

def get_models(random_state=42):
    """
    Returns a dictionary of the 7 classical ML models we're evaluating.
    """
    models = {
        "Naïve Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state, algorithm="SAMME")
    }
    return models

def train_models(models, X_train, y_train):
    """
    Loops through the model dictionary and fits each one to the training data.
    """
    print("Training models... This might take a bit (SVM is usually the slowest).")
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Failed to train {name}: {e}")
    print("Done. All models are trained.")
    return trained_models
