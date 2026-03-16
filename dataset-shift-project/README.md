# Dataset Shift: Evaluating Classical ML Robustness

A comparative analysis of how seven classical machine learning models respond to distribution changes using the UCI Adult Income dataset.

---

## Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Dataset Shift Types](#dataset-shift-types)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Results Preview](#results-preview)
- [References](#references)

---

## Motivation
In real-world machine learning deployments, the assumption that training and testing data come from the same distribution (i.i.d.) is frequently violated. This phenomenon, known as **Dataset Shift**, can lead to significant performance degradation when a model encounters environments different from its training context. Understanding which classical models are inherently more robust to specific types of shift is critical for building reliable AI systems.

## Dataset
This project utilizes the **UCI Adult Income Dataset** (OpenML ID: [1590](https://www.openml.org/d/1590)), also known as the "Census Income" dataset. It contains demographic information used to predict whether an individual's annual income exceeds $50,000.
- **Samples**: ~48,842
- **Features**: 14 (Age, Workclass, Education, Occupation, etc.)
- **Task**: Binary Classification

## Models Evaluated
We evaluate seven fundamental machine learning architectures to compare their stability under distribution changes:

| Model | Key Property |
| :--- | :--- |
| **Naïve Bayes** | High bias, assumes feature independence. |
| **Logistic Regression** | Linear decision boundary with probabilistic output. |
| **SVM (RBF Kernel)** | Non-linear boundary using the kernel trick. |
| **Decision Tree** | Interpretable, non-parametric hierarchy. |
| **Random Forest** | Ensemble of trees reducing variance via bagging. |
| **Gradient Boosting** | Sequential ensemble focusing on residual errors. |
| **AdaBoost** | Iteratively re-weights samples based on error. |

## Dataset Shift Types
We simulate three distinct categories of dataset shift with varying levels of intensity:

### 1. Covariate Shift
Occurs when the distribution of the independent variables $P(X)$ changes between training and testing, but the conditional distribution $P(Y|X)$ remains constant. We simulate this by injecting Gaussian noise and scaling continuous features.

### 2. Prior Probability Shift
Occurs when the class distribution $P(Y)$ changes, while $P(X|Y)$ remains constant. This is simulated by aggressively resampling the test set to significantly alter the class balance toward the majority class.

### 3. Concept-Adjacent Shift
A variation of concept shift where the relationship between features and labels is disrupted. We simulate this by corrupting the most informative features (as determined by a Random Forest) for a subset of the test samples.

## Evaluation Metrics
To capture a holistic view of model degradation, we track four primary metrics:
- **Accuracy**: Overall correctness frequency.
- **F1-Score**: Weighted harmonic mean of precision and recall.
- **ROC-AUC**: Ability to discriminate between classes across thresholds.
- **Brier Score**: Measures the accuracy of probabilistic predictions (lower is better).

## Project Structure
```text
dataset-shift-project/
├── app.py                # Streamlit dashboard for interactive exploration
├── requirements.txt      # Project dependencies
├── results/              # CSV storage for experimental results
│   └── experiment_results.csv
└── src/
    ├── data_loader.py    # Fetches and preprocesses the UCI dataset
    ├── evaluation.py     # Metric calculation and feature importance
    ├── models.py         # Model instantiation and training logic
    ├── run_experiments.py # Pipeline execution script
    ├── shift_simulators.py # Algorithms for simulating data shifts
    └── visualizations.py  # Plotting logic for curves and heatmaps
```

## Installation and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
Execute the experimental pipeline to generate results:
```bash
python src/run_experiments.py
```

### 3. Launch Dashboard
Visualize the results interactively:
```bash
streamlit run app.py
```

## Results Preview
The following figures illustrate the performance decay observed across models as shift intensity increases.

*Placeholder: Run the experiment and view the Streamlit app for dynamic visualizations.*

## References
1. Quinonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.
2. Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid*. KDD.
3. UCI Machine Learning Repository. *Adult Income Dataset*.
