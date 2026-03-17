# Dataset Shift: Longitudinal Robustness Analysis

This repository explores how classical machine learning models respond to distribution changes (dataset shift). We utilize the UCI Adult Income dataset to simulate various environmental stressors and quantify their impact on model reliability.

## Project Structure
- [`dataset-shift-project/`](file:///e:/Math-4/DSCI441_amandeep_koushik/dataset-shift-project): Core experimental codebase and Streamlit application.
- `results/`: Serialization of experimental benchmarks and visualizations.

---

## Milestone 1: Baseline Benchmarking (Completed)
Milestone 1 established a foundational pipeline for quantifying model degradation.
- **Models**: Naïve Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, AdaBoost.
- **Shift Regimes**: Covariate Shift (Noise), Prior Shift (Class Imbalance), Concept-Adjacent Shift (Feature Corruption).
- **Deliverables**: Professionalized codebase and an interactive exploration dashboard.

---

## Milestone 2: Research-Grade Robustness & Interpretability (Roadmap)

### 1. Deeper Shift Simulation
Upgrade simulations to represent more realistic environmental challenges:
- **Covariate Shift**: Addition of feature scaling drift alongside noise injection to distinguish between precision loss and distribution bias.
- **Prior Shift**: Implementation of majority oversampling to analyze asymmetric classification effects.
- **Concept Shift**: Transition to feature permutation (value shuffling) to simulate realistic information loss.

### 2. Robustness Scoring
Transition from raw performance tracking to a unified robustness metric:
- **`robustness_score()`**: Calculation of the AUC degradation rate per unit of shift intensity. This enables objective ranking of models by their "Resilience Factor."

### 3. Interpretability Layer
A dedicated `interpretability.py` module to analyze *why* models fail:
- **Feature Importance Drifts**: Analysis of weight/importance shifts before vs. after distribution change.
- **Decision Boundary Visualization**: 2D projections showing how separation hyperplanes collapse under shift.
- **Confidence Distribution**: Quantifying model calibration and overconfidence during degradation.

### 4. Statistical Validation
Moving beyond single-point estimates to defensible results:
- **Multi-seed Cross-validation**: Aggregating results across 3–5 random initializations.
- **Variance Analysis**: Error bands on all performance curves to visualize statistical significance.

### 5. Streamlit App Upgrade
Evolution of the UI into a research dashboard with three functional tabs:
- **Tab 1: Explorer**: Polished version of the current distribution visualizer.
- **Tab 2: Robustness Leaderboard**: A ranked comparison highlighting the "Best-in-Class" robust model per shift type.
- **Tab 3: Model Insights**: Interactive inspection of failure modes (Boundaries, Confidence, Importance).

### 6. Automated Reporting
- **Saved Figures**: `run_experiments.py` will automatically export all benchmark plots to `figures/` for use in technical posters and reports.
