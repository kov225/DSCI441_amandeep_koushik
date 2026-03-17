# Dataset Shift: Classical ML Robustness Analysis

This research project investigates the resilience of classical machine learning models under three distinct categories of distribution change. Using the UCI Adult Income dataset, we quantify performance degradation as a function of shift intensity to identify inherent algorithmic vulnerabilities.



---

## Milestone 2 Upgrade Plan

### 1. Deeper Shift Simulation
Upgrade simulations to represent more realistic environmental challenges:
- **Covariate Shift**: Add feature scaling drift as a separate experiment from noise injection to distinguish between precision loss and distribution bias.
- **Prior Shift**: Currently only drops minority samples. Add the reverse — oversample majority to see asymmetric effects.
- **Concept Shift**: Upgrade to feature permutation (shuffle values between samples) instead of random noise injection for more realistic corruption.

### 2. Robustness Scoring
Transition from raw performance tracking to a unified robustness metric:
- **`robustness_score()`**: Calculation of the AUC degradation rate per unit of shift intensity. This enables objective ranking of models by their "Resilience Factor."

### 3. Interpretability Layer
A dedicated `interpretability.py` module to analyze why models fail:
- **Feature Importance Drifts**: Analysis of rank changes before vs. after shift.
- **Decision Boundary Visualization**: 2D projections showing how separation hyperplanes collapse.
- **Confidence Distribution**: Quantifying model calibration and overconfidence during degradation.

### 4. Statistical Validation
Moving beyond single-point estimates to defensible results:
- **Multi-seed Cross-validation**: Aggregating results across 3–5 random seeds.
- **Error bands**: Visualizing variance on performance curves to ensure results are defensible.

### 5. Streamlit App Upgrade
Evolution of the UI into a research dashboard with three functional tabs:
- **Tab 1: Explorer**: Polished distribution visualizer.
- **Tab 2: Robustness Leaderboard**: Ranked comparison highlighting the "Best-in-Class" model per shift type.
- **Tab 3: Model Insights**: Interactive inspection of failure modes (Boundaries, Confidence, Importance).

### 6. Automated Reporting
- **Auto-save Figures**: Automatically export all benchmark plots to `figures/` for use in technical posters and reports.
