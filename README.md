# Dataset Shift: Assessing Model Robustness to Environmental Change

This is my Milestone 2 project for DSCI 441 Machine Learning. The idea is to study how nine common machine learning models behave when the data they see at test time is different from what they were trained on. I built an experiment pipeline that applies eight kinds of synthetic shift to the test set, computes a long list of metrics with bootstrap confidence intervals, and shows everything inside a Streamlit dashboard.

Course: DSCI 441 Machine Learning
Authors: Amandeep and Koushik

## 1. Project Description

In real life, models almost never see the same kind of data after deployment that they saw during training. Sensors drift, populations change, labeling rules get updated, and pipelines silently break. This problem is usually called dataset shift, and it is one of the easiest ways for a model to fail without anyone noticing.

The goal of this project is to actually measure that failure in a careful way. The pipeline does the following.

1. Loads the UCI Adult Income dataset from OpenML and preprocesses it.
2. Trains nine classical models on the clean training set.
3. Applies eight families of dataset shift to the test set, each at ten different intensity levels from 0.1 to 1.0.
4. For every (model, shift, intensity) combination it computes Accuracy, Precision, Recall, F1, ROC AUC and Brier Score, and then runs 200 bootstrap iterations to get 95 percent confidence intervals on each.
5. It also computes the Kolmogorov Smirnov statistic and the Population Stability Index between the clean and shifted feature distributions, so we know how big the shift actually is.
6. From all of that it produces a Robustness Score and a Relative Drop value for each (model, shift) pair, plus a one sided Welch t test that says whether the degradation is statistically significant.
7. Finally everything is dumped into a tidy CSV and shown in a four tab Streamlit dashboard.

### Shift families implemented

| ID | Family | What the simulator does | What changes |
|----|--------|-------------------------|---------------|
| A  | Covariate Shift   | Adds Gaussian noise plus a multiplicative drift to continuous features | P(X) |
| A2 | Scaling Drift     | Rescales each continuous feature with a log normal factor (sensor calibration drift) | P(X) |
| B  | Label Shift       | Drops minority class samples to change class prior | P(Y) |
| C  | Concept Drift     | Corrupts top informative features and flips a fraction of labels | P(Y given X) |
| D  | Gaussian Noise    | Adds pure additive i.i.d. Gaussian noise | P(X) |
| E  | MCAR Missingness  | Masks cells uniformly at random | P(X) |
| E2 | MAR Missingness   | Masks features only when an observed feature is above its median | P(X) |
| F  | Feature Removal   | Zeros out the top informative columns (pipeline failure simulation) | P(X) |

### Models compared

Naive Baseline (DummyClassifier), Naive Bayes, Logistic Regression, SVM with RBF kernel, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost when it is installed.

## 2. Data Source

Dataset: UCI Adult Income, also known as the Census Income dataset.
Source : OpenML, dataset ID 1590, available at https://www.openml.org/d/1590
Size   : roughly 48,842 rows and 14 features. The target is binary (income greater than 50K or not).

The dataset is fetched automatically. The first time you run `python main.py`, the function `sklearn.datasets.fetch_openml(data_id=1590, as_frame=True)` downloads it and scikit learn caches the file on disk, so later runs do not hit the network.

If the download fails for any reason (no internet, OpenML is down, etc.), the loader falls back to a synthetic dataset of 10,000 samples generated with `make_classification` using a similar 76 / 24 class imbalance. That way the pipeline always finishes, but with slightly different numbers.

You do not need to download the data manually. If you want to do it manually anyway, the instructions are inside `dataset-shift-project/data/readme_data.txt`.

Preprocessing is standard. Numeric columns are imputed with the median and standardized. Categorical columns are imputed with the most frequent value and one hot encoded. The data is then split 80 / 20 with stratification on the target.

## 3. Packages Required

The exact versions used during development are pinned in `dataset-shift-project/requirements.txt`. To install everything at once run:

```
pip install -r dataset-shift-project/requirements.txt
```

| Package | Minimum version | Used for |
|---------|------------------|----------|
| scikit-learn | 1.3.0 | Models, preprocessing pipeline, OpenML fetch |
| pandas       | 2.0.0 | Tidy result tables and CSV io |
| numpy        | 1.24.0 | Numerical arrays |
| scipy        | 1.11.0 | KS test and Welch t test |
| matplotlib   | 3.7.0 | Plotting |
| seaborn      | 0.12.0 | Heatmaps and styling |
| streamlit    | 1.32.0 | Interactive dashboard |
| xgboost      | 2.0.0 | Optional ninth model, skipped silently if not installed |

I tested everything on Python 3.10 and 3.11.

## 4. How to Run the Code

Once the dependencies are installed, the workflow has two main steps. First you run the experiment pipeline once to produce the results CSV. Then you launch the Streamlit dashboard, which just reads that CSV.

```
git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>/dataset-shift-project

pip install -r requirements.txt

python main.py
```

`python main.py` does the full benchmark. It downloads the data on the first run, trains all nine models, sweeps the eight shift families across ten intensity levels, runs 200 bootstrap iterations for every metric, computes the KS statistic, PSI, Robustness Score and Welch p values, then saves `results/experiment_results.csv` and writes around 40 figures to the `figures/` folder.

On a normal laptop this takes about 15 to 25 minutes. The slowest step is fitting the SVM with RBF kernel on roughly 39,000 training rows.

After the CSV is created, launch the dashboard:

```
streamlit run app.py
```

The dashboard has four tabs.

1. Performance Curves. Shows each metric vs intensity per model, with shaded 95 percent bootstrap confidence intervals.
2. Comparative Heatmap. A model by intensity heatmap for the chosen metric, plus a global Relative Drop heatmap that shows which (model, shift) combinations are the most damaging.
3. Robustness Analysis. Horizontal bar ranking by mean Robustness Score, plus the Brier Score calibration decay curves.
4. Distribution Shift. KS statistic and PSI as a function of intensity, plus a per intensity statistical significance table from the Welch t test.

There are also sidebar controls to switch shift type, primary metric, model and intensity range without rerunning anything.

## 5. Repository Structure

```
.
├── README.md
└── dataset-shift-project/
    ├── main.py                       Entry point that runs the full experiment
    ├── app.py                        Streamlit dashboard
    ├── requirements.txt
    ├── data/
    │   └── readme_data.txt           How to obtain and place the data
    ├── src/
    │   ├── data_loader.py            OpenML fetch and preprocessing
    │   ├── models.py                 Nine model registry and training
    │   ├── shift_simulators.py       Eight shift functions
    │   ├── evaluation.py             Metrics, bootstrap, robustness, t test
    │   ├── visualizations.py         Plot functions used by the dashboard
    │   ├── stats_utils.py            KS, PSI, CLT and LLN helpers
    │   ├── utils.py                  Logging, seeding and path helpers
    │   └── run_experiments.py        Orchestration loop
    ├── results/
    │   └── experiment_results.csv    Tidy results table (auto generated)
    ├── figures/                      Plots saved by main.py (auto generated)
    └── docs/screenshots/             Dashboard screenshots used in docs
```

## 6. Reproducibility

A single seed (`RANDOM_STATE = 42`) is set inside `utils.set_global_seed` and is also passed to the train test split and to every model. Re running `python main.py` on the same machine produces the same `experiment_results.csv` byte for byte.

This project is the Milestone 2 deliverable for DSCI 441 Machine Learning.
