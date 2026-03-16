"""
Streamlit app to explore how dataset shift affects ML models.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Link to local src for visualizations
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from visualizations import plot_performance_curves, plot_model_comparison_heatmap
except ImportError:
    pass 

# Basic page setup
st.set_page_config(page_title="Dataset Shift Explorer", layout="wide", page_icon="📈")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "experiment_results.csv")

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)

df = load_results()

st.title("Dataset Shift: Classical ML Robustness")
st.markdown("""
Watch how 7 different ML models handle three kinds of dataset shift.
Adjust the filters on the left to see the results.
""")

if df is None:
    st.error("Results not found. Please run the experiments first to see data.")
    st.stop()
    
# Get list of models and shifts for the UI
models = sorted(df["Model"].unique().tolist())
shift_types = sorted([s for s in df["Shift_Type"].unique() if s != "Baseline"])
metrics = ["Accuracy", "F1_Score", "ROC_AUC", "Brier_Score"]

# Sidebar controls
st.sidebar.header("Controls")
selected_shift = st.sidebar.selectbox("Select Shift Type", shift_types)
selected_metric = st.sidebar.selectbox("Select Evaluation Metric", metrics)

st.sidebar.markdown("---")
st.sidebar.subheader("Single Model View")
selected_model = st.sidebar.selectbox("Select a specific model", models)

st.sidebar.markdown("---")
st.sidebar.info("Note: The baseline (intensity = 0.0) is the same for all shifts.")

# Merge baseline data so the plots start at zero
df_baseline = df[df["Shift_Type"] == "Baseline"].copy()
df_baseline["Shift_Type"] = selected_shift

df_current_shift = df[df["Shift_Type"] == selected_shift].copy()
if 0.0 not in df_current_shift["Intensity"].values:
    df_plot = pd.concat([df_baseline, df_current_shift], ignore_index=True)
else:
    df_plot = df_current_shift

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Performance Curves: {selected_metric} under {selected_shift}")
    st.markdown("See how performance drops as the shift gets stronger.")
    try:
        fig_curve = plot_performance_curves(df_plot, selected_shift, selected_metric)
        st.pyplot(fig_curve)
    except Exception as e:
        st.error(f"Plotting error: {e}")

with col2:
    st.subheader(f"Focus on {selected_model}")
    model_df = df_plot[df_plot["Model"] == selected_model].sort_values("Intensity")
    
    st.dataframe(
        model_df[["Intensity", selected_metric]].style.format({selected_metric: "{:.3f}"}),
        hide_index=True,
        use_container_width=True
    )
    
st.markdown("---")
st.subheader("Interactive Heatmap")
st.markdown(f"Side by side comparison of all models for {selected_shift}.")

try:
    fig_heatmap = plot_model_comparison_heatmap(df_plot, selected_shift, selected_metric)
    st.pyplot(fig_heatmap)
except Exception as e:
    st.error(f"Heatmap error: {e}")
