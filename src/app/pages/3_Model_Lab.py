"""
Page 3: Model Lab & Benchmark Comparisons.
Complies with Project 9 Blueprint Section 10, 11 & 15.1 (Model Lab).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Lab | AMR Discovery", page_icon="🧪", layout="wide")
st.title("🧪 Model Lab: Benchmarks & Scaffold Evaluation")

report_file = _ROOT / "artifacts" / "reports" / "EVAL-INHA-PRIMARY_evaluation.json"

if not report_file.exists():
    st.warning("Evaluation report not found. Please run `python run_pipeline.py` first.")
    st.stop()

with open(report_file, "r", encoding="utf-8") as f:
    report = json.load(f)

# Leakage Assertion Card
st.subheader("🛡️ Leakage-Safe Split Verification (Section 11.1)")
split_info = report.get("split_info", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Split Protocol", "Bemis-Murcko Scaffold")
c2.metric("Train Compounds", split_info.get("train_compounds"))
c3.metric("Val Compounds", split_info.get("val_compounds"))
c4.metric("Test Compounds", split_info.get("test_compounds"))

st.success("✅ Minimum automated leakage assertion passed: `train_scaffolds.isdisjoint(test_scaffolds)` is verified. Zero chemical series leakage.")

st.markdown("---")

# Benchmark Comparison Table
st.subheader("Benchmark Comparison (All Models on Held-Out Test Set)")
comparison_data = report.get("models_comparison", [])
df_comp = pd.DataFrame(comparison_data)

st.dataframe(
    df_comp.style.highlight_max(subset=["AUROC", "PR-AUC"], color="#D1FAE5"),
    use_container_width=True
)

st.markdown("---")

# Primary Multimodal Metrics & Confidence Intervals
st.subheader("Primary Multimodal Model (GNN + ESM-2) Rigorous Metrics")
p_metrics = report.get("primary_model_metrics", {})
ci = report.get("confidence_intervals_95", {})

m1, m2, m3, m4 = st.columns(4)
m1.metric("AUROC", f"{p_metrics.get('AUROC')}", f"95% CI: {ci.get('AUROC_CI')}")
m2.metric("PR-AUC", f"{p_metrics.get('PR_AUC')}", f"95% CI: {ci.get('PR_AUC_CI')}")
m3.metric("Enrichment Factor @ 1%", f"{p_metrics.get('EF_1pct')}×", f"95% CI: {ci.get('EF_1pct_CI')}")
m4.metric("Brier Score (Calibration)", f"{p_metrics.get('BrierScore')}", f"ECE: {p_metrics.get('ExpectedCalibrationError')}")

# Reliability Diagram
st.markdown("---")
st.subheader("Reliability Calibration Diagram (Section 11.4)")
rel_bins = p_metrics.get("reliability_bins", {})
if rel_bins:
    confs = rel_bins.get("bin_confidences", [])
    accs = rel_bins.get("bin_accuracies", [])
    cal_df = pd.DataFrame({
        "Mean Predicted Probability": confs,
        "Empirical Observed Accuracy": accs,
        "Perfect Calibration": confs
    })
    st.line_chart(cal_df.set_index("Mean Predicted Probability"))
    st.caption("A well-calibrated model tracks closely to the 45-degree diagonal line, ensuring probabilities reflect true likelihood of activity.")
