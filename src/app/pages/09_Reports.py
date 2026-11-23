"""
Reports: Exportable scientific reports, experiment history, and dataset/model cards.
Organized per Priority 2: Reports.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd
from src.storage.db import db_manager

st.set_page_config(page_title="Scientific Reports | DeepTarget-AMR", page_icon="📑", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Scientific Reports</span>
</div>
""", unsafe_allow_html=True)

st.title("📑 Scientific Governance, Reports & History")
st.markdown("Access transparent audit artifacts, download candidate screening summaries, and inspect historical database predictions:")

tab_cards, tab_history, tab_export = st.tabs([
    "📇 Dataset & Model Cards",
    "📜 Database Prediction History",
    "💾 Export Scientific Data"
])

with tab_cards:
    col_dcard, col_mcard = st.columns(2)
    with col_dcard:
        st.subheader("Dataset Card (InhA Benchmark v1)")
        dcard_path = _ROOT / "docs" / "dataset_card.md"
        if dcard_path.exists():
            with open(dcard_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Dataset card file not found.")

    with col_mcard:
        st.subheader("Model Card (Multimodal GNN + ESM-2)")
        mcard_path = _ROOT / "docs" / "model_card.md"
        if mcard_path.exists():
            with open(mcard_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Model card file not found.")

with tab_history:
    st.subheader("Recent Database Prediction Records")
    recent_preds = db_manager.list_predictions(limit=50)
    if recent_preds:
        st.dataframe(pd.DataFrame(recent_preds), use_container_width=True)
    else:
        st.info("No prediction records found in the database. Run candidate discovery to populate records.")

with tab_export:
    st.subheader("Export Evaluation Artifacts")
    eval_json = _ROOT / "artifacts" / "reports" / "EVAL-INHA-PRIMARY_evaluation.json"
    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            raw_text = f.read()
        st.download_button("📥 Download Benchmark Evaluation JSON", data=raw_text, file_name="evaluation_report.json", mime="application/json")
    else:
        st.info("Run `python run_pipeline.py` to generate evaluation artifacts.")
