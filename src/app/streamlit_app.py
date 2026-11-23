"""
AMR Drug Target Discovery - Streamlit Research Platform.
Complies with Project 9 Blueprint Section 15 (Frontend and dashboard specification).
"""

import sys
from pathlib import Path

# Ensure project root directory is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd

from src.storage.db import db_manager

# Page config
st.set_page_config(
    page_title="AMR Drug Target Discovery | GNN + ESM-2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #92400E;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: Database & Storage Status
with st.sidebar:
    st.markdown("### 🗄️ Persistence & Database")
    db_health = db_manager.health_check()
    if db_health.get("status") == "HEALTHY":
        st.success(f"Status: **Connected** ({db_health.get('engine')})")
        with st.expander("Storage Details"):
            st.caption(f"Location: `{db_health.get('location')}`")
            counts = db_health.get("table_counts", {})
            st.write(f"- Targets: **{counts.get('targets', 0)}**")
            st.write(f"- Predictions: **{counts.get('predictions', 0)}**")
            st.write(f"- Docking Runs: **{counts.get('docking_runs', 0)}**")
            st.write(f"- ADMET Runs: **{counts.get('admet_runs', 0)}**")
    else:
        st.error(f"DB Error: {db_health.get('error')}")

# Title & Abstract
st.markdown('<div class="main-title">🧬 Antimicrobial-Resistance Drug Target Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Molecular Graph Neural Networks + ESM-2 Protein Language Models under Scaffold-Aware Validation</div>', unsafe_allow_html=True)

# Scientific Disclaimer Alert (Blueprint Section 1.4 & 15.3)
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Scientific Interpretation & Responsible Research Caveat:</strong><br>
    This research platform generates in-silico computational predictions. Predicted drug-target interactions,
    surrogate docking scores, and in-silico ADMET filters are prioritisation evidence and 
    <strong>do NOT constitute proof of microbiological efficacy, safety, target engagement, or clinical benefit</strong>.
    All findings are candidate hypotheses intended to triage laboratory review.
</div>
""", unsafe_allow_html=True)

# Project Status & Current Pinned Target (Pure HTML Cards to prevent dynamic chunk fetch errors)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 14px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Primary Target</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8; margin: 4px 0;">InhA (Rv1484)</div>
        <div style="font-size: 0.75rem; color: #64748B;">M. tuberculosis H37Rv</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 14px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Validated Dataset</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #34D399; margin: 4px 0;">54 Compounds</div>
        <div style="font-size: 0.75rem; color: #64748B;">Curated direct IC50 records</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 14px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Validation Split</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #C084FC; margin: 4px 0;">Bemis-Murcko</div>
        <div style="font-size: 0.75rem; color: #64748B;">Zero train/test scaffold overlap</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 14px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Primary Architecture</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24; margin: 4px 0;">GNN + ESM-2</div>
        <div style="font-size: 0.75rem; color: #64748B;">PyG GCN + ESM-2 320d projection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Architectural Overview & System Status
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("System Architecture & Pipeline Flow")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Data["1. Curation & QC"]
            A["ChEMBL Bioactivities"] --> B["Standardize & Desalt"]
            B --> C["Label Harmonization"]
            C --> D["Bemis-Murcko Split"]
        end
        subgraph Models["2. Multimodal Learning"]
            D --> E["PyG Molecular Graph"]
            D --> F["UniProt -> ESM-2 Embedding"]
            E --> G["GNN Encoder (43d -> 64d)"]
            F --> H["ESM-2 Projector (320d -> 64d)"]
            G --> I["Multimodal Fusion & Head"]
            H --> I
        end
        subgraph Triage["3. Validation & Triage"]
            I --> J["Deep Ensemble Uncertainty"]
            J --> K["Platt Probability Calibration"]
            K --> L["AutoDock Vina & ADMET Filters"]
            L --> M["Ranked Candidate Shortlist"]
        end
    ```
    """)
    st.info("Navigate through the multi-page sidebar on the left to inspect data, compare models, score candidates, and explore active learning.")

with right_col:
    st.subheader("Current Pinned Run Metadata")
    
    # Load recent evaluation metadata if available
    eval_json_path = PROJECT_ROOT / "artifacts" / "reports" / "EVAL-INHA-PRIMARY_evaluation.json"
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            st.json({
                "Report ID": eval_data.get("report_id"),
                "Target ID": eval_data.get("target_id"),
                "Split": eval_data.get("split_info", {}).get("type"),
                "Train/Val/Test Count": f"{eval_data.get('split_info', {}).get('train_compounds')} / {eval_data.get('split_info', {}).get('val_compounds')} / {eval_data.get('split_info', {}).get('test_compounds')}",
                "Multimodal AUROC": eval_data.get("primary_model_metrics", {}).get("AUROC"),
                "Multimodal PR-AUC": eval_data.get("primary_model_metrics", {}).get("PR_AUC"),
                "Enrichment Factor @ 1%": f"{eval_data.get('primary_model_metrics', {}).get('EF_1pct')}x",
                "Expected Calibration Error": eval_data.get("primary_model_metrics", {}).get("ExpectedCalibrationError"),
            })
    else:
        st.warning("Run pipeline via `python run_pipeline.py` to populate run metadata.")

st.markdown("---")
st.subheader("Key Scientific Validation Highlights")
h_col1, h_col2, h_col3 = st.columns(3)

with h_col1:
    st.markdown("""
    **Zero-Leakage Scaffold Split**
    - Compounds grouped by Bemis-Murcko frameworks.
    - Automated assertions verify 0% scaffold sharing between train, validation, and test partitions.
    """)

with h_col2:
    st.markdown("""
    **Uncertainty & Calibration**
    - 3-seed Deep Ensemble computes epistemic uncertainty.
    - Platt probability calibration prevents overconfident predictions on new chemical series.
    """)

with h_col3:
    st.markdown("""
    **Secondary Computational Triage**
    - AutoDock Vina redocking with co-crystal control.
    - Rule of 5, Veber oral bioavailability, and PAINS substructure alert screening.
    """)
