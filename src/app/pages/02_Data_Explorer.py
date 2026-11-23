"""
Data Explorer: ChEMBL records, assay values, and compound filtering.
Organized per Priority 2: Data Explorer.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(page_title="Data Explorer | DeepTarget-AMR", page_icon="🔍", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Data Explorer & QC</span>
</div>
""", unsafe_allow_html=True)

st.title("🔍 Curated Bioactivity Data Explorer & QC")
st.markdown("Audit standardized bioactivity measurements, replicate agreement, and filter compound libraries:")

data_file = _ROOT / "data" / "processed" / "inha_benchmark_v1.json"

if not data_file.exists():
    st.error("Processed benchmark dataset not found. Please run `python run_pipeline.py` first.")
    st.stop()

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

compounds = data.get("compounds", [])
records = data.get("records", [])
qc_report = data.get("qc_report", {})

cid_to_comp = {c["compound_id"]: c for c in compounds}

# QC Summary Metrics (Native Streamlit API)
st.subheader("Data Hygiene & QC Audit Summary")
m1, m2, m3, m4, m5 = st.columns(5)
c_sum = qc_report.get("compound_summary", {})
a_sum = qc_report.get("activity_summary", {})

with m1:
    st.metric("Total Compounds", c_sum.get("total", 0), "Curated")
with m2:
    st.metric("Standardized", f"{c_sum.get('valid', 0)} ({c_sum.get('success_rate_pct', 0)}%)", "RDKit Desalted")
with m3:
    st.metric("Actives (≤ 1 µM)", a_sum.get("labeled_active", 0), "pIC50 ≥ 6.0")
with m4:
    st.metric("Inactives (> 10 µM)", a_sum.get("labeled_inactive", 0), "pIC50 < 5.0")
with m5:
    st.metric("Quarantined", a_sum.get("quarantined_intermediate", 0), "Intermediate Band")

st.markdown("---")

# Filter Controls
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    status_filter = st.multiselect(
        "Activity Label Status",
        options=["ACTIVE", "INACTIVE", "INTERMEDIATE"],
        default=["ACTIVE", "INACTIVE"]
    )
with f_col2:
    available_endpoints = sorted(list({r.get("endpoint", "IC50") for r in records}))
    endpoint_filter = st.multiselect(
        "Experimental Endpoint",
        options=available_endpoints,
        default=available_endpoints
    )
with f_col3:
    mw_range = st.slider("Molecular Weight Filter (g/mol)", 50.0, 800.0, (100.0, 600.0))

# Filter records
filtered_records = []
for r in records:
    c = cid_to_comp.get(r["compound_id"], {})
    mw = c.get("molecular_weight", 300.0)
    if r.get("label_status") in status_filter and r.get("endpoint") in endpoint_filter:
        if mw_range[0] <= mw <= mw_range[1]:
            filtered_records.append({
                "Record ID": r["record_id"],
                "Compound ID": r["compound_id"],
                "SMILES": c.get("smiles_standard", ""),
                "Endpoint": r["endpoint"],
                "Relation": r["relation"],
                "Value": r["value_raw"],
                "Units": r["units_raw"],
                "pIC50": r["pic50"],
                "Label Status": r["label_status"],
                "MW": mw,
                "LogP": c.get("logp", 0.0),
                "Year": r.get("year", 2020)
            })

df_display = pd.DataFrame(filtered_records)
st.write(f"Showing **{len(df_display)}** of **{len(records)}** records:")
st.dataframe(df_display, use_container_width=True)

# 2D Structure Preview of Selected Compound
st.markdown("### 2D Chemical Structure Inspector")
if not df_display.empty:
    selected_idx = st.selectbox(
        "Select Compound to Render:",
        options=range(len(df_display)),
        format_func=lambda i: f"{df_display.iloc[i]['Compound ID']} (pIC50={df_display.iloc[i]['pIC50']}, {df_display.iloc[i]['Label Status']})"
    )
    smi = df_display.iloc[selected_idx]["SMILES"]
    mol = Chem.MolFromSmiles(smi)
    if mol:
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 180)
        drawer.drawOptions().clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        st.image(drawer.GetDrawingText(), use_container_width=False, width=400)
