"""
Page 1: Data Explorer & QC Audit.
Complies with Project 9 Blueprint Section 15.1 (Data Explorer).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(page_title="Data Explorer | AMR Discovery", page_icon="🔍", layout="wide")
st.title("🔍 Curated Bioactivity Data Explorer & QC")

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

# QC Summary Metrics
st.subheader("Data Hygiene & QC Audit Summary")
m1, m2, m3, m4, m5 = st.columns(5)
c_sum = qc_report.get("compound_summary", {})
a_sum = qc_report.get("activity_summary", {})

with m1:
    st.markdown(f"""
    <div style="background: rgba(100, 116, 139, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Total Compounds</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #F1F5F9;">{c_sum.get('total', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div style="background: rgba(59, 130, 246, 0.08); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Standardized</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #38BDF8;">{c_sum.get('valid', 0)} ({c_sum.get('success_rate_pct', 0)}%)</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.08); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Actives (≤ 1 µM)</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #34D399;">{a_sum.get('labeled_active', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div style="background: rgba(239, 68, 68, 0.08); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Inactives (> 10 µM)</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #F87171;">{a_sum.get('labeled_inactive', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with m5:
    st.markdown(f"""
    <div style="background: rgba(245, 158, 11, 0.08); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Quarantined</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #FBBF24;">{a_sum.get('quarantined_intermediate', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Filter Sidebar
st.sidebar.header("Filter Criteria")
status_filter = st.sidebar.multiselect(
    "Activity Status",
    options=["ACTIVE", "INACTIVE", "INTERMEDIATE"],
    default=["ACTIVE", "INACTIVE"]
)

min_year, max_year = 2005, 2024
year_range = st.sidebar.slider("Publication Year", min_year, max_year, (2005, 2024))

# Build DataFrame
table_rows = []
for r in records:
    cid = r["compound_id"]
    comp = cid_to_comp.get(cid, {})
    yr = r.get("year") or 2015
    if r.get("label_status") in status_filter and (year_range[0] <= yr <= year_range[1]):
        table_rows.append({
            "Compound ID": cid,
            "Common Name": comp.get("source_ids", ["-"])[1] if len(comp.get("source_ids", [])) > 1 else cid,
            "SMILES": comp.get("smiles_standard", comp.get("smiles_raw")),
            "Endpoint": r["endpoint"],
            "IC50 (nM)": r["value_normalized_nm"],
            "pIC50": r["pic50"],
            "Label Status": r["label_status"],
            "Year": yr,
            "Document ID": r.get("assay_id", "-"),
            "MW": comp.get("molecular_weight"),
            "LogP": comp.get("logp"),
        })

df = pd.DataFrame(table_rows)

st.subheader(f"Filtered Records ({len(df)} compounds)")
st.dataframe(df, use_container_width=True)

# Export Data
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Records as CSV",
    data=csv_data,
    file_name="inha_curated_bioactivity.csv",
    mime="text/csv"
)

st.markdown("---")

# 2D Structure Preview Selector
st.subheader("2D Chemical Structure Inspector")
selected_cid = st.selectbox("Select compound to inspect structure:", options=df["Compound ID"].tolist() if not df.empty else [])

if selected_cid:
    comp_info = cid_to_comp.get(selected_cid, {})
    smi = comp_info.get("smiles_standard", comp_info.get("smiles_raw"))

    c_left, c_right = st.columns([1, 1])

    with c_left:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            drawer = rdMolDraw2D.MolDraw2DSVG(400, 250)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            st.image(svg, caption=f"{selected_cid} Structure", use_container_width=False)
        else:
            st.warning("Structure parsing failed.")

    with c_right:
        st.write(f"**Compound ID**: `{selected_cid}`")
        st.write(f"**Canonical SMILES**: `{smi}`")
        st.write(f"**InChIKey**: `{comp_info.get('inchikey')}`")
        st.write(f"**Molecular Weight**: `{comp_info.get('molecular_weight')} g/mol`")
        st.write(f"**Calculated LogP**: `{comp_info.get('logp')}`")
        st.write(f"**Standardization Status**: `{comp_info.get('standardisation_status')}`")
