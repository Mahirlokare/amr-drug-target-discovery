"""
Page 01: Projects & Pathogen Selection.
Complies with Project 9 Blueprint Section 15.1 (Projects Overview).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from src.common.config import config_manager

st.set_page_config(page_title="Projects | DeepTarget-AMR", page_icon="🗂️", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Research Projects & Pathogens</span>
</div>
""", unsafe_allow_html=True)

st.title("🗂️ AMR Research Projects & Pathogen Selection")
st.markdown("Select an active antimicrobial resistance research portfolio aligned with the **WHO Bacterial Priority Pathogens List (BPPL 2024)**:")

projects = [
    {
        "id": "PROJ-TB-01",
        "title": "MDR-TB InhA Direct Inhibition Pipeline",
        "pathogen": "Mycobacterium tuberculosis (H37Rv)",
        "priority_tier": "CRITICAL PRIORITY (WHO Group 1)",
        "primary_target": "InhA (Rv1484, FAS-II Enoyl-ACP Reductase)",
        "pdb_id": "4TZK",
        "status": "Active Screening / Validated Benchmark",
        "description": "Circumventing isoniazid prodrug resistance caused by KatG mutations by designing direct InhA binding inhibitors (NITD-916 and GEQ series). Validated on 54 curated bioactivities under zero-leakage Bemis-Murcko scaffold splits.",
        "mortality": "1.3 million deaths annually worldwide (leading infectious killer).",
        "key_compounds": "Isoniazid, NITD-916, GEQ, Triclosan, Diazaborines"
    },
    {
        "id": "PROJ-TB-02",
        "title": "Mycobacterial Cell-Wall DprE1 Epimerase",
        "pathogen": "Mycobacterium tuberculosis",
        "priority_tier": "CRITICAL PRIORITY (WHO Group 1)",
        "primary_target": "DprE1 (Rv3790, Decaprenylphosphoryl-beta-D-ribose 2'-epimerase)",
        "pdb_id": "4KW5",
        "status": "Active Pre-Clinical Target",
        "description": "Essential flavoenzyme responsible for decaprenylphosphoryl arabinose (DPA) biosynthesis in the mycobacterial cell wall. Clinically validated target of Macozinone (PBTZ169) and OPC-167832.",
        "mortality": "Essential in latent and replicating TB phenotypes.",
        "key_compounds": "Macozinone (PBTZ169), BTZ043, OPC-167832, TBA-7371"
    },
    {
        "id": "PROJ-MRSA-01",
        "title": "MRSA DNA Gyrase Topoisomerase Rescue",
        "pathogen": "Staphylococcus aureus (MRSA)",
        "priority_tier": "HIGH PRIORITY (WHO BPPL)",
        "primary_target": "GyrA (DNA gyrase subunit A)",
        "pdb_id": "2XCS",
        "status": "Target Registry Active",
        "description": "Overcoming classical S84L/E88K quinolone resistance-determining region (QRDR) mutations that disrupt fluoroquinolone-mediated water-metal ion bridges.",
        "mortality": "Major cause of hospital-acquired bacteremia and endocarditis.",
        "key_compounds": "Ciprofloxacin, Moxifloxacin, Levofloxacin"
    },
    {
        "id": "PROJ-PSA-01",
        "title": "Pseudomonas aeruginosa PBP3 Peptidoglycan Inhibitors",
        "pathogen": "Pseudomonas aeruginosa (Carbapenem-Resistant)",
        "priority_tier": "CRITICAL PRIORITY (WHO Group 1)",
        "primary_target": "PBP3 (FtsI, Transpeptidase)",
        "pdb_id": "3PBQ",
        "status": "Target Registry Active",
        "description": "Inhibiting septal cell division and peptidoglycan crosslinking in multi-drug resistant Pseudomonas strains displaying outer membrane porin loss.",
        "mortality": "Severe ventilator-associated pneumonia and cystic fibrosis exacerbations.",
        "key_compounds": "Aztreonam, Meropenem, Cefepime"
    },
    {
        "id": "PROJ-KPN-01",
        "title": "Pan-Resistant NDM-1 Metallo-Beta-Lactamase Blockers",
        "pathogen": "Klebsiella pneumoniae (Carbapenemase-Producing)",
        "priority_tier": "CRITICAL PRIORITY (WHO Group 1)",
        "primary_target": "NDM-1 (New Delhi metallo-beta-lactamase 1)",
        "pdb_id": "3SPU",
        "status": "Target Registry Active",
        "description": "Discovery of non-chelating, boronic acid and thiol-derived inhibitors of bimetallic dizinc carbapenemases that hydrolyze carbapenems and penicillins.",
        "mortality": "High-fatality nosocomial sepsis and urinary tract infections.",
        "key_compounds": "Boronic acid transition-state analogs, Taniborbactam"
    }
]

# Set active project
if "active_project_id" not in st.session_state:
    st.session_state["active_project_id"] = "PROJ-TB-01"

proj_choice = st.selectbox(
    "Set Active Research Project Workspace:",
    options=[p["id"] for p in projects],
    format_func=lambda pid: f"{pid}: {next(p['title'] for p in projects if p['id'] == pid)} ({next(p['pathogen'] for p in projects if p['id'] == pid)})"
)
st.session_state["active_project_id"] = proj_choice

sel_proj = next(p for p in projects if p["id"] == proj_choice)

st.markdown("---")

c_left, c_right = st.columns([1.2, 0.8])

with c_left:
    st.subheader(sel_proj["title"])
    st.markdown(f"**Pathogen**: *{sel_proj['pathogen']}*")
    st.markdown(f"**Priority Category**: `{sel_proj['priority_tier']}`")
    st.markdown(f"**Primary Target**: `{sel_proj['primary_target']}`")
    st.info(sel_proj["description"])
    st.write(f"**Clinical Impact**: {sel_proj['mortality']}")
    st.write(f"**Benchmark Chemotypes**: `{sel_proj['key_compounds']}`")

with c_right:
    st.markdown("### Workspace Quick Actions")
    st.success(f"Active Workspace: **{sel_proj['id']}**")
    st.page_link("pages/02_Target_Registry.py", label="View Target & UniProt Info", icon="🎯")
    st.page_link("pages/03_Data_Explorer.py", label="Explore Curated ChEMBL/PubChem QC", icon="🔍")
    st.page_link("pages/05_Model_Lab.py", label="Open Model Lab & Benchmarks", icon="🧪")
    st.page_link("pages/07_Candidate_Ranking.py", label="Screen Small-Molecule Library", icon="📊")
    st.page_link("streamlit_app.py", label="Return to Executive Overview", icon="🏠")
