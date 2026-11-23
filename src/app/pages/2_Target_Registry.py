"""
Page 2: Target Registry.
Complies with Project 9 Blueprint Section 15.1 (Target Registry).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import yaml

from src.common.config import config_manager

st.set_page_config(page_title="Target Registry | AMR Discovery", page_icon="🎯", layout="wide")
st.title("🎯 Bacterial Protein Target Registry")

registry = config_manager.get_config("target_registry")["approved_targets"]

target_choice = st.selectbox(
    "Select Approved Target:",
    options=list(registry.keys()),
    format_func=lambda tid: f"{tid}: {registry[tid]['name']} ({registry[tid]['organism']})"
)

target = registry[target_choice]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{target['name']}")
    st.markdown(f"**Biological Rationale & AMR Significance**:")
    st.info(target["selection_rationale"])

    st.markdown("### Organism & Taxonomic Identity")
    t1, t2, t3, t4 = st.columns(4)
    t1.write(f"**Organism**:\n*{target['organism']}*")
    t2.write(f"**Strain Scope**:\n`{target['strain_scope']}`")
    t3.write(f"**Taxonomy ID**:\n`{target['taxonomy_id']}`")
    t4.write(f"**Gene Symbol**:\n`{target['gene']}` (`{target['locus_tag']}`)")

    st.markdown("---")
    st.markdown("### Primary Amino Acid Sequence")
    st.code(target["sequence"], language="text")
    st.write(f"**Sequence Length**: {target['sequence_length']} aa | **SHA-256 Checksum**: `{target['sequence_sha256']}`")

with col2:
    st.subheader("Structural & Assay Mappings")
    st.write(f"**UniProt Accession**: [`{target['accession']}`](https://www.uniprot.org/uniprotkb/{target['accession']})")
    st.write(f"**ChEMBL Target ID**: [`{target['chembl_target_id']}`](https://www.ebi.ac.uk/chembl/target_report_card/{target['chembl_target_id']})")
    st.write(f"**Primary PDB Structure**: [`{target['pdb_id']}`](https://www.rcsb.org/structure/{target['pdb_id']}) (Chain {target['chain_id']})")

    st.markdown("---")
    st.subheader("Docking Pocket Definition")
    pocket = target.get("pocket", {})
    st.write("**Pocket Center (Å)**:")
    st.code(f"X: {pocket.get('center_x')}, Y: {pocket.get('center_y')}, Z: {pocket.get('center_z')}")
    st.write("**Search Grid Size (Å)**:")
    st.code(f"{pocket.get('size_x')} x {pocket.get('size_y')} x {pocket.get('size_z')}")

    if target_choice == "TARGET-001":
        st.success("Co-crystallized cofactor: **NAD+** is retained in receptor binding cavity.")
