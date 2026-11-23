"""
Page 2: Target Registry.
Complies with Project 9 Blueprint Section 15.1 (Target Registry).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from src.common.config import config_manager

st.set_page_config(page_title="Target Registry | AMR Discovery", page_icon="🎯", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Target Registry</span>
</div>
""", unsafe_allow_html=True)

st.title("🎯 Bacterial Protein Target Registry")

registry = config_manager.get_config("target_registry")["approved_targets"]

target_choice = st.selectbox(
    "Select Approved Target:",
    options=list(registry.keys()),
    format_func=lambda tid: f"{tid}: {registry[tid]['name']} ({registry[tid]['organism']})"
)

target = registry[target_choice]

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader(f"{target['name']}")
    st.markdown("**Biological Rationale & AMR Significance**:")
    st.info(target["selection_rationale"])

    st.markdown("### Organism & Taxonomic Identity")
    t1, t2, t3, t4 = st.columns(4)
    t1.write(f"**Organism**:\n*{target['organism']}*")
    t2.write(f"**Strain Scope**:\n`{target['strain_scope']}`")
    t3.write(f"**Taxonomy ID**:\n`{target['taxonomy_id']}`")
    t4.write(f"**Gene Symbol**:\n`{target['gene']}` (`{target.get('locus_tag', '-')}`)")

    st.markdown("---")
    st.markdown("### Primary Amino Acid Sequence")
    st.code(target["sequence"], language="text")
    st.write(f"**Sequence Length**: {target['sequence_length']} aa | **SHA-256 Checksum**: `{target['sequence_sha256']}`")

    # Known Resistance Mutations
    mutations = target.get("known_resistance_mutations", [])
    if mutations:
        st.markdown("---")
        st.markdown("### ⚠️ Documented Clinical Resistance Mutations")
        for mut in mutations:
            st.warning(mut)

with col2:
    st.subheader("Structural & Assay Mappings")
    st.write(f"**UniProt Accession**: [`{target['accession']}`](https://www.uniprot.org/uniprotkb/{target['accession']})")
    st.write(f"**ChEMBL Target ID**: [`{target['chembl_target_id']}`](https://www.ebi.ac.uk/chembl/target_report_card/{target['chembl_target_id']})")
    st.write(f"**Primary PDB Structure**: [`{target['pdb_id']}`](https://www.rcsb.org/structure/{target['pdb_id']}) (Chain {target['chain_id']})")

    st.markdown("---")
    st.subheader("Docking Grid Definition")
    pocket = target.get("pocket", {})
    st.write("**Pocket Center (Å)**:")
    st.code(f"X: {pocket.get('center_x')}, Y: {pocket.get('center_y')}, Z: {pocket.get('center_z')}")
    st.write("**Search Box (Å)**:")
    st.code(f"{pocket.get('size_x')} x {pocket.get('size_y')} x {pocket.get('size_z')}")

    # Interactive 3D WebGL Structure Viewer
    st.markdown("---")
    st.subheader(f"3D Structure: PDB {target['pdb_id']}")
    pdb_id = target["pdb_id"]
    viewer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
      <style>
        body {{ margin: 0; padding: 0; background-color: #0F172A; }}
        #viewport {{ width: 100%; height: 280px; position: relative; border-radius: 8px; overflow: hidden; }}
      </style>
    </head>
    <body>
      <div id="viewport"></div>
      <script>
        $(function() {{
          let viewer = $3Dmol.createViewer($('#viewport'), {{ backgroundColor: '#0B0F19' }});
          $3Dmol.download("pdb:{pdb_id}", viewer, {{}}, function() {{
            viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum' }} }});
            viewer.setStyle({{ hetflag: true }}, {{ stick: {{ colorscheme: 'greenCarbon' }} }});
            viewer.zoomTo();
            viewer.render();
            viewer.spin("y", 0.6);
          }});
        }});
      </script>
    </body>
    </html>
    """
    components.html(viewer_html, height=290)
