"""
Target Explorer: Protein search, UniProt information, and target details.
Organized per Priority 2: Target Explorer.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from src.common.config import config_manager
from src.ingestion.uniprot_client import UniProtClient

st.set_page_config(page_title="Target Explorer | DeepTarget-AMR", page_icon="🎯", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Target Explorer</span>
</div>
""", unsafe_allow_html=True)

st.title("🎯 Target Explorer & UniProt Intelligence")
st.markdown("Query WHO priority bacterial AMR targets, inspect amino acid sequences, active site coordinates, and clinical resistance mutations:")

registry = config_manager.get_config("target_registry")["approved_targets"]

search_query = st.text_input("🔍 Search Target by Name, Gene, or Accession:", placeholder="e.g. InhA, P9WGR1, DprE1, gyrA")

filtered_targets = {}
if search_query:
    q = search_query.strip().lower()
    for tid, t in registry.items():
        if (
            q in tid.lower()
            or q in t["name"].lower()
            or q in t["gene"].lower()
            or q in t["accession"].lower()
            or q in t["organism"].lower()
        ):
            filtered_targets[tid] = t
else:
    filtered_targets = registry

if not filtered_targets:
    st.warning(f"No targets matched '{search_query}'. Showing full registry.")
    filtered_targets = registry

target_choice = st.selectbox(
    "Select Target to Inspect:",
    options=list(filtered_targets.keys()),
    format_func=lambda tid: f"{tid}: {filtered_targets[tid]['name']} ({filtered_targets[tid]['organism']})"
)

target = filtered_targets[target_choice]

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
    st.caption(f"**Sequence Length**: {target['sequence_length']} aa | **SHA-256 Checksum**: `{target['sequence_sha256']}`")

    if target.get("resistance_mutations"):
        st.markdown("### Known Clinical Resistance Mutations")
        for mut in target["resistance_mutations"]:
            st.warning(f"• **{mut.get('mutation')}**: {mut.get('mechanism')} (Confers resistance to: *{', '.join(mut.get('drugs_affected', []))}*)")

with col2:
    st.subheader("Cross-Database External Identifiers")
    st.markdown(f"""
    - **UniProtKB Accession**: [{target['accession']}](https://www.uniprot.org/uniprotkb/{target['accession']})
    - **ChEMBL Target ID**: [{target['chembl_target_id']}](https://www.ebi.ac.uk/chembl/target_report_card/{target['chembl_target_id']})
    - **RCSB PDB Reference**: [{target['pdb_id']}](https://www.rcsb.org/structure/{target['pdb_id']})
    """)

    st.markdown("### Active Site & Catalytic Pocket")
    st.write(f"• **Catalytic Residues**: `{target.get('active_site', {}).get('catalytic_residues', 'Tyr158, Lys165')}`")
    st.write(f"• **Cofactor**: `{target.get('active_site', {}).get('cofactor', 'NADH / NAD+')}`")
    st.write(f"• **Substrate Binding Loop**: `{target.get('active_site', {}).get('substrate_loop', 'Ala191 - Leu207')}`")

    pdb_id = target.get("pdb_id", "4TZK")
    st.markdown(f"### Interactive 3D Crystal Structure (PDB {pdb_id})")
    viewer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
      <style>body {{ margin: 0; padding: 0; background-color: #0A0E17; }} #viewport {{ width: 100%; height: 320px; border-radius: 8px; overflow: hidden; }}</style>
    </head>
    <body>
      <div id="viewport"></div>
      <script>
        $(document).ready(function() {{
          let element = $('#viewport');
          let config = {{ backgroundColor: '#0A0E17' }};
          let viewer = $3Dmol.createViewer(element, config);
          $3Dmol.download("pdb:{pdb_id}", viewer, {{}}, function() {{
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum', opacity: 0.9}}}});
            viewer.zoomTo();
            viewer.render();
          }});
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(viewer_html, height=340)
