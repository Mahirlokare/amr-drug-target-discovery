"""
Page 09: Explainability & Graph Attribution.
Complies with Project 9 Blueprint Section 13 & 15.1 (Explainability).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.explainability.graph_attribution import GraphAttributionService
from src.data.resolver import chemical_resolver

st.set_page_config(page_title="Explainability | DeepTarget-AMR", page_icon="🔬", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Model Explainability & Saliency</span>
</div>
""", unsafe_allow_html=True)

st.title("🔬 Molecular Graph Explainability & Attribution")
st.markdown("Interpret atom-level features and substructure motifs driving predicted antimicrobial activity via **Integrated Gradients & GNN Saliency**:")

candidate_presets = {
    "GEQ (InhA Co-crystal Active)": "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl",
    "NITD-916 (Direct InhA Inhibitor)": "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1",
    "Isoniazid (Hydrazide Prodrug)": "O=C(NN)c1ccncc1",
    "Triclosan (Diphenyl Ether)": "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl",
    "Custom Molecule": ""
}

selected_cand = st.selectbox("Select Molecule to Analyze:", list(candidate_presets.keys()))

if "Custom Molecule" in selected_cand:
    raw_query = st.text_input("Enter SMILES or Compound Name:", value="Bedaquiline")
else:
    raw_query = candidate_presets[selected_cand]

resolved = chemical_resolver.resolve_input(raw_query)

if not resolved["valid"]:
    st.error(resolved["error"])
    st.stop()

smi = resolved["canonical_smiles"]

c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("2D Chemical Topology")
    mol = Chem.MolFromSmiles(smi)
    if mol:
        drawer = rdMolDraw2D.MolDraw2DSVG(420, 240)
        drawer.drawOptions().clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        st.image(drawer.GetDrawingText(), use_container_width=True)

with c_right:
    st.subheader("Molecule Provenance & Descriptors")
    st.write(f"• **Canonical SMILES**: `{smi}`")
    st.write(f"• **Formula**: `{resolved['formula']}` | **MW**: `{resolved['molecular_weight']} g/mol`")
    st.write(f"• **Calculated LogP**: `{resolved['logp']}` | **TPSA**: `{resolved['tpsa']} Å²`")
    st.write(f"• **H-Bond Donors/Acceptors**: `{resolved['hbd']} / {resolved['hba']}`")

st.markdown("---")

st.subheader("Atom-Level Integrated Gradients & Saliency Attribution")
st.caption("Gradient of the predicted binding logit with respect to each atom's 43 invariant input descriptors:")

attr_service = GraphAttributionService()
attribution_map = attr_service.compute_atom_attributions(smi)

if attribution_map:
    attr_rows = []
    for atom_idx, data in attribution_map.items():
        attr_rows.append({
            "Atom Index": atom_idx,
            "Element": data["element"],
            "Attribution Weight": round(data["weight"], 4),
            "Influence Direction": "▲ Positive (Pro-Binding)" if data["weight"] >= 0 else "▼ Negative (Attenuating)"
        })
    df_attr = pd.DataFrame(attr_rows)
    st.dataframe(df_attr, use_container_width=True)

st.markdown("---")
st.info("""
**Scientific Caution & Interpretability Governance**:
Integrated gradients and GNN saliency mappings represent computational model attributions, not direct physical free-energy contributions.
Always corroborate computational substructure attributions with crystallographic difference maps and SAR series data.
""")
