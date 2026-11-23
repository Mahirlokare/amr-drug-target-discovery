"""
Molecular Analysis: SMILES input, RDKit descriptors, molecular graphs, and atom attribution.
Organized per Priority 2: Molecular Analysis.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from src.data.resolver import chemical_resolver
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.fingerprints import FingerprintGenerator

st.set_page_config(page_title="Molecular Analysis | DeepTarget-AMR", page_icon="⚛️", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Molecular Analysis</span>
</div>
""", unsafe_allow_html=True)

st.title("⚛️ Molecular Analysis & Graph Featurization")
st.markdown("Analyze chemical structures, generate 43-dimensional invariant atom node graphs for PyTorch Geometric GNNs, and inspect physicochemical descriptors:")

user_input = st.text_input("Enter Molecule Name or SMILES:", value="NITD-916")

resolved = chemical_resolver.resolve_input(user_input)

if not resolved["valid"]:
    st.error(f"Chemical resolution failed: {resolved.get('error')}")
    st.stop()

clean_smi = resolved["canonical_smiles"]
mol = Chem.MolFromSmiles(clean_smi)

col_viz, col_desc = st.columns([1, 1.2])

with col_viz:
    st.subheader("2D Chemical Topology")
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 200)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    st.image(drawer.GetDrawingText(), use_container_width=True)
    st.caption(f"Canonical SMILES: `{clean_smi}`")

with col_desc:
    st.subheader("Physicochemical RDKit Descriptors")
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Molecular Weight", f"{mw:.1f} g/mol")
        st.metric("H-Bond Donors", hbd)
    with d2:
        st.metric("LogP (Lipophilicity)", f"{logp:.2f}")
        st.metric("H-Bond Acceptors", hba)
    with d3:
        st.metric("TPSA", f"{tpsa:.1f} Å²")
        st.metric("Rotatable Bonds", rot_bonds)

st.markdown("---")

# Graph Featurization
st.subheader("PyTorch Geometric Molecular Graph (43-d Invariant Node Features)")
featurizer = MolecularGraphFeaturizer()
graph = featurizer.featurize(clean_smi)

if graph is not None:
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Atom Nodes (|V|)", graph.num_nodes, "Heavy atoms")
    with g2:
        st.metric("Directed Edges (|E|)", graph.num_edges, "Covalent bonds")
    with g3:
        st.metric("Node Feature Dim", graph.num_node_features, "Per-atom invariant channels")

    st.write(f"• **Node Feature Matrix (`x`)**: Shape `{list(graph.x.shape)}` (dtype: `torch.float32`)")
    st.write(f"• **Edge Index (`edge_index`)**: Shape `{list(graph.edge_index.shape)}` (dtype: `torch.int64`)")

    with st.expander("Inspect Atom Node Feature Matrix (|V| x 43)"):
        atom_df = pd.DataFrame(graph.x.numpy())
        atom_df.columns = [f"channel_{i}" for i in range(43)]
        atom_df.index = [f"Atom_{i} ({mol.GetAtomWithIdx(i).GetSymbol()})" for i in range(mol.GetNumAtoms())]
        st.dataframe(atom_df, use_container_width=True)

    with st.expander("Inspect 2048-bit Morgan Circular Fingerprint"):
        fp_gen = FingerprintGenerator(radius=2, n_bits=2048)
        fp_arr = fp_gen.smiles_to_morgan_numpy(clean_smi)
        st.write(f"Active On-Bits: **{int(np.sum(fp_arr))}** / 2048 (Bit density: `{float(np.mean(fp_arr)):.2%}`)")
