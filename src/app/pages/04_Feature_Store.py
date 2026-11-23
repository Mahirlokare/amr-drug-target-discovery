"""
Page 04: Feature Store.
Complies with Project 9 Blueprint Section 8.6 & 10.2 (Feature Store).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.fingerprints import FingerprintGenerator
from src.features.protein_esm import ESMEmbeddingService
from src.data.resolver import chemical_resolver

st.set_page_config(page_title="Feature Store | DeepTarget-AMR", page_icon="🧬", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Feature Store</span>
</div>
""", unsafe_allow_html=True)

st.title("🧬 Multimodal Feature Store")
st.markdown("Inspect pre-computed and on-demand representations powering the **PyG Graph Convolutional Networks** and **ESM-2 Protein Language Models**:")

tab_graphs, tab_esm, tab_fp = st.tabs([
    "⚛️ PyG Molecular Graphs (43 Invariant Atom Descriptors)",
    "🧪 ESM-2 Protein Embeddings (320-dim PLM)",
    "🔬 Morgan Fingerprints (2048-bit ECFP4)"
])

with tab_graphs:
    st.subheader("Molecular Graph Featurization (PyTorch Geometric)")
    st.caption("Each small molecule is converted into a bidirectional graph G = (V, E) where each atom node holds 43 invariant chemical features:")

    test_smiles = st.text_input("Enter SMILES to Featurize:", value="Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl")
    res = chemical_resolver.resolve_input(test_smiles)

    if res["valid"]:
        clean_smi = res["canonical_smiles"]
        featurizer = MolecularGraphFeaturizer()
        graph = featurizer.featurize(clean_smi)

        if graph is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Number of Atom Nodes (|V|)", graph.num_nodes)
            c2.metric("Number of Directed Edges (|E|)", graph.num_edges)
            c3.metric("Node Feature Dimension", graph.num_node_features)

            st.write(f"• **Node Feature Matrix (`x`)**: Shape `{list(graph.x.shape)}` (dtype: `torch.float32`)")
            st.write(f"• **Edge Index (`edge_index`)**: Shape `{list(graph.edge_index.shape)}` (dtype: `torch.int64`)")

            with st.expander("Inspect Atom Node Feature Vectors (|V| x 43)"):
                atom_df = pd.DataFrame(graph.x.numpy())
                atom_df.columns = [f"feat_{i}" for i in range(43)]
                st.dataframe(atom_df, use_container_width=True)
        else:
            st.error("Graph featurization returned empty.")
    else:
        st.error(res["error"])

with tab_esm:
    st.subheader("Protein Language Model Embeddings (ESM-2)")
    st.caption("Target protein sequences encoded into 320-dimensional evolutionary representations using Meta's `esm2_t6_8M_UR50D` architecture:")

    esm_service = ESMEmbeddingService()
    sample_seq = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLTDAKVAAEVAAAGANAVALHCDVTDESQISALVAAAVERYGTVDIVVSNAGIAVPADGVEHDVTSDGWETMLNVNAAGVFLGLQAAIPLLAKGGSVVINMSSVSGQMGYPALGAYSASKAAVNGLVRNLALEHAPVRVNCINPGLVRTRLQERLLADPYQAVQAGWAPETLDKIAALAGVDTSGYYSLTPRVLRVTPGDAEYVGKPVTVIQPGGFTQVDAYD"

    st.write(f"• **InhA (Rv1484) Benchmark Sequence**: {len(sample_seq)} amino acids")
    emb, _ = esm_service.get_embedding(sample_seq, target_id="TARGET-001")

    e1, e2, e3 = st.columns(3)
    e1.metric("Embedding Dimension", emb.shape[0])
    e2.metric("L2 Vector Norm", f"{float(np.linalg.norm(emb.numpy())):.3f}")
    e3.metric("Pooling Strategy", "Mean Sequence Pooling")

    st.write(f"• **Embedding Tensor**: Shape `{list(emb.shape)}` (dtype: `torch.float32`)")
    with st.expander("Inspect First 50 Dimensions"):
        st.write(emb.numpy()[:50])

with tab_fp:
    st.subheader("Topological Morgan Fingerprints (ECFP4)")
    st.caption("Circular Morgan fingerprints (radius=2, 2048 bits) used as baseline descriptors and for Tanimoto chemical similarity:")

    fp_gen = FingerprintGenerator(radius=2, n_bits=2048)
    fp_vec = fp_gen.smiles_to_morgan_numpy(test_smiles)

    f1, f2, f3 = st.columns(3)
    f1.metric("Bit Length", len(fp_vec))
    f2.metric("Active Bits Set (On-Bits)", int(np.sum(fp_vec)))
    f3.metric("Bit Density", f"{float(np.mean(fp_vec)):.3%}")
