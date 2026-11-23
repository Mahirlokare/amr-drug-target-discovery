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
            with c1:
                st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Atom Nodes (|V|)</div><div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8; margin-top: 0.2rem;">{graph.num_nodes}</div><div style="font-size: 0.72rem; color: #64748B;">Heavy atom vertices</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Directed Edges (|E|)</div><div style="font-size: 1.4rem; font-weight: 700; color: #34D399; margin-top: 0.2rem;">{graph.num_edges}</div><div style="font-size: 0.72rem; color: #64748B;">Bidirectional covalent bonds</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Node Feature Dim</div><div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24; margin-top: 0.2rem;">{graph.num_node_features}</div><div style="font-size: 0.72rem; color: #64748B;">Invariant descriptors</div></div>', unsafe_allow_html=True)

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
    esm_status = esm_service.get_service_status()
    st.info(f"**PLM Engine Status**: `{esm_status['active_engine']}` | Model: `{esm_status['model_name']}` (dim={esm_status['embedding_dim']})")

    sample_seq = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLTDAKVAAEVAAAGANAVALHCDVTDESQISALVAAAVERYGTVDIVVSNAGIAVPADGVEHDVTSDGWETMLNVNAAGVFLGLQAAIPLLAKGGSVVINMSSVSGQMGYPALGAYSASKAAVNGLVRNLALEHAPVRVNCINPGLVRTRLQERLLADPYQAVQAGWAPETLDKIAALAGVDTSGYYSLTPRVLRVTPGDAEYVGKPVTVIQPGGFTQVDAYD"

    st.write(f"• **InhA (Rv1484) Benchmark Sequence**: {len(sample_seq)} amino acids")
    emb, _ = esm_service.get_embedding(sample_seq, target_id="TARGET-001")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Embedding Dim</div><div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8; margin-top: 0.2rem;">{emb.shape[0]}</div><div style="font-size: 0.72rem; color: #64748B;">Dense representation</div></div>', unsafe_allow_html=True)
    with e2:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">L2 Vector Norm</div><div style="font-size: 1.4rem; font-weight: 700; color: #34D399; margin-top: 0.2rem;">{float(np.linalg.norm(emb.numpy())):.3f}</div><div style="font-size: 0.72rem; color: #64748B;">Unit sphere projected</div></div>', unsafe_allow_html=True)
    with e3:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Pooling Strategy</div><div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24; margin-top: 0.2rem;">Mean</div><div style="font-size: 0.72rem; color: #64748B;">Global sequence pooling</div></div>', unsafe_allow_html=True)

    st.write(f"• **Embedding Tensor**: Shape `{list(emb.shape)}` (dtype: `torch.float32`)")
    with st.expander("Inspect First 50 Dimensions"):
        st.write(emb.numpy()[:50])

with tab_fp:
    st.subheader("Topological Morgan Fingerprints (ECFP4)")
    st.caption("Circular Morgan fingerprints (radius=2, 2048 bits) used as baseline descriptors and for Tanimoto chemical similarity:")

    fp_gen = FingerprintGenerator(radius=2, n_bits=2048)
    fp_vec = fp_gen.smiles_to_morgan_numpy(test_smiles)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Bit Length</div><div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8; margin-top: 0.2rem;">{len(fp_vec)}</div><div style="font-size: 0.72rem; color: #64748B;">Fixed length bit vector</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Active Bits Set</div><div style="font-size: 1.4rem; font-weight: 700; color: #34D399; margin-top: 0.2rem;">{int(np.sum(fp_vec))}</div><div style="font-size: 0.72rem; color: #64748B;">On-bits in fingerprint</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown(f'<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem;"><div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Bit Density</div><div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24; margin-top: 0.2rem;">{float(np.mean(fp_vec)):.2%}</div><div style="font-size: 0.72rem; color: #64748B;">Sparsity factor</div></div>', unsafe_allow_html=True)
