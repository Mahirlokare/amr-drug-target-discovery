"""
Protein Analysis: Sequence input, ESM-2 embeddings, and evolutionary representations.
Organized per Priority 2: Protein Analysis.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import torch

from src.features.protein_esm import ESMEmbeddingService
from src.common.config import config_manager

st.set_page_config(page_title="Protein Analysis | DeepTarget-AMR", page_icon="🧪", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Protein Analysis</span>
</div>
""", unsafe_allow_html=True)

st.title("🧪 Protein Language Model Analysis (ESM-2)")
st.markdown("Generate and inspect 320-dimensional evolutionary sequence embeddings for bacterial AMR enzyme targets:")

registry = config_manager.get_config("target_registry")["approved_targets"]

target_key = st.selectbox(
    "Select Target Sequence Preset:",
    options=list(registry.keys()),
    format_func=lambda k: f"{k}: {registry[k]['name']} ({registry[k]['organism']})"
)

seq_input = st.text_area(
    "Primary Amino Acid Sequence (FASTA or bare string):",
    value=registry[target_key]["sequence"],
    height=140
)

clean_seq = "".join(seq_input.split()).upper()

if clean_seq:
    esm_service = ESMEmbeddingService()
    status = esm_service.get_service_status()

    st.info(f"**PLM Engine Status**: `{status['active_engine']}` | Model Architecture: `{status['model_name']}` (dim={status['embedding_dim']})")

    emb, seq_hash = esm_service.get_embedding(clean_seq, target_id=target_key)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Sequence Length", f"{len(clean_seq)} aa")
    with p2:
        st.metric("Embedding Dimension", emb.shape[0])
    with p3:
        st.metric("L2 Vector Norm", f"{float(torch.norm(emb)):.3f}")
    with p4:
        st.metric("Pooling Strategy", "Mean Sequence")

    st.markdown("---")

    # Amino Acid Composition Chart
    st.subheader("Amino Acid Composition Breakdown")
    aa_counts = {}
    for aa in clean_seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    aa_df = pd.DataFrame(list(aa_counts.items()), columns=["Amino Acid", "Count"]).sort_values("Count", ascending=False)
    st.bar_chart(aa_df.set_index("Amino Acid"), height=250)

    with st.expander("Inspect Embedding Tensor Vector (First 50 Dimensions)"):
        st.write(emb.numpy()[:50])
