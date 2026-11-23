"""
Page 7: Documentation, Dataset Card, Model Card & Governance.
Complies with Project 9 Blueprint Section 16.3 & 15.1 (Reports).
"""

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Reports & Cards | AMR Discovery", page_icon="📑", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Scientific Governance & Cards</span>
</div>
""", unsafe_allow_html=True)

st.title("📑 Scientific Governance & Model / Dataset Cards")

tab1, tab2, tab3 = st.tabs(["Dataset Card", "Model Card", "Validation Protocol & Governance"])

with tab1:
    st.subheader("Dataset Card: M. tuberculosis InhA Bioactivity Benchmark")
    st.markdown("""
    ### 1. Dataset Description
    - **Dataset Name**: `inha_benchmark_v1`
    - **Target Organism**: *Mycobacterium tuberculosis* (strain ATCC 25618 / H37Rv)
    - **Taxonomy ID**: `83332`
    - **Target Protein**: Enoyl-[acyl-carrier-protein] reductase [NADH] (InhA, Rv1484)
    - **Primary Endpoint**: Direct enzyme biochemical inhibition IC50 (nM)
    - **Primary Label Policy**:
      - Active (1): IC50 <= 1000 nM (pIC50 >= 6.0)
      - Inactive (0): IC50 > 10000 nM (pIC50 < 5.0)
      - Intermediate / Quarantined: 1000 nM < IC50 <= 10000 nM

    ### 2. Provenance & Curation
    - **Source**: ChEMBL Target CHEMBL2014 & peer-reviewed antimycobacterial literature.
    - **Standardization**: RDKit MolStandardize (largest organic fragment selection, uncharging, canonical tautomer generation).
    - **Deduplication**: Geometric mean aggregation for replicates with < 1.0 log span; discordant replicates quarantined.

    ### 3. Limitations & Known Failure Modes
    - Whole-cell MIC and cell-wall penetration are NOT measured by this enzyme assay dataset.
    - Assay conditions (NADH cofactor concentration, buffer pH) may introduce batch effects across documents.
    """)

with tab2:
    st.subheader("Model Card: Multimodal DTI (Molecular GNN + ESM-2)")
    st.markdown("""
    ### 1. Model Architecture
    - **Molecule Encoder**: 3-layer Graph Convolutional Network (GCN) with 43-dimensional invariant atom features and global pooling (mean + max).
    - **Protein Encoder**: ESM-2 (320-dimensional precomputed mean-pooled sequence representation).
    - **Multimodal Fusion**: Concatenation + LayerNorm + 3-layer MLP interaction head.
    - **Uncertainty Quantification**: 3-seed Deep Ensemble + Monte Carlo Dropout.
    - **Calibration**: Validation-fitted Platt scaling.

    ### 2. Intended Use
    - Prioritisation and ranking of candidate small molecules for follow-up biochemical in-vitro assay testing against InhA.
    - Chemical series triage and scaffold diversity analysis.

    ### 3. Prohibited Interpretations (Section 1.4)
    - **NOT a clinical recommendation**: Must never be used to prescribe antibiotics or select patient therapies.
    - **NOT proof of efficacy**: A high computational score does not guarantee in-vivo bacterial clearance.
    """)

with tab3:
    st.subheader("Validation Protocol & Scientific Integrity")
    st.markdown("""
    ### 1. Leakage Controls (Section 11)
    - **Scaffold-Aware Split**: Bemis-Murcko framework clustering with automated assertion:
      `assert train_scaffolds.isdisjoint(test_scaffolds)`
    - **Zero Lookahead**: Model hyperparameters, early stopping, and calibration thresholds are tuned exclusively on training/validation partitions. Test data remains untouched until final scoring.

    ### 2. Statistical Reporting (Section 11.5)
    - Primary reported metrics: AUROC, PR-AUC, Enrichment Factor @ 1% (EF@1%), Brier Score, and Expected Calibration Error (ECE).
    - 95% Confidence intervals derived via 500-iteration bootstrap resampling.
    """)
