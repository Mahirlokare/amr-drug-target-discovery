"""
Candidate Discovery: Ranked compounds and prediction evidence.
Organized per Priority 2: Candidate Discovery.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.common.config import config_manager
from src.data.schemas import Target
from src.models.calibration import ProbabilityCalibrator
from src.models.fusion import MultimodalDTIFusionModel
from src.models.predict import CandidateScoringPipeline
from src.admet.adapters import ADMETAdapter
from src.storage.db import db_manager
from src.data.resolver import chemical_resolver

st.set_page_config(page_title="Candidate Discovery | DeepTarget-AMR", page_icon="📊", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Candidate Discovery</span>
</div>
""", unsafe_allow_html=True)

st.title("📊 Candidate Discovery & Triage Ranking")
st.markdown("High-throughput in-silico screening with calibrated probabilities, epistemic uncertainties, and chemical space applicability domain verification:")

# Target Selection
target_reg = config_manager.get_config("target_registry")["approved_targets"]
selected_tid = st.selectbox(
    "Select Target Protein for Screening:",
    list(target_reg.keys()),
    format_func=lambda x: f"{x}: {target_reg[x]['name']} ({target_reg[x]['organism']})"
)

# Candidate Input Source
st.subheader("Input Chemical Library")
input_mode = st.radio("Choose Input Method:", ["Preset Benchmark Panel", "Paste Multi-Line SMILES / Names", "Upload CSV/TXT File"], horizontal=True)

raw_inputs = []
if input_mode == "Preset Benchmark Panel":
    default_panel = [
        ("Isoniazid", "O=C(NN)c1ccncc1"),
        ("GEQ (4TZK Co-crystal)", "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl"),
        ("Triclosan", "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl"),
        ("NITD-916", "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1"),
        ("Ethionamide", "CCc1cc(C(N)=S)ccn1"),
        ("Diazaborine InhA hit", "OB1ON(Cc2ccccc2)c3ccccc13"),
        ("Pyridomycin core", "CC(C)CC1NC(=O)C(Cc2ccncc2)NC(=O)C(CC(C)C)OC(=O)c3cccnc3N1C"),
        ("Aspirin (Decoy Inactive)", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Acetaminophen (Decoy)", "CC(=O)Nc1ccc(O)cc1"),
        ("Nicotinic Acid (Decoy)", "O=C(O)c1cccnc1")
    ]
    st.write(f"Pre-loaded panel of **{len(default_panel)}** benchmark compounds.")
    raw_inputs = [smi for _, smi in default_panel]
elif input_mode == "Paste Multi-Line SMILES / Names":
    text_data = st.text_area(
        "Enter one compound per line (SMILES or Drug Names like Bedaquiline, Isoniazid):",
        value="O=C(NN)c1ccncc1\nBedaquiline\nCc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl\nCC(=O)Oc1ccccc1C(=O)O",
        height=140
    )
    raw_inputs = [line.strip() for line in text_data.split("\n") if line.strip()]
else:
    uploaded = st.file_uploader("Upload CSV or TXT file (must contain 'smiles' column or lines):", type=["csv", "txt"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
            col_match = [c for c in df_up.columns if "smiles" in c.lower()]
            if col_match:
                raw_inputs = df_up[col_match[0]].dropna().astype(str).tolist()
            else:
                raw_inputs = df_up.iloc[:, 0].dropna().astype(str).tolist()
        else:
            raw_inputs = [line.decode("utf-8").strip() for line in uploaded if line.strip()]

if st.button("🚀 Run Multimodal Scoring & ADMET Triage", type="primary"):
    with st.spinner("Executing PyG GNN + ESM-2 scoring across candidate panel..."):
        valid_candidates = []
        for inp in raw_inputs:
            resolved = chemical_resolver.resolve_input(inp)
            if resolved["valid"]:
                valid_candidates.append(resolved["canonical_smiles"])

        if not valid_candidates:
            st.error("No valid chemical structures could be parsed.")
            st.stop()

        pipeline = CandidateScoringPipeline.load_default_pipeline(_ROOT)

        t_meta = target_reg[selected_tid]
        t_obj = Target(
            target_id=selected_tid,
            name=t_meta["name"],
            accession=t_meta["accession"],
            organism=t_meta["organism"],
            gene=t_meta["gene"],
            sequence=t_meta["sequence"],
            sequence_hash=t_meta["sequence_sha256"]
        )

        preds, warnings = pipeline.score_smiles_list(valid_candidates, t_obj)
        admet = ADMETAdapter()

        results = []
        for p in preds:
            admet_res = admet.evaluate_candidate(p.smiles, p.compound_id, p.prediction_id)
            # Save prediction to DB
            try:
                db_manager.save_prediction({
                    "prediction_id": p.prediction_id,
                    "run_id": p.model_run_id,
                    "compound_id": p.compound_id,
                    "target_id": p.target_id,
                    "smiles": p.smiles,
                    "calibrated_prob": p.calibrated_prob,
                    "uncertainty": p.uncertainty,
                    "is_uncertain": p.is_uncertain,
                    "is_out_of_domain": p.is_out_of_domain,
                    "decision_status": p.decision_status
                })
            except Exception:
                pass

            results.append({
                "Compound ID": p.compound_id,
                "SMILES": p.smiles,
                "Calibrated Active Prob": f"{p.calibrated_probability:.1%}",
                "Uncertainty (σ)": f"{p.uncertainty:.3f}",
                "Domain Status": "OOD" if p.is_out_of_domain else "In-Domain",
                "Lipinski Ro5": "Pass" if admet_res.lipinski_pass else "Fail",
                "PAINS Alerts": admet_res.pains_alert_count,
                "QED": f"{admet_res.qed_score:.2f}",
                "Decision Status": p.decision_status
            })

        df_res = pd.DataFrame(results)
        st.subheader("Screened Candidate Ranking Table")
        st.dataframe(df_res, use_container_width=True)

        csv_data = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Triage CSV", data=csv_data, file_name=f"triage_results_{selected_tid}.csv", mime="text/csv")
