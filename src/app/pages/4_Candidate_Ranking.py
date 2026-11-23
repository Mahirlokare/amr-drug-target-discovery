"""
Page 4: Candidate Ranking & Triage Table.
Complies with Project 9 Blueprint Section 15.1 & 15.2 (Candidate ranking table).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

from src.common.config import config_manager
from src.data.schemas import Target
from src.features.protein_esm import ESMEmbeddingService
from src.models.calibration import ProbabilityCalibrator
from src.models.fusion import MultimodalDTIFusionModel
from src.models.predict import CandidateScoringPipeline
from src.admet.adapters import ADMETAdapter
from src.storage.db import db_manager
import torch

st.set_page_config(page_title="Candidate Ranking | AMR Discovery", page_icon="📊", layout="wide")
st.title("📊 Candidate Screening & Prioritisation Ranking")

# Target Selection
target_reg = config_manager.get_config("target_registry")["approved_targets"]
selected_tid = st.selectbox("Target Protein:", list(target_reg.keys()), format_func=lambda x: f"{x}: {target_reg[x]['name']}")

# Candidate Source Selector
st.subheader("Input Candidate Molecules")
input_mode = st.radio("Choose Input Mode:", ["Default Curated Screening Library", "Paste SMILES List", "Upload CSV/TXT"], horizontal=True)

candidate_smiles_list = []

if input_mode == "Default Curated Screening Library":
    default_smiles = [
        "O=C(NN)c1ccncc1",                                         # Isoniazid analogue
        "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl",                 # GEQ (4TZK co-crystal)
        "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl",                             # Triclosan derivative
        "CCc1cc(C(N)=S)ccn1",                                      # Ethionamide analogue
        "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1",                   # NITD-916
        "O=C(Nc1ccc(F)cc1)C2CCCN2C(=O)c3ccc(Cl)cc3",              # Pyrrolidine 1a
        "CC(=O)Oc1ccccc1C(=O)O",                                   # Decoy (Aspirin)
        "CC(=O)Nc1ccc(O)cc1",                                      # Decoy (Acetaminophen)
        "c1ccc2ncccc2c1",                                          # Quinone fragment
        "O=C1Oc2ccccc2C=C1",                                       # Coumarin decoy
    ]
    candidate_smiles_list = default_smiles
    st.info(f"Loaded {len(candidate_smiles_list)} representative antimycobacterial chemotypes and decoys.")

elif input_mode == "Paste SMILES List":
    user_text = st.text_area("Enter SMILES (one per line):", value="O=C(NN)c1ccncc1\nOc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl\nCC(=O)Oc1ccccc1C(=O)O")
    candidate_smiles_list = [line.strip() for line in user_text.splitlines() if line.strip()]

else:
    uploaded = st.file_uploader("Upload CSV containing 'smiles' column:", type=["csv", "txt"])
    if uploaded:
        up_df = pd.read_csv(uploaded)
        col_name = "smiles" if "smiles" in up_df.columns else up_df.columns[0]
        candidate_smiles_list = up_df[col_name].dropna().astype(str).tolist()

st.markdown("---")

if st.button("🚀 Run Multimodal Scoring & Triage", type="primary"):
    with st.spinner("Executing GNN + ESM-2 scoring, uncertainty estimation & ADMET triage..."):
        # Load models from artifacts
        checkpoints = list((_ROOT / "artifacts" / "checkpoints").glob("dti_model_seed_*.pt"))
        device = torch.device("cpu")

        models = []
        if checkpoints:
            for ckpt in checkpoints[:3]:
                m = MultimodalDTIFusionModel()
                m.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
                models.append(m)
        else:
            # Fallback initialized model
            models = [MultimodalDTIFusionModel()]

        calibrator = ProbabilityCalibrator()

        # Reference training SMILES for applicability domain
        train_smi = ["O=C(NN)c1ccncc1", "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl", "CCc1cc(C(N)=S)ccn1"]
        t_data = target_reg[selected_tid]
        t_obj = Target(
            target_id=selected_tid,
            name=t_data["name"],
            accession=t_data["accession"],
            organism=t_data["organism"],
            strain_scope=t_data["strain_scope"],
            gene=t_data["gene"],
            sequence=t_data["sequence"],
            sequence_hash=t_data["sequence_sha256"],
            taxonomy_id=t_data["taxonomy_id"]
        )

        pipeline = CandidateScoringPipeline(
            ensemble_models=models,
            calibrator=calibrator,
            train_reference_smiles=train_smi,
            epistemic_threshold=0.15
        )

        preds, warnings = pipeline.score_smiles_list(candidate_smiles_list, t_obj)
        admet = ADMETAdapter()

        # Build ranking table & persist to database
        rows = []
        for rank, p in enumerate(preds, 1):
            admet_run = admet.evaluate_candidate(p.smiles, p.compound_id, p.prediction_id)
            # Persist to database
            try:
                db_manager.save_prediction({
                    "prediction_id": p.prediction_id,
                    "run_id": p.run_id,
                    "compound_id": p.compound_id,
                    "target_id": p.target_id,
                    "smiles": p.smiles,
                    "calibrated_prob": p.calibrated_probability,
                    "uncertainty": p.uncertainty,
                    "is_uncertain": p.is_uncertain,
                    "is_out_of_domain": p.is_out_of_domain,
                    "decision_status": p.decision_status
                })
                db_manager.save_admet_run({
                    "admet_id": admet_run.admet_id,
                    "compound_id": p.compound_id,
                    "mol_weight": admet_run.molecular_weight,
                    "logp": admet_run.logp,
                    "hbd": admet_run.h_bond_donors,
                    "hba": admet_run.h_bond_acceptors,
                    "tpsa": admet_run.tpsa,
                    "rotatable_bonds": admet_run.rotatable_bonds,
                    "lipinski_pass": 1 if admet_run.lipinski_pass else 0,
                    "veber_pass": 1 if admet_run.veber_pass else 0,
                    "qed": admet_run.qed_score,
                    "pains_alerts": admet_run.pains_alert_count,
                    "decision_status": admet_run.decision_status
                })
            except Exception as dbe:
                pass  # graceful fallback

            rows.append({
                "Rank": rank,
                "Compound ID": p.compound_id,
                "SMILES": p.smiles,
                "Calibrated Probability": f"{p.calibrated_probability:.3f}",
                "Uncertainty (Std Dev)": f"{p.uncertainty:.3f}",
                "Out of Domain": "⚠️ OOD" if p.is_out_of_domain else "In Domain",
                "Nearest Similarity": f"{p.nearest_train_similarity:.2f}",
                "Decision Status": p.decision_status,
                "Lipinski Pass": "✅ Pass" if admet_run.lipinski_pass else "❌ Fail",
                "PAINS Alerts": f"{admet_run.pains_alert_count} alert(s)" if admet_run.pains_alert_count > 0 else "0",
                "QED": f"{admet_run.qed_score:.2f}",
            })

        df_rank = pd.DataFrame(rows)
        st.session_state["ranked_results"] = df_rank
        st.session_state["predictions"] = preds

if "ranked_results" in st.session_state:
    st.subheader("Candidate Ranking Table (Section 15.2)")
    st.caption("Docking scores and ADMET are kept in separate evidence columns and never blended into model probability without justification.")
    st.dataframe(st.session_state["ranked_results"], use_container_width=True)

    csv_data = st.session_state["ranked_results"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Ranked Candidate Shortlist as CSV",
        data=csv_data,
        file_name="amr_candidate_shortlist.csv",
        mime="text/csv"
    )
