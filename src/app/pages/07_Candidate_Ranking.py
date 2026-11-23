"""
Page 4: Candidate Screening & Prioritisation Ranking.
Complies with Project 9 Blueprint Section 15.2 (Candidate Ranking Table).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
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

st.set_page_config(page_title="Candidate Ranking | AMR Discovery", page_icon="📊", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">Candidate Screening</span>
</div>
""", unsafe_allow_html=True)

st.title("📊 Candidate Screening & Prioritisation Ranking")

# Target Selection
target_reg = config_manager.get_config("target_registry")["approved_targets"]
selected_tid = st.selectbox(
    "Select Target Protein for Screening:",
    list(target_reg.keys()),
    format_func=lambda x: f"{x}: {target_reg[x]['name']} ({target_reg[x]['organism']})"
)

# Candidate Source Selector
st.subheader("Input Chemical Library")
input_mode = st.radio("Choose Input Method:", ["Preset Benchmark Panel", "Paste Multi-Line SMILES / Names", "Upload CSV/TXT File"], horizontal=True)

raw_inputs = []

if input_mode == "Preset Benchmark Panel":
    default_panel = [
        ("Isoniazid", "O=C(NN)c1ccncc1"),
        ("GEQ (4TZK Co-crystal)", "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl"),
        ("Triclosan", "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl"),
        ("Ethionamide", "CCc1cc(C(N)=S)ccn1"),
        ("NITD-916", "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1"),
        ("Macozinone", "CC1CCN(CC1)c2ccc(c(c2)[N+](=O)[O-])C(=O)N3CC4(CC3)CCC(CC4)F"),
        ("Ciprofloxacin", "Ciprofloxacin"),
        ("Bedaquiline", "Bedaquiline"),
        ("Meropenem", "Meropenem"),
        ("Aspirin (Decoy)", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Acetaminophen (Decoy)", "CC(=O)Nc1ccc(O)cc1"),
        ("Coumarin (Decoy)", "O=C1Oc2ccccc2C=C1")
    ]
    raw_inputs = [smi for _, smi in default_panel]
    st.info(f"Loaded {len(raw_inputs)} diverse antimicrobial leads, clinical therapeutics, and negative controls.")

elif input_mode == "Paste Multi-Line SMILES / Names":
    user_text = st.text_area(
        "Enter SMILES or Compound Names (one per line, e.g. 'Isoniazid' or 'O=C(NN)c1ccncc1'):",
        value="Isoniazid\nNITD-916\nCiprofloxacin\nBedaquiline\nO=C(NN)c1ccncc1.Cl\nCC(=O)Oc1ccccc1C(=O)O",
        height=140
    )
    raw_inputs = [line.strip() for line in user_text.splitlines() if line.strip()]

else:
    uploaded = st.file_uploader("Upload CSV containing 'smiles' or 'structure' column:", type=["csv", "txt"])
    if uploaded:
        up_df = pd.read_csv(uploaded)
        col = None
        for c in up_df.columns:
            if c.lower() in ["smiles", "structure", "compound", "name"]:
                col = c
                break
        col = col or up_df.columns[0]
        raw_inputs = up_df[col].dropna().astype(str).tolist()
        st.success(f"Extracted {len(raw_inputs)} records from column '{col}'.")

# Resolve and standardize compounds
valid_candidates = []
resolve_warnings = []

for item in raw_inputs:
    res = chemical_resolver.resolve_input(item)
    if res["valid"]:
        valid_candidates.append(res["canonical_smiles"])
    else:
        resolve_warnings.append(f"'{item[:25]}...': {res['error']}")

if valid_candidates:
    st.caption(f"✓ **{len(valid_candidates)}** valid chemical structures ready for screening against **{selected_tid}**.")
    if resolve_warnings:
        with st.expander(f"⚠️ {len(resolve_warnings)} Input Parsing Notice(s)"):
            for w in resolve_warnings:
                st.caption(w)

st.markdown("---")

run_scoring = st.button("🚀 Run Multimodal Scoring & Triage", type="primary", disabled=len(valid_candidates) == 0)

if run_scoring:
    with st.spinner(f"Executing GNN convolutions, ESM-2 PLM projection for {selected_tid}, and ADMET filters..."):
        try:
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
                models = [MultimodalDTIFusionModel()]

            calibrator = ProbabilityCalibrator()

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

            preds, warnings = pipeline.score_smiles_list(valid_candidates, t_obj)
            admet = ADMETAdapter()

            rows = []
            for rank, p in enumerate(preds, 1):
                admet_run = admet.evaluate_candidate(p.smiles, p.compound_id, p.prediction_id)
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
                except Exception:
                    pass

                rows.append({
                    "Rank": rank,
                    "Compound ID": p.compound_id,
                    "SMILES": p.smiles,
                    "Calibrated Prob": round(float(p.calibrated_probability), 3),
                    "Uncertainty (σ)": round(float(p.uncertainty), 3),
                    "OOD Flag": "⚠️ OOD" if p.is_out_of_domain else "In-Domain",
                    "Nearest Train Sim": round(float(p.nearest_train_similarity), 2),
                    "Decision Status": p.decision_status,
                    "Ro5 Pass": "✅ Pass" if admet_run.lipinski_pass else "❌ Fail",
                    "PAINS Alerts": admet_run.pains_alert_count,
                    "QED": round(float(admet_run.qed_score), 2),
                    "MW": round(float(admet_run.molecular_weight), 1),
                    "LogP": round(float(admet_run.logp), 2)
                })

            df_rank = pd.DataFrame(rows)
            st.session_state["ranked_results"] = df_rank
            st.session_state["current_target"] = selected_tid
        except Exception as e:
            st.error(f"Candidate ranking execution notice: {e}")

if "ranked_results" in st.session_state:
    df_res = st.session_state["ranked_results"]
    st.subheader(f"Triage Leaderboard: {st.session_state.get('current_target', 'Screened')}")
    st.caption("Multimodal calibrated predictions and ADMET safety triage under scaffold separation:")

    # Interactive filtering controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        min_prob = st.slider("Min Calibrated Probability:", 0.0, 1.0, 0.0, 0.05)
    with f_col2:
        max_unc = st.slider("Max Epistemic Uncertainty (σ):", 0.0, 0.5, 0.5, 0.02)
    with f_col3:
        pains_only = st.checkbox("PAINS-Free Only (0 alerts)", value=False)

    filtered_df = df_res[
        (df_res["Calibrated Prob"] >= min_prob) &
        (df_res["Uncertainty (σ)"] <= max_unc)
    ]
    if pains_only:
        filtered_df = filtered_df[filtered_df["PAINS Alerts"] == 0]

    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Shortlist (CSV)",
        data=csv_data,
        file_name=f"amr_screened_{selected_tid}.csv",
        mime="text/csv"
    )

    # 2D Structure Preview Selector
    st.markdown("---")
    st.subheader("2D Chemical Inspector for Screened Candidates")
    cid_choices = filtered_df["Compound ID"].tolist()
    if cid_choices:
        inspect_cid = st.selectbox("Inspect Compound Structure:", cid_choices)
        matched_row = filtered_df[filtered_df["Compound ID"] == inspect_cid].iloc[0]
        ins_smi = matched_row["SMILES"]

        i_c1, i_c2 = st.columns([1, 1])
        with i_c1:
            try:
                mol = Chem.MolFromSmiles(ins_smi)
                if mol:
                    drawer = rdMolDraw2D.MolDraw2DSVG(360, 200)
                    drawer.drawOptions().clearBackground = False
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    st.image(drawer.GetDrawingText(), use_container_width=True)
            except Exception:
                st.warning("Could not render 2D depiction.")
        with i_c2:
            st.write(f"**Compound ID**: `{inspect_cid}`")
            st.write(f"**Canonical SMILES**: `{ins_smi}`")
            st.write(f"**Calibrated Prob**: `{matched_row['Calibrated Prob']}`")
            st.write(f"**Epistemic Uncertainty**: `{matched_row['Uncertainty (σ)']}`")
            st.write(f"**Decision Status**: `{matched_row['Decision Status']}`")
            st.write(f"**Molecular Weight**: `{matched_row['MW']} g/mol` | **LogP**: `{matched_row['LogP']}`")
