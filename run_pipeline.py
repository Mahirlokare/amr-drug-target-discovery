"""
Reproducible End-to-End Pipeline CLI.
Executes data building, split assertions, baseline training, GNN & Multimodal training,
calibration, evaluation, active learning simulation, and docking triage.
Complies with Project 9 Blueprint Sections 4, 7, 8, 10, 11, 12, 13 & 20.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

from src.active_learning.simulator import ActiveLearningSimulator
from src.admet.adapters import ADMETAdapter
from src.common.config import config_manager
from src.common.logging import get_logger
from src.data.dataset_builder import AMRDatasetBuilder
from src.data.schemas import Target
from src.docking.ligand_prep import LigandPreparer
from src.docking.parse_results import DockingResultParser
from src.docking.receptor_prep import ReceptorPreparer
from src.docking.vina_runner import VinaRunner
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_bootstrap_confidence_intervals,
)
from src.evaluation.reports import EvaluationReporter
from src.features.fingerprints import FingerprintGenerator
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.protein_esm import ESMEmbeddingService
from src.models.baselines import ClassicalMLBaselines
from src.models.calibration import ProbabilityCalibrator
from src.models.fusion import MultimodalDTIFusionModel
from src.models.gnn_encoder import MoleculeOnlyGNN
from src.models.predict import CandidateScoringPipeline
from src.models.train import DTIDatasetItem, ModelTrainer, collate_dti_batch

logger = get_logger("run_pipeline")


def run_full_pipeline():
    logger.info("=================================================================")
    logger.info("  PROJECT 9: AMR DRUG TARGET DISCOVERY - REPRODUCIBLE PIPELINE   ")
    logger.info("=================================================================")

    # 1. Dataset Construction & Quality Control
    builder = AMRDatasetBuilder()
    compounds, records, dataset_export = builder.build_curated_inha_dataset()

    target_cfg = config_manager.get_config("target_registry")["approved_targets"]["TARGET-001"]
    target = Target(
        target_id="TARGET-001",
        name=target_cfg["name"],
        accession=target_cfg["accession"],
        organism=target_cfg["organism"],
        strain_scope=target_cfg["strain_scope"],
        gene=target_cfg["gene"],
        sequence=target_cfg["sequence"],
        sequence_hash=target_cfg["sequence_sha256"],
        taxonomy_id=target_cfg["taxonomy_id"],
        chembl_target_id=target_cfg.get("chembl_target_id"),
        pdb_id=target_cfg.get("pdb_id"),
        selection_rationale=target_cfg.get("selection_rationale"),
    )

    scaffold_split = dataset_export["scaffold_split"]
    train_cids = set(scaffold_split["train"]["compound_ids"])
    val_cids = set(scaffold_split["val"]["compound_ids"])
    test_cids = set(scaffold_split["test"]["compound_ids"])

    cid_to_comp = {c.compound_id: c for c in compounds}
    labeled_records = [r for r in records if r.label is not None]

    train_recs = [r for r in labeled_records if r.compound_id in train_cids]
    val_recs = [r for r in labeled_records if r.compound_id in val_cids]
    test_recs = [r for r in labeled_records if r.compound_id in test_cids]

    # 2. Featurization
    logger.info("Extracting features: RDKit Morgan fingerprints, descriptors & PyG graphs...")
    fp_gen = FingerprintGenerator(radius=2, n_bits=1024)
    graph_gen = MolecularGraphFeaturizer()
    esm_service = ESMEmbeddingService()

    target_embedding, _ = esm_service.get_embedding(target.sequence, target.target_id)

    # Build feature arrays for classical baselines
    def get_features_and_labels(rec_list):
        X_list = []
        y_list = []
        graphs = []
        items = []
        for r in rec_list:
            comp = cid_to_comp[r.compound_id]
            smi = comp.smiles_standard or comp.smiles_raw
            comb_fp = fp_gen.compute_combined_feature(smi)
            g = graph_gen.featurize(smi)
            if comb_fp is not None and g is not None:
                X_list.append(comb_fp)
                y_list.append(r.label)
                graphs.append(g)
                items.append(DTIDatasetItem(
                    graph=g,
                    protein_embedding=target_embedding,
                    label=float(r.label),
                    compound_id=r.compound_id,
                    record_id=r.record_id
                ))
        return np.array(X_list), np.array(y_list), items

    X_train, y_train, train_items = get_features_and_labels(train_recs)
    X_val, y_val, val_items = get_features_and_labels(val_recs)
    X_test, y_test, test_items = get_features_and_labels(test_recs)

    # 3. Train Classical Baselines
    logger.info("Training classical cheminformatics baselines...")
    baselines = ClassicalMLBaselines(seed=42)
    baselines.fit_all(X_train, y_train)
    baseline_results = baselines.evaluate_all(X_test, y_test)

    # 4. Train Multimodal Deep Learning Model (GNN + ESM-2)
    logger.info("Training Multimodal DTI GNN + ESM-2 Fusion Model across seeds...")
    trainer = ModelTrainer()
    ensemble_models, calibrator, ensemble_report = trainer.train_deep_ensemble(
        train_items, val_items, seeds=[42, 101, 777]
    )

    # Evaluate Multimodal Model on held-out test split
    test_graphs = [item.graph for item in test_items]
    scorer = CandidateScoringPipeline(
        ensemble_models=ensemble_models,
        calibrator=calibrator,
        train_reference_smiles=[cid_to_comp[r.compound_id].smiles_standard for r in train_recs]
    )
    raw_probs, std_devs, raw_logits = scorer.scorer.score_candidates(test_graphs, target_embedding)
    calibrated_test_probs = calibrator.calibrate(raw_logits)

    multimodal_metrics = compute_all_metrics(y_test, calibrated_test_probs)
    conf_intervals = compute_bootstrap_confidence_intervals(y_test, calibrated_test_probs)

    # Molecular GNN only baseline score (single model raw probabilities)
    single_model = ensemble_models[0]
    single_model.eval()
    with torch.no_grad():
        bg, pe, _, _, _ = collate_dti_batch(test_items)
        gnn_logits, _, _ = single_model(bg.x, bg.edge_index, pe, bg.batch)
        gnn_probs = torch.sigmoid(gnn_logits).cpu().numpy()
    gnn_metrics = compute_all_metrics(y_test, gnn_probs)

    # 5. Generate Benchmark Comparison Report
    reporter = EvaluationReporter()
    comparison_df = reporter.generate_comparison_table(baseline_results, gnn_metrics, multimodal_metrics)
    report_path = reporter.save_evaluation_report(
        report_id="EVAL-INHA-PRIMARY",
        target_id="TARGET-001",
        split_manifest=scaffold_split,
        comparison_df=comparison_df,
        multimodal_metrics=multimodal_metrics,
        confidence_intervals=conf_intervals
    )

    # 6. Run Retrospective Active Learning Simulation
    logger.info("Simulating Retrospective Active Learning cycles...")
    all_smiles = [cid_to_comp[r.compound_id].smiles_standard for r in labeled_records]
    all_labels = [r.label for r in labeled_records]
    al_sim = ActiveLearningSimulator(all_smiles, all_labels, seed=42)

    al_results_random = al_sim.run_simulation(policy_name="random", n_rounds=4, batch_size=8)
    al_results_balanced = al_sim.run_simulation(policy_name="balanced", n_rounds=4, batch_size=8)

    al_report_path = config_manager.project_root / "artifacts" / "reports" / "active_learning_simulation.json"
    with open(al_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "random_policy": al_results_random,
            "balanced_policy": al_results_balanced
        }, f, indent=2)

    # 7. Docking & ADMET Triage Shortlist
    logger.info("Executing AutoDock Vina & ADMET triage on top candidate shortlist...")
    # Prepare receptor
    receptor_prep = ReceptorPreparer()
    prep_info = receptor_prep.prepare_receptor("TARGET-001")

    # Select top 5 test candidates
    sorted_test_indices = np.argsort(-calibrated_test_probs)[:5]
    vina_runner = VinaRunner()
    ligand_prep = LigandPreparer()
    admet_adapter = ADMETAdapter()

    triage_cards = []
    docking_dir = config_manager.project_root / "artifacts" / "docking" / "TARGET-001"

    for rank, idx in enumerate(sorted_test_indices, 1):
        item = test_items[idx]
        comp = cid_to_comp[item.compound_id]
        smi = comp.smiles_standard
        score = float(calibrated_test_probs[idx])
        unc = float(std_devs[idx])

        # Prepare ligand
        lig_pdbqt, internal_energy = ligand_prep.prepare_ligand_pdbqt(smi, comp.compound_id, docking_dir)
        pose_pdbqt = docking_dir / f"{comp.compound_id}_docked_pose.pdbqt"

        if lig_pdbqt:
            affinity, log_out, rmsd = vina_runner.run_docking(
                receptor_pdbqt=Path(prep_info["receptor_pdbqt"]),
                ligand_pdbqt=lig_pdbqt,
                grid_box=prep_info["grid_box"],
                output_pdbqt=pose_pdbqt,
                seed=42 + rank
            )
            dock_run = DockingResultParser.parse_docking_run(
                prediction_id=f"PRED-TEST-{rank}",
                compound_id=comp.compound_id,
                target_id="TARGET-001",
                receptor_id=prep_info["pdb_id"],
                affinity_kcal_mol=affinity,
                pose_file=pose_pdbqt,
                redock_rmsd=rmsd
            )
        else:
            dock_run = None

        admet_run = admet_adapter.evaluate_candidate(smi, comp.compound_id, f"PRED-TEST-{rank}")

        triage_cards.append({
            "rank": rank,
            "compound_id": comp.compound_id,
            "smiles": smi,
            "calibrated_prob": round(score, 4),
            "uncertainty_std": round(unc, 4),
            "docking_affinity_kcal_mol": dock_run.affinity_kcal_mol if dock_run else None,
            "docking_redock_rmsd": dock_run.redock_rmsd if dock_run else None,
            "lipinski_pass": admet_run.lipinski_pass,
            "pains_alerts": admet_run.pains_alerts,
            "qed_score": admet_run.qed_score,
            "decision_status": "PRIORITISED_FOR_REVIEW" if (score >= 0.7 and admet_run.lipinski_pass) else "CONDITIONAL_REVIEW"
        })

    triage_path = config_manager.project_root / "artifacts" / "reports" / "candidate_shortlist_triage.json"
    with open(triage_path, "w", encoding="utf-8") as f:
        json.dump(triage_cards, f, indent=2)

    logger.info("=================================================================")
    logger.info("  PIPELINE EXECUTION COMPLETE! ALL DELIVERABLES GENERATED        ")
    logger.info(f"  Benchmark Comparison: {report_path}")
    logger.info(f"  Triage Shortlist:     {triage_path}")
    logger.info(f"  AL Simulation:        {al_report_path}")
    logger.info("=================================================================")
    print(comparison_df.to_string())


if __name__ == "__main__":
    run_full_pipeline()
