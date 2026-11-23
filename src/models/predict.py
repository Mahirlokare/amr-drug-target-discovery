"""
High-level candidate scoring, calibration, and prediction pipeline.
Complies with Project 9 Blueprint Section 7.5 & 9.3.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import uuid

from src.common.config import config_manager
from src.common.logging import get_logger
from src.data.schemas import Compound, Prediction, Target
from src.data.standardise import CompoundStandardizer
from src.features.fingerprints import check_applicability_domain
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.protein_esm import ESMEmbeddingService
from src.models.calibration import ProbabilityCalibrator
from src.models.uncertainty import DeepEnsembleScorer

logger = get_logger("predict")


class CandidateScoringPipeline:
    """Orchestrates candidate validation, featurization, ensemble scoring and uncertainty."""

    def __init__(
        self,
        ensemble_models: List[torch.nn.Module],
        calibrator: ProbabilityCalibrator,
        train_reference_smiles: Optional[List[str]] = None,
        epistemic_threshold: float = 0.15
    ):
        self.ensemble_models = ensemble_models
        self.calibrator = calibrator
        self.scorer = DeepEnsembleScorer(ensemble_models, epistemic_threshold=epistemic_threshold)
        self.standardizer = CompoundStandardizer()
        self.graph_featurizer = MolecularGraphFeaturizer()
        self.esm_service = ESMEmbeddingService()
        self.train_reference_smiles = train_reference_smiles or []
        self.epistemic_threshold = epistemic_threshold

    @classmethod
    def load_default_pipeline(cls, project_root: Optional[Path] = None) -> "CandidateScoringPipeline":
        """
        Loads the production model release deterministically from model_release_v1.json.
        Loads training reference SMILES from curated benchmark data.
        """
        import json
        from src.models.fusion import MultimodalDTIFusionModel

        root = project_root or config_manager.project_root
        ckpt_dir = root / "artifacts" / "checkpoints"
        manifest_path = ckpt_dir / "model_release_v1.json"

        device = torch.device("cpu")
        models: List[torch.nn.Module] = []

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                ckpt_files = manifest.get("ensemble", {}).get("checkpoints", [])
                for fname in ckpt_files:
                    fpath = ckpt_dir / fname
                    if fpath.exists():
                        m = MultimodalDTIFusionModel()
                        m.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
                        m.eval()
                        models.append(m)
            except Exception as e:
                logger.warning(f"Error loading model manifest: {e}")

        if not models:
            # Fallback to sorted checkpoints
            found_ckpts = sorted(list(ckpt_dir.glob("dti_model_seed_*.pt")))
            for fpath in found_ckpts[:3]:
                m = MultimodalDTIFusionModel()
                m.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
                m.eval()
                models.append(m)

        if not models:
            logger.warning("No checkpoint files found. Initializing architecture baseline.")
            models = [MultimodalDTIFusionModel()]

        # Load reference training SMILES
        train_smi: List[str] = []
        bench_file = root / "data" / "processed" / "inha_benchmark_v1.json"
        if bench_file.exists():
            try:
                with open(bench_file, "r", encoding="utf-8") as bf:
                    bdata = json.load(bf)
                    train_smi = [c["smiles_standard"] for c in bdata.get("compounds", []) if "smiles_standard" in c]
            except Exception as e:
                logger.warning(f"Could not load benchmark training SMILES: {e}")

        if not train_smi:
            train_smi = ["NNC(=O)c1ccncc1", "Oc1ccc(Cl)cc1Oc1ccc(Cl)cc1Cl", "CCc1cc(C(N)=S)ccn1"]

        calibrator = ProbabilityCalibrator(method="platt")

        return cls(
            ensemble_models=models,
            calibrator=calibrator,
            train_reference_smiles=train_smi,
            epistemic_threshold=0.15
        )

    def score_smiles_list(
        self,
        smiles_list: List[str],
        target: Target,
        run_id: Optional[str] = None
    ) -> Tuple[List[Prediction], List[Dict[str, Any]]]:
        """
        Scores a list of raw SMILES strings against a target protein.
        Returns:
            predictions: List of typed Prediction objects sorted by calibrated score descending.
            warnings: List of validation / standardization failure notices.
        """
        active_run_id = run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"
        target_emb, _ = self.esm_service.get_embedding(target.sequence, target.target_id)

        valid_graphs = []
        valid_compounds: List[Compound] = []
        warnings = []

        for idx, raw_smi in enumerate(smiles_list):
            cid = f"CAND-{idx + 1:04d}"
            compound = self.standardizer.standardize_to_compound(raw_smi, cid)

            if compound.standardisation_status == "FAILED":
                warnings.append({
                    "compound_id": cid,
                    "smiles_raw": raw_smi,
                    "error": compound.metadata.get("error", "Standardization failed")
                })
                continue

            graph = self.graph_featurizer.featurize(compound.smiles_standard)
            if graph is None:
                warnings.append({
                    "compound_id": cid,
                    "smiles_raw": raw_smi,
                    "error": "Graph featurization returned empty"
                })
                continue

            valid_graphs.append(graph)
            valid_compounds.append(compound)

        if not valid_graphs:
            return [], warnings

        # Deep ensemble scoring
        mean_probs, std_devs, raw_logits = self.scorer.score_candidates(valid_graphs, target_emb)

        # Calibration
        calibrated_probs = self.calibrator.calibrate(raw_logits)

        predictions: List[Prediction] = []

        for i, comp in enumerate(valid_compounds):
            # Applicability domain check against training distribution
            is_ood, max_sim, _ = check_applicability_domain(
                comp.smiles_standard, self.train_reference_smiles, k=5, min_similarity_threshold=0.35
            )

            is_uncertain = bool(std_devs[i] > self.epistemic_threshold)

            # Documented score policy decision status
            if is_ood:
                status = "OUT_OF_DOMAIN"
            elif is_uncertain:
                status = "UNCERTAIN_PREDICTION"
            elif calibrated_probs[i] >= 0.70:
                status = "PRIORITISED_CANDIDATE"
            elif calibrated_probs[i] <= 0.30:
                status = "PREDICTED_INACTIVE"
            else:
                status = "INTERMEDIATE_SCORE"

            pred = Prediction(
                prediction_id=f"PRED-{uuid.uuid4().hex[:8].upper()}",
                run_id=active_run_id,
                compound_id=comp.compound_id,
                target_id=target.target_id,
                smiles=comp.smiles_standard,
                raw_logit=round(float(raw_logits[i]), 4),
                raw_score=round(float(mean_probs[i]), 4),
                calibrated_probability=round(float(calibrated_probs[i]), 4),
                uncertainty=round(float(std_devs[i]), 4),
                is_uncertain=is_uncertain,
                is_out_of_domain=is_ood,
                nearest_train_similarity=round(max_sim, 4),
                decision_status=status
            )
            predictions.append(pred)

        # Sort by calibrated probability descending
        predictions.sort(key=lambda p: p.calibrated_probability, reverse=True)
        return predictions, warnings
