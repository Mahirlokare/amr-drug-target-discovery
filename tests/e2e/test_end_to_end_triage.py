"""
End-to-end candidate triage workflow integration test.
Tests the complete execution pipeline from raw user input to chemical standardization,
PyG graph featurization, DTI ensemble scoring, calibration, ADMET filtering, and DB persistence.
Complies with Section 14 & 16.
"""

import pytest
import torch

from src.data.resolver import chemical_resolver
from src.data.standardise import CompoundStandardizer
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.protein_esm import ESMEmbeddingService
from src.models.fusion import MultimodalDTIFusionModel
from src.models.calibration import ProbabilityCalibrator
from src.models.predict import CandidateScoringPipeline
from src.admet.adapters import ADMETAdapter
from src.data.schemas import Target
from src.storage.db import DatabaseManager


@pytest.fixture
def isolated_db(tmp_path):
    db_file = tmp_path / "e2e_amr.db"
    manager = DatabaseManager(db_url=None)
    manager.db_path = db_file
    manager.init_schema()
    return manager


def test_full_pipeline_triage(isolated_db):
    # 1. Resolve raw user input (Drug name or SMILES)
    resolved = chemical_resolver.resolve_input("Isoniazid")
    assert resolved["valid"] is True
    smi = resolved["canonical_smiles"]
    assert "c1ccncc1" in smi

    # 2. Chemical Standardisation
    standardizer = CompoundStandardizer()
    std_smi, inchikey, status, _ = standardizer.standardize_smiles(smi)
    assert status in ["SUCCESS", "MODIFIED"]
    assert len(inchikey) == 27

    # 3. Protein representation
    esm_service = ESMEmbeddingService()
    target_seq = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLTDAKVAAEVAAAGANAVALHCDVTDESQISALVAAAVERYGTVDIVVSNAGIAVPADGVEHDVTSDGWETMLNVNAAGVFLGLQAAIPLLAKGGSVVINMSSVSGQMGYPALGAYSASKAAVNGLVRNLALEHAPVRVNCINPGLVRTRLQERLLADPYQAVQAGWAPETLDKIAALAGVDTSGYYSLTPRVLRVTPGDAEYVGKPVTVIQPGGFTQVDAYD"
    emb, seq_hash = esm_service.get_embedding(target_seq, target_id="TARGET-001")
    assert emb.shape == (320,)

    # 4. Multimodal DTI Scoring Pipeline
    model = MultimodalDTIFusionModel()
    calibrator = ProbabilityCalibrator()
    pipeline = CandidateScoringPipeline(
        ensemble_models=[model],
        calibrator=calibrator,
        train_reference_smiles=[std_smi],
        epistemic_threshold=0.15
    )

    target_obj = Target(
        target_id="TARGET-001",
        name="InhA",
        accession="P9WGR1",
        organism="Mycobacterium tuberculosis",
        gene="inhA",
        sequence=target_seq,
        sequence_hash=seq_hash
    )

    preds, warnings = pipeline.score_smiles_list([std_smi], target_obj)
    assert len(preds) == 1
    pred = preds[0]
    assert 0.0 <= pred.calibrated_prob <= 1.0

    # 5. ADMET & Safety Triage
    admet_adapter = ADMETAdapter()
    admet_run = admet_adapter.evaluate_candidate(std_smi, pred.compound_id, pred.prediction_id)
    assert admet_run.lipinski_pass is True
    assert admet_run.pains_alert_count == 0

    # 6. Database Persistence
    isolated_db.save_target({
        "target_id": target_obj.target_id,
        "name": target_obj.name,
        "accession": target_obj.accession,
        "organism": target_obj.organism,
        "gene": target_obj.gene,
        "sequence_hash": target_obj.sequence_hash
    })
    isolated_db.save_prediction({
        "prediction_id": pred.prediction_id,
        "run_id": pred.model_run_id,
        "compound_id": pred.compound_id,
        "target_id": pred.target_id,
        "smiles": pred.smiles,
        "calibrated_prob": pred.calibrated_prob,
        "uncertainty": pred.uncertainty,
        "is_uncertain": pred.is_uncertain,
        "is_out_of_domain": pred.is_out_of_domain,
        "decision_status": pred.decision_status
    })

    db_preds = isolated_db.list_predictions(target_id="TARGET-001")
    assert len(db_preds) == 1
    assert db_preds[0]["prediction_id"] == pred.prediction_id
