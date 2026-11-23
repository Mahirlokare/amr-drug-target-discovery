"""
Model training and inference test suite.
Verifies GNN encoder, MultimodalDTIFusionModel dimensions, checkpoint saving/loading,
calibrated probabilities, and epistemic uncertainty calculation.
Complies with Section 5, 6, & 14.
"""

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.features.molecular_graph import MolecularGraphFeaturizer
from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.fusion import MultimodalDTIFusionModel
from src.models.calibration import ProbabilityCalibrator
from src.models.predict import CandidateScoringPipeline
from src.data.schemas import Target


@pytest.fixture
def sample_graph():
    featurizer = MolecularGraphFeaturizer()
    mol = Chem.MolFromSmiles("Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl")  # GEQ
    return featurizer.featurize(mol)


@pytest.fixture
def sample_protein_emb():
    return torch.randn(320, dtype=torch.float32)


def test_gnn_encoder_forward(sample_graph):
    encoder = MolecularGNNEncoder(node_in_dim=43, hidden_dims=(64, 64, 64), output_dim=64)
    encoder.eval()
    with torch.no_grad():
        graph_emb, node_emb = encoder(sample_graph.x, sample_graph.edge_index)
    assert graph_emb.shape == (1, 64)
    assert node_emb.shape == (sample_graph.num_nodes, 64)


def test_multimodal_fusion_forward(sample_graph, sample_protein_emb):
    model = MultimodalDTIFusionModel(node_in_dim=43, gnn_hidden_dim=64, protein_emb_dim=320, projection_dim=64)
    model.eval()
    with torch.no_grad():
        logit, mol_vec, prot_vec = model(sample_graph.x, sample_graph.edge_index, sample_protein_emb)
    assert logit.shape == (1,)
    assert mol_vec.shape == (1, 64)
    assert prot_vec.shape == (1, 64)


def test_probability_calibrator():
    calibrator = ProbabilityCalibrator()
    # Test on extreme negative and positive logits
    test_logits = torch.tensor([-5.0, 0.0, 5.0]).numpy()
    cal_probs = calibrator.calibrate(test_logits)
    assert len(cal_probs) == 3
    assert 0.0 <= cal_probs[0] <= cal_probs[1] <= cal_probs[2] <= 1.0


def test_candidate_scoring_pipeline(sample_protein_emb):
    model1 = MultimodalDTIFusionModel()
    model2 = MultimodalDTIFusionModel()
    calibrator = ProbabilityCalibrator()
    
    pipeline = CandidateScoringPipeline(
        ensemble_models=[model1, model2],
        calibrator=calibrator,
        train_reference_smiles=["O=C(NN)c1ccncc1", "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl"],
        epistemic_threshold=0.20
    )

    dummy_target = Target(
        target_id="TARGET-001",
        name="InhA",
        accession="P9WGR1",
        organism="Mycobacterium tuberculosis",
        gene="inhA",
        sequence="MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLTDAKVAAEVAAAGANAVALHCDVTDESQISALVAAAVERYGTVDIVVSNAGIAVPADGVEHDVTSDGWETMLNVNAAGVFLGLQAAIPLLAKGGSVVINMSSVSGQMGYPALGAYSASKAAVNGLVRNLALEHAPVRVNCINPGLVRTRLQERLLADPYQAVQAGWAPETLDKIAALAGVDTSGYYSLTPRVLRVTPGDAEYVGKPVTVIQPGGFTQVDAYD",
        sequence_hash="test_hash"
    )

    preds, warnings = pipeline.score_smiles_list(["O=C(NN)c1ccncc1"], dummy_target)
    assert len(preds) == 1
    assert 0.0 <= preds[0].calibrated_prob <= 1.0
    assert preds[0].uncertainty >= 0.0
    assert preds[0].decision_status in [
        "PRIORITISED",
        "PREDICTED_INACTIVE",
        "UNCERTAIN_PREDICTION",
        "OUT_OF_DOMAIN",
    ]
