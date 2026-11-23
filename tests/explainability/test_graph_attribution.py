"""
Explainability and graph attribution test suite.
Verifies Input x Gradient saliency, atom score normalization,
and medicinal chemistry pharmacophore mapping.
"""

import pytest
import torch
from src.explainability.graph_attribution import GraphAttributionService
from src.models.fusion import MultimodalDTIFusionModel


@pytest.fixture
def attribution_service():
    model = MultimodalDTIFusionModel()
    return GraphAttributionService(model)


def test_atom_attribution_forward_pass(attribution_service):
    smiles = "O=C(NN)c1ccncc1"  # Isoniazid
    prot_emb = torch.randn(320, dtype=torch.float32)

    res = attribution_service.attribute_atoms_gradient(smiles, prot_emb)

    assert "error" not in res
    assert res["atom_count"] > 0
    assert len(res["atom_symbols"]) == res["atom_count"]
    assert len(res["atom_importances"]) == res["atom_count"]

    # Scores must be bounded in [0.0, 1.0]
    for score in res["atom_importances"]:
        assert 0.0 <= score <= 1.0

    # Top contributing atoms list
    assert len(res["top_contributing_atoms"]) <= 5


def test_attribution_invalid_smiles(attribution_service):
    prot_emb = torch.randn(320, dtype=torch.float32)
    res = attribution_service.attribute_atoms_gradient("INVALID_SMILES_STRING", prot_emb)
    assert "error" in res


def test_pharmacophore_detection(attribution_service):
    smiles = "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl"  # GEQ
    prot_emb = torch.randn(320, dtype=torch.float32)

    res = attribution_service.attribute_atoms_gradient(smiles, prot_emb)
    assert "pharmacophores" in res
    pharms = res["pharmacophores"]
    assert isinstance(pharms, list)
    if pharms:
        assert "pharmacophore" in pharms[0]
        assert "mean_attribution" in pharms[0]
        assert pharms[0]["mean_attribution"] >= 0.0
