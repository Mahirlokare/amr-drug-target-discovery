"""
Unit tests for data standardization, molecular graph featurization, and metrics.
Complies with Project 9 Blueprint Section 17 (Unit tests).
"""

import pytest
import numpy as np
import torch
from rdkit import Chem

from src.data.standardise import CompoundStandardizer
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.fingerprints import FingerprintGenerator, compute_tanimoto_similarity
from src.evaluation.metrics import compute_enrichment_factor, compute_all_metrics
from src.models.calibration import ProbabilityCalibrator


def test_smiles_standardization():
    standardizer = CompoundStandardizer()
    # Test valid SMILES
    smi, inchikey, status, err = standardizer.standardize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert status in ["SUCCESS", "MODIFIED"]
    assert inchikey is not None
    assert len(inchikey) == 27
    assert err is None

    # Test salt stripping
    salt_smi = "CCN.Cl"
    smi_desalt, _, status_salt, _ = standardizer.standardize_smiles(salt_smi)
    assert "Cl" not in smi_desalt
    assert status_salt == "MODIFIED"

    # Test invalid SMILES
    _, _, status_fail, err_fail = standardizer.standardize_smiles("invalid_chem_string_xyz")
    assert status_fail == "FAILED"
    assert err_fail is not None


def test_molecular_graph_dimensions():
    featurizer = MolecularGraphFeaturizer()
    mol = Chem.MolFromSmiles("O=C(NN)c1ccncc1")  # Isoniazid
    graph = featurizer.featurize(mol)

    assert graph is not None
    assert graph.x.size(1) == 43  # Invariant 43-d atom feature schema
    assert graph.edge_index.size(0) == 2
    assert graph.edge_attr.size(1) == 6
    assert graph.num_nodes == mol.GetNumAtoms()


def test_fingerprints_and_tanimoto():
    fp_gen = FingerprintGenerator(radius=2, n_bits=1024)
    fp = fp_gen.smiles_to_morgan_numpy("O=C(NN)c1ccncc1")
    assert fp is not None
    assert fp.shape == (1024,)

    sim_identical = compute_tanimoto_similarity("O=C(NN)c1ccncc1", "O=C(NN)c1ccncc1")
    assert sim_identical == 1.0


def test_enrichment_factor_calculation():
    # 100 samples with 10 actives (prevalence = 0.10)
    # Perfect model ranks 1 active at top 1% (sample 1)
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1
    y_score = np.linspace(1.0, 0.0, 100)

    # Top 1% contains 1 sample which is active -> hit rate = 1.0 / 0.10 = 10.0x
    ef_1pct = compute_enrichment_factor(y_true, y_score, top_fraction=0.01)
    assert ef_1pct == 10.0


def test_probability_calibrator():
    calibrator = ProbabilityCalibrator(method="platt")
    raw_logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y_val = np.array([0, 0, 0, 1, 1, 1])
    calibrator.fit(raw_logits, y_val)

    test_logits = np.array([-3.0, 0.0, 3.0])
    cal_probs = calibrator.calibrate(test_logits)
    assert len(cal_probs) == 3
    assert cal_probs[0] < cal_probs[1] < cal_probs[2]
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))
