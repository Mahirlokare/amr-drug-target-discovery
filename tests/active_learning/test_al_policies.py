"""
Active learning policies test suite.
Verifies random, greedy, uncertainty, and balanced diversity acquisition policies.
"""

import numpy as np
import pytest
from src.active_learning.policies import ActiveLearningPolicies


def test_random_policy():
    indices = ActiveLearningPolicies.random_policy(n_candidates=20, batch_size=5, seed=42)
    assert len(indices) == 5
    assert len(set(indices)) == 5  # No duplicates
    assert all(0 <= i < 20 for i in indices)


def test_greedy_activity_policy():
    pred_probs = np.array([0.1, 0.85, 0.4, 0.95, 0.2])
    indices = ActiveLearningPolicies.greedy_activity_policy(pred_probs, batch_size=2)
    assert indices == [3, 1]  # 0.95, then 0.85


def test_uncertainty_policy():
    uncertainties = np.array([0.05, 0.22, 0.18, 0.02, 0.35])
    indices = ActiveLearningPolicies.uncertainty_policy(uncertainties, batch_size=2)
    assert indices == [4, 1]  # 0.35, then 0.22


def test_balanced_diversity_policy():
    smiles = [
        "O=C(NN)c1ccncc1",
        "CCc1cc(C(N)=S)ccn1",
        "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl",
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl"
    ]
    pred_probs = np.array([0.8, 0.75, 0.7, 0.1, 0.9])
    uncertainties = np.array([0.1, 0.12, 0.05, 0.02, 0.18])

    indices = ActiveLearningPolicies.balanced_diversity_policy(
        smiles_list=smiles,
        pred_probs=pred_probs,
        uncertainties=uncertainties,
        batch_size=3
    )

    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert all(0 <= i < 5 for i in indices)
