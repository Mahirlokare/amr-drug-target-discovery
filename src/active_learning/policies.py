"""
Acquisition functions and selection policies for active learning.
Complies with Project 9 Blueprint Section 8.11 & 12.3.
"""

from typing import List, Tuple
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors


def calculate_fps(smiles_list: List[str]):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
        else:
            fps.append(None)
    return fps


class ActiveLearningPolicies:
    """Implements candidate acquisition policies for iterative screening."""

    @staticmethod
    def random_policy(n_candidates: int, batch_size: int, seed: int = 42) -> List[int]:
        """Uniform random sampling baseline."""
        rng = np.random.RandomState(seed)
        indices = np.arange(n_candidates)
        rng.shuffle(indices)
        return list(indices[:batch_size])

    @staticmethod
    def greedy_activity_policy(pred_probs: np.ndarray, batch_size: int) -> List[int]:
        """Greedy exploitation: selects highest predicted activity probabilities."""
        sorted_indices = np.argsort(-pred_probs)
        return list(sorted_indices[:batch_size])

    @staticmethod
    def uncertainty_policy(uncertainties: np.ndarray, batch_size: int) -> List[int]:
        """Pure exploration: selects highest model epistemic uncertainty."""
        sorted_indices = np.argsort(-uncertainties)
        return list(sorted_indices[:batch_size])

    @staticmethod
    def balanced_diversity_policy(
        smiles_list: List[str],
        pred_probs: np.ndarray,
        uncertainties: np.ndarray,
        batch_size: int,
        alpha_activity: float = 0.5,
        beta_uncertainty: float = 0.3,
        gamma_diversity: float = 0.2
    ) -> List[int]:
        """
        Multi-objective policy: Combines predicted activity, epistemic uncertainty,
        and maximum-minimum Tanimoto distance to ensure chemical scaffold diversity.
        """
        n = len(pred_probs)
        if n <= batch_size:
            return list(range(n))

        # Normalize score components to [0, 1]
        p_norm = (pred_probs - np.min(pred_probs)) / (np.ptp(pred_probs) + 1e-8)
        u_norm = (uncertainties - np.min(uncertainties)) / (np.ptp(uncertainties) + 1e-8)

        base_utility = alpha_activity * p_norm + beta_uncertainty * u_norm

        fps = calculate_fps(smiles_list)

        # Greedy diversity selection: Pick highest utility first
        selected: List[int] = [int(np.argmax(base_utility))]
        remaining: List[int] = [i for i in range(n) if i != selected[0]]

        while len(selected) < batch_size and remaining:
            best_idx = -1
            best_score = -float("inf")

            for cand_idx in remaining:
                cand_fp = fps[cand_idx]
                if cand_fp is None:
                    min_dist = 0.5
                else:
                    # Compute minimum Tanimoto distance to already selected candidates
                    sims = [
                        DataStructs.TanimotoSimilarity(cand_fp, fps[s_idx])
                        for s_idx in selected if fps[s_idx] is not None
                    ]
                    max_sim = max(sims) if sims else 0.0
                    min_dist = 1.0 - max_sim

                score = base_utility[cand_idx] + gamma_diversity * min_dist
                if score > best_score:
                    best_score = score
                    best_idx = cand_idx

            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected
