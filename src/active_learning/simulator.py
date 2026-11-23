"""
Retrospective active learning simulation engine.
Complies with Project 9 Blueprint Section 8.11 & 12.3.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.active_learning.policies import ActiveLearningPolicies
from src.common.logging import get_logger
from src.features.fingerprints import FingerprintGenerator

logger = get_logger("al_simulator")


class ActiveLearningSimulator:
    """Simulates iterative wet-lab screening cycles using held-out historical labels."""

    def __init__(
        self,
        candidate_smiles: List[str],
        ground_truth_labels: List[int],
        seed: int = 42
    ):
        assert len(candidate_smiles) == len(ground_truth_labels)
        self.candidate_smiles = candidate_smiles
        self.ground_truth_labels = np.array(ground_truth_labels)
        self.n_total = len(candidate_smiles)
        self.seed = seed
        self.fp_gen = FingerprintGenerator(radius=2, n_bits=1024)
        self.X = self._build_feature_matrix()

    def _build_feature_matrix(self) -> np.ndarray:
        feats = []
        for s in self.candidate_smiles:
            fp = self.fp_gen.smiles_to_morgan_numpy(s)
            if fp is None:
                fp = np.zeros(1024, dtype=np.float32)
            feats.append(fp)
        return np.array(feats)

    def run_simulation(
        self,
        policy_name: str = "balanced",
        n_rounds: int = 5,
        batch_size: int = 10,
        initial_seed_size: int = 15
    ) -> Dict[str, Any]:
        """
        Runs iterative simulation.
        Hidden ground truth labels are strictly masked from the policy until selection!
        """
        rng = np.random.RandomState(self.seed)

        # Initial random seed training set
        all_indices = np.arange(self.n_total)
        rng.shuffle(all_indices)
        train_indices = set(all_indices[:initial_seed_size])
        pool_indices = [idx for idx in all_indices[initial_seed_size:]]

        cumulative_budget = [len(train_indices)]
        cumulative_actives = [int(np.sum(self.ground_truth_labels[list(train_indices)]))]
        round_histories = []

        logger.info(f"Running AL simulation '{policy_name}': {n_rounds} rounds, batch_size={batch_size}...")

        for r in range(1, n_rounds + 1):
            if not pool_indices:
                break

            # 1. Fit surrogate model on current labeled set
            X_train = self.X[list(train_indices)]
            y_train = self.ground_truth_labels[list(train_indices)]

            clf = RandomForestClassifier(n_estimators=100, random_state=self.seed + r, max_depth=8)
            clf.fit(X_train, y_train)

            # 2. Score unlabelled pool
            X_pool = self.X[pool_indices]
            # Predict probabilities
            probs_2d = clf.predict_proba(X_pool)
            pred_probs = probs_2d[:, 1] if probs_2d.shape[1] > 1 else probs_2d[:, 0]

            # Estimate tree ensemble variance as epistemic uncertainty
            tree_preds = np.array([tree.predict_proba(X_pool)[:, 1] if tree.predict_proba(X_pool).shape[1] > 1 else tree.predict_proba(X_pool)[:, 0] for tree in clf.estimators_])
            uncertainties = np.std(tree_preds, axis=0)

            # 3. Apply selection policy
            pool_smiles = [self.candidate_smiles[idx] for idx in pool_indices]

            if policy_name == "random":
                selected_rel_indices = ActiveLearningPolicies.random_policy(
                    len(pool_indices), batch_size, seed=self.seed + r
                )
            elif policy_name == "greedy_activity":
                selected_rel_indices = ActiveLearningPolicies.greedy_activity_policy(
                    pred_probs, batch_size
                )
            elif policy_name == "uncertainty":
                selected_rel_indices = ActiveLearningPolicies.uncertainty_policy(
                    uncertainties, batch_size
                )
            else:  # balanced
                selected_rel_indices = ActiveLearningPolicies.balanced_diversity_policy(
                    pool_smiles, pred_probs, uncertainties, batch_size
                )

            selected_global_indices = [pool_indices[i] for i in selected_rel_indices]

            # 4. Reveal ground truth labels ONLY for chosen candidates
            revealed_labels = self.ground_truth_labels[selected_global_indices]
            new_actives = int(np.sum(revealed_labels))

            # 5. Update labeled pool
            train_indices.update(selected_global_indices)
            pool_indices = [idx for idx in pool_indices if idx not in train_indices]

            cumulative_budget.append(len(train_indices))
            cumulative_actives.append(cumulative_actives[-1] + new_actives)

            round_histories.append({
                "round": r,
                "batch_size": len(selected_global_indices),
                "actives_found": new_actives,
                "hit_rate_pct": round((new_actives / len(selected_global_indices)) * 100, 2),
                "total_actives_so_far": cumulative_actives[-1]
            })

        total_tested = cumulative_budget[-1]
        total_actives_found = cumulative_actives[-1]
        overall_hit_rate = round((total_actives_found / total_tested) * 100, 2) if total_tested > 0 else 0.0

        return {
            "policy": policy_name,
            "cumulative_budget": cumulative_budget,
            "cumulative_actives": cumulative_actives,
            "overall_hit_rate_pct": overall_hit_rate,
            "round_histories": round_histories
        }
