"""
Attribution stability verification across model seeds and chemical analogues.
Complies with Project 9 Blueprint Section 8.12 & 12.4.
"""

from typing import Any, Dict, List
import numpy as np
import torch
from scipy.stats import spearmanr

from src.explainability.graph_attribution import GraphAttributionService


class AttributionStabilityChecker:
    """Measures explanation consistency across multiple ensemble seeds."""

    def __init__(self, ensemble_models: List[torch.nn.Module]):
        self.explainers = [GraphAttributionService(m) for m in ensemble_models]

    def check_ensemble_stability(self, smiles: str, protein_embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Computes atom saliency across all ensemble members and calculates Spearman
        rank correlation between seeds to measure attribution stability.
        """
        all_importances: List[List[float]] = []

        for explainer in self.explainers:
            res = explainer.attribute_atoms_gradient(smiles, protein_embedding)
            if "atom_importances" in res:
                all_importances.append(res["atom_importances"])

        if len(all_importances) < 2:
            return {"stability_score": 1.0, "is_stable": True, "note": "Single model evaluated"}

        # Pairwise Spearman correlation between seeds
        correlations = []
        n_models = len(all_importances)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                r, _ = spearmanr(all_importances[i], all_importances[j])
                if not np.isnan(r):
                    correlations.append(r)

        avg_corr = float(np.mean(correlations)) if correlations else 1.0
        is_stable = avg_corr >= 0.50

        # Mean atom importance across ensemble
        mean_atom_importances = np.mean(np.array(all_importances), axis=0).round(4).tolist()

        return {
            "mean_atom_importances": mean_atom_importances,
            "spearman_stability_score": round(avg_corr, 4),
            "is_stable": is_stable,
            "warning": None if is_stable else "Substructure attribution exhibits high variance across ensemble seeds; interpret with caution."
        }
