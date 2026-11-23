"""
Uncertainty quantification (Deep Ensembles & Monte Carlo Dropout).
Complies with Project 9 Blueprint Section 8.10 & 12.1.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.common.logging import get_logger

logger = get_logger("uncertainty")


class DeepEnsembleScorer:
    """Estimates predictive mean and epistemic uncertainty across model ensemble."""

    def __init__(self, models: List[nn.Module], epistemic_threshold: float = 0.15):
        self.models = models
        self.epistemic_threshold = epistemic_threshold

    def score_candidates(
        self,
        graph_data_list: List[Data],
        protein_embedding: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs batch inference across all ensemble members.
        Returns:
            mean_probs: average predicted active probability [N]
            std_devs: epistemic uncertainty standard deviation [N]
            raw_logits_mean: average raw logit [N]
        """
        device = next(self.models[0].parameters()).device
        protein_emb = protein_embedding.to(device)

        batch_graph = Batch.from_data_list(graph_data_list).to(device)

        all_probs = []
        all_logits = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(batch_graph.x, batch_graph.edge_index, protein_emb, batch_graph.batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_logits.append(logits.cpu().numpy())

        # Shape: [num_models, N]
        all_probs_arr = np.array(all_probs)
        all_logits_arr = np.array(all_logits)

        mean_probs = np.mean(all_probs_arr, axis=0)
        std_devs = np.std(all_probs_arr, axis=0)
        raw_logits_mean = np.mean(all_logits_arr, axis=0)

        return mean_probs, std_devs, raw_logits_mean


def monte_carlo_dropout_inference(
    model: nn.Module,
    graph: Data,
    protein_embedding: torch.Tensor,
    n_samples: int = 20
) -> Tuple[float, float]:
    """
    Evaluates epistemic uncertainty via Monte Carlo dropout forward passes.
    Returns:
        mean_probability, standard_deviation
    """
    model.train()  # Keep dropout enabled
    device = next(model.parameters()).device
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    prot_emb = protein_embedding.to(device)

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logit, _, _ = model(x, edge_index, prot_emb)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs))
    return round(mean_p, 4), round(std_p, 4)
