"""
Atom-level and substructure attribution for molecular GNN predictions.
Complies with Project 9 Blueprint Section 8.12 & 12.4.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem

from src.common.logging import get_logger
from src.features.molecular_graph import MolecularGraphFeaturizer

logger = get_logger("explainability")


class GraphAttributionService:
    """Computes atom saliency and substructure attribution maps."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.featurizer = MolecularGraphFeaturizer()

    def attribute_atoms_gradient(
        self,
        smiles: str,
        protein_embedding: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes atom-level saliency using Input x Gradient attribution on atom node features.
        Returns:
            atom_importances: List of positive normalized saliency scores per atom.
            atom_symbols: List of atom symbols.
            key_pharmacophores: Detected functional groups with highest contribution.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        graph = self.featurizer.featurize(mol)
        if graph is None:
            return {"error": "Featurization failed"}

        self.model.eval()
        device = next(self.model.parameters()).device
        x = graph.x.clone().detach().to(device).requires_grad_(True)
        edge_index = graph.edge_index.to(device)
        prot_emb = protein_embedding.to(device)

        # Forward pass
        logit, _, _ = self.model(x, edge_index, prot_emb)
        # Backward pass on target prediction logit
        logit.backward()

        # Input x Gradient
        grads = x.grad.data.cpu().numpy()
        inputs = x.data.cpu().numpy()
        atom_scores = np.sum(inputs * grads, axis=-1)
        # ReLU to focus on positive contributions towards active class
        atom_scores = np.maximum(atom_scores, 0.0)

        # Normalize to [0, 1]
        max_score = np.max(atom_scores) if len(atom_scores) > 0 else 1.0
        if max_score > 0:
            norm_scores = (atom_scores / max_score).tolist()
        else:
            norm_scores = [0.0] * len(atom_scores)

        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # Identify chemically meaningful functional group hits
        functional_groups = self._detect_pharmacophores(mol, norm_scores)

        return {
            "smiles": smiles,
            "atom_count": len(atom_symbols),
            "atom_symbols": atom_symbols,
            "atom_importances": [round(s, 4) for s in norm_scores],
            "top_contributing_atoms": sorted(
                range(len(norm_scores)), key=lambda i: norm_scores[i], reverse=True
            )[:5],
            "pharmacophores": functional_groups
        }

    def _detect_pharmacophores(self, mol: Chem.Mol, atom_scores: List[float]) -> List[Dict[str, Any]]:
        """Identifies standard medicinal chemistry pharmacophores and their mean attribution."""
        patterns = {
            "Aromatic Ring": "c1ccccc1",
            "Pyridine Ring": "c1ccncc1",
            "Hydrazide / Amide": "C(=O)NN",
            "Phenol / Hydroxyl": "[OX2H][c,C]",
            "Halogen (Cl/Br/F)": "[F,Cl,Br,I]",
            "Ether Linkage": "[OD2]([c,C])[c,C]",
            "Carboxylic Acid / Ester": "C(=O)[O]",
            "Sulfonamide": "S(=O)(=O)N",
        }

        detected = []
        for name, smarts in patterns.items():
            patt = Chem.MolFromSmarts(smarts)
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                avg_sal = float(np.mean([atom_scores[idx] for idx in match])) if match else 0.0
                detected.append({
                    "name": name,
                    "pharmacophore": name,
                    "atom_indices": list(match),
                    "mean_attribution": round(avg_sal, 4),
                    "atom_count": len(match)
                })

        detected.sort(key=lambda x: x["mean_attribution"], reverse=True)
        return detected[:6]
