"""
Molecular graph construction using RDKit and PyTorch Geometric.
Complies with Project 9 Blueprint Section 8.6 & 10.2.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("molecular_graph")

# Allowed atom symbols and hybridizations for one-hot encoding
ALLOWED_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ALLOWED_CHARGES = [-2, -1, 0, 1, 2]
ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ALLOWED_HS = [0, 1, 2, 3, 4]


def one_hot_encode(value, allowable_set: List):
    """Encodes value into one-hot list, with an extra slot for unknown values."""
    encoded = [1.0 if value == s else 0.0 for s in allowable_set]
    encoded.append(1.0 if value not in allowable_set else 0.0)
    return encoded


class MolecularGraphFeaturizer:
    """Extracts node features, edge indices, and edge features for PyTorch Geometric GNN."""

    def __init__(self):
        self.node_dim = self._compute_node_dim()

    def _compute_node_dim(self) -> int:
        # Atomic number (len+1) + degree (len+1) + charge (len+1) + hyb (len+1) + hs (len+1) + aromatic (1) + ring (1) + chirality (1)
        dim = (len(ALLOWED_ATOMS) + 1) + \
              (len(ALLOWED_DEGREES) + 1) + \
              (len(ALLOWED_CHARGES) + 1) + \
              (len(ALLOWED_HYBRIDIZATIONS) + 1) + \
              (len(ALLOWED_HS) + 1) + 3
        return dim

    def atom_to_features(self, atom: Chem.Atom) -> List[float]:
        """Calculates 32-dimensional invariant atom feature vector."""
        feats = []
        # 1. Atomic Number
        feats.extend(one_hot_encode(atom.GetAtomicNum(), ALLOWED_ATOMS))
        # 2. Degree
        feats.extend(one_hot_encode(atom.GetTotalDegree(), ALLOWED_DEGREES))
        # 3. Formal Charge
        feats.extend(one_hot_encode(atom.GetFormalCharge(), ALLOWED_CHARGES))
        # 4. Hybridization
        feats.extend(one_hot_encode(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS))
        # 5. Number of Hydrogens
        feats.extend(one_hot_encode(atom.GetTotalNumHs(), ALLOWED_HS))
        # 6. Aromaticity
        feats.append(1.0 if atom.GetIsAromatic() else 0.0)
        # 7. In Ring
        feats.append(1.0 if atom.IsInRing() else 0.0)
        # 8. Chirality
        feats.append(1.0 if atom.HasProp("_CIPCode") else 0.0)

        return feats

    def bond_to_features(self, bond: Chem.Bond) -> List[float]:
        """Calculates bond feature vector."""
        bt = bond.GetBondType()
        feats = [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
        return feats

    def featurize(self, smiles_or_mol) -> Optional[Data]:
        """Converts molecule to PyG Data instance."""
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol

        if mol is None:
            return None

        # Build node features
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(self.atom_to_features(atom))

        if not atom_feats:
            return None

        x = torch.tensor(atom_feats, dtype=torch.float32)

        # Build edge index & edge attributes (bidirectional)
        edge_indices = []
        edge_feats = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            b_feat = self.bond_to_features(bond)

            # i -> j
            edge_indices.append([i, j])
            edge_feats.append(b_feat)

            # j -> i
            edge_indices.append([j, i])
            edge_feats.append(b_feat)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        return data
