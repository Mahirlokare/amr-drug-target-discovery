"""
Molecular fingerprint extraction and chemical space similarity calculation.
Complies with Project 9 Blueprint Section 8.6, 10.1 & 12.2.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.common.logging import get_logger

logger = get_logger("fingerprints")


class FingerprintGenerator:
    """Generates Morgan circular fingerprints and RDKit 2D descriptors."""

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_morgan_bitvect(self, smiles: str) -> Optional[DataStructs.ExplicitBitVect]:
        """Returns RDKit ExplicitBitVect for Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits, useChirality=True
        )

    def smiles_to_morgan_numpy(self, smiles: str) -> Optional[np.ndarray]:
        """Returns 1D numpy binary array of length n_bits."""
        bv = self.smiles_to_morgan_bitvect(smiles)
        if bv is None:
            return None
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def morgan_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Returns 1D numpy binary array of length n_bits (alias for smiles_to_morgan_numpy)."""
        return self.smiles_to_morgan_numpy(smiles)

    def compute_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Calculates 8 standardized 2D physicochemical descriptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            feats = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.RingCount(mol),
            ]
            return np.array(feats, dtype=np.float32)
        except Exception:
            return None

    def compute_combined_feature(self, smiles: str) -> Optional[np.ndarray]:
        """Combines Morgan fingerprint (2048-d) and physicochemical descriptors (8-d)."""
        fp = self.smiles_to_morgan_numpy(smiles)
        desc = self.compute_descriptors(smiles)
        if fp is None or desc is None:
            return None
        return np.concatenate([fp, desc])


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Calculates Tanimoto coefficient between two SMILES."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_a, radius=radius, nBits=n_bits)
    fp_b = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_b, radius=radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def check_applicability_domain(
    candidate_smiles: str,
    reference_smiles_list: List[str],
    k: int = 5,
    min_similarity_threshold: float = 0.35
) -> Tuple[bool, float, List[float]]:
    """
    Checks if a candidate molecule falls within the chemical applicability domain
    of the training/reference compounds based on nearest neighbor Tanimoto similarity.
    Returns:
        is_out_of_domain: True if maximum similarity is below threshold
        max_similarity: highest Tanimoto similarity in reference set
        top_k_similarities: similarities of top-k nearest neighbors
    """
    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    if cand_mol is None:
        return True, 0.0, []

    cand_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(cand_mol, radius=2, nBits=2048)
    sims: List[float] = []

    for ref_smi in reference_smiles_list:
        ref_mol = Chem.MolFromSmiles(ref_smi)
        if ref_mol:
            ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
            sims.append(DataStructs.TanimotoSimilarity(cand_fp, ref_fp))

    if not sims:
        return True, 0.0, []

    sims.sort(reverse=True)
    max_sim = sims[0]
    top_k = sims[:k]
    is_ood = max_sim < min_similarity_threshold

    return is_ood, round(max_sim, 4), [round(s, 4) for s in top_k]
