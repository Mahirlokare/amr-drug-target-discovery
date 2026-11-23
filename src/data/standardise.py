"""
Compound standardization pipeline adhering to RDKit best practices.
Complies with Project 9 Blueprint Section 5.4 & 8.3.
"""

from typing import Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from src.common.logging import get_logger
from src.data.schemas import Compound

logger = get_logger("standardise")


class CompoundStandardizer:
    """Standardizes small molecules using RDKit MolStandardize routines."""

    def __init__(self):
        # Largest fragment chooser to remove salts and counterions
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        # Uncharger to neutralize ionized functional groups
        self.uncharger = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smiles_raw: str, compound_id: str = "") -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
        """
        Standardizes a raw SMILES string.
        Returns:
            canonical_smiles, inchikey, status, error_message
        Status is one of: "SUCCESS", "MODIFIED", "FAILED"
        """
        if not smiles_raw or not isinstance(smiles_raw, str):
            return None, None, "FAILED", "Empty or non-string SMILES"

        clean_smiles = smiles_raw.strip()
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return None, None, "FAILED", "Invalid SMILES parse failure"

        try:
            # 1. Basic sanitization & cleanup
            mol = rdMolStandardize.Cleanup(mol)

            # 2. Desalt / choose largest organic fragment
            mol = self.fragment_chooser.choose(mol)

            # 3. Uncharge / neutralize
            mol = self.uncharger.uncharge(mol)

            # 4. Standardize stereochemistry / aromaticity
            Chem.SanitizeMol(mol)

            # 5. Generate canonical SMILES & InChIKey
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)

            # Determine whether structure was modified from raw input
            raw_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles), isomericSmiles=True, canonical=True)
            status = "MODIFIED" if canonical_smiles != raw_canonical else "SUCCESS"

            return canonical_smiles, inchikey, status, None

        except Exception as e:
            logger.warning(f"Standardization exception for '{smiles_raw}': {e}")
            return None, None, "FAILED", str(e)

    def standardize_to_compound(self, smiles_raw: str, compound_id: str) -> Compound:
        """Standardizes a compound and returns a typed Compound instance."""
        can_smiles, inchikey, status, err = self.standardize_smiles(smiles_raw, compound_id)
        
        mw = None
        logp = None
        if can_smiles:
            mol = Chem.MolFromSmiles(can_smiles)
            if mol:
                mw = round(Descriptors.MolWt(mol), 2)
                logp = round(Descriptors.MolLogP(mol), 2)

        return Compound(
            compound_id=compound_id,
            smiles_raw=smiles_raw,
            smiles_standard=can_smiles or "",
            inchikey=inchikey or "",
            standardisation_status=status,
            molecular_weight=mw,
            logp=logp,
            metadata={"error": err} if err else {}
        )
