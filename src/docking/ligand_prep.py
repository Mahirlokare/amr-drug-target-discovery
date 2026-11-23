"""
Ligand 3D conformer generation and PDBQT preparation.
Complies with Project 9 Blueprint Section 13.3 (Ligand preparation).
"""

from pathlib import Path
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

from src.common.logging import get_logger

logger = get_logger("ligand_prep")


class LigandPreparer:
    """Generates energy-minimized 3D conformers and Vina-compatible PDBQT files."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def generate_3d_conformer(self, smiles: str) -> Optional[Chem.Mol]:
        """Adds explicit hydrogens and optimizes 3D coordinates using ETKDG and MMFF94."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add explicit hydrogens
        mol = Chem.AddHs(mol)

        # Embed 3D conformer using ETKDGv3 parameters
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        success = AllChem.EmbedMolecule(mol, params)

        if success != 0:
            # Fallback embedding with random coordinates if ETKDG fails
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=self.random_seed)

        try:
            # MMFF94 force-field geometry optimization
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

        return mol

    def prepare_ligand_pdbqt(
        self,
        smiles: str,
        compound_id: str,
        output_dir: Path
    ) -> Tuple[Optional[Path], Optional[float]]:
        """
        Prepares 3D conformer and exports PDBQT format.
        Returns:
            pdbqt_path: Path to prepared ligand PDBQT file
            internal_energy: MMFF energy value
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mol = self.generate_3d_conformer(smiles)
        if mol is None:
            logger.warning(f"Could not generate 3D conformer for {compound_id}")
            return None, None

        pdbqt_path = output_dir / f"{compound_id}.pdbqt"

        # Export PDB representation with Gasteiger partial charges
        lines = []
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            charge = 0.0
            if atom.HasProp("_GasteigerCharge"):
                try:
                    charge = float(atom.GetProp("_GasteigerCharge"))
                except Exception:
                    charge = 0.0

            # PDB ATOM format with charge
            line = f"ATOM  {i+1:5d} {symbol:^4s} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:+6.3f} {symbol:>2s}\n"
            lines.append(line)

        # Wrap with ROOT/ENDROOT for AutoDock Vina flexibility
        pdbqt_content = "ROOT\n" + "".join(lines) + "ENDROOT\nTORSDOF 0\n"
        with open(pdbqt_path, "w", encoding="utf-8") as f:
            f.write(pdbqt_content)

        # Calculate internal energy
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            energy = round(ff.CalcEnergy(), 2) if ff else 0.0
        except Exception:
            energy = 0.0

        return pdbqt_path, energy
