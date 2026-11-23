"""
Physicochemical, drug-likeness, and PAINS filter algorithms.
Complies with Project 9 Blueprint Section 8.14 & 13.6.
"""

from typing import Any, Dict, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from src.common.logging import get_logger

logger = get_logger("admet_filters")

# Common PAINS (Pan-Assay Interference Compounds) alert SMARTS patterns
PAINS_SMARTS = {
    "Quinone / Hydroquinone": "O=C1C=CC(=O)C=C1",
    "Rhodanine": "O=C1NC(=S)SC1",
    "Catechol": "Oc1ccccc1O",
    "Michael Acceptor": "C=CC(=O)",
    "Azo group": "[#6]-N=N-[#6]",
    "Thiazolidinedione": "O=C1NC(=O)SC1",
    "Aniline": "c1ccccc1N",
    "Aliphatic halide": "[CX4][Cl,Br,I]",
    "Epoxide": "C1OC1",
}


class ADMETFilterEngine:
    """Calculates physicochemical properties, rules of five/veber, and PAINS warnings."""

    def __init__(self):
        self.pains_patterns = {
            name: Chem.MolFromSmarts(smarts) for name, smarts in PAINS_SMARTS.items()
        }

    def compute_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculates molecular properties for triage."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid structure"}

        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        hbd = int(Descriptors.NumHDonors(mol))
        hba = int(Descriptors.NumHAcceptors(mol))
        tpsa = round(Descriptors.TPSA(mol), 2)
        rot_bonds = int(Descriptors.NumRotatableBonds(mol))
        qed_val = round(QED.qed(mol), 3)

        # 1. Lipinski Rule of 5 violations
        lipinski_violations = 0
        if mw > 500.0:
            lipinski_violations += 1
        if logp > 5.0:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1

        lipinski_pass = bool(lipinski_violations <= 1)

        # 2. Veber Filter
        veber_pass = bool(rot_bonds <= 10 and tpsa <= 140.0)

        # 3. PAINS alerts
        pains_alerts = []
        for name, patt in self.pains_patterns.items():
            if patt and mol.HasSubstructMatch(patt):
                pains_alerts.append(name)

        # 4. Synthetic Accessibility estimation (1 easy to 10 difficult)
        # Based on ring count, stereocenters, and molecular size
        n_rings = Descriptors.RingCount(mol)
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        sa_score = round(min(10.0, max(1.0, 1.5 + 0.5 * n_rings + 0.6 * n_chiral + 0.005 * mw)), 2)

        warnings = []
        if lipinski_violations > 1:
            warnings.append(f"Lipinski violations: {lipinski_violations}")
        if not veber_pass:
            warnings.append("Veber filter failed (poor oral bioavailability predicted)")
        if pains_alerts:
            warnings.append(f"PAINS alerts detected: {', '.join(pains_alerts)}")
        if sa_score > 6.0:
            warnings.append(f"High synthetic complexity: SA score {sa_score}")

        return {
            "mw": mw,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "tpsa": tpsa,
            "rotatable_bonds": rot_bonds,
            "lipinski_violations": lipinski_violations,
            "lipinski_pass": lipinski_pass,
            "veber_pass": veber_pass,
            "pains_alert_count": len(pains_alerts),
            "pains_alerts": pains_alerts,
            "qed_score": qed_val,
            "synthetic_accessibility": sa_score,
            "warnings": warnings,
        }
