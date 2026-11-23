"""
Universal Chemical Input Resolver & Drug Name Dictionary.
Parses, sanitizes, resolves, and diagnostically evaluates small-molecule inputs.
Supports SMILES, salts, common drug names, and multi-line batch libraries.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from src.common.logging import get_logger

logger = get_logger("resolver")

# Comprehensive dictionary of antimicrobials, inhibitors, decoys, and benchmark compounds
DRUG_NAME_DICTIONARY: Dict[str, str] = {
    # Frontline & Second-line TB Therapeutics
    "isoniazid": "O=C(NN)c1ccncc1",
    "inh": "O=C(NN)c1ccncc1",
    "rifampicin": "CC1C=CC=C(C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)C(O4)(OC=CC(C(C(C(C(C(C1O)C)O)C)C)OC(=O)C)C)OC)C)O)O)C=NN5CCN(CC5)C)C",
    "rifampin": "CC1C=CC=C(C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)C(O4)(OC=CC(C(C(C(C(C(C1O)C)O)C)C)OC(=O)C)C)OC)C)O)O)C=NN5CCN(CC5)C)C",
    "pyrazinamide": "NC(=O)c1ncccn1",
    "ethambutol": "CC[C@@H](CO)NCCN[C@@H](CC)CO",
    "streptomycin": "CNC1[C@H](O)[C@@H](O[C@H]2[C@H](O[C@H]3[C@H](O)[C@@H](O)[C@H](N=C(N)N)[C@@H](O)[C@@H]3N=C(N)N)O[C@@H](CO)[C@@H](O)[C@@H]2O)O[C@@H](C)[C@@H]1O",
    "bedaquiline": "CC(C)CC(c1ccc(cc1)Br)(c2ccc3ccccc3n2)CC(c4ccccc4)c5cc(ccc5OC)N(C)C",
    "delamanid": "CC1CN(CCN1c2ccc(cc2)OC(F)(F)F)C(=O)c3ccc(cc3)Oc4ccc(c(c4)F)[C@H]5CN(CC5)C(=O)c6cccc(n6)[N+](=O)[O-]",
    "pretomanid": "CC1(CN(CCN1c2ccc(cc2)OC(F)(F)F)C(=O)c3ccc(cc3)OCc4ccccc4)C[N+](=O)[O-]",
    "linezolid": "CC(=O)NCC1CN(C(=O)O1)c2ccc(c(c2)F)N3CCOCC3",
    "ethionamide": "CCc1cc(C(N)=S)ccn1",
    "prothionamide": "CCCc1cc(C(N)=S)ccn1",
    "clofazimine": "CC(C)NC1=NC(=C2C(=N1)C3=CC=CC=C3N2c4ccc(cc4)Cl)Nc5ccc(cc5)Cl",
    "cycloserine": "NC1CON=C1",
    "para-aminosalicylic acid": "Nc1ccc(C(=O)O)c(O)c1",
    "pas": "Nc1ccc(C(=O)O)c(O)c1",

    # Direct InhA Inhibitors & Cocrystal Benchmarks
    "nitd-916": "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1",
    "nitd916": "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1",
    "geq": "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl",
    "4tzk inhibitor": "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl",
    "triclosan": "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl",
    "pt70": "CCCCCCCOc1ccc(cc1)c2ccccc2",
    "genz-10850": "COc1ccc(cc1)C(=O)Nc2ccc(s2)C(=O)Nc3ccc(OC)cc3",

    # DprE1 Inhibitors
    "macozinone": "CC1CCN(CC1)c2ccc(c(c2)[N+](=O)[O-])C(=O)N3CC4(CC3)CCC(CC4)F",
    "pbtz169": "CC1CCN(CC1)c2ccc(c(c2)[N+](=O)[O-])C(=O)N3CC4(CC3)CCC(CC4)F",
    "btz043": "CC1CCN(CC1)c2ccc(c(c2)[N+](=O)[O-])C(=O)N3CCN(CC3)c4ccc(cc4)OC(F)(F)F",
    "opc-167832": "COc1ccc(cc1)C2CCN(CC2)c3cc(F)cc(c3)C(=O)Nc4cc(OC)cc(n4)C#N",

    # Fluoroquinolones (GyrA Targets)
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c3cc(N4CCNCC4)c(F)cc3c1=O",
    "moxifloxacin": "COc1c(N2CC3CC(CNC3)C2)c(F)cc4c(=O)c(C(=O)O)cn(C5CC5)c14",
    "levofloxacin": "CC1COc2c(N1)c(F)cc3c(=O)c(C(=O)O)cn2c3C4CCN(C)CC4",
    "gatifloxacin": "CC1CN(CCN1)c2c(OC)c(F)cc3c(=O)c(C(=O)O)cn(C4CC4)c23",
    "norfloxacin": "CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21",

    # Beta-Lactams & Carbapenems (PBP3 & NDM-1 Targets)
    "meropenem": "CC1C2CC(=C(N2C1=O)C(=O)O)SC3CNC(C3)C(=O)N(C)C",
    "imipenem": "CC(O)C1C2CC(=C(N2C1=O)C(=O)O)SCCN=CN",
    "biapenem": "CC1C2CC(=C(N2C1=O)C(=O)O)SC3CN4C(C3)N=CN4",
    "aztreonam": "CC1C(C(=O)N1S(=O)(=O)O)NC(=O)C(=NOC(C)(C)C(=O)O)c2csc(n2)N",
    "ceftriaxone": "COc1c(n(c(=O)n1C)C)C2=C(N3C(C(=O)C3SCC(=O)O)CS2)C(=O)O",
    "ceftazidime": "CC(C)(C(=O)O)ON=C(C(=O)NC1C2SCC(=C(N2C1=O)C(=O)[O-])C[n+]3ccccc3)c4csc(n4)N",
    "amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C",
    "ampicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccccc3)N)C(=O)O)C",
    "penicillin g": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",

    # Control Decoys & Non-antibiotics
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "coumarin": "O=C1Oc2ccccc2C=C1",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "benzene": "c1ccccc1"
}


class UniversalChemicalResolver:
    """Intelligently handles, parses, and normalizes chemical input queries."""

    def __init__(self):
        self.cleaner = rdMolStandardize.Cleanup
        self.frag_chooser = rdMolStandardize.LargestFragmentChooser()
        self.uncharger = rdMolStandardize.Uncharger()

    def resolve_input(self, raw_query: str) -> Dict[str, Any]:
        """
        Resolves a user-submitted chemical string into structured standardized data.
        Accepts:
            - Drug common names (e.g. 'Isoniazid', 'NITD-916')
            - Canonical / Isomeric SMILES
            - Salted or charged SMILES
        Returns:
            Dictionary with standardized canonical SMILES, formula, MW, LogP, and status.
        """
        query = (raw_query or "").strip()
        if not query:
            return {
                "valid": False,
                "error": "Input is empty. Please enter a chemical SMILES or compound name.",
                "input_type": "EMPTY"
            }

        # Check drug name dictionary
        norm_key = query.lower()
        matched_name = None
        target_smiles = query

        if norm_key in DRUG_NAME_DICTIONARY:
            target_smiles = DRUG_NAME_DICTIONARY[norm_key]
            matched_name = query.title()
        else:
            # Check for partial or alias match
            for k, smi in DRUG_NAME_DICTIONARY.items():
                if norm_key == k or norm_key == k.replace("-", "").replace(" ", ""):
                    target_smiles = smi
                    matched_name = k.title()
                    break

        # Attempt to parse SMILES with RDKit
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            # Analyze exact syntax error
            error_hint = self._diagnose_invalid_smiles(target_smiles)
            return {
                "valid": False,
                "error": f"Invalid chemical syntax: {error_hint}",
                "input_type": "SYNTAX_ERROR",
                "raw_query": query
            }

        try:
            # Standardize: Clean, desalt, uncharge, sanitize
            mol = self.cleaner(mol)
            mol = self.frag_chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)
            Chem.SanitizeMol(mol)

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            mw = round(Descriptors.MolWt(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            tpsa = round(Descriptors.TPSA(mol), 2)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)

            return {
                "valid": True,
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "formula": formula,
                "molecular_weight": mw,
                "logp": logp,
                "tpsa": tpsa,
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotb,
                "matched_name": matched_name,
                "raw_query": query,
                "input_type": "NAME_RESOLVED" if matched_name else "SMILES_RESOLVED"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Standardization failed: {str(e)}",
                "input_type": "STANDARDIZATION_ERROR",
                "raw_query": query
            }

    def parse_batch_input(self, text_input: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parses multi-line or tabular SMILES inputs from CSV, TSV, or raw lists.
        Returns:
            parsed_compounds: List of standardized compound dictionaries.
            warnings: List of skipped or invalid line messages.
        """
        lines = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        results = []
        warnings = []

        for idx, line in enumerate(lines, 1):
            # Skip header lines
            if idx == 1 and any(h in line.lower() for h in ["smiles", "structure", "compound", "name"]):
                continue

            # Split on comma, tab, or space
            tokens = re.split(r"[\t,;]+", line)
            tokens = [t.strip().strip('"').strip("'") for t in tokens if t.strip()]

            if not tokens:
                continue

            # Identify SMILES candidate token
            candidate_str = tokens[0]
            compound_id = f"LIB-{idx:04d}"

            if len(tokens) >= 2:
                # Check if first token is ID and second is SMILES
                if Chem.MolFromSmiles(tokens[1]) is not None or tokens[1].lower() in DRUG_NAME_DICTIONARY:
                    compound_id = tokens[0]
                    candidate_str = tokens[1]
                else:
                    compound_id = tokens[1]
                    candidate_str = tokens[0]

            res = self.resolve_input(candidate_str)
            if res["valid"]:
                res["compound_id"] = compound_id
                results.append(res)
            else:
                warnings.append(f"Line {idx} ('{line[:30]}...'): {res['error']}")

        return results, warnings

    def _diagnose_invalid_smiles(self, smi: str) -> str:
        """Heuristic diagnostics for common SMILES syntax mistakes."""
        # Unclosed ring numbers
        digits = re.findall(r"\d", smi)
        from collections import Counter
        counts = Counter(digits)
        unpaired = [d for d, c in counts.items() if c % 2 != 0]
        if unpaired:
            return f"Unpaired ring closure number(s): {', '.join(unpaired)}."

        # Unmatched brackets
        if smi.count("[") != smi.count("]"):
            return "Mismatched square brackets '[' and ']'."
        if smi.count("(") != smi.count(")"):
            return "Mismatched branch parentheses '(' and ')'."

        # Lowercase non-aromatic
        bad_lower = re.findall(r"[a-z]", smi)
        valid_aromatic = set("bcnops")
        invalid_aromatic = [ch for ch in bad_lower if ch not in valid_aromatic]
        if invalid_aromatic:
            return f"Invalid lowercase element token: '{invalid_aromatic[0]}'. Only b, c, n, o, p, s are aromatic."

        return "Unparseable molecular valence or connectivity."


# Singleton instance
chemical_resolver = UniversalChemicalResolver()
