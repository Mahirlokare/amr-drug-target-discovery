"""
Leakage-resistant dataset splitting and validation assertions.
Complies with Project 9 Blueprint Section 8.5, 11.1, 11.2 & 17.1.
"""

from collections import defaultdict
import random
from typing import Any, Dict, List, Set, Tuple
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.common.logging import get_logger
from src.data.schemas import ActivityRecord, Compound

logger = get_logger("splits")


def get_bemis_murcko_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Computes Bemis-Murcko scaffold for a molecule.
    If generic=True, converts all atoms to carbon and bonds to single bonds (framework).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "UNPARSEABLE"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            # Molecule has no ring systems (e.g. acyclic chain)
            return "ACYCLIC"
        if generic:
            scaffold_mol = Chem.MolFromSmiles(scaffold)
            if scaffold_mol:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
                return Chem.MolToSmiles(scaffold)
        return scaffold
    except Exception:
        return "SCAFFOLD_ERROR"


class SplitManager:
    """Manages scaffold and temporal splits with automated zero-leakage assertions."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def create_scaffold_split(
        self,
        compounds: List[Compound],
        records: List[ActivityRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Partitions dataset by Bemis-Murcko scaffold ensuring zero scaffold overlap.
        Sorts scaffold clusters by size to distribute large scaffolds balanced across partitions.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        # Map compound_id to compound
        cid_to_comp = {c.compound_id: c for c in compounds}
        # Filter records with binary labels only
        labeled_records = [r for r in records if r.label is not None and r.compound_id in cid_to_comp]

        # Calculate scaffold for each compound
        scaffold_to_cids = defaultdict(list)
        cid_to_scaffold = {}
        for r in labeled_records:
            cid = r.compound_id
            if cid in cid_to_scaffold:
                continue
            comp = cid_to_comp[cid]
            scaffold = get_bemis_murcko_scaffold(comp.smiles_standard or comp.smiles_raw)
            cid_to_scaffold[cid] = scaffold
            scaffold_to_cids[scaffold].append(cid)

        # Sort scaffolds by number of compounds descending (balanced bin-packing)
        rng = random.Random(self.seed)
        scaffold_list = list(scaffold_to_cids.items())
        # Random shuffle first, then sort by length to avoid deterministic ordering of ties
        rng.shuffle(scaffold_list)
        scaffold_list.sort(key=lambda x: len(x[1]), reverse=True)

        total_compounds = len(cid_to_scaffold)
        target_train = int(total_compounds * train_ratio)
        target_val = int(total_compounds * val_ratio)

        train_cids: Set[str] = set()
        val_cids: Set[str] = set()
        test_cids: Set[str] = set()

        train_scaffolds: Set[str] = set()
        val_scaffolds: Set[str] = set()
        test_scaffolds: Set[str] = set()

        for scaffold, cids in scaffold_list:
            if len(train_cids) + len(cids) <= target_train or (len(val_cids) >= target_val and len(test_cids) >= (total_compounds - target_train - target_val)):
                train_cids.update(cids)
                train_scaffolds.add(scaffold)
            elif len(val_cids) + len(cids) <= target_val:
                val_cids.update(cids)
                val_scaffolds.add(scaffold)
            else:
                test_cids.update(cids)
                test_scaffolds.add(scaffold)

        # Assertions as demanded in blueprint 17.1
        self.assert_scaffold_leakage(train_scaffolds, val_scaffolds, test_scaffolds, train_cids, test_cids)

        train_records = [r for r in labeled_records if r.compound_id in train_cids]
        val_records = [r for r in labeled_records if r.compound_id in val_cids]
        test_records = [r for r in labeled_records if r.compound_id in test_cids]

        split_manifest = {
            "split_type": "bemis_murcko_scaffold",
            "seed": self.seed,
            "train": {
                "num_compounds": len(train_cids),
                "num_scaffolds": len(train_scaffolds),
                "num_records": len(train_records),
                "num_actives": sum(1 for r in train_records if r.label == 1),
                "num_inactives": sum(1 for r in train_records if r.label == 0),
                "compound_ids": sorted(list(train_cids)),
                "scaffold_ids": sorted(list(train_scaffolds)),
            },
            "val": {
                "num_compounds": len(val_cids),
                "num_scaffolds": len(val_scaffolds),
                "num_records": len(val_records),
                "num_actives": sum(1 for r in val_records if r.label == 1),
                "num_inactives": sum(1 for r in val_records if r.label == 0),
                "compound_ids": sorted(list(val_cids)),
                "scaffold_ids": sorted(list(val_scaffolds)),
            },
            "test": {
                "num_compounds": len(test_cids),
                "num_scaffolds": len(test_scaffolds),
                "num_records": len(test_records),
                "num_actives": sum(1 for r in test_records if r.label == 1),
                "num_inactives": sum(1 for r in test_records if r.label == 0),
                "compound_ids": sorted(list(test_cids)),
                "scaffold_ids": sorted(list(test_scaffolds)),
            },
            "zero_scaffold_overlap_verified": True,
        }

        logger.info(
            f"Scaffold Split created: Train={len(train_cids)} cmpds ({len(train_scaffolds)} scaffolds), "
            f"Val={len(val_cids)} cmpds ({len(val_scaffolds)} scaffolds), "
            f"Test={len(test_cids)} cmpds ({len(test_scaffolds)} scaffolds)."
        )
        return split_manifest

    def create_temporal_split(
        self,
        compounds: List[Compound],
        records: List[ActivityRecord],
        cutoff_year: int = 2018
    ) -> Dict[str, Any]:
        """
        Creates a time-aware split: records <= cutoff_year in train, records > cutoff_year in test.
        """
        dated_records = [r for r in records if r.label is not None and r.year is not None]
        if len(dated_records) < 30:
            logger.warning(f"Insufficient dated records ({len(dated_records)}) for reliable temporal split.")
            return {"split_type": "temporal", "status": "INSUFFICIENT_DATA", "record_count": len(dated_records)}

        train_records = [r for r in dated_records if r.year <= cutoff_year]
        test_records = [r for r in dated_records if r.year > cutoff_year]

        # Minimum automated temporal leakage assertion (blueprint 17.1)
        if train_records and test_records:
            max_train_year = max(r.year for r in train_records)
            min_test_year = min(r.year for r in test_records)
            assert max_train_year <= cutoff_year, f"Temporal leakage: train year {max_train_year} > {cutoff_year}"
            assert min_test_year > cutoff_year, f"Temporal leakage: test year {min_test_year} <= {cutoff_year}"

        split_manifest = {
            "split_type": "temporal",
            "cutoff_year": cutoff_year,
            "train": {
                "num_records": len(train_records),
                "num_actives": sum(1 for r in train_records if r.label == 1),
                "num_inactives": sum(1 for r in train_records if r.label == 0),
                "years_range": [min((r.year for r in train_records), default=0), max((r.year for r in train_records), default=0)],
                "compound_ids": sorted(list({r.compound_id for r in train_records})),
            },
            "test": {
                "num_records": len(test_records),
                "num_actives": sum(1 for r in test_records if r.label == 1),
                "num_inactives": sum(1 for r in test_records if r.label == 0),
                "years_range": [min((r.year for r in test_records), default=0), max((r.year for r in test_records), default=0)],
                "compound_ids": sorted(list({r.compound_id for r in test_records})),
            },
            "zero_future_leakage_verified": True,
        }
        return split_manifest

    @staticmethod
    def assert_scaffold_leakage(
        train_scaffolds: Set[str],
        val_scaffolds: Set[str],
        test_scaffolds: Set[str],
        train_cids: Set[str],
        test_cids: Set[str],
    ) -> None:
        """Enforces Section 17.1 Minimum automated leakage assertions."""
        assert train_scaffolds.isdisjoint(val_scaffolds), f"Overlap between train and val scaffolds: {train_scaffolds & val_scaffolds}"
        assert train_scaffolds.isdisjoint(test_scaffolds), f"Overlap between train and test scaffolds: {train_scaffolds & test_scaffolds}"
        assert val_scaffolds.isdisjoint(test_scaffolds), f"Overlap between val and test scaffolds: {val_scaffolds & test_scaffolds}"
        assert train_cids.isdisjoint(test_cids), f"Overlap between train and test compounds: {train_cids & test_cids}"

    def scaffold_split(
        self,
        dataset: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partitions a record list by Bemis-Murcko scaffold ensuring zero scaffold overlap.
        Returns {'train': [...], 'val': [...], 'test': [...]}.
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold

        def _get_bm(smi: str) -> str:
            m = Chem.MolFromSmiles(smi)
            if not m:
                return "UNPARSEABLE"
            try:
                sc = MurckoScaffold.GetScaffoldForMol(m)
                s = Chem.MolToSmiles(sc) if sc else ""
                return s if s else "ACYCLIC"
            except Exception:
                return "ERROR"

        scaffold_to_records = defaultdict(list)
        for r in dataset:
            sc = _get_bm(r.get("smiles", ""))
            scaffold_to_records[sc].append(r)

        rng = random.Random(self.seed)
        scaffold_items = list(scaffold_to_records.items())
        rng.shuffle(scaffold_items)
        # Sort descending by group length
        scaffold_items.sort(key=lambda x: len(x[1]), reverse=True)

        total = len(dataset)
        target_train = int(total * train_ratio)
        target_val = max(1, int(total * val_ratio))
        target_test = max(1, total - target_train - target_val)

        train_groups, val_groups, test_groups = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        for sc, recs in scaffold_items:
            count = len(recs)
            if train_count + count <= target_train or (val_count >= target_val and test_count >= target_test):
                train_groups.append(recs)
                train_count += count
            elif val_count + count <= target_val or val_count == 0:
                val_groups.append(recs)
                val_count += count
            else:
                test_groups.append(recs)
                test_count += count

        # Ensure all partitions non-empty by moving whole groups if necessary
        if not val_groups and len(train_groups) > 1:
            val_groups.append(train_groups.pop())
        if not test_groups and len(train_groups) > 1:
            test_groups.append(train_groups.pop())

        train_records = [r for grp in train_groups for r in grp]
        val_records = [r for grp in val_groups for r in grp]
        test_records = [r for grp in test_groups for r in grp]

        return {
            "train": train_records,
            "val": val_records,
            "test": test_records
        }
