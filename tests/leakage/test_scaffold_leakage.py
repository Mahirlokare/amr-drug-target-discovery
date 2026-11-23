"""
Leakage tests: Formally assert scaffold disjointness between all partitions.
Bemis-Murcko scaffolds must be completely disjoint across train/val/test.
This is a hard scientific requirement — any failure is a critical bug.
"""
import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.dataset_builder import AMRDatasetBuilder
from src.data.splits import SplitManager


def get_scaffold(smiles: str) -> str:
    """Return the canonical Bemis-Murcko scaffold SMILES, or '' if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""


@pytest.fixture(scope="module")
def scaffold_splits(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("leak_test")
    builder = AMRDatasetBuilder(processed_dir=str(tmp))
    dataset = builder.build()
    splitter = SplitManager()
    splits = splitter.scaffold_split(dataset)
    return splits


class TestScaffoldLeakage:
    """Zero scaffold leakage across all partition pairs."""

    def _scaffold_set(self, records):
        scaffolds = set()
        for r in records:
            sc = get_scaffold(r["smiles"])
            if sc:  # skip acyclic/linear molecules (empty scaffold)
                scaffolds.add(sc)
        return scaffolds

    def test_no_scaffold_leakage_train_val(self, scaffold_splits):
        train_sc = self._scaffold_set(scaffold_splits["train"])
        val_sc = self._scaffold_set(scaffold_splits["val"])
        overlap = train_sc & val_sc
        assert len(overlap) == 0, (
            f"SCAFFOLD LEAKAGE DETECTED between train and val!\n"
            f"Overlapping scaffolds ({len(overlap)}): {overlap}"
        )

    def test_no_scaffold_leakage_train_test(self, scaffold_splits):
        train_sc = self._scaffold_set(scaffold_splits["train"])
        test_sc = self._scaffold_set(scaffold_splits["test"])
        overlap = train_sc & test_sc
        assert len(overlap) == 0, (
            f"SCAFFOLD LEAKAGE DETECTED between train and test!\n"
            f"Overlapping scaffolds ({len(overlap)}): {overlap}"
        )

    def test_no_scaffold_leakage_val_test(self, scaffold_splits):
        val_sc = self._scaffold_set(scaffold_splits["val"])
        test_sc = self._scaffold_set(scaffold_splits["test"])
        overlap = val_sc & test_sc
        assert len(overlap) == 0, (
            f"SCAFFOLD LEAKAGE DETECTED between val and test!\n"
            f"Overlapping scaffolds ({len(overlap)}): {overlap}"
        )

    def test_all_partitions_non_empty(self, scaffold_splits):
        for partition, records in scaffold_splits.items():
            assert len(records) > 0, f"Partition '{partition}' is empty!"

    def test_scaffold_coverage(self, scaffold_splits):
        """Scaffolds in test should not appear in training data at all."""
        train_sc = self._scaffold_set(scaffold_splits["train"])
        val_sc = self._scaffold_set(scaffold_splits["val"])
        test_sc = self._scaffold_set(scaffold_splits["test"])
        seen_in_training = train_sc | val_sc
        leaked = test_sc & seen_in_training
        assert len(leaked) == 0, (
            f"Test scaffolds seen during training/validation!\n"
            f"Leaked ({len(leaked)}): {leaked}"
        )
