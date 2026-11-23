"""
Integration tests: Dataset build → split → feature extraction chain.
Tests run against the in-memory seed dataset without network access.
"""
import sys
import os
import json
import pytest
import tempfile
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.dataset_builder import AMRDatasetBuilder
from src.data.splits import SplitManager
from src.features.molecular_graph import MolecularGraphFeaturizer
from src.features.fingerprints import FingerprintGenerator


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    """Build the benchmark dataset into a temp directory."""
    tmp = tmp_path_factory.mktemp("processed")
    builder = AMRDatasetBuilder(processed_dir=str(tmp))
    dataset = builder.build()
    return dataset, tmp


class TestDatasetBuildIntegration:
    """Full dataset build produces valid, consistent records."""

    def test_dataset_has_records(self, built_dataset):
        dataset, _ = built_dataset
        assert len(dataset) > 0, "Dataset must be non-empty"

    def test_dataset_has_required_fields(self, built_dataset):
        dataset, _ = built_dataset
        required = {"smiles", "pIC50", "binary_label"}
        for rec in dataset:
            keys = set(rec.keys())
            assert required.issubset(keys), f"Missing fields in record: {keys}"

    def test_dataset_pIC50_range(self, built_dataset):
        dataset, _ = built_dataset
        for rec in dataset:
            if rec.get("binary_label") in (0, 1):
                pic50 = rec["pIC50"]
                assert 3.0 <= pic50 <= 12.0, f"Implausible pIC50: {pic50}"

    def test_dataset_persisted_to_disk(self, built_dataset):
        _, tmp = built_dataset
        json_files = list(tmp.glob("*.json"))
        assert len(json_files) >= 1, "At least one JSON file must be persisted"

    def test_dataset_json_loadable(self, built_dataset):
        _, tmp = built_dataset
        # Find any persisted dataset JSON file and assert it is loadable
        json_files = list(tmp.glob("*.json"))
        assert len(json_files) > 0, "No JSON dataset files found"
        for json_file in json_files:
            with open(json_file, encoding="utf-8") as f:
                loaded = json.load(f)
            assert isinstance(loaded, (list, dict))
            assert len(loaded) > 0


class TestSplitIntegration:
    """Split produces train/val/test with correct disjointness guarantees."""

    def test_scaffold_split_produces_three_partitions(self, built_dataset):
        dataset, _ = built_dataset
        splitter = SplitManager()
        splits = splitter.scaffold_split(dataset)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_all_records_assigned(self, built_dataset):
        dataset, _ = built_dataset
        splitter = SplitManager()
        splits = splitter.scaffold_split(dataset)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(dataset), "Split must account for all records"

    def test_no_smiles_overlap_across_partitions(self, built_dataset):
        dataset, _ = built_dataset
        splitter = SplitManager()
        splits = splitter.scaffold_split(dataset)
        train_smiles = {r["smiles"] for r in splits["train"]}
        val_smiles = {r["smiles"] for r in splits["val"]}
        test_smiles = {r["smiles"] for r in splits["test"]}
        assert train_smiles.isdisjoint(val_smiles), "Train/val SMILES overlap!"
        assert train_smiles.isdisjoint(test_smiles), "Train/test SMILES overlap!"
        assert val_smiles.isdisjoint(test_smiles), "Val/test SMILES overlap!"

    def test_train_partition_is_largest(self, built_dataset):
        dataset, _ = built_dataset
        splitter = SplitManager()
        splits = splitter.scaffold_split(dataset)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])


class TestFeatureExtractionIntegration:
    """Feature extraction produces tensors of expected shape and dtype."""

    VALID_SMILES = [
        "CC1=CC2=C(C=C1)N=C(S2)NC(=O)C3=CC=CO3",   # active-like
        "c1ccc(cc1)S(=O)(=O)N",                       # inactive-like
        "CC(=O)Nc1ccc(O)cc1",                          # paracetamol scaffold
    ]

    def test_graph_featurizer_shape(self):
        feat = MolecularGraphFeaturizer()
        for smi in self.VALID_SMILES:
            g = feat.featurize(smi)
            assert g is not None, f"Graph None for {smi}"
            assert g.x.shape[1] == 43, f"Expected 43 node features, got {g.x.shape[1]}"
            assert g.edge_index.shape[0] == 2

    def test_fingerprint_shape(self):
        gen = FingerprintGenerator()
        for smi in self.VALID_SMILES:
            fp = gen.morgan_fingerprint(smi)
            assert fp is not None
            assert fp.shape == (2048,), f"Expected 2048-d fingerprint, got {fp.shape}"

    def test_graph_featurizer_invalid_smiles_returns_none(self):
        feat = MolecularGraphFeaturizer()
        result = feat.featurize("NOT_A_VALID_SMILES_STRING!!!!")
        assert result is None, "Invalid SMILES should return None"

    def test_full_chain_dataset_to_features(self, built_dataset):
        """Build dataset → split → extract graph features for train partition."""
        dataset, _ = built_dataset
        splitter = SplitManager()
        splits = splitter.scaffold_split(dataset)
        feat = MolecularGraphFeaturizer()

        failures = []
        for rec in splits["train"][:10]:  # sample first 10
            g = feat.featurize(rec["smiles"])
            if g is None:
                failures.append(rec["smiles"])

        assert len(failures) == 0, f"Graph featurization failed for: {failures}"
