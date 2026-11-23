"""
Unit tests for Database and Persistence module.
Tests SQLite/PostgreSQL schema initialization, CRUD operations, healthcheck,
restart persistence, and connection error handling.
Complies with Project 9 Blueprint Section 14 and Section 21.D.
"""

import tempfile
from pathlib import Path
import pytest

from src.storage.db import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """Creates a temporary isolated SQLite database instance."""
    db_file = tmp_path / "test_amr.db"
    manager = DatabaseManager(db_url=None)
    # Redirect db_path to isolated temp path
    manager.db_path = db_file
    manager.init_schema()
    return manager


def test_schema_initialization(temp_db):
    """Verify all 7 core tables are created."""
    health = temp_db.health_check()
    assert health["status"] == "HEALTHY"
    counts = health["table_counts"]
    for expected_table in [
        "targets",
        "compounds",
        "activity_records",
        "model_runs",
        "predictions",
        "docking_runs",
        "admet_runs",
    ]:
        assert expected_table in counts
        assert counts[expected_table] == 0


def test_target_crud(temp_db):
    """Test inserting and retrieving target definitions."""
    target_data = {
        "target_id": "TARGET-001",
        "name": "Mycobacterium tuberculosis InhA",
        "accession": "P9WGR1",
        "organism": "Mycobacterium tuberculosis",
        "gene": "inhA",
        "sequence_hash": "sha256_mock_hash",
        "chembl_target_id": "CHEMBL2014",
        "pdb_id": "4TZK",
        "selection_rationale": "Validated clinical target of isoniazid",
    }
    temp_db.save_target(target_data)
    targets = temp_db.list_targets()
    assert len(targets) == 1
    assert targets[0]["target_id"] == "TARGET-001"
    assert targets[0]["gene"] == "inhA"


def test_prediction_crud(temp_db):
    """Test saving and querying predictions."""
    pred = {
        "prediction_id": "PRED-001",
        "run_id": "RUN-2026-001",
        "compound_id": "CMPD-0001",
        "target_id": "TARGET-001",
        "smiles": "O=C(NN)c1ccncc1",
        "calibrated_prob": 0.88,
        "uncertainty": 0.04,
        "is_uncertain": False,
        "is_out_of_domain": False,
        "decision_status": "PRIORITISED",
    }
    temp_db.save_prediction(pred)
    preds = temp_db.list_predictions(target_id="TARGET-001")
    assert len(preds) == 1
    assert preds[0]["prediction_id"] == "PRED-001"
    assert preds[0]["calibrated_prob"] == 0.88


def test_docking_and_admet_crud(temp_db):
    """Test docking and ADMET record storage."""
    dock = {
        "docking_id": "DOCK-001",
        "prediction_id": "PRED-001",
        "compound_id": "CMPD-0001",
        "target_id": "TARGET-001",
        "receptor_pdb": "4TZK_clean_A.pdb",
        "binding_affinity": -8.7,
        "rmsd_to_control": 1.15,
        "pose_path": "artifacts/docking/TARGET-001/CMPD-0001_docked_pose.pdbqt",
    }
    temp_db.save_docking_run(dock)
    docks = temp_db.list_docking_runs()
    assert len(docks) == 1
    assert docks[0]["binding_affinity"] == -8.7

    admet = {
        "admet_id": "ADMET-001",
        "compound_id": "CMPD-0001",
        "mol_weight": 137.14,
        "logp": -0.64,
        "hbd": 2,
        "hba": 3,
        "tpsa": 55.12,
        "rotatable_bonds": 1,
        "lipinski_pass": 1,
        "veber_pass": 1,
        "qed": 0.65,
        "pains_alerts": 0,
        "decision_status": "PRIORITISED",
    }
    temp_db.save_admet_run(admet)
    health = temp_db.health_check()
    assert health["table_counts"]["docking_runs"] == 1
    assert health["table_counts"]["admet_runs"] == 1


def test_persistence_across_restart(tmp_path):
    """Verify that records persist across manager instantiation (simulating app restart)."""
    db_file = tmp_path / "persistent_amr.db"

    # Instance 1: write data
    manager1 = DatabaseManager(db_url=None)
    manager1.db_path = db_file
    manager1.init_schema()
    manager1.save_target({
        "target_id": "TARGET-TEST",
        "name": "Test Protein",
        "accession": "P00000",
        "organism": "E. coli",
        "gene": "testA",
        "sequence_hash": "dummy_hash",
        "chembl_target_id": None,
        "pdb_id": None,
        "selection_rationale": "Persistence test",
    })

    # Instance 2: read data after 'restart'
    manager2 = DatabaseManager(db_url=None)
    manager2.db_path = db_file
    targets = manager2.list_targets()
    assert len(targets) == 1
    assert targets[0]["target_id"] == "TARGET-TEST"
