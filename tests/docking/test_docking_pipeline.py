"""
Docking pipeline and AutoDock Vina runner test suite.
Verifies receptor/ligand handling, Vina execution/surrogate fallback,
disclaimer tagging, and typed DockingRun record generation.
"""

from pathlib import Path
import pytest
from src.docking.vina_runner import VinaRunner
from src.docking.parse_results import DockingResultParser
from src.data.schemas import DockingRun


def test_vina_runner_execution(tmp_path):
    runner = VinaRunner()
    rec_pdbqt = tmp_path / "rec.pdbqt"
    lig_pdbqt = tmp_path / "lig.pdbqt"
    out_pdbqt = tmp_path / "out.pdbqt"

    rec_pdbqt.write_text("ATOM      1  N   MET A   1      20.154  12.304  10.201  1.00 20.00           N\n")
    lig_pdbqt.write_text("ATOM      1  C   INH A   1       2.154   1.304   0.201  1.00 20.00           C\n")

    grid_box = {
        "center": [-12.5, 34.2, 18.0],
        "size": [20.0, 20.0, 20.0]
    }

    score, stdout, rmsd, meta = runner.run_docking(
        receptor_pdbqt=rec_pdbqt,
        ligand_pdbqt=lig_pdbqt,
        grid_box=grid_box,
        output_pdbqt=out_pdbqt,
        exhaustiveness=4
    )

    assert isinstance(score, float)
    assert score < 0.0  # Favorable negative binding energy
    assert "is_vina_binary" in meta
    assert "engine" in meta
    assert out_pdbqt.exists()


def test_docking_result_parser(tmp_path):
    pose_file = tmp_path / "pose.pdbqt"
    pose_file.write_text("REMARK DOCKED POSE\n")

    run_record = DockingResultParser.parse_docking_run(
        prediction_id="PRED-TEST-001",
        compound_id="CMPD-0001",
        target_id="TARGET-001",
        receptor_id="4TZK",
        affinity_kcal_mol=-8.4,
        pose_file=pose_file,
        redock_rmsd=1.2,
        is_vina_binary=False
    )

    assert isinstance(run_record, DockingRun)
    assert run_record.affinity_kcal_mol == -8.4
    assert run_record.status == "SURROGATE_ESTIMATE"
    assert "Surrogate" in run_record.vina_version
    assert "Surrogate" in run_record.notes
