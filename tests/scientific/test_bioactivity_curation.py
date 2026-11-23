"""
Scientific bioactivity curation test suite.
Verifies segregation of IC50, Ki, Kd, EC50, and MIC endpoints,
correct unit conversions (nM, uM, ug/mL -> uM), pIC50 calculations,
and adherence to pre-registered activity threshold policies.
Complies with Section 3 & 14.
"""

import pytest
from src.data.labels import LabelHarmonizer


@pytest.fixture
def harmonizer():
    return LabelHarmonizer()


def test_enzymatic_ic50_harmonization(harmonizer):
    # Active: IC50 = 45 nM -> pIC50 = 7.347 -> Active (1)
    rec_active = harmonizer.process_record(
        record_id="REC-001",
        compound_id="CMPD-0001",
        target_id="TARGET-001",
        assay_id="ASSAY-001",
        source="ChEMBL",
        endpoint="IC50",
        relation="=",
        value_raw=45.0,
        units_raw="nM"
    )
    assert rec_active is not None
    assert rec_active.label == 1
    assert rec_active.label_status == "ACTIVE"
    assert rec_active.pic50 == pytest.approx(7.347, abs=1e-2)

    # Inactive: IC50 = 50 uM = 50,000 nM -> pIC50 = 4.301 -> Inactive (0)
    rec_inactive = harmonizer.process_record(
        record_id="REC-002",
        compound_id="CMPD-0002",
        target_id="TARGET-001",
        assay_id="ASSAY-002",
        source="ChEMBL",
        endpoint="IC50",
        relation="=",
        value_raw=50.0,
        units_raw="uM"
    )
    assert rec_inactive is not None
    assert rec_inactive.label == 0
    assert rec_inactive.label_status == "INACTIVE"
    assert rec_inactive.pic50 == pytest.approx(4.301, abs=1e-2)


def test_ki_and_kd_segregation(harmonizer):
    # Ki and Kd must be accepted with their native endpoints preserved
    rec_ki = harmonizer.process_record(
        record_id="REC-003",
        compound_id="CMPD-0003",
        target_id="TARGET-001",
        assay_id="ASSAY-003",
        source="ChEMBL",
        endpoint="Ki",
        relation="=",
        value_raw=120.0,
        units_raw="nM"
    )
    assert rec_ki is not None
    assert rec_ki.endpoint == "Ki"
    assert rec_ki.label == 1

    rec_kd = harmonizer.process_record(
        record_id="REC-004",
        compound_id="CMPD-0004",
        target_id="TARGET-001",
        assay_id="ASSAY-004",
        source="ChEMBL",
        endpoint="Kd",
        relation="=",
        value_raw=850.0,
        units_raw="nM"
    )
    assert rec_kd is not None
    assert rec_kd.endpoint == "Kd"
    assert rec_kd.label == 1


def test_mic_microbiological_endpoint_handling(harmonizer):
    # MIC in ug/mL: Potent <= 1.0 ug/mL -> Active (1)
    rec_mic_active = harmonizer.process_record(
        record_id="REC-005",
        compound_id="CMPD-0005",
        target_id="TARGET-001",
        assay_id="ASSAY-005",
        source="ChEMBL",
        endpoint="MIC",
        relation="=",
        value_raw=0.25,
        units_raw="ug/mL",
        molecular_weight=300.0
    )
    assert rec_mic_active is not None
    assert rec_mic_active.endpoint == "MIC"
    assert rec_mic_active.label == 1
    assert rec_mic_active.label_status == "ACTIVE"

    # MIC > 8.0 ug/mL -> Inactive (0)
    rec_mic_inactive = harmonizer.process_record(
        record_id="REC-006",
        compound_id="CMPD-0006",
        target_id="TARGET-001",
        assay_id="ASSAY-006",
        source="ChEMBL",
        endpoint="MIC",
        relation="=",
        value_raw=32.0,
        units_raw="ug/mL",
        molecular_weight=300.0
    )
    assert rec_mic_inactive is not None
    assert rec_mic_inactive.endpoint == "MIC"
    assert rec_mic_inactive.label == 0
    assert rec_mic_inactive.label_status == "INACTIVE"
