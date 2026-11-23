"""
Data contracts and schemas for AMR Drug Target Discovery.
Complies with Project 9 Blueprint Section 6.1 (Core Entities).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Compound:
    """Canonical internal chemical identity."""
    compound_id: str
    smiles_raw: str
    smiles_standard: str
    inchikey: str
    standardisation_status: str  # "SUCCESS", "MODIFIED", "FAILED"
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Target:
    """Stable protein target definition."""
    target_id: str
    name: str
    accession: str
    organism: str
    strain_scope: str
    gene: str
    sequence: str
    sequence_hash: str
    taxonomy_id: int
    chembl_target_id: Optional[str] = None
    pdb_id: Optional[str] = None
    selection_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Assay:
    """Experimental context for bioactivity measurements."""
    assay_id: str
    source: str
    endpoint: str
    relation: str
    value: float
    units: str
    assay_type: str
    document_id: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityRecord:
    """Standardized training/evaluation observation."""
    record_id: str
    compound_id: str
    target_id: str
    assay_id: str
    source: str
    endpoint: str
    relation: str
    value_raw: float
    units_raw: str
    value_normalized_nm: float
    pic50: float
    label: Optional[int]  # 1: Active, 0: Inactive, None: Intermediate/Quarantined
    label_status: str  # "ACTIVE", "INACTIVE", "INTERMEDIATE", "CONFLICTING", "EXCLUDED"
    year: Optional[int] = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSet:
    """Molecular graph, fingerprint or protein embedding reference."""
    feature_id: str
    entity_id: str  # compound_id or target_id
    feature_type: str  # "graph", "fingerprint", "esm2_embedding"
    version: str
    shape: List[int]
    checksum: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """Scored candidate result with calibration and uncertainty."""
    prediction_id: str
    run_id: str
    compound_id: str
    target_id: str
    smiles: str
    raw_logit: float
    raw_score: float
    calibrated_probability: float
    uncertainty: float  # std dev or entropy
    is_uncertain: bool
    is_out_of_domain: bool
    nearest_train_similarity: float
    substructure_attribution: Optional[Dict[str, float]] = None
    decision_status: str = "NEEDS_REVIEW"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DockingRun:
    """AutoDock Vina docking provenance and result."""
    docking_id: str
    prediction_id: str
    compound_id: str
    target_id: str
    receptor_id: str
    vina_version: str
    exhaustiveness: int
    affinity_kcal_mol: float
    pose_path: Optional[str] = None
    redock_rmsd: Optional[float] = None
    status: str = "COMPLETED"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ADMETRun:
    """Physicochemical and early safety filter triage results."""
    admet_id: str
    prediction_id: str
    compound_id: str
    tool_version: str
    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    lipinski_violations: int
    lipinski_pass: bool
    veber_pass: bool
    pains_alert_count: int
    pains_alerts: List[str]
    qed_score: float
    synthetic_accessibility: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
