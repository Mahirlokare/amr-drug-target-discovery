"""
Provenance and data lineage tracker.
Complies with Project 9 Blueprint Section 6 (Data contracts, schemas and provenance).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


def compute_sha256(data: str) -> str:
    """Computes SHA-256 checksum of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_provenance_record(
    source: str,
    source_record_id: str,
    target_id: str,
    endpoint: str,
    value: float,
    units: str,
    relation: str = "=",
    assay_type: str = "B",
    document_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds an immutable provenance-tracked activity record adhering to blueprint 6.2."""
    now = datetime.now(timezone.utc).isoformat()
    record_id = f"REC-{uuid.uuid4().hex[:12].upper()}"

    record = {
        "record_id": record_id,
        "source": source,
        "source_record_id": str(source_record_id),
        "target_id": target_id,
        "assay": {
            "endpoint": endpoint,
            "relation": relation,
            "value": float(value),
            "units": units,
            "assay_type": assay_type,
            "document_id": document_id or "UNKNOWN",
        },
        "retrieved_at": now,
        "label_policy_version": "label-v1.0",
        "provenance_uri": f"{source}:{source_record_id}",
    }
    if extra:
        record["extra_metadata"] = extra
    return record


class RunManifest:
    """Manages immutable run manifests for experiments, scoring, and docking triage."""

    def __init__(self, run_type: str, config: Dict[str, Any]):
        self.run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_type = run_type
        self.config = config
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.metrics: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
        self.status = "INITIALIZED"

    def complete(self, metrics: Dict[str, Any], artifacts: Dict[str, str]) -> None:
        self.metrics = metrics
        self.artifacts = artifacts
        self.status = "COMPLETED"
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error_message: str) -> None:
        self.status = "FAILED"
        self.error = error_message
        self.failed_at = datetime.now(timezone.utc).isoformat()

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{self.run_id}_manifest.json"
        data = {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": getattr(self, "completed_at", None),
            "config": self.config,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return manifest_path
