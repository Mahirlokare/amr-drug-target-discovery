"""
Label harmonisation and threshold assignment.
Complies with Project 9 Blueprint Section 5.3 & 8.4.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from src.common.config import config_manager
from src.common.logging import get_logger
from src.data.schemas import ActivityRecord

logger = get_logger("labels")


class LabelHarmonizer:
    """Harmonizes bioactivity records to standard pIC50 and binary activity labels."""

    def __init__(self, labels_config: Optional[Dict[str, Any]] = None):
        self.config = labels_config or config_manager.get_config("labels")
        self.active_max_nm = self.config["classification"]["active_max_nm"]
        self.active_min_pic50 = self.config["classification"]["active_min_pic50"]
        self.inactive_min_nm = self.config["classification"]["inactive_min_nm"]
        self.inactive_max_pic50 = self.config["classification"]["inactive_max_pic50"]
        self.policy_version = self.config.get("policy_version", "label-v1.0")

    def normalize_value_to_nm(self, value: float, units: str) -> Optional[float]:
        """Converts raw concentration to nanomolar (nM)."""
        if value is None or value <= 0:
            return None
        
        unit_clean = units.strip() if isinstance(units, str) else ""
        factors = self.config["units_conversion"]["factors"]
        factor = factors.get(unit_clean)
        
        if factor is not None:
            return float(value) * factor
        elif unit_clean.lower() in ["nm", "nanomolar"]:
            return float(value)
        elif unit_clean.lower() in ["um", "micromolar", "µm"]:
            return float(value) * 1000.0
        elif unit_clean.lower() in ["mm", "millimolar"]:
            return float(value) * 1000000.0
        elif unit_clean.lower() in ["m", "molar"]:
            return float(value) * 1000000000.0
        
        # Unconvertible units
        return None

    def calculate_pic50(self, value_nm: float) -> float:
        """pIC50 = -log10(IC50 in M) = 9 - log10(IC50 in nM)"""
        if value_nm <= 0:
            return 0.0
        return round(9.0 - math.log10(value_nm), 3)

    def assign_label(self, value_nm: float, relation: str = "=") -> Tuple[Optional[int], str]:
        """
        Assigns binary label according to pre-registered policy:
        Active (1): IC50 <= 1000 nM (pIC50 >= 6.0)
        Inactive (0): IC50 > 10000 nM (pIC50 < 5.0)
        Intermediate/Quarantine (None): 1000 nM < IC50 <= 10000 nM
        Censored relations:
          '>' 10000 nM -> Inactive (0)
          '<' 1000 nM -> Active (1)
          '>' 500 nM -> Ambiguous / Quarantined
        """
        rel = relation.strip() if relation else "="

        if rel == ">":
            if value_nm >= self.inactive_min_nm:
                return 0, "INACTIVE"
            else:
                return None, "INTERMEDIATE"

        if rel == "<":
            if value_nm <= self.active_max_nm:
                return 1, "ACTIVE"
            else:
                return None, "INTERMEDIATE"

        if rel in ["=", "~"]:
            if value_nm <= self.active_max_nm:
                return 1, "ACTIVE"
            elif value_nm > self.inactive_min_nm:
                return 0, "INACTIVE"
            else:
                return None, "INTERMEDIATE"

        return None, "EXCLUDED"

    def process_record(
        self,
        record_id: str,
        compound_id: str,
        target_id: str,
        assay_id: str,
        source: str,
        endpoint: str,
        relation: str,
        value_raw: float,
        units_raw: str,
        year: Optional[int] = None,
        provenance_uri: Optional[str] = None
    ) -> Optional[ActivityRecord]:
        """Processes and harmonizes a raw assay measurement into a typed ActivityRecord."""
        # Endpoint check - strict IC50 only
        if endpoint.upper() not in ["IC50"]:
            logger.debug(f"Skipping non-IC50 endpoint: {endpoint}")
            return None

        val_nm = self.normalize_value_to_nm(value_raw, units_raw)
        if val_nm is None:
            return None

        pic50 = self.calculate_pic50(val_nm)
        label, label_status = self.assign_label(val_nm, relation)

        return ActivityRecord(
            record_id=record_id,
            compound_id=compound_id,
            target_id=target_id,
            assay_id=assay_id,
            source=source,
            endpoint=endpoint,
            relation=relation,
            value_raw=value_raw,
            units_raw=units_raw,
            value_normalized_nm=round(val_nm, 2),
            pic50=pic50,
            label=label,
            label_status=label_status,
            year=year,
            provenance_uri=provenance_uri
        )
