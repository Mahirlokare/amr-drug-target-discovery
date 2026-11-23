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
    """Harmonizes bioactivity records to standard pIC50/pEC50 and binary activity labels across distinct endpoints."""

    def __init__(self, labels_config: Optional[Dict[str, Any]] = None):
        self.config = labels_config or config_manager.get_config("labels")
        self.supported_endpoints = [
            ep.upper() for ep in self.config.get("supported_endpoints", ["IC50", "Ki", "Kd", "EC50", "MIC"])
        ]
        
        # Classification thresholds
        cls_cfg = self.config.get("classification", {})
        enz_cfg = cls_cfg.get("enzymatic", {})
        self.active_max_nm = enz_cfg.get("active_max_nm", cls_cfg.get("active_max_nm", 1000.0))
        self.active_min_pic50 = enz_cfg.get("active_min_pic50", cls_cfg.get("active_min_pic50", 6.0))
        self.inactive_min_nm = enz_cfg.get("inactive_min_nm", cls_cfg.get("inactive_min_nm", 10000.0))
        self.inactive_max_pic50 = enz_cfg.get("inactive_max_pic50", cls_cfg.get("inactive_max_pic50", 5.0))

        # Cellular thresholds
        cell_cfg = cls_cfg.get("cellular_ec50", {})
        self.ec50_active_max_nm = cell_cfg.get("active_max_nm", 2000.0)
        self.ec50_inactive_min_nm = cell_cfg.get("inactive_min_nm", 20000.0)

        # Microbiological whole-cell thresholds (ug/mL)
        mic_cfg = cls_cfg.get("microbiological_mic", {})
        self.mic_active_max_ug_ml = mic_cfg.get("active_max_ug_ml", 1.0)
        self.mic_inactive_min_ug_ml = mic_cfg.get("inactive_min_ug_ml", 8.0)

        self.policy_version = self.config.get("policy_version", "label-v2.0")

    def normalize_value_to_nm(
        self, value: float, units: str, endpoint: str = "IC50", mw: Optional[float] = None
    ) -> Optional[float]:
        """
        Converts raw concentration to nanomolar (nM).
        Handles ug/mL conversion when molecular weight is available.
        """
        if value is None or value <= 0:
            return None
        
        unit_clean = units.strip().lower() if isinstance(units, str) else ""
        factors = self.config["units_conversion"]["factors"]
        factor = factors.get(units.strip() if isinstance(units, str) else "")
        
        if factor is not None:
            return float(value) * factor
        elif unit_clean in ["nm", "nanomolar"]:
            return float(value)
        elif unit_clean in ["um", "micromolar", "µm", "umol/l"]:
            return float(value) * 1000.0
        elif unit_clean in ["mm", "millimolar"]:
            return float(value) * 1000000.0
        elif unit_clean in ["m", "molar"]:
            return float(value) * 1000000000.0
        elif unit_clean in ["ug/ml", "ug.ml-1", "mg/l"]:
            # Molar mass conversion: nM = (ug/mL / MW) * 1,000,000
            if mw and mw > 0:
                return (float(value) / float(mw)) * 1000000.0
            # If MW is unavailable, return rough 350 g/mol surrogate for ranking or None
            return None
        
        return None

    def calculate_pic50(self, value_nm: float) -> float:
        """pActivity = -log10(M) = 9 - log10(nM)"""
        if value_nm <= 0:
            return 0.0
        return round(9.0 - math.log10(value_nm), 3)

    def assign_label(
        self, value_nm: float, relation: str = "=", endpoint: str = "IC50"
    ) -> Tuple[Optional[int], str]:
        """
        Assigns binary label according to endpoint-specific policy:
        - Enzymatic (IC50, Ki, Kd): Active <= 1000 nM, Inactive > 10000 nM
        - Cellular (EC50): Active <= 2000 nM, Inactive > 20000 nM
        - Intermediate/Quarantine: between active and inactive thresholds
        """
        rel = relation.strip() if relation else "="
        ep_upper = endpoint.upper()

        if ep_upper == "EC50":
            act_thresh = self.ec50_active_max_nm
            inact_thresh = self.ec50_inactive_min_nm
        else:
            act_thresh = self.active_max_nm
            inact_thresh = self.inactive_min_nm

        if rel == ">":
            if value_nm >= inact_thresh:
                return 0, "INACTIVE"
            else:
                return None, "INTERMEDIATE"

        if rel == "<":
            if value_nm <= act_thresh:
                return 1, "ACTIVE"
            else:
                return None, "INTERMEDIATE"

        if rel in ["=", "~"]:
            if value_nm <= act_thresh:
                return 1, "ACTIVE"
            elif value_nm > inact_thresh:
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
        provenance_uri: Optional[str] = None,
        molecular_weight: Optional[float] = None
    ) -> Optional[ActivityRecord]:
        """Processes and harmonizes a raw assay measurement into a typed ActivityRecord with endpoint segregation."""
        ep_clean = endpoint.strip()
        ep_upper = ep_clean.upper()

        # Endpoint check - must be among supported endpoints
        if ep_upper not in self.supported_endpoints:
            logger.debug(f"Skipping unsupported endpoint: {endpoint}")
            return None

        # Special handling for whole-cell microbiological MIC (ug/mL)
        if ep_upper == "MIC" and units_raw.lower() in ["ug/ml", "ug.ml-1", "mg/l"]:
            val_nm = (float(value_raw) / (molecular_weight or 350.0)) * 1000000.0
            pic50 = self.calculate_pic50(val_nm)
            # Label by ug/mL breakpoint directly
            if value_raw <= self.mic_active_max_ug_ml and relation in ["=", "<="]:
                label, label_status = 1, "ACTIVE"
            elif value_raw >= self.mic_inactive_min_ug_ml and relation in ["=", ">", ">="]:
                label, label_status = 0, "INACTIVE"
            else:
                label, label_status = None, "INTERMEDIATE"
        else:
            val_nm = self.normalize_value_to_nm(value_raw, units_raw, endpoint=ep_upper, mw=molecular_weight)
            if val_nm is None:
                return None
            pic50 = self.calculate_pic50(val_nm)
            label, label_status = self.assign_label(val_nm, relation, endpoint=ep_upper)

        return ActivityRecord(
            record_id=record_id,
            compound_id=compound_id,
            target_id=target_id,
            assay_id=assay_id,
            source=source,
            endpoint=ep_clean,
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
