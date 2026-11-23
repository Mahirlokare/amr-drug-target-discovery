"""
Data quality control, audits, and quarantine management.
Complies with Project 9 Blueprint Section 5.4, 8.1 & 17.
"""

from typing import Any, Dict, List, Tuple
from src.common.logging import get_logger
from src.data.schemas import ActivityRecord, Compound

logger = get_logger("quality")


class QualityAuditor:
    """Performs rigorous scientific data hygiene audits and produces QC manifests."""

    def __init__(self):
        pass

    def audit_dataset(
        self,
        compounds: List[Compound],
        records: List[ActivityRecord]
    ) -> Dict[str, Any]:
        """
        Audits raw and standardized records for quality, missingness, and label distributions.
        """
        total_compounds = len(compounds)
        valid_compounds = [c for c in compounds if c.standardisation_status in ["SUCCESS", "MODIFIED"]]
        failed_compounds = [c for c in compounds if c.standardisation_status == "FAILED"]

        total_records = len(records)
        active_records = [r for r in records if r.label == 1]
        inactive_records = [r for r in records if r.label == 0]
        intermediate_records = [r for r in records if r.label is None and r.label_status == "INTERMEDIATE"]
        excluded_records = [r for r in records if r.label_status in ["EXCLUDED", "CONFLICTING"]]

        # Extreme value check (e.g. pIC50 < 2 or pIC50 > 12)
        extreme_records = [
            r for r in records if r.pic50 is not None and (r.pic50 < 2.0 or r.pic50 > 12.0)
        ]

        # Year distribution
        year_counts: Dict[str, int] = {}
        for r in records:
            yr = str(r.year) if r.year is not None else "UNKNOWN"
            year_counts[yr] = year_counts.get(yr, 0) + 1

        active_ratio = len(active_records) / (len(active_records) + len(inactive_records)) if (active_records or inactive_records) else 0.0

        report = {
            "compound_summary": {
                "total": total_compounds,
                "valid": len(valid_compounds),
                "failed": len(failed_compounds),
                "modified": len([c for c in compounds if c.standardisation_status == "MODIFIED"]),
                "success_rate_pct": round(len(valid_compounds) / total_compounds * 100, 2) if total_compounds > 0 else 0.0,
            },
            "activity_summary": {
                "total_records": total_records,
                "labeled_active": len(active_records),
                "labeled_inactive": len(inactive_records),
                "quarantined_intermediate": len(intermediate_records),
                "excluded_or_conflicting": len(excluded_records),
                "active_prevalence_pct": round(active_ratio * 100, 2),
            },
            "anomalies": {
                "extreme_pic50_count": len(extreme_records),
                "extreme_sample_ids": [r.record_id for r in extreme_records[:5]],
            },
            "temporal_distribution": year_counts,
            "qc_passed": len(failed_compounds) == 0 and len(extreme_records) == 0
        }

        logger.info(
            f"QC Audit complete: {report['compound_summary']['valid']}/{total_compounds} valid compounds; "
            f"{len(active_records)} actives, {len(inactive_records)} inactives, "
            f"{len(intermediate_records)} quarantined intermediates."
        )
        return report
