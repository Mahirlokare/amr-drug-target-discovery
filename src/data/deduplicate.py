"""
Deduplication and replicate aggregation for bioactivity records.
Complies with Project 9 Blueprint Section 5.4 & 8.4.
"""

from collections import defaultdict
import math
from typing import Any, Dict, List, Tuple
from src.common.logging import get_logger
from src.data.schemas import ActivityRecord

logger = get_logger("deduplicate")


class ReplicateAggregator:
    """Aggregates replicates per compound-target pair under documented scientific rules."""

    def __init__(self, max_log_discrepancy: float = 1.0):
        self.max_log_discrepancy = max_log_discrepancy

    def deduplicate_records(
        self, records: List[ActivityRecord]
    ) -> Tuple[List[ActivityRecord], List[Dict[str, Any]]]:
        """
        Groups records by (compound_id, target_id).
        Returns:
            clean_records: List of aggregated ActivityRecords
            quarantine_records: List of conflicting/ambiguous records with audit info
        """
        grouped = defaultdict(list)
        for rec in records:
            key = (rec.compound_id, rec.target_id)
            grouped[key].append(rec)

        clean_records: List[ActivityRecord] = []
        quarantine_audit: List[Dict[str, Any]] = []

        for (cid, tid), group in grouped.items():
            if len(group) == 1:
                clean_records.append(group[0])
                continue

            # Multiple measurements exist
            pic50_values = [r.pic50 for r in group]
            min_p = min(pic50_values)
            max_p = max(pic50_values)
            span = max_p - min_p

            if span > self.max_log_discrepancy:
                # Conflicting replicates (> 10-fold discrepancy)
                logger.info(
                    f"Quarantining conflicting replicate for compound {cid}: "
                    f"pIC50 span {span:.2f} > threshold {self.max_log_discrepancy}"
                )
                quarantine_audit.append({
                    "compound_id": cid,
                    "target_id": tid,
                    "reason": "HIGH_REPLICATE_VARIANCE",
                    "pic50_span": round(span, 2),
                    "measurements": [r.value_normalized_nm for r in group],
                    "record_ids": [r.record_id for r in group]
                })
                continue

            # Consistent replicates: geometric mean of IC50
            # log10(geo_mean) = avg(log10) -> avg(pIC50)
            avg_pic50 = round(sum(pic50_values) / len(pic50_values), 3)
            # Recompute IC50 in nM from avg_pic50
            geo_mean_nm = round(10.0 ** (9.0 - avg_pic50), 2)

            # Determine label under active_min_pic50 >= 6.0 and inactive <= 5.0
            label = 1 if avg_pic50 >= 6.0 else (0 if avg_pic50 < 5.0 else None)
            label_status = "ACTIVE" if label == 1 else ("INACTIVE" if label == 0 else "INTERMEDIATE")

            # Document years (pick median/latest)
            years = [r.year for r in group if r.year is not None]
            med_year = sorted(years)[len(years)//2] if years else None

            rep_record = ActivityRecord(
                record_id=f"AGG-{cid}-{tid}",
                compound_id=cid,
                target_id=tid,
                assay_id="AGGREGATED",
                source=f"REPLICATE_AGG({len(group)})",
                endpoint="IC50",
                relation="=",
                value_raw=geo_mean_nm,
                units_raw="nM",
                value_normalized_nm=geo_mean_nm,
                pic50=avg_pic50,
                label=label,
                label_status=label_status,
                year=med_year,
                provenance_uri=f"Aggregated from {len(group)} records"
            )
            clean_records.append(rep_record)

        return clean_records, quarantine_audit
