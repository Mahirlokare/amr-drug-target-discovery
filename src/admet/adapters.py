"""
ADMET triage adapter and evidence card integration.
Complies with Project 9 Blueprint Section 8.14, 13.6 & 13.7.
"""

from typing import Any, Dict, List, Optional
import uuid

from src.admet.filters import ADMETFilterEngine
from src.data.schemas import ADMETRun


class ADMETAdapter:
    """Triage adapter executing in-silico property screening on shortlist candidates."""

    def __init__(self):
        self.engine = ADMETFilterEngine()

    def evaluate_candidate(
        self,
        smiles: str,
        compound_id: str,
        prediction_id: str
    ) -> ADMETRun:
        """Evaluates candidate structure and returns typed ADMETRun record."""
        props = self.engine.compute_properties(smiles)

        return ADMETRun(
            admet_id=f"ADMET-{uuid.uuid4().hex[:8].upper()}",
            prediction_id=prediction_id,
            compound_id=compound_id,
            tool_version="RDKit ADMET Filter v1.0",
            mw=props.get("mw", 0.0),
            logp=props.get("logp", 0.0),
            hbd=props.get("hbd", 0),
            hba=props.get("hba", 0),
            tpsa=props.get("tpsa", 0.0),
            rotatable_bonds=props.get("rotatable_bonds", 0),
            lipinski_violations=props.get("lipinski_violations", 0),
            lipinski_pass=props.get("lipinski_pass", False),
            veber_pass=props.get("veber_pass", False),
            pains_alert_count=props.get("pains_alert_count", 0),
            pains_alerts=props.get("pains_alerts", []),
            qed_score=props.get("qed_score", 0.0),
            synthetic_accessibility=props.get("synthetic_accessibility", 0.0),
            warnings=props.get("warnings", [])
        )
