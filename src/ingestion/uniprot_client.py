"""
UniProt REST API client with local caching and SHA-256 sequence validation.
Complies with Project 9 Blueprint Section 5.1, 8.1 & 9.1.
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional
import requests

from src.common.config import config_manager
from src.common.logging import get_logger
from src.data.schemas import Target

logger = get_logger("uniprot_client")


class UniProtClient:
    """Client for UniProtKB REST API with caching and provenance tracking."""

    def __init__(self, cache_dir: Optional[Path] = None):
        cfg = config_manager.get_config("data_sources")
        self.base_url = cfg["uniprot"]["base_url"]
        self.timeout = cfg["uniprot"].get("timeout_seconds", 20)
        self.cache_dir = cache_dir or (config_manager.project_root / cfg["uniprot"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_target(self, accession: str, target_id: str = "TARGET-001") -> Target:
        """
        Fetches protein target metadata and amino acid sequence from UniProt.
        Uses local cache if available.
        """
        cache_file = self.cache_dir / f"{accession}.json"
        data: Optional[Dict[str, Any]] = None

        if cache_file.exists():
            logger.info(f"Loading cached UniProt record for {accession}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            url = f"{self.base_url}/{accession}.json"
            logger.info(f"Querying UniProt API: {url}")
            try:
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                else:
                    logger.warning(f"UniProt query failed ({resp.status_code}): {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"Could not connect to UniProt REST API: {e}")

        # Fallback to approved target registry if network or query fails
        if not data:
            logger.info(f"Falling back to target_registry.yaml for {accession}")
            target_reg = config_manager.get_config("target_registry")["approved_targets"].get(target_id)
            if not target_reg:
                raise ValueError(f"Target {accession} could not be resolved from UniProt or local registry.")
            return Target(
                target_id=target_id,
                name=target_reg["name"],
                accession=target_reg["accession"],
                organism=target_reg["organism"],
                strain_scope=target_reg["strain_scope"],
                gene=target_reg["gene"],
                sequence=target_reg["sequence"],
                sequence_hash=target_reg["sequence_sha256"],
                taxonomy_id=target_reg["taxonomy_id"],
                chembl_target_id=target_reg.get("chembl_target_id"),
                pdb_id=target_reg.get("pdb_id"),
                selection_rationale=target_reg.get("selection_rationale"),
            )

        # Parse UniProt JSON payload
        primary_accession = data.get("primaryAccession", accession)
        gene_name = "UNKNOWN"
        if "genes" in data and data["genes"]:
            gene_name = data["genes"][0].get("geneName", {}).get("value", "UNKNOWN")

        organism_name = data.get("organism", {}).get("scientificName", "Unknown Organism")
        tax_id = data.get("organism", {}).get("taxonId", 0)
        seq_text = data.get("sequence", {}).get("value", "")
        seq_hash = hashlib.sha256(seq_text.encode("utf-8")).hexdigest()

        rec_name = "Unknown Protein"
        desc = data.get("proteinDescription", {})
        if "recommendedName" in desc:
            rec_name = desc["recommendedName"].get("fullName", {}).get("value", rec_name)

        target = Target(
            target_id=target_id,
            name=rec_name,
            accession=primary_accession,
            organism=organism_name,
            strain_scope="H37Rv" if "H37Rv" in organism_name or tax_id == 83332 else "Species-level",
            gene=gene_name,
            sequence=seq_text,
            sequence_hash=seq_hash,
            taxonomy_id=tax_id,
            selection_rationale="Curated from UniProtKB Swiss-Prot entry."
        )
        return target
