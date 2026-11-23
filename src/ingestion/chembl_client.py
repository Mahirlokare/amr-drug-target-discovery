"""
ChEMBL REST API client with local raw snapshot caching and provenance tracking.
Complies with Project 9 Blueprint Section 5.1, 8.1 & 9.1.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import requests

from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("chembl_client")


class ChEMBLClient:
    """Fetches and caches raw bioactivity records from ChEMBL Web Services."""

    def __init__(self, cache_dir: Optional[Path] = None):
        cfg = config_manager.get_config("data_sources")
        self.base_url = cfg["chembl"]["base_url"]
        self.timeout = cfg["chembl"].get("timeout_seconds", 30)
        self.max_retries = cfg["chembl"].get("max_retries", 3)
        self.backoff_factor = cfg["chembl"].get("backoff_factor", 1.5)
        self.page_size = cfg["chembl"].get("page_size", 100)
        self.cache_dir = cache_dir or (config_manager.project_root / cfg["chembl"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_target_activities(
        self,
        target_chembl_id: str = "CHEMBL2014",
        standard_type: str = "IC50",
        limit: int = 500,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves bioactivity records for a target.
        Stores an immutable raw JSON snapshot before parsing.
        """
        cache_file = self.cache_dir / f"{target_chembl_id}_{standard_type}_raw.json"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached raw ChEMBL snapshot: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info(f"Querying ChEMBL API for {target_chembl_id} ({standard_type})...")
        activities: List[Dict[str, Any]] = []
        offset = 0

        # Endpoint format: https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=...
        while len(activities) < limit:
            batch_limit = min(self.page_size, limit - len(activities))
            url = (
                f"{self.base_url}/activity.json"
                f"?target_chembl_id={target_chembl_id}"
                f"&standard_type={standard_type}"
                f"&limit={batch_limit}"
                f"&offset={offset}"
            )
            
            success = False
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.get(url, timeout=self.timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        batch = data.get("activities", [])
                        if not batch:
                            break
                        activities.extend(batch)
                        offset += len(batch)
                        success = True
                        break
                    elif resp.status_code == 404:
                        logger.warning(f"ChEMBL target not found: {target_chembl_id}")
                        break
                    else:
                        logger.warning(f"ChEMBL HTTP {resp.status_code} on attempt {attempt}")
                        time.sleep(self.backoff_factor ** attempt)
                except Exception as e:
                    logger.warning(f"ChEMBL connection error on attempt {attempt}: {e}")
                    time.sleep(self.backoff_factor ** attempt)

            if not success or not batch:
                break

        # Save immutable raw snapshot
        if activities:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(activities, f, indent=2)
            logger.info(f"Saved {len(activities)} raw ChEMBL records to {cache_file}")

        return activities
