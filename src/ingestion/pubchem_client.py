"""
PubChem PUG REST API client with local caching and rate limiting.
Complies with Project 9 Blueprint Section 5.1 & 9.1.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional
import requests

from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("pubchem_client")


class PubChemClient:
    """Queries PubChem PUG REST API for chemical metadata enrichment."""

    def __init__(self, cache_dir: Optional[Path] = None):
        cfg = config_manager.get_config("data_sources")
        self.base_url = cfg["pubchem"]["base_url"]
        self.timeout = cfg["pubchem"].get("timeout_seconds", 20)
        self.rate_delay = cfg["pubchem"].get("rate_limit_delay_seconds", 0.2)
        self.cache_dir = cache_dir or (config_manager.project_root / cfg["pubchem"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_time = 0.0

    def _throttle(self) -> None:
        """Enforces rate limit delay between successive external calls."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_call_time = time.time()

    def get_compound_by_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves compound properties by SMILES using PUG REST API.
        Caches results locally.
        """
        import hashlib
        cache_key = hashlib.md5(smiles.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        self._throttle()
        url = f"{self.base_url}/compound/smiles/{requests.utils.quote(smiles)}/property/MolecularWeight,MolecularFormula,XLogP,IUPACName/JSON"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(props, f, indent=2)
                return props
            else:
                logger.debug(f"PubChem query returned {resp.status_code} for SMILES: {smiles[:30]}")
                return None
        except Exception as e:
            logger.debug(f"PubChem request exception: {e}")
            return None
