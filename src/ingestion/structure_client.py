"""
RCSB PDB structure client with local caching and metadata inspection.
Complies with Project 9 Blueprint Section 5.1 & 9.1.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import requests

from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("structure_client")


class StructureClient:
    """Retrieves experimental macromolecular 3D structures from RCSB PDB."""

    def __init__(self, cache_dir: Optional[Path] = None):
        cfg = config_manager.get_config("data_sources")
        self.files_url = cfg["rcsb"].get("files_url", "https://files.rcsb.org/download")
        self.cache_dir = cache_dir or (config_manager.project_root / cfg["rcsb"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_pdb_file(self, pdb_id: str) -> Optional[Path]:
        """
        Downloads PDB file from RCSB.
        Returns Path to the local .pdb file.
        """
        clean_id = pdb_id.strip().upper()
        pdb_path = self.cache_dir / f"{clean_id}.pdb"

        if pdb_path.exists() and pdb_path.stat().st_size > 500:
            logger.info(f"Using cached PDB structure: {pdb_path}")
            return pdb_path

        url = f"{self.files_url}/{clean_id}.pdb"
        logger.info(f"Downloading PDB structure {clean_id} from {url}...")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(pdb_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.info(f"Saved PDB structure {clean_id} ({len(resp.text)} chars) to {pdb_path}")
                return pdb_path
            else:
                logger.warning(f"RCSB download failed with status {resp.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch PDB {clean_id}: {e}")
            return None
