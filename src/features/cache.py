"""
Feature set persistence and checksum management.
Complies with Project 9 Blueprint Section 8.6 & 14.1.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.common.logging import get_logger
from src.data.schemas import FeatureSet

logger = get_logger("feature_cache")


class FeatureCache:
    """Stores and loads versioned molecular features and protein embeddings."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_numpy(self, name: str, array: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Saves a numpy array and corresponding metadata manifest."""
        file_path = self.cache_dir / f"{name}.npy"
        meta_path = self.cache_dir / f"{name}_meta.json"

        np.save(file_path, array)
        chk = hashlib.sha256(array.tobytes()).hexdigest()

        meta = {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "checksum_sha256": chk,
            "extra": metadata or {}
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return file_path

    def load_numpy(self, name: str) -> Optional[np.ndarray]:
        """Loads a cached numpy array if available."""
        file_path = self.cache_dir / f"{name}.npy"
        if file_path.exists():
            return np.load(file_path)
        return None
