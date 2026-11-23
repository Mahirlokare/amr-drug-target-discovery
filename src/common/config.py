"""
Configuration management service.
Loads, validates, and provides typed access to YAML configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """Manages project configuration paths and settings."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Default to repo root (2 levels up from src/common)
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.configs_dir = self.project_root / "configs"
        self.data_dir = self.project_root / "data"
        self.artifacts_dir = self.project_root / "artifacts"
        self._configs: Dict[str, Dict[str, Any]] = {}

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Loads and caches a configuration file from configs/<config_name>.yaml."""
        if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
            config_name = f"{config_name}.yaml"

        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.configs_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Environment variable overrides
        self._apply_env_overrides(cfg)

        self._configs[config_name] = cfg
        return cfg

    def _apply_env_overrides(self, cfg: Dict[str, Any]) -> None:
        """Applies environment variable overrides if present."""
        if "chembl" in cfg and "CHEMBL_BASE_URL" in os.environ:
            cfg["chembl"]["base_url"] = os.environ["CHEMBL_BASE_URL"]
        if "pubchem" in cfg and "PUBCHEM_BASE_URL" in os.environ:
            cfg["pubchem"]["base_url"] = os.environ["PUBCHEM_BASE_URL"]
        if "uniprot" in cfg and "UNIPROT_BASE_URL" in os.environ:
            cfg["uniprot"]["base_url"] = os.environ["UNIPROT_BASE_URL"]
        if "vina" in cfg and "VINA_BINARY_PATH" in os.environ:
            cfg["vina"]["binary_path"] = os.environ["VINA_BINARY_PATH"]

    def reload(self) -> None:
        """Clears cached configurations."""
        self._configs.clear()


# Default singleton instance
config_manager = ConfigManager()
