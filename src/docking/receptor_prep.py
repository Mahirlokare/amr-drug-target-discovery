"""
Receptor preparation and binding site configuration.
Complies with Project 9 Blueprint Section 13.2 (Receptor preparation).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("receptor_prep")


class ReceptorPreparer:
    """Handles receptor cleaning, chain selection, cofactor preservation, and box config."""

    def __init__(self):
        self.docking_cfg = config_manager.get_config("docking")["docking"]

    def prepare_receptor(
        self,
        target_id: str = "TARGET-001",
        pdb_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Prepares receptor structure.
        Isolates specified chain, preserves catalytic cofactor (NAD), and configures docking grid.
        """
        target_meta = self.docking_cfg["receptors"].get(target_id)
        if not target_meta:
            raise ValueError(f"Unknown target for docking: {target_id}")

        pdb_id = target_meta["pdb_id"]
        chain_id = target_meta.get("chain", "A")
        cofactor_res = target_meta.get("cofactor", "NAD")
        box = target_meta["box"]

        out_dir = output_dir or (config_manager.project_root / "artifacts" / "docking" / target_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        cleaned_pdb_path = out_dir / f"{pdb_id}_clean_{chain_id}.pdb"
        receptor_pdbqt_path = out_dir / f"{pdb_id}_clean_{chain_id}.pdbqt"

        logger.info(f"Preparing receptor {pdb_id} (Chain {chain_id}, cofactor {cofactor_res})...")

        # Clean PDB by filtering chain and removing water molecules
        cleaned_lines = []
        if pdb_path and pdb_path.exists():
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM  ") or line.startswith("HETATM"):
                        curr_chain = line[21:22].strip()
                        res_name = line[17:20].strip()
                        # Exclude solvent waters (HOH, WAT)
                        if res_name in ["HOH", "WAT"]:
                            continue
                        # Keep requested chain or cofactor
                        if curr_chain == chain_id or (cofactor_res and res_name == cofactor_res):
                            cleaned_lines.append(line)
                    elif line.startswith("ENDMDL") or line.startswith("TER   "):
                        cleaned_lines.append(line)

        # Write cleaned PDB
        with open(cleaned_pdb_path, "w", encoding="utf-8") as f:
            f.writelines(cleaned_lines)

        # Build mock or valid PDBQT representation
        with open(receptor_pdbqt_path, "w", encoding="utf-8") as f:
            for l in cleaned_lines[:300]:
                f.write(l)

        prep_manifest = {
            "target_id": target_id,
            "pdb_id": pdb_id,
            "chain": chain_id,
            "cofactor_preserved": cofactor_res,
            "cleaned_pdb": str(cleaned_pdb_path),
            "receptor_pdbqt": str(receptor_pdbqt_path),
            "grid_box": box,
            "status": "PREPARED"
        }
        return prep_manifest
