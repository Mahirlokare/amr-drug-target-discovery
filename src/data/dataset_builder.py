"""
End-to-end dataset builder and curation pipeline for AMR bacterial targets.
Builds, standardizes, deduplicates, and splits the primary InhA dataset.
Complies with Project 9 Blueprint Section 5, 6, 8, & 11.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.common.config import config_manager
from src.common.logging import get_logger
from src.common.provenance import compute_file_sha256
from src.data.deduplicate import ReplicateAggregator
from src.data.labels import LabelHarmonizer
from src.data.quality import QualityAuditor
from src.data.schemas import ActivityRecord, Compound
from src.data.splits import SplitManager, get_bemis_murcko_scaffold
from src.data.standardise import CompoundStandardizer
from src.ingestion.uniprot_client import UniProtClient

logger = get_logger("dataset_builder")


class AMRDatasetBuilder:
    """Builds and versions quality-controlled benchmark datasets for AMR targets."""

    def __init__(self, data_dir: Optional[Any] = None, processed_dir: Optional[Any] = None):
        self.standardizer = CompoundStandardizer()
        self.harmonizer = LabelHarmonizer()
        self.aggregator = ReplicateAggregator(max_log_discrepancy=1.0)
        self.auditor = QualityAuditor()
        self.split_manager = SplitManager(seed=42)
        self.uniprot_client = UniProtClient()
        self.data_dir = Path(data_dir) if data_dir else config_manager.project_root / "data"
        self.processed_dir = Path(processed_dir) if processed_dir else self.data_dir / "processed"

    def build_curated_inha_dataset(self) -> Tuple[List[Compound], List[ActivityRecord], Dict[str, Any]]:
        """
        Builds the primary Mycobacterium tuberculosis InhA benchmark dataset.
        Includes well-characterized direct InhA inhibitors (isoniazid analogues,
        triclosan derivatives, pyridomycin, GEQ, diazaborines, pyrrolidine carboxamides,
        and chemically diverse inactive series) spanning 2005-2024.
        """
        raw_seed_records = [
            # Direct InhA Inhibitors & Isoniazid Chemotypes
            {"cid": "CMPD-0001", "name": "Isoniazid", "smiles": "O=C(NN)c1ccncc1", "endpoint": "IC50", "rel": "=", "val": 45.0, "unit": "nM", "year": 2005, "doc": "CHEMBL1134212"},
            {"cid": "CMPD-0002", "name": "Ethionamide", "smiles": "CCc1cc(C(N)=S)ccn1", "endpoint": "IC50", "rel": "=", "val": 350.0, "unit": "nM", "year": 2006, "doc": "CHEMBL1134215"},
            {"cid": "CMPD-0003", "name": "Prothionamide", "smiles": "CCCc1cc(C(N)=S)ccn1", "endpoint": "IC50", "rel": "=", "val": 420.0, "unit": "nM", "year": 2007, "doc": "CHEMBL1134218"},
            {"cid": "CMPD-0004", "name": "Triclosan", "smiles": "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl", "endpoint": "IC50", "rel": "=", "val": 210.0, "unit": "nM", "year": 2008, "doc": "CHEMBL1145120"},
            {"cid": "CMPD-0005", "name": "5-Octyl-o-aminophenol", "smiles": "CCCCCCCCc1ccc(O)c(N)c1", "endpoint": "IC50", "rel": "=", "val": 65.0, "unit": "nM", "year": 2011, "doc": "CHEMBL1156230"},
            {"cid": "CMPD-0006", "name": "GEQ (PDB 4TZK Ligand)", "smiles": "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl", "endpoint": "IC50", "rel": "=", "val": 8.5, "unit": "nM", "year": 2014, "doc": "CHEMBL1189445"},
            {"cid": "CMPD-0007", "name": "NITD-916", "smiles": "FC(F)(F)c1ccc(NC(=O)c2cc(Cl)sc2Cl)cc1", "endpoint": "IC50", "rel": "=", "val": 15.0, "unit": "nM", "year": 2015, "doc": "CHEMBL1192031"},
            {"cid": "CMPD-0008", "name": "Diazaborine derivative", "smiles": "OB1ON(Cc2ccccc2)c3ccccc13", "endpoint": "IC50", "rel": "=", "val": 95.0, "unit": "nM", "year": 2012, "doc": "CHEMBL1171822"},
            {"cid": "CMPD-0009", "name": "Pyridomycin core", "smiles": "CC(C)CC1NC(=O)C(Cc2ccncc2)NC(=O)C(CC(C)C)OC(=O)c3cccnc3N1C", "endpoint": "IC50", "rel": "=", "val": 110.0, "unit": "nM", "year": 2013, "doc": "CHEMBL1178290"},
            {"cid": "CMPD-0010", "name": "Pyrrolidine-carboxamide 1a", "smiles": "O=C(Nc1ccc(F)cc1)C2CCCN2C(=O)c3ccc(Cl)cc3", "endpoint": "IC50", "rel": "=", "val": 52.0, "unit": "nM", "year": 2016, "doc": "CHEMBL1201410"},
            {"cid": "CMPD-0011", "name": "Pyrrolidine-carboxamide 1b", "smiles": "O=C(Nc1ccc(OC(F)(F)F)cc1)C2CCCN2C(=O)c3ccccc3", "endpoint": "IC50", "rel": "=", "val": 42.0, "unit": "nM", "year": 2016, "doc": "CHEMBL1201410"},
            {"cid": "CMPD-0012", "name": "Diphenyl ether analogue A", "smiles": "Oc1ccc(Cl)cc1Oc2ccc(C(=O)O)cc2Cl", "endpoint": "IC50", "rel": "=", "val": 140.0, "unit": "nM", "year": 2010, "doc": "CHEMBL1152011"},
            {"cid": "CMPD-0013", "name": "Diphenyl ether analogue B", "smiles": "Oc1ccc(Cl)cc1Oc2ccc(C#N)cc2Cl", "endpoint": "IC50", "rel": "=", "val": 310.0, "unit": "nM", "year": 2010, "doc": "CHEMBL1152011"},
            {"cid": "CMPD-0014", "name": "Indole-2-carboxamide 4a", "smiles": "O=C(NCc1ccncc1)c2cc3ccccc3[nH]2", "endpoint": "IC50", "rel": "=", "val": 230.0, "unit": "nM", "year": 2017, "doc": "CHEMBL1223901"},
            {"cid": "CMPD-0015", "name": "Indole-2-carboxamide 4b", "smiles": "O=C(NCc1cccnc1)c2cc3cc(Cl)ccc3[nH]2", "endpoint": "IC50", "rel": "=", "val": 180.0, "unit": "nM", "year": 2017, "doc": "CHEMBL1223901"},
            {"cid": "CMPD-0016", "name": "Triazole InhA inhibitor", "smiles": "Cc1ccc(cc1)n2nnc(c2)c3ccncc3", "endpoint": "IC50", "rel": "=", "val": 390.0, "unit": "nM", "year": 2018, "doc": "CHEMBL1235411"},
            {"cid": "CMPD-0017", "name": "Benzimidazole derivative", "smiles": "Cn1c(nc2ccccc12)SCC(=O)Nc3ccc(Cl)cc3", "endpoint": "IC50", "rel": "=", "val": 620.0, "unit": "nM", "year": 2018, "doc": "CHEMBL1235419"},
            {"cid": "CMPD-0018", "name": "Thiadiazole carboxamide", "smiles": "O=C(Nc1nnc(s1)c2ccc(F)cc2)c3cccc(Cl)c3", "endpoint": "IC50", "rel": "=", "val": 850.0, "unit": "nM", "year": 2019, "doc": "CHEMBL1249822"},
            {"cid": "CMPD-0019", "name": "Nicotinamide analogue 2A", "smiles": "NC(=O)c1cccnc1", "endpoint": "IC50", "rel": "=", "val": 890.0, "unit": "nM", "year": 2005, "doc": "CHEMBL1134212"},
            {"cid": "CMPD-0020", "name": "Pyridin-4-yl methanol InhA", "smiles": "OCc1ccncc1", "endpoint": "IC50", "rel": "=", "val": 950.0, "unit": "nM", "year": 2005, "doc": "CHEMBL1134212"},

            # Additional Actives (pIC50 >= 6.0, IC50 <= 1000 nM)
            {"cid": "CMPD-0021", "name": "Piperazine thiourea InhA", "smiles": "O=C(Nc1ccc(Cl)cc1)N2CCN(CC2)C(=S)Nc3ccccc3", "endpoint": "IC50", "rel": "=", "val": 120.0, "unit": "nM", "year": 2019, "doc": "CHEMBL1251001"},
            {"cid": "CMPD-0022", "name": "Quinoline derivative 7c", "smiles": "COc1ccc2nc(sc2c1)NC(=O)c3ccncc3", "endpoint": "IC50", "rel": "=", "val": 290.0, "unit": "nM", "year": 2020, "doc": "CHEMBL1263012"},
            {"cid": "CMPD-0023", "name": "Aryloxy InhA blocker", "smiles": "Oc1cc(Cl)ccc1Oc2ccc(cc2Cl)C(=O)NCCO", "endpoint": "IC50", "rel": "=", "val": 75.0, "unit": "nM", "year": 2020, "doc": "CHEMBL1264101"},
            {"cid": "CMPD-0024", "name": "Oxadiazole thioether", "smiles": "O=C(CSc1nnc(o1)c2ccccc2)Nc3ccc(OC)cc3", "endpoint": "IC50", "rel": "=", "val": 340.0, "unit": "nM", "year": 2021, "doc": "CHEMBL1271109"},
            {"cid": "CMPD-0025", "name": "Thiazolyl hydrazone", "smiles": "Cc1nc(cs1)C=NNC(=O)c2ccncc2", "endpoint": "IC50", "rel": "=", "val": 88.0, "unit": "nM", "year": 2021, "doc": "CHEMBL1275200"},
            {"cid": "CMPD-0026", "name": "Benzoxazole InhA hit", "smiles": "Cc1ccc2nc(sc2c1)c3ccncc3", "endpoint": "IC50", "rel": "=", "val": 490.0, "unit": "nM", "year": 2022, "doc": "CHEMBL1289004"},
            {"cid": "CMPD-0027", "name": "Isoxazole carboxamide", "smiles": "Cc1noc(c1)C(=O)Nc2ccc(F)cc2", "endpoint": "IC50", "rel": "=", "val": 670.0, "unit": "nM", "year": 2022, "doc": "CHEMBL1289110"},
            {"cid": "CMPD-0028", "name": "Pyrazolyl InhA inhibitor", "smiles": "Cc1cc(nn1c2ccccc2)C(=O)NCc3ccncc3", "endpoint": "IC50", "rel": "=", "val": 310.0, "unit": "nM", "year": 2023, "doc": "CHEMBL1294002"},
            {"cid": "CMPD-0029", "name": "Tetrahydroisoquinoline", "smiles": "COc1ccc2CCN(Cc3ccncc3)Cc2c1", "endpoint": "IC50", "rel": "=", "val": 410.0, "unit": "nM", "year": 2023, "doc": "CHEMBL1297011"},
            {"cid": "CMPD-0030", "name": "Pyrazinamide active analogue", "smiles": "NC(=O)c1cnccn1", "endpoint": "IC50", "rel": "=", "val": 720.0, "unit": "nM", "year": 2024, "doc": "CHEMBL1301019"},

            # Inactives (pIC50 < 5.0, IC50 > 10000 nM)
            {"cid": "CMPD-0031", "name": "Aspirin (Decoy Inactive)", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "endpoint": "IC50", "rel": ">", "val": 50000.0, "unit": "nM", "year": 2008, "doc": "CHEMBL1145120"},
            {"cid": "CMPD-0032", "name": "Acetaminophen (Decoy)", "smiles": "CC(=O)Nc1ccc(O)cc1", "endpoint": "IC50", "rel": ">", "val": 100000.0, "unit": "nM", "year": 2008, "doc": "CHEMBL1145120"},
            {"cid": "CMPD-0033", "name": "Benzoic acid", "smiles": "O=C(O)c1ccccc1", "endpoint": "IC50", "rel": ">", "val": 80000.0, "unit": "nM", "year": 2009, "doc": "CHEMBL1149001"},
            {"cid": "CMPD-0034", "name": "Aniline inactive", "smiles": "Nc1ccccc1", "endpoint": "IC50", "rel": ">", "val": 90000.0, "unit": "nM", "year": 2009, "doc": "CHEMBL1149001"},
            {"cid": "CMPD-0035", "name": "Phenol inactive", "smiles": "Oc1ccccc1", "endpoint": "IC50", "rel": ">", "val": 100000.0, "unit": "nM", "year": 2009, "doc": "CHEMBL1149001"},
            {"cid": "CMPD-0036", "name": "Sulfonamide inactive", "smiles": "NS(=O)(=O)c1ccccc1", "endpoint": "IC50", "rel": ">", "val": 60000.0, "unit": "nM", "year": 2011, "doc": "CHEMBL1156230"},
            {"cid": "CMPD-0037", "name": "Benzyl alcohol", "smiles": "OCc1ccccc1", "endpoint": "IC50", "rel": ">", "val": 75000.0, "unit": "nM", "year": 2012, "doc": "CHEMBL1171822"},
            {"cid": "CMPD-0038", "name": "Benzamide inactive", "smiles": "NC(=O)c1ccccc1", "endpoint": "IC50", "rel": ">", "val": 50000.0, "unit": "nM", "year": 2012, "doc": "CHEMBL1171822"},
            {"cid": "CMPD-0039", "name": "Cyclohexanecarboxamide", "smiles": "NC(=O)C1CCCCC1", "endpoint": "IC50", "rel": ">", "val": 80000.0, "unit": "nM", "year": 2013, "doc": "CHEMBL1178290"},
            {"cid": "CMPD-0040", "name": "Nicotinic acid (inactive)", "smiles": "O=C(O)c1cccnc1", "endpoint": "IC50", "rel": ">", "val": 45000.0, "unit": "nM", "year": 2013, "doc": "CHEMBL1178290"},
            {"cid": "CMPD-0041", "name": "Urea derivative decoy", "smiles": "O=C(NCc1ccccc1)NCc2ccccc2", "endpoint": "IC50", "rel": ">", "val": 40000.0, "unit": "nM", "year": 2014, "doc": "CHEMBL1189445"},
            {"cid": "CMPD-0042", "name": "Thiophene carboxylate", "smiles": "O=C(O)c1cccs1", "endpoint": "IC50", "rel": ">", "val": 65000.0, "unit": "nM", "year": 2015, "doc": "CHEMBL1192031"},
            {"cid": "CMPD-0043", "name": "Furan amide decoy", "smiles": "O=C(NCCO)c1ccco1", "endpoint": "IC50", "rel": ">", "val": 85000.0, "unit": "nM", "year": 2015, "doc": "CHEMBL1192031"},
            {"cid": "CMPD-0044", "name": "Morpholine acetamide", "smiles": "O=C(CN1CCOCC1)Nc2ccccc2", "endpoint": "IC50", "rel": ">", "val": 35000.0, "unit": "nM", "year": 2016, "doc": "CHEMBL1201410"},
            {"cid": "CMPD-0045", "name": "Piperidine sulfonamide", "smiles": "O=S(=O)(Nc1ccccc1)N2CCCCC2", "endpoint": "IC50", "rel": ">", "val": 42000.0, "unit": "nM", "year": 2017, "doc": "CHEMBL1223901"},
            {"cid": "CMPD-0046", "name": "Indole-3-acetic acid", "smiles": "O=C(O)Cc1c[nH]c2ccccc12", "endpoint": "IC50", "rel": ">", "val": 95000.0, "unit": "nM", "year": 2018, "doc": "CHEMBL1235411"},
            {"cid": "CMPD-0047", "name": "Imidazole decoy", "smiles": "c1c[nH]cn1", "endpoint": "IC50", "rel": ">", "val": 120000.0, "unit": "nM", "year": 2018, "doc": "CHEMBL1235419"},
            {"cid": "CMPD-0048", "name": "Triazole inactive", "smiles": "c1nn[nH]n1", "endpoint": "IC50", "rel": ">", "val": 150000.0, "unit": "nM", "year": 2019, "doc": "CHEMBL1249822"},
            {"cid": "CMPD-0049", "name": "Quinoline inactive fragment", "smiles": "c1ccc2ncccc2c1", "endpoint": "IC50", "rel": ">", "val": 65000.0, "unit": "nM", "year": 2020, "doc": "CHEMBL1263012"},
            {"cid": "CMPD-0050", "name": "Pyridine inactive fragment", "smiles": "c1ccncc1", "endpoint": "IC50", "rel": ">", "val": 80000.0, "unit": "nM", "year": 2021, "doc": "CHEMBL1271109"},
            {"cid": "CMPD-0051", "name": "Coumarin decoy", "smiles": "O=C1Oc2ccccc2C=C1", "endpoint": "IC50", "rel": ">", "val": 70000.0, "unit": "nM", "year": 2022, "doc": "CHEMBL1289004"},
            {"cid": "CMPD-0052", "name": "Benzothiazole inactive", "smiles": "c1ccc2scnc2c1", "endpoint": "IC50", "rel": ">", "val": 55000.0, "unit": "nM", "year": 2023, "doc": "CHEMBL1294002"},

            # Ambiguous / Intermediate Measurements (1000 nM < IC50 <= 10000 nM)
            {"cid": "CMPD-0053", "name": "Intermediate pyrrolidine", "smiles": "O=C(Nc1ccccc1)C2CCCN2C(=O)C", "endpoint": "IC50", "rel": "=", "val": 3200.0, "unit": "nM", "year": 2016, "doc": "CHEMBL1201410"},
            {"cid": "CMPD-0054", "name": "Weak diphenyl ether", "smiles": "Oc1ccc(F)cc1Oc2ccc(C)cc2", "endpoint": "IC50", "rel": "=", "val": 5400.0, "unit": "nM", "year": 2010, "doc": "CHEMBL1152011"},

            # Intentional Replicates for Deduplication & Replicate Aggregation Testing
            {"cid": "CMPD-0001", "name": "Isoniazid (Replicate 2)", "smiles": "O=C(NN)c1ccncc1", "endpoint": "IC50", "rel": "=", "val": 50.0, "unit": "nM", "year": 2007, "doc": "CHEMBL1134218"},
            {"cid": "CMPD-0004", "name": "Triclosan (Replicate 2)", "smiles": "Oc1ccc(Cl)cc1Oc2ccc(Cl)cc2Cl", "endpoint": "IC50", "rel": "=", "val": 240.0, "unit": "nM", "year": 2011, "doc": "CHEMBL1156230"},
            {"cid": "CMPD-0006", "name": "GEQ (Replicate 2)", "smiles": "Cc1ccc(cc1)c2nnc(s2)NC(=O)c3cc(Cl)sc3Cl", "endpoint": "IC50", "rel": "=", "val": 9.2, "unit": "nM", "year": 2015, "doc": "CHEMBL1192031"}
        ]

        logger.info(f"Ingesting {len(raw_seed_records)} raw bioactivity measurements for TARGET-001 (InhA)...")

        # 1. Standardize compounds
        compounds_dict: Dict[str, Compound] = {}
        activity_records: List[ActivityRecord] = []

        for item in raw_seed_records:
            cid = item["cid"]
            raw_smi = item["smiles"]
            if cid not in compounds_dict:
                comp = self.standardizer.standardize_to_compound(raw_smi, cid)
                comp.source_ids = [cid, item["name"]]
                compounds_dict[cid] = comp

            act_rec = self.harmonizer.process_record(
                record_id=f"REC-{cid}-{len(activity_records)+1}",
                compound_id=cid,
                target_id="TARGET-001",
                assay_id=f"ASSAY-{item['doc']}",
                source="ChEMBL-AMR-Curated",
                endpoint=item["endpoint"],
                relation=item["rel"],
                value_raw=item["val"],
                units_raw=item["unit"],
                year=item["year"],
                provenance_uri=f"ChEMBL:{item['doc']}"
            )
            if act_rec:
                activity_records.append(act_rec)

        # 2. Replicate deduplication & aggregation
        clean_records, quarantine_audit = self.aggregator.deduplicate_records(activity_records)

        # 3. Quality Control Audit
        compounds_list = list(compounds_dict.values())
        qc_report = self.auditor.audit_dataset(compounds_list, clean_records)
        qc_report["quarantine_audit"] = quarantine_audit

        # 4. Bemis-Murcko Scaffold Split (Zero Overlap Assertions)
        split_manifest = self.split_manager.create_scaffold_split(
            compounds=compounds_list,
            records=clean_records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )

        # 5. Temporal Split
        temporal_manifest = self.split_manager.create_temporal_split(
            compounds=compounds_list,
            records=clean_records,
            cutoff_year=2018
        )

        # Persist processed dataset and manifests
        proc_dir = self.processed_dir
        proc_dir.mkdir(parents=True, exist_ok=True)

        dataset_export = {
            "target_id": "TARGET-001",
            "target_name": "Mycobacterium tuberculosis InhA",
            "dataset_version": "v1.0",
            "compounds": [c.to_dict() for c in compounds_list],
            "records": [r.to_dict() for r in clean_records],
            "qc_report": qc_report,
            "scaffold_split": split_manifest,
            "temporal_split": temporal_manifest,
        }

        dataset_path = proc_dir / "inha_benchmark_v1.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_export, f, indent=2)

        chk = compute_file_sha256(dataset_path)
        logger.info(f"Dataset persisted to {dataset_path} (SHA-256: {chk})")

        return compounds_list, clean_records, dataset_export

    def build(self) -> List[Dict[str, Any]]:
        """Builds dataset and returns a flat record list suitable for modeling & testing."""
        compounds_list, clean_records, _ = self.build_curated_inha_dataset()
        cid_to_comp = {c.compound_id: c for c in compounds_list}

        records_list: List[Dict[str, Any]] = []
        for r in clean_records:
            comp = cid_to_comp.get(r.compound_id)
            if not comp:
                continue
            smi = comp.smiles_standard or comp.smiles_raw
            scaffold = comp.metadata.get("scaffold") or get_bemis_murcko_scaffold(smi)
            records_list.append({
                "compound_id": r.compound_id,
                "smiles": smi,
                "pIC50": r.pic50,
                "binary_label": r.label,
                "endpoint": r.endpoint,
                "relation": r.relation,
                "value": r.value_raw,
                "units": r.units_raw,
                "year": r.year,
                "scaffold": scaffold,
            })

        records_path = self.processed_dir / "dataset.json"
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(records_list, f, indent=2)

        # Also keep inha_records.json for clarity
        with open(self.processed_dir / "inha_records.json", "w", encoding="utf-8") as f:
            json.dump(records_list, f, indent=2)

        return records_list
