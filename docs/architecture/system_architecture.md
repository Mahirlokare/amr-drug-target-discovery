# End-to-End System Architecture

## Logical Architecture
```
+---------------------------------------------------------------+
|                      STREAMLIT PRESENTATION                   |
|  Overview | Data Explorer | Target Registry | Model Lab        |
|  Candidate Ranking | Candidate Detail | Active Learning | Docs |
+------------------------------+--------------------------------+
                               |
                        Application Layer
                               |
+------------------------------v--------------------------------+
|                      BACKEND / PIPELINE                       |
|  Ingestion (ChEMBL / UniProt / PDB / PubChem)                 |
|  QC Auditor & Replicate Aggregator                             |
|  Bemis-Murcko Split Manager with Leakage Assertions          |
|  Molecular Graph Featurizer (PyG) & ESM-2 Embedding Service   |
|  Multimodal DTI Fusion Network (GNN + ESM-2)                  |
|  Deep Ensemble Uncertainty & Probability Calibrator          |
|  Retrospective Active Learning Simulator                      |
|  AutoDock Vina Docking & In-Silico ADMET Filters              |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                    ARTIFACT & DATA STORE                      |
|  data/raw (API snapshots) | data/processed (Cleaned datasets) |
|  artifacts/embeddings     | artifacts/checkpoints (Weights)   |
|  artifacts/docking        | artifacts/reports (JSON / MD)     |
+---------------------------------------------------------------+
```

## Data Lineage & Entity Relationships
```
Target (P9WGR1 / InhA) 1 ---- * ActivityRecord (IC50 nM) * ---- 1 Compound (SMILES / InChIKey)
Compound 1 ---- * MolecularGraph (PyG Data)
Target 1 ---- * ESM2Embedding (320d tensor)
DatasetVersion 1 ---- * ModelRun (Checkpoints, Seeds, Metrics)
ModelRun 1 ---- * Prediction (Calibrated prob, Uncertainty, OOD flag)
Prediction 1 ---- * DockingRun (Vina affinity kcal/mol, PDBQT pose)
Prediction 1 ---- * ADMETRun (Lipinski, Veber, PAINS alerts)
```
