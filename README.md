# Antimicrobial-Resistance (AMR) Drug Target Discovery
### Molecular GNNs and Protein Language Models under Scaffold-Aware Validation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.8-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.09-green.svg)](https://www.rdkit.org/)
[![CI Pipeline](https://github.com/Mahirlokare/amr-drug-target-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahirlokare/amr-drug-target-discovery/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg)](https://amr-drug-target-discovery.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> 🚀 **Live Cloud Application**: [https://amr-drug-target-discovery.onrender.com](https://amr-drug-target-discovery.onrender.com)  
> 🏥 **Healthcheck Endpoint**: [https://amr-drug-target-discovery.onrender.com/_stcore/health](https://amr-drug-target-discovery.onrender.com/_stcore/health)

An end-to-end reproducible computational research platform designed to prioritize small-molecule therapeutics against high-impact bacterial targets relevant to antimicrobial resistance (AMR), specifically focusing on **_Mycobacterium tuberculosis_ Enoyl-[acyl-carrier-protein] reductase [NADH] (InhA)**.

---

## 🎯 Primary Research Question
> *Can a molecular Graph Neural Network combined with ESM-2 protein-language-model embeddings prioritise potentially active small molecules against bacterial AMR targets more reliably than standard baselines under leakage-resistant, scaffold-aware evaluation?*

---

## 🧬 Scientific Focus & Target Profile
- **Pathogen**: *Mycobacterium tuberculosis* (strain ATCC 25618 / H37Rv)
- **Target Protein**: InhA (Enoyl-ACP reductase [NADH], locus tag `Rv1484`)
- **UniProt Accession**: [`P9WGR1`](https://www.uniprot.org/uniprotkb/P9WGR1)
- **ChEMBL Target**: [`CHEMBL2014`](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL2014/)
- **Primary PDB Structure**: [`4TZK`](https://www.rcsb.org/structure/4TZK) (Chain A, with cofactor NAD+)
- **Biological Role**: Essential enzyme in the Type II fatty acid synthesis (FAS-II) pathway responsible for mycolic acid envelope assembly. Clinically validated target of isoniazid (INH) and direct InhA inhibitors (such as triclosan analogues, GEQ, NITD-916, and pyridomycin). Target of key mutations conferring clinical drug resistance.

---

## 🔬 Core Innovations & Methodological Rigor

1. **Strict Data Hygiene & Label Policy**:
   - Single standardized endpoint: direct biochemical **IC50 (nM)**.
   - Active: `IC50 <= 1000 nM` (`pIC50 >= 6.0`).
   - Inactive: `IC50 > 10000 nM` (`pIC50 < 5.0`).
   - Intermediate / Ambiguous: `1000 nM < IC50 <= 10000 nM` are quarantined, not silently labeled as negative!
   - Replicate aggregation via geometric mean; conflicting measurements (>1.0 log span) are quarantined.

2. **Zero-Leakage Bemis-Murcko Scaffold Partitioning**:
   - Automated assertions verify zero scaffold overlap between Train, Validation, and Test partitions:
     ```python
     assert train_scaffolds.isdisjoint(test_scaffolds)
     assert train_compounds.isdisjoint(test_compounds)
     ```

3. **Multimodal Deep Learning Architecture**:
   - **Molecular Graph**: PyG GCN with 43-dimensional invariant atom node features.
   - **Protein Embedding**: ESM-2 (`esm2_t6_8M_UR50D`) 320-dimensional mean-pooled representation.
   - **Interaction Fusion**: Multimodal concatenation + LayerNorm + MLP interaction head.

4. **Honest Uncertainty & Calibration**:
   - 3-seed Deep Ensemble (`seeds=[42, 101, 777]`) and Monte Carlo Dropout to quantify epistemic uncertainty.
   - Platt probability calibration fitted strictly on validation logits.
   - Applicability domain warning based on nearest-neighbor Tanimoto similarity to training chemical space.

5. **Multi-Stream Computational Triage**:
   - **AutoDock Vina Docking**: Standardized grid search box on InhA substrate pocket near Tyr158 with redocking control (RMSD < 2.0 Å).
   - **In-Silico ADMET**: Lipinski Rule of 5, Veber oral bioavailability, QED, Synthetic Accessibility, and 9 PAINS substructure alert checks.

6. **Retrospective Active Learning Simulation**:
   - Compares Uniform Random, Greedy Activity, Uncertainty Sampling, and Balanced Multi-Objective (Activity + Uncertainty + MaxMin Tanimoto Diversity) policies over sequential screening rounds.

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```powershell
git clone https://github.com/example/amr-drug-discovery.git
cd amr-drug-discovery
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
Run data ingestion, quality audits, scaffold splitting, baseline comparisons, GNN + ESM-2 training, active learning simulation, and docking triage with one command:
```powershell
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
Launch the multi-page Streamlit application:
```powershell
streamlit run src/app/streamlit_app.py
```

---

## 📊 Benchmark Evaluation Summary

| Model | Architecture | Test AUROC | Test PR-AUC | EF@1% | Brier Score | ECE |
|---|---|---|---|---|---|---|
| Majority Baseline | Class Prevalence | 0.50 | 0.44 | - | 0.2747 | - |
| Logistic Regression | Morgan Fingerprints | 0.95 | 0.95 | - | 0.0881 | - |
| Random Forest | Morgan + Descriptors | 1.00 | 1.00 | - | 0.1143 | - |
| Hist Gradient Boosting | Morgan + Descriptors | 0.50 | 0.44 | - | 0.2500 | - |
| Molecular GNN (PyG) | Graph Convolution (GCN) | 0.90 | 0.89 | 2.25× | 0.1507 | 0.158 |
| **Multimodal GNN + ESM-2** | **PyG GCN + ESM-2 Fusion** | **0.85** | **0.80** | **2.25×** | **0.1924** | **0.267** |

*All models evaluated under strictly identical, frozen Bemis-Murcko scaffold test partitions with zero data leakage.*

---

---

## 🌐 Production Deployment & Hosting (Render & Docker)

The application is fully architected and configured for cloud hosting on **Render** (as a unified Streamlit Web Service paired with persistent storage and optional PostgreSQL).

### 1. Render Blueprint (`render.yaml`)
Deploy instantly using Render's Infrastructure-as-Code Blueprint:
1. Connect your GitHub repository to [Render Dashboard](https://dashboard.render.com).
2. Choose **New > Blueprint** and select `render.yaml`.
3. Render automatically provisions:
   - **Streamlit Web Service**: `amr-drug-target-discovery` (Python 3.11 starter instance)
   - **Persistent PostgreSQL Database**: `amr-discovery-db`
   - **Persistent Disk Volume**: 1GB mounted at `/var/data` (for SQLite/pose storage)
   - **Automated Healthcheck**: Ping endpoint `/_stcore/health`

### 2. Docker Container Deployment
Run with Docker locally or on any container platform:
```bash
# Build the production container
docker build -t amr-drug-target-discovery:latest .

# Run the container
docker run -d -p 8501:8501 --name amr-discovery amr-drug-target-discovery:latest

# Verify health status
curl -f http://localhost:8501/_stcore/health
```

### 3. Database Persistence & Restart Safety
- **Dual SQLite / PostgreSQL Support**: Configured via `src/storage/db.py`.
- **Local / Ephemeral Avoidance**: When running in production, state, candidates, docking poses, and predictions persist to `/var/data/amr_discovery.db` (or PostgreSQL if `DATABASE_URL` is configured). Data survives restarts and redeployments.
- **Automated Schema Initialization**: All 7 core entity tables (`targets`, `compounds`, `activity_records`, `model_runs`, `predictions`, `docking_runs`, `admet_runs`) are created on boot with zero manual migration overhead.

---

## 🔄 GitHub Actions CI/CD Pipeline

Continuous Integration runs on every `push` and `pull_request` to `main`:
1. **Configuration Validation**: Asserts all YAML configs in `configs/` are well-formed.
2. **Code Safety & Lint**: Scans for syntax errors and unsafe constructs.
3. **Unit Tests**: Full suite for standardisation, graphs, calibration, and database CRUD.
4. **Leakage & Scientific Integrity**: Hard assertions verify zero scaffold overlap.
5. **Regression Benchmarks**: Validates AUROC and Enrichment Factor against stored reports.
6. **Container Build Check**: Tests Docker build and healthcheck response.

---

## 📁 Repository Structure

```
amr-drug-discovery/
├── README.md
├── pyproject.toml
├── run_pipeline.py
├── configs/
│   ├── data_sources.yaml
│   ├── target_registry.yaml
│   ├── labels.yaml
│   ├── features.yaml
│   ├── model_gnn.yaml
│   ├── evaluation.yaml
│   └── docking.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── inha_benchmark_v1.json
├── artifacts/
│   ├── checkpoints/
│   ├── embeddings/
│   ├── docking/
│   └── reports/
├── src/
│   ├── common/
│   ├── ingestion/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── active_learning/
│   ├── explainability/
│   ├── docking/
│   ├── admet/
│   ├── evaluation/
│   └── app/
│       ├── streamlit_app.py
│       └── pages/
│           ├── 1_Data_Explorer.py
│           ├── 2_Target_Registry.py
│           ├── 3_Model_Lab.py
│           ├── 4_Candidate_Ranking.py
│           ├── 5_Candidate_Detail.py
│           ├── 6_Active_Learning.py
│           └── 7_Reports.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   └── regression/
└── docs/
    ├── dataset_card.md
    ├── model_card.md
    ├── validation_protocol.md
    └── architecture/
```

---

## ⚖️ Scientific Governance & Responsible Research
**Guiding Principle**: Build a scientifically careful system that can explain what it knows, what it does not know, and why a candidate was prioritised.
- Predicted affinities, in-silico ADMET scores, and model classifications are computational evidence and **do not constitute proof of antimicrobial efficacy, bacterial clearance, or safety**.
- In accordance with biological safety protocols, this software provides no pathogen-handling or wet-lab experimental synthesis protocols.
