# Model Card: Multimodal DTI (Molecular GNN + ESM-2)

## 1. Model Details
- **Architecture**: Multimodal Drug-Target Interaction Fusion Network
- **Components**:
  1. **Molecular Graph Encoder**: 3-layer Graph Convolutional Network (GCNConv) with 43-dimensional invariant atom node features, global mean + max readout projection to 64 dimensions.
  2. **Protein Representation**: ESM-2 (Evolutionary Scale Modeling) mean-pooled 320-dimensional sequence embedding with 2-layer linear projection to 64 dimensions.
  3. **Multimodal Fusion Head**: Representation concatenation (128-d) -> LayerNorm -> Dropout(0.2) -> 3-layer MLP -> Logit output.
- **Uncertainty Quantification**: 3-seed Deep Ensemble (`seeds=[42, 101, 777]`) and Monte Carlo Dropout.
- **Probability Calibration**: Platt scaling fitted exclusively on validation fold logits.

## 2. Intended Use & Application Scope
- **Primary Use**: Computational triage and rank-ordering of candidate small molecules for follow-up biochemical in-vitro assay testing against *Mycobacterium tuberculosis* InhA.
- **Applicability Domain**: Organic drug-like molecules within 0.35 Tanimoto similarity to the training chemical space.

## 3. Evaluation Metrics (Held-Out Scaffold Test Split)
- **AUROC**: Measured on zero-overlap Bemis-Murcko scaffolds.
- **PR-AUC**: Evaluated with empirical prevalence baseline.
- **Enrichment Factor @ 1% (EF@1%)**: Fold concentration of confirmed actives in the top 1% ranked pool.
- **Expected Calibration Error (ECE)** & **Brier Score**: Reliability across confidence bins.

## 4. Prohibited Uses & Ethical Guardrails
- **No Clinical Decisions**: This model is strictly an in-silico discovery triage tool. It must never be used for clinical treatment selection, patient prescription, or veterinary use.
- **No Efficacy Guarantees**: A predicted active score does not guarantee in-vivo bacterial clearance, pharmacokinetic bioavailability, or absence of human cytotoxicity.
