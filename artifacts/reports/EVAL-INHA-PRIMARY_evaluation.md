# Model Evaluation & Benchmark Report: TARGET-001

**Report ID**: `EVAL-INHA-PRIMARY`  
**Split Type**: Bemis-Murcko Scaffold Split (Zero Overlap Verified)  
**Test Set Size**: 9 compounds  

## Model Benchmark Comparison

| Model | Type | AUROC | PR-AUC | Brier Score | EF@1% | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| majority_prevalence | Classical Baseline | 0.5 | 0.4444 | 0.2747 | - | - |
| morgan_logistic_regression | Classical Baseline | 0.95 | 0.95 | 0.0881 | - | - |
| morgan_random_forest | Classical Baseline | 1.0 | 1.0 | 0.1143 | - | - |
| morgan_gradient_boosting | Classical Baseline | 0.5 | 0.4444 | 0.25 | - | - |
| Molecular GNN (PyG) | Graph Deep Learning | 0.9 | 0.8875 | 0.1507 | 2.25 | 0.1576 |
| Multimodal GNN + ESM-2 | Multimodal DTI | 0.85 | 0.8042 | 0.1924 | 2.25 | 0.2669 |

## Statistical Rigor & 95% Confidence Intervals

- **AUROC**: 0.85 (95% CI: None)
- **PR-AUC**: 0.8042 (95% CI: None)
- **Enrichment Factor @ 1%**: 2.25× (95% CI: None)
- **Expected Calibration Error (ECE)**: 0.2669
- **Brier Score**: 0.1924

## Leakage Protocol Audit
- Scaffold disjointness check passed: Zero train/test scaffold sharing.
- Calibration and thresholds fitted strictly on validation partition.
