# Scientific Validation & Leakage Prevention Protocol

## 1. Research Gap & Motivation
Standard machine learning benchmarks in drug discovery frequently rely on uniform random train/test splits. In small-molecule screening, random splitting allows close structural analogues (and identical Bemis-Murcko core scaffolds) to appear simultaneously in training and test sets, severely inflating apparent generalisation performance. Models trained under random splits often collapse when evaluated prospectively against novel chemical series.

## 2. Scaffold Partitioning Protocol
1. **Scaffold Extraction**: Standardised SMILES are converted to Bemis-Murcko molecular frameworks using RDKit `MurckoScaffold.MurckoScaffoldSmiles`.
2. **Cluster Grouping**: All molecules sharing an identical core scaffold are assigned strictly to a single partition (Train, Validation, or Test).
3. **Partition Budget**: ~70% Train, ~15% Validation, ~15% Test.
4. **Mandatory Leakage Assertions**:
   ```python
   assert train_scaffolds.isdisjoint(val_scaffolds), "Overlap between train and val scaffolds"
   assert train_scaffolds.isdisjoint(test_scaffolds), "Overlap between train and test scaffolds"
   assert val_scaffolds.isdisjoint(test_scaffolds), "Overlap between val and test scaffolds"
   assert train_compounds.isdisjoint(test_compounds), "Overlap between train and test compounds"
   ```

## 3. Temporal Evaluation Protocol
Where publication or deposition dates exist in source records:
- Training cutoff: Records deposited on or before cutoff year (e.g., 2018).
- Evaluation set: Records published after cutoff year (e.g., > 2018).
- Preprocessing and feature selection are strictly fit on historical training data.

## 4. Probability Calibration & Uncertainty Reporting
- Raw neural network logits are passed through validation-fitted Platt scaling.
- Reliability diagrams bin test probabilities into 10 intervals to report empirical calibration.
- Expected Calibration Error (ECE) and Brier Score are tracked alongside AUROC.
- Epistemic uncertainty standard deviation is reported for every scored molecule.
