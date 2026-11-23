# Dataset Card: Mycobacterium tuberculosis InhA Bioactivity Benchmark

## 1. Summary and Scope
- **Dataset Version**: `v1.0.0`
- **Target Organism**: *Mycobacterium tuberculosis* (strain ATCC 25618 / H37Rv)
- **NCBI Taxonomy ID**: `83332`
- **Target Protein**: Enoyl-[acyl-carrier-protein] reductase [NADH] (InhA, Rv1484)
- **UniProt Accession**: `P9WGR1` (Reviewed / Swiss-Prot)
- **ChEMBL Target ID**: `CHEMBL2014`
- **Primary Assay Endpoint**: Direct biochemical enzyme inhibition IC50 (nanomolar, nM)

## 2. Biological Relevance
InhA is an essential enoyl-ACP reductase involved in the Type II fatty acid synthesis (FAS-II) pathway responsible for the biosynthesis of mycolic acids—the primary lipid component of the *Mycobacterium tuberculosis* protective cell envelope. InhA is the validated primary target of the first-line antitubercular prodrug isoniazid (INH) and direct InhA inhibitors (such as triclosan derivatives, GEQ, NITD-916, and pyridomycin). Drug resistance mutations in InhA or its promoter (e.g. S94A, I194T, and promoter -15C>T) represent a major global public health challenge in multidrug-resistant tuberculosis (MDR-TB).

## 3. Label Definition & Harmonization Policy
- **Active (Class 1)**: `IC50 <= 1000 nM` (`pIC50 >= 6.0`) with `=` or `<` relation.
- **Inactive (Class 0)**: `IC50 > 10000 nM` (`pIC50 < 5.0`) with `=` or `>` relation.
- **Intermediate / Quarantined**: `1000 nM < IC50 <= 10000 nM` (`5.0 <= pIC50 < 6.0`). These compounds are flagged and separated to prevent label noise and silent negative bias.
- **Replicate Policy**: Replicates within 1.0 log10 span are aggregated by geometric mean of IC50; conflicting measurements exceeding 1.0 log10 span are quarantined into an audit table.

## 4. Leakage-Controlled Partitioning
- **Partitioning Algorithm**: Bemis-Murcko scaffold clustering.
- **Partition Ratios**: 70% Train, 15% Validation, 15% Test.
- **Automated Assertions**:
  ```python
  assert train_scaffold_ids.isdisjoint(val_scaffold_ids)
  assert train_scaffold_ids.isdisjoint(test_scaffold_ids)
  assert val_scaffold_ids.isdisjoint(test_scaffold_ids)
  assert train_compound_ids.isdisjoint(test_compound_ids)
  ```

## 5. Limitations
1. In-vitro enzyme inhibition does not model mycobacterial outer membrane permeability or active efflux pump kinetics.
2. Assay protocols across different literature sources may vary in buffer conditions and NADH cofactor concentrations.
