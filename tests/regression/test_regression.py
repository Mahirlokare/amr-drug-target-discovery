"""
Regression tests: Run the pipeline and assert minimum scientific performance thresholds.

Thresholds (conservative, appropriate for the seed dataset size):
  - Classical ML (Random Forest):  AUROC >= 0.75
  - Multimodal GNN + ESM-2:        AUROC >= 0.70, EF@1% >= 1.5

These are intentionally lenient — the seed dataset is 57 records.
Tighten thresholds when the full ChEMBL dataset is ingested.
"""
import sys
import json
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

EVAL_REPORT = (
    pathlib.Path(__file__).parents[2]
    / "artifacts"
    / "reports"
    / "EVAL-INHA-PRIMARY_evaluation.json"
)

REQUIRED_AUROC_CLASSICAL = 0.75
REQUIRED_AUROC_MULTIMODAL = 0.70
REQUIRED_EF_MULTIMODAL = 1.5


@pytest.fixture(scope="module")
def eval_report():
    if not EVAL_REPORT.exists():
        pytest.skip(
            "Evaluation report not found — run `python run_pipeline.py` first."
        )
    with open(EVAL_REPORT) as f:
        return json.load(f)


def _find_model(report, name_fragment):
    """Return metrics dict for the first model whose name contains name_fragment."""
    models = report.get("models_comparison", report) if isinstance(report, dict) else report
    for entry in models:
        name = entry.get("Model", entry.get("model", ""))
        if name_fragment.lower() in name.lower():
            # Return normalized dictionary with lowercased keys
            normalized = {}
            for k, v in entry.items():
                norm_key = k.lower().replace("@", "_at_").replace("%", "pct").replace("-", "_").replace(" ", "_")
                normalized[norm_key] = v
            return normalized
    return None


class TestRegressionPerformance:
    """Assert minimum AUROC and EF thresholds on the held-out test partition."""

    def test_random_forest_auroc(self, eval_report):
        rf = _find_model(eval_report, "random_forest")
        if rf is None:
            pytest.skip("Random Forest model not found in evaluation report.")
        auroc = rf.get("auroc", 0.0)
        assert auroc >= REQUIRED_AUROC_CLASSICAL, (
            f"Random Forest AUROC {auroc:.3f} < threshold {REQUIRED_AUROC_CLASSICAL}"
        )

    def test_multimodal_gnn_auroc(self, eval_report):
        mm = _find_model(eval_report, "multimodal")
        if mm is None:
            pytest.skip("Multimodal model not found in evaluation report.")
        auroc = mm.get("auroc", 0.0)
        assert auroc >= REQUIRED_AUROC_MULTIMODAL, (
            f"Multimodal GNN AUROC {auroc:.3f} < threshold {REQUIRED_AUROC_MULTIMODAL}"
        )

    def test_multimodal_ef_at_1pct(self, eval_report):
        mm = _find_model(eval_report, "multimodal")
        if mm is None:
            pytest.skip("Multimodal model not found in evaluation report.")
        ef = mm.get("ef_at_1pct", None)
        if ef is None or ef == "-":
            pytest.skip("EF@1% not computed (test partition too small).")
        assert float(ef) >= REQUIRED_EF_MULTIMODAL, (
            f"Multimodal EF@1% {ef} < threshold {REQUIRED_EF_MULTIMODAL}"
        )

    def test_evaluation_report_structure(self, eval_report):
        models = eval_report.get("models_comparison", eval_report) if isinstance(eval_report, dict) else eval_report
        assert isinstance(models, list)
        assert len(models) >= 2, "Report must contain at least 2 model entries"
        for entry in models:
            assert "Model" in entry or "model" in entry
            assert "AUROC" in entry or "auroc" in entry

    def test_no_model_has_auroc_below_random(self, eval_report):
        """No fitted model should score worse than 0.45 AUROC (sanity check)."""
        models = eval_report.get("models_comparison", eval_report) if isinstance(eval_report, dict) else eval_report
        for entry in models:
            model_name = entry.get("Model", entry.get("model", ""))
            auroc = entry.get("AUROC", entry.get("auroc", 1.0))
            # Prevalence baseline is exempt
            if isinstance(auroc, (int, float)) and "prevalence" not in model_name.lower():
                assert auroc >= 0.45, (
                    f"Model '{model_name}' has AUROC {auroc:.3f} — worse than random!"
                )
