"""
Baseline models for drug-target interaction benchmarking.
Complies with Project 9 Blueprint Section 10.1 (Baseline-first strategy).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.common.logging import get_logger

logger = get_logger("baselines")


class PrevalenceBaseline:
    """Predicts empirical class prevalence from training data."""

    def __init__(self):
        self.dummy = DummyClassifier(strategy="prior")
        self.prevalence = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.dummy.fit(X, y)
        self.prevalence = float(np.mean(y)) if len(y) > 0 else 0.5
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.dummy.predict_proba(X)


class ClassicalMLBaselines:
    """Manages training and evaluation of classical cheminformatics baselines."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models: Dict[str, Any] = {
            "majority_prevalence": PrevalenceBaseline(),
            "morgan_logistic_regression": LogisticRegression(
                C=1.0, max_iter=1000, random_state=seed, class_weight="balanced"
            ),
            "morgan_random_forest": RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=seed, class_weight="balanced", n_jobs=-1
            ),
            "morgan_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=100, random_state=seed, class_weight="balanced"
            ),
        }
        self.fitted_models: Dict[str, Any] = {}

    def fit_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """Trains all classical baseline models on the exact training partition."""
        for name, model in self.models.items():
            logger.info(f"Training baseline model: {name} (N={len(y_train)})...")
            model.fit(X_train, y_train)
            self.fitted_models[name] = model

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates a baseline model on the test partition without leakage."""
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} is not trained.")

        model = self.fitted_models[model_name]
        proba = model.predict_proba(X_test)
        # Select positive class probability
        if proba.shape[1] > 1:
            pos_proba = proba[:, 1]
        else:
            pos_proba = proba[:, 0]

        # Calculate metrics
        try:
            auroc = float(roc_auc_score(y_test, pos_proba))
        except Exception:
            auroc = 0.5

        try:
            pr_auc = float(average_precision_score(y_test, pos_proba))
        except Exception:
            pr_auc = float(np.mean(y_test))

        brier = float(brier_score_loss(y_test, pos_proba))

        return {
            "model_name": model_name,
            "AUROC": round(auroc, 4),
            "PR_AUC": round(pr_auc, 4),
            "BrierScore": round(brier, 4),
            "test_samples": len(y_test),
            "active_prevalence": round(float(np.mean(y_test)), 4)
        }

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates all baselines against the test set."""
        results = {}
        for name in self.fitted_models:
            results[name] = self.evaluate_model(name, X_test, y_test)
        return results
