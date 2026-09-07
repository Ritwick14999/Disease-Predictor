"""Attribution: why did the model produce *this* ranking?

A shortlist of diagnoses with no reasoning attached is not usable by the person
reading it. Two complementary views are provided:

:func:`global_importance`
    Which symptoms drive the model overall. Read from the estimator when it
    exposes importances, and computed by permutation when it does not, so the
    rule-based baseline and the boosted ensemble can be compared like for like.

:func:`explain_prediction`
    Which of *this patient's* symptoms moved *this* probability. Implemented by
    occlusion -- toggle one reported symptom off and measure how far the
    probability falls -- which is model-agnostic and needs no extra dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

__all__ = ["SymptomContribution", "global_importance", "explain_prediction", "counterfactual_symptoms"]


@dataclass(frozen=True)
class SymptomContribution:
    """How much one reported symptom supports a predicted disease.

    Attributes:
        symptom: Canonical symptom name.
        contribution: Drop in the predicted probability when this symptom is
            removed. Positive means the symptom supports the prediction.
        share: The contribution as a share of all positive contributions, which
            is what a UI should display as "% of the evidence".
    """

    symptom: str
    contribution: float
    share: float


def _unwrap(estimator: BaseEstimator) -> BaseEstimator:
    """Return the final estimator of a pipeline, or the estimator itself."""
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator


def global_importance(
    estimator: BaseEstimator,
    feature_names: Sequence[str],
    *,
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    top_n: int = 20,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank symptoms by overall influence on the model.

    Uses the estimator's native importances when available. Otherwise -- or when
    ``features`` and ``labels`` are supplied -- falls back to permutation
    importance, which measures the accuracy actually lost when a symptom is
    scrambled, and is comparable across model families.

    Returns:
        Frame with ``symptom``, ``importance`` and ``method`` columns, sorted
        by importance, truncated to ``top_n`` rows.
    """
    model = _unwrap(estimator)
    method = "native"
    values: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        # Multi-class linear model: aggregate magnitude across the one-vs-rest rows.
        values = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
        method = "mean_abs_coefficient"

    if values is None:
        if features is None or labels is None:
            raise ValueError(
                "This estimator exposes no native importances; pass features and labels "
                "so permutation importance can be computed."
            )
        result = permutation_importance(
            estimator, features, labels, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        values = np.asarray(result.importances_mean, dtype=float)
        method = "permutation"

    if len(values) != len(feature_names):
        raise ValueError(
            f"Importance vector has {len(values)} entries but there are {len(feature_names)} features."
        )

    frame = pd.DataFrame({"symptom": list(feature_names), "importance": values})
    frame["method"] = method
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def explain_prediction(
    estimator: BaseEstimator,
    vector: np.ndarray,
    feature_names: Sequence[str],
    *,
    class_index: int | None = None,
    top_n: int = 5,
) -> list[SymptomContribution]:
    """Attribute a single prediction to the symptoms the patient reported.

    For every reported symptom the model is re-scored with that symptom switched
    off; the resulting drop in probability is its contribution. Only reported
    symptoms are probed, so cost is linear in the number of symptoms a person
    actually entered rather than in the size of the vocabulary.

    Args:
        estimator: A fitted classifier exposing ``predict_proba``.
        vector: Binary row vector of shape ``(1, n_features)`` or ``(n_features,)``.
        feature_names: Names matching the vector's columns.
        class_index: Which class to explain; defaults to the top prediction.
        top_n: How many contributions to return.

    Raises:
        AttributeError: If the estimator has no ``predict_proba``.
        ValueError: If the vector's width does not match ``feature_names``.
    """
    if not hasattr(estimator, "predict_proba"):
        raise AttributeError("explain_prediction needs an estimator with predict_proba.")

    row = np.asarray(vector).reshape(1, -1)
    if row.shape[1] != len(feature_names):
        raise ValueError(
            f"Vector has {row.shape[1]} features but {len(feature_names)} names were given."
        )

    baseline = estimator.predict_proba(row)[0]
    if class_index is None:
        class_index = int(np.argmax(baseline))
    baseline_probability = float(baseline[class_index])

    active = np.flatnonzero(row[0])
    if active.size == 0:
        return []

    # One batched call instead of one call per symptom.
    occluded = np.repeat(row, active.size, axis=0)
    occluded[np.arange(active.size), active] = 0
    occluded_probabilities = estimator.predict_proba(occluded)[:, class_index]

    contributions = baseline_probability - occluded_probabilities
    positive_total = float(contributions[contributions > 0].sum())

    results = [
        SymptomContribution(
            symptom=str(feature_names[feature_index]),
            contribution=round(float(delta), 6),
            share=round(float(delta) / positive_total, 4) if positive_total > 0 and delta > 0 else 0.0,
        )
        for feature_index, delta in zip(active, contributions, strict=True)
    ]
    results.sort(key=lambda item: item.contribution, reverse=True)
    return results[:top_n]


def counterfactual_symptoms(
    estimator: BaseEstimator,
    vector: np.ndarray,
    feature_names: Sequence[str],
    *,
    class_index: int | None = None,
    top_n: int = 5,
) -> list[SymptomContribution]:
    """Find the *unreported* symptoms that would most change the prediction.

    This is the "what should I ask about next" view: it ranks absent symptoms by
    how much switching them on would raise the current top class. In a triage
    setting that ordering is the follow-up question list.
    """
    if not hasattr(estimator, "predict_proba"):
        raise AttributeError("counterfactual_symptoms needs an estimator with predict_proba.")

    row = np.asarray(vector).reshape(1, -1)
    baseline = estimator.predict_proba(row)[0]
    if class_index is None:
        class_index = int(np.argmax(baseline))
    baseline_probability = float(baseline[class_index])

    inactive = np.flatnonzero(row[0] == 0)
    if inactive.size == 0:
        return []

    candidates = np.repeat(row, inactive.size, axis=0)
    candidates[np.arange(inactive.size), inactive] = 1
    probabilities = estimator.predict_proba(candidates)[:, class_index]
    deltas = probabilities - baseline_probability

    order = np.argsort(deltas)[::-1][:top_n]
    total = float(np.abs(deltas[order]).sum()) or 1.0
    return [
        SymptomContribution(
            symptom=str(feature_names[inactive[i]]),
            contribution=round(float(deltas[i]), 6),
            share=round(float(deltas[i]) / total, 4),
        )
        for i in order
    ]
