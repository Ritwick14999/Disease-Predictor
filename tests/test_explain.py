"""Tests for global importance and per-prediction attribution."""

from __future__ import annotations

import numpy as np
import pytest

from disease_predictor.explain import (
    counterfactual_symptoms,
    explain_prediction,
    global_importance,
)
from disease_predictor.models import build_pipeline


@pytest.fixture(scope="module")
def separable() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Two classes with disjoint, unambiguous symptom signatures."""
    features = np.array([[1, 1, 0, 0]] * 20 + [[0, 0, 1, 1]] * 20)
    labels = np.array([0] * 20 + [1] * 20)
    names = ["fever", "cough", "rash", "itch"]
    return features, labels, names


@pytest.mark.parametrize("model_name", ["logreg", "tree", "random_forest"])
def test_global_importance_works_across_model_families(model_name, encoded, dataset) -> None:
    features, labels, _ = encoded
    model = build_pipeline(model_name).fit(features, labels)
    table = global_importance(model, dataset.feature_names, top_n=5)
    assert len(table) == 5
    assert list(table.columns) == ["symptom", "importance", "method"]
    assert table["importance"].is_monotonic_decreasing


def test_global_importance_falls_back_to_permutation(encoded, dataset) -> None:
    features, labels, _ = encoded
    model = build_pipeline("rules").fit(features, labels)
    table = global_importance(
        model, dataset.feature_names, features=features, labels=labels, top_n=3, n_repeats=2
    )
    assert (table["method"] == "permutation").all()


def test_global_importance_needs_data_when_there_are_no_native_importances(
    encoded, dataset
) -> None:
    features, labels, _ = encoded
    model = build_pipeline("rules").fit(features, labels)
    with pytest.raises(ValueError, match="permutation importance"):
        global_importance(model, dataset.feature_names)


def test_global_importance_rejects_a_name_length_mismatch(encoded) -> None:
    features, labels, _ = encoded
    model = build_pipeline("logreg").fit(features, labels)
    with pytest.raises(ValueError, match="Importance vector"):
        global_importance(model, ["only", "two"])


class TestExplainPrediction:
    def test_the_signature_symptoms_carry_the_evidence(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        contributions = explain_prediction(model, features[0], names, top_n=4)
        assert {c.symptom for c in contributions} == {"fever", "cough"}
        assert all(c.contribution > 0 for c in contributions)

    def test_only_reported_symptoms_are_attributed(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        contributions = explain_prediction(model, np.array([1, 0, 0, 0]), names)
        assert [c.symptom for c in contributions] == ["fever"]

    def test_shares_of_positive_evidence_sum_to_one(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        contributions = explain_prediction(model, features[0], names, top_n=4)
        assert sum(c.share for c in contributions) == pytest.approx(1.0, abs=1e-3)

    def test_an_empty_symptom_vector_yields_no_attribution(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        assert explain_prediction(model, np.zeros(4), names) == []

    def test_contributions_are_returned_strongest_first(self, encoded, dataset) -> None:
        features, labels, _ = encoded
        model = build_pipeline("logreg").fit(features, labels)
        contributions = explain_prediction(model, features[0], dataset.feature_names, top_n=5)
        values = [c.contribution for c in contributions]
        assert values == sorted(values, reverse=True)

    def test_a_vector_width_mismatch_is_rejected(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        with pytest.raises(ValueError, match="features but"):
            explain_prediction(model, np.zeros(9), names)

    def test_a_model_without_probabilities_is_rejected(self, separable) -> None:
        from sklearn.svm import LinearSVC

        features, labels, names = separable
        model = LinearSVC().fit(features, labels)
        with pytest.raises(AttributeError, match="predict_proba"):
            explain_prediction(model, features[0], names)


class TestCounterfactuals:
    def test_suggests_only_unreported_symptoms(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        suggestions = counterfactual_symptoms(model, np.array([1, 0, 0, 0]), names, top_n=3)
        assert "fever" not in {c.symptom for c in suggestions}

    def test_the_best_suggestion_completes_the_signature(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        suggestions = counterfactual_symptoms(model, np.array([1, 0, 0, 0]), names, top_n=1)
        assert suggestions[0].symptom == "cough"

    def test_a_fully_reported_vector_has_nothing_to_suggest(self, separable) -> None:
        features, labels, names = separable
        model = build_pipeline("logreg").fit(features, labels)
        assert counterfactual_symptoms(model, np.ones(4), names) == []
