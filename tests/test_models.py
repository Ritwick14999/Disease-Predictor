"""Tests for the model registry and the rule-based baseline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from disease_predictor.models import (
    MODEL_REGISTRY,
    SymptomSignatureMatcher,
    UnknownModelError,
    available_models,
    build_estimator,
    build_pipeline,
    build_sampler,
)


@pytest.mark.parametrize("name", sorted(set(MODEL_REGISTRY) - {"xgboost"}))
def test_every_registered_model_fits_and_scores(name: str, encoded) -> None:
    features, labels, _ = encoded
    model = build_pipeline(name)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(labels), len(np.unique(labels)))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(UnknownModelError, match="Unknown model"):
        build_estimator("not_a_model")


def test_available_models_can_exclude_optional_backends() -> None:
    assert "xgboost" not in available_models(include_optional=False)
    assert "rules" in available_models(include_optional=False)


def test_pipeline_without_a_sampler_has_a_single_step() -> None:
    assert [name for name, _ in build_pipeline("logreg").steps] == ["model"]


def test_pipeline_with_a_sampler_resamples_before_the_model() -> None:
    pipeline = build_pipeline("logreg", sampler="random")
    assert [name for name, _ in pipeline.steps] == ["resample", "model"]
    assert isinstance(pipeline, Pipeline)


def test_build_sampler_returns_none_for_no_resampling() -> None:
    assert build_sampler("none") is None


def test_build_sampler_rejects_unknown_strategies() -> None:
    with pytest.raises(ValueError, match="Unknown sampler"):
        build_sampler("bootstrap")


class TestSymptomSignatureMatcher:
    """The transparent baseline that keeps the headline number honest."""

    def test_learns_one_signature_per_class(self, encoded) -> None:
        features, labels, _ = encoded
        matcher = SymptomSignatureMatcher().fit(features, labels)
        assert matcher.signatures_.shape == (len(np.unique(labels)), features.shape[1])

    def test_recovers_cleanly_separated_classes(self) -> None:
        features = np.array([[1, 1, 0, 0]] * 5 + [[0, 0, 1, 1]] * 5)
        labels = np.array([0] * 5 + [1] * 5)
        matcher = SymptomSignatureMatcher().fit(features, labels)
        np.testing.assert_array_equal(matcher.predict(features), labels)

    def test_probabilities_form_a_distribution(self, encoded) -> None:
        features, labels, _ = encoded
        matcher = SymptomSignatureMatcher().fit(features, labels)
        probabilities = matcher.predict_proba(features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        assert (probabilities >= 0).all()

    def test_handles_an_all_zero_symptom_vector(self, encoded) -> None:
        features, labels, _ = encoded
        matcher = SymptomSignatureMatcher().fit(features, labels)
        probabilities = matcher.predict_proba(np.zeros((1, features.shape[1])))
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-9)

    def test_rejects_a_wrong_feature_count(self, encoded) -> None:
        features, labels, _ = encoded
        matcher = SymptomSignatureMatcher().fit(features, labels)
        with pytest.raises(ValueError, match="Expected"):
            matcher.predict(np.zeros((1, features.shape[1] + 3)))

    def test_rejects_an_out_of_range_threshold(self, encoded) -> None:
        features, labels, _ = encoded
        with pytest.raises(ValueError, match="prevalence_threshold"):
            SymptomSignatureMatcher(prevalence_threshold=1.5).fit(features, labels)
