"""Tests for the inference layer's contract.

These pin the behaviours that separate a service from a bare ``predict_proba``:
unknown input is surfaced, thin evidence is flagged, and every response carries
its disclaimer.
"""

from __future__ import annotations

import pytest

from disease_predictor.predict import (
    MIN_RECOMMENDED_SYMPTOMS,
    DiseasePredictor,
    PredictionResult,
)
from disease_predictor.training import load_bundle, save_bundle


@pytest.fixture(scope="module")
def symptoms(predictor: DiseasePredictor) -> list[str]:
    return list(predictor.symptoms[:4])


def test_prediction_returns_a_ranked_shortlist(predictor, symptoms) -> None:
    result = predictor.predict(symptoms, top_k=3)
    assert len(result.predictions) == 3
    assert [p.rank for p in result.predictions] == [1, 2, 3]
    probabilities = [p.probability for p in result.predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_probabilities_are_valid(predictor, symptoms) -> None:
    result = predictor.predict(symptoms, top_k=len(predictor.diseases))
    assert all(0.0 <= p.probability <= 1.0 for p in result.predictions)
    assert sum(p.probability for p in result.predictions) == pytest.approx(1.0, abs=1e-3)


def test_top_k_is_clamped_to_the_number_of_diseases(predictor, symptoms) -> None:
    result = predictor.predict(symptoms, top_k=999)
    assert len(result.predictions) == len(predictor.diseases)


def test_top_k_must_be_positive(predictor, symptoms) -> None:
    with pytest.raises(ValueError, match="top_k"):
        predictor.predict(symptoms, top_k=0)


def test_predicted_diseases_come_from_the_trained_label_set(predictor, symptoms) -> None:
    result = predictor.predict(symptoms)
    assert all(p.disease in predictor.diseases for p in result.predictions)


def test_unknown_symptoms_are_reported_not_silently_dropped(predictor, symptoms) -> None:
    result = predictor.predict([*symptoms, "definitely_not_a_symptom"])
    assert result.unknown_symptoms == ("definitely_not_a_symptom",)
    assert any("not recognised" in warning for warning in result.warnings)
    assert result.predictions


def test_no_recognised_symptoms_yields_no_prediction(predictor) -> None:
    result = predictor.predict(["xyzzy", "plugh"])
    assert result.predictions == ()
    assert result.top is None
    assert any("No known symptoms" in warning for warning in result.warnings)


def test_thin_evidence_is_flagged(predictor, symptoms) -> None:
    result = predictor.predict(symptoms[:1])
    assert any(str(MIN_RECOMMENDED_SYMPTOMS) in warning for warning in result.warnings)


def test_ample_evidence_is_not_flagged_as_thin(predictor, symptoms) -> None:
    result = predictor.predict(symptoms)
    assert not any("at least" in warning for warning in result.warnings)


def test_casing_and_spacing_do_not_change_the_answer(predictor, symptoms) -> None:
    spelled_out = [s.replace("_", " ").upper() for s in symptoms]
    assert (
        predictor.predict(symptoms).top.disease == predictor.predict(spelled_out).top.disease
    )


def test_explanations_are_attached_by_default(predictor, symptoms) -> None:
    result = predictor.predict(symptoms)
    assert result.top.supporting_symptoms
    assert all(
        c.symptom in result.recognised_symptoms for c in result.top.supporting_symptoms
    )


def test_explanations_can_be_switched_off(predictor, symptoms) -> None:
    result = predictor.predict(symptoms, explain=False)
    assert result.top.supporting_symptoms == ()


def test_follow_up_suggests_unreported_symptoms(predictor, symptoms) -> None:
    result = predictor.predict(symptoms, follow_up=True)
    assert result.follow_up_symptoms
    assert all(c.symptom not in result.recognised_symptoms for c in result.follow_up_symptoms)


def test_every_response_carries_the_disclaimer(predictor, symptoms) -> None:
    result = predictor.predict(symptoms)
    assert "not medical advice" in result.disclaimer
    assert "not medical advice" in result.to_dict()["disclaimer"]


def test_result_serialises_to_json_ready_primitives(predictor, symptoms) -> None:
    import json

    payload = predictor.predict(symptoms, follow_up=True).to_dict()
    assert json.loads(json.dumps(payload))["predictions"][0]["rank"] == 1


def test_batch_prediction_matches_single_prediction(predictor, symptoms) -> None:
    batch = predictor.predict_batch([symptoms, symptoms[:2]])
    assert len(batch) == 2
    assert isinstance(batch[0], PredictionResult)
    assert batch[0].top.disease == predictor.predict(symptoms, explain=False).top.disease


def test_symptom_search_is_exposed_for_autocomplete(predictor) -> None:
    needle = predictor.symptoms[0]
    assert needle in predictor.search_symptoms(needle)


def test_metadata_records_provenance(predictor) -> None:
    metadata = predictor.metadata
    assert metadata["config"]["model_name"] == "logreg"
    assert "data_fingerprint" in metadata
    assert "environment" in metadata


class TestBundleRoundTrip:
    def test_a_saved_bundle_reloads_and_predicts_identically(
        self, trained, tmp_path, symptoms
    ) -> None:
        path = save_bundle(trained.bundle, tmp_path / "bundle.joblib")
        reloaded = DiseasePredictor(load_bundle(path))
        assert reloaded.predict(symptoms).top.disease == DiseasePredictor(
            trained.bundle
        ).predict(symptoms).top.disease

    def test_feature_order_survives_serialisation(self, trained, tmp_path) -> None:
        path = save_bundle(trained.bundle, tmp_path / "bundle.joblib")
        assert load_bundle(path).feature_names == trained.bundle.feature_names

    def test_loading_a_missing_bundle_explains_how_to_train_one(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="dp train"):
            DiseasePredictor.load(tmp_path / "absent.joblib")

    def test_loading_a_non_bundle_file_is_rejected(self, tmp_path) -> None:
        import joblib

        path = tmp_path / "not_a_bundle.joblib"
        joblib.dump({"model": None}, path)
        with pytest.raises(TypeError, match="ModelBundle"):
            DiseasePredictor.load(path)

    def test_the_environment_variable_locates_the_bundle(
        self, trained, tmp_path, monkeypatch
    ) -> None:
        from disease_predictor.config import BUNDLE_PATH_ENV_VAR

        path = save_bundle(trained.bundle, tmp_path / "bundle.joblib")
        monkeypatch.setenv(BUNDLE_PATH_ENV_VAR, str(path))
        assert DiseasePredictor.load().diseases == trained.bundle.class_names
