"""Tests for the REST API contract, including its behaviour with no model."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from disease_predictor.config import BUNDLE_PATH_ENV_VAR
from disease_predictor.training import save_bundle


@pytest.fixture(scope="module")
def client(trained, tmp_path_factory):
    """A client backed by the session's trained bundle."""
    path = save_bundle(trained.bundle, tmp_path_factory.mktemp("api") / "bundle.joblib")
    import os

    previous = os.environ.get(BUNDLE_PATH_ENV_VAR)
    os.environ[BUNDLE_PATH_ENV_VAR] = str(path)
    api = importlib.reload(importlib.import_module("disease_predictor.api"))
    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
        if previous is None:
            os.environ.pop(BUNDLE_PATH_ENV_VAR, None)
        else:
            os.environ[BUNDLE_PATH_ENV_VAR] = previous


@pytest.fixture(scope="module")
def known_symptoms(client) -> list[str]:
    return client.get("/symptoms", params={"limit": 4}).json()["symptoms"]


class TestOps:
    def test_health_reports_a_loaded_model(self, client) -> None:
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["model_loaded"] is True
        assert payload["n_diseases"] > 0

    def test_model_info_exposes_provenance(self, client) -> None:
        payload = client.get("/model-info").json()
        assert payload["model_name"] == "logreg"
        assert payload["trained_at"]
        assert payload["leakage_audit"]["n_rows"] > 0
        assert "not medical advice" in payload["disclaimer"]

    def test_openapi_schema_is_served(self, client) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestReference:
    def test_symptoms_are_listed(self, client) -> None:
        payload = client.get("/symptoms").json()
        assert payload["count"] > 0
        assert payload["count"] <= payload["total"]

    def test_symptoms_can_be_filtered(self, client, known_symptoms) -> None:
        needle = known_symptoms[0]
        assert needle in client.get("/symptoms", params={"q": needle}).json()["symptoms"]

    def test_symptom_limit_is_validated(self, client) -> None:
        assert client.get("/symptoms", params={"limit": 0}).status_code == 422

    def test_diseases_are_listed(self, client) -> None:
        payload = client.get("/diseases").json()
        assert payload["count"] == len(payload["diseases"])


class TestPredict:
    def test_returns_a_ranked_shortlist(self, client, known_symptoms) -> None:
        response = client.post("/predict", json={"symptoms": known_symptoms, "top_k": 3})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["predictions"]) == 3
        assert [p["rank"] for p in payload["predictions"]] == [1, 2, 3]

    def test_response_carries_the_disclaimer(self, client, known_symptoms) -> None:
        payload = client.post("/predict", json={"symptoms": known_symptoms}).json()
        assert "not medical advice" in payload["disclaimer"]

    def test_unknown_symptoms_come_back_with_suggestions(self, client, known_symptoms) -> None:
        typo = known_symptoms[0][:-1]
        payload = client.post(
            "/predict", json={"symptoms": [*known_symptoms, typo]}
        ).json()
        assert typo in payload["unknown_symptoms"]
        assert payload["predictions"]

    def test_explanations_are_included_by_default(self, client, known_symptoms) -> None:
        payload = client.post("/predict", json={"symptoms": known_symptoms}).json()
        assert payload["predictions"][0]["supporting_symptoms"]

    def test_follow_up_can_be_requested(self, client, known_symptoms) -> None:
        payload = client.post(
            "/predict", json={"symptoms": known_symptoms, "follow_up": True}
        ).json()
        assert payload["follow_up_symptoms"]

    def test_an_empty_symptom_list_is_rejected(self, client) -> None:
        assert client.post("/predict", json={"symptoms": []}).status_code == 422

    def test_blank_symptoms_are_rejected(self, client) -> None:
        assert client.post("/predict", json={"symptoms": ["  ", ""]}).status_code == 422

    def test_too_many_symptoms_are_rejected(self, client) -> None:
        response = client.post("/predict", json={"symptoms": [f"s{i}" for i in range(200)]})
        assert response.status_code == 422

    @pytest.mark.parametrize("top_k", [0, 11])
    def test_out_of_range_top_k_is_rejected(self, client, known_symptoms, top_k: int) -> None:
        response = client.post("/predict", json={"symptoms": known_symptoms, "top_k": top_k})
        assert response.status_code == 422

    def test_a_missing_symptoms_field_is_rejected(self, client) -> None:
        assert client.post("/predict", json={"top_k": 3}).status_code == 422


class TestWithoutAModel:
    """A service that boots without an artifact should degrade, not crash."""

    @pytest.fixture
    def bare_client(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BUNDLE_PATH_ENV_VAR, str(tmp_path / "absent.joblib"))
        api = importlib.reload(importlib.import_module("disease_predictor.api"))
        with TestClient(api.app) as test_client:
            yield test_client
        importlib.reload(importlib.import_module("disease_predictor.api"))

    def test_health_reports_degraded_rather_than_failing(self, bare_client) -> None:
        payload = bare_client.get("/health").json()
        assert payload["status"] == "degraded"
        assert payload["model_loaded"] is False

    def test_prediction_returns_503_with_a_next_step(self, bare_client) -> None:
        response = bare_client.post("/predict", json={"symptoms": ["itching"]})
        assert response.status_code == 503
        assert "dp train" in response.json()["detail"]
