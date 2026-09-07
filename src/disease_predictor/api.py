"""A FastAPI service around the trained model.

The service exists to make one point concrete: the difference between a notebook
that predicts and a system that serves. It loads the model once at startup,
validates input with a schema rather than trusting it, returns structured errors,
and exposes the model's own provenance at ``/model-info`` so a caller can tell
which artifact answered them.

Run it with ``dp serve`` or ``uvicorn disease_predictor.api:app``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from disease_predictor.__about__ import __version__
from disease_predictor.config import DISCLAIMER
from disease_predictor.predict import DiseasePredictor

LOGGER = logging.getLogger(__name__)

#: Populated at startup so every request reuses one loaded model.
_PREDICTOR: DiseasePredictor | None = None

MAX_SYMPTOMS_PER_REQUEST = 50


class SymptomContributionOut(BaseModel):
    """How much one symptom supported a prediction."""

    symptom: str
    contribution: float = Field(description="Probability lost if this symptom were removed.")
    share: float = Field(description="Share of the total supporting evidence, in [0, 1].")


class DiseaseScoreOut(BaseModel):
    """One entry of the ranked shortlist."""

    disease: str
    probability: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    supporting_symptoms: list[SymptomContributionOut] = []


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    symptoms: list[str] = Field(
        min_length=1,
        max_length=MAX_SYMPTOMS_PER_REQUEST,
        description="Reported symptom names; casing and spacing are normalised.",
        examples=[["itching", "skin_rash", "nodal_skin_eruptions"]],
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Length of the shortlist.")
    explain: bool = Field(default=True, description="Attach per-symptom attribution.")
    follow_up: bool = Field(
        default=False, description="Also suggest which unreported symptoms to ask about next."
    )

    @field_validator("symptoms")
    @classmethod
    def _reject_blank_symptoms(cls, value: list[str]) -> list[str]:
        cleaned = [item for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty symptom is required.")
        return cleaned


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``."""

    predictions: list[DiseaseScoreOut]
    recognised_symptoms: list[str]
    unknown_symptoms: list[str] = []
    suggestions: dict[str, list[str]] = {}
    warnings: list[str] = []
    follow_up_symptoms: list[dict[str, Any]] = []
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    version: str
    model_loaded: bool
    n_symptoms: int = 0
    n_diseases: int = 0


class ModelInfoResponse(BaseModel):
    """Provenance of the artifact currently serving requests."""

    version: str
    trained_at: str | None = None
    model_name: str | None = None
    n_symptoms: int
    n_diseases: int
    trained_on_synthetic_data: bool = False
    data_fingerprint: str | None = None
    holdout_metrics: dict[str, float] = {}
    leakage_audit: dict[str, Any] = {}
    environment: dict[str, str] = {}
    disclaimer: str = DISCLAIMER


def get_predictor() -> DiseasePredictor:
    """FastAPI dependency returning the loaded predictor.

    Raises:
        HTTPException: 503 when no model bundle could be loaded at startup, so
            the caller sees "not ready" rather than a 500.
    """
    if _PREDICTOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is loaded. Train one with `dp train` "
                "(or `dp train --synthetic` to try the service without data)."
            ),
        )
    return _PREDICTOR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup instead of once per request."""
    global _PREDICTOR
    try:
        _PREDICTOR = DiseasePredictor.load()
        LOGGER.info(
            "Loaded model: %d symptoms, %d diseases.",
            len(_PREDICTOR.symptoms),
            len(_PREDICTOR.diseases),
        )
    except (FileNotFoundError, TypeError) as exc:
        # A service that boots and reports 503 is more useful than one that
        # refuses to start: /health still answers and says what is missing.
        _PREDICTOR = None
        LOGGER.warning("Starting without a model: %s", exc)
    yield
    _PREDICTOR = None


app = FastAPI(
    title="Disease Predictor API",
    version=__version__,
    description=(
        "Ranks likely diseases from reported symptoms, with per-symptom "
        "attribution.\n\n**Educational project — not a medical device and not "
        "medical advice.**"
    ),
    lifespan=lifespan,
)

PredictorDep = Annotated[DiseasePredictor, Depends(get_predictor)]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness and readiness check; always 200 so probes can distinguish states."""
    if _PREDICTOR is None:
        return HealthResponse(status="degraded", version=__version__, model_loaded=False)
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=True,
        n_symptoms=len(_PREDICTOR.symptoms),
        n_diseases=len(_PREDICTOR.diseases),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info(predictor: PredictorDep) -> ModelInfoResponse:
    """Return how the serving model was trained and how it scored."""
    metadata = predictor.metadata
    config = metadata.get("config", {})
    return ModelInfoResponse(
        version=__version__,
        trained_at=metadata.get("created_at"),
        model_name=config.get("model_name"),
        n_symptoms=len(predictor.symptoms),
        n_diseases=len(predictor.diseases),
        trained_on_synthetic_data=bool(metadata.get("synthetic", False)),
        data_fingerprint=str(metadata.get("data_fingerprint", ""))[:16] or None,
        holdout_metrics=metadata.get("holdout", {}),
        leakage_audit=metadata.get("leakage_audit", {}),
        environment=metadata.get("environment", {}),
    )


@app.get("/symptoms", tags=["reference"])
def list_symptoms(
    predictor: PredictorDep,
    q: Annotated[str | None, Query(description="Optional substring/fuzzy filter.")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    """List the symptom vocabulary, optionally filtered for autocomplete."""
    names = predictor.search_symptoms(q, limit=limit) if q else predictor.symptoms[:limit]
    return {"count": len(names), "total": len(predictor.symptoms), "symptoms": list(names)}


@app.get("/diseases", tags=["reference"])
def list_diseases(predictor: PredictorDep) -> dict[str, Any]:
    """List every disease the model can rank."""
    return {"count": len(predictor.diseases), "diseases": list(predictor.diseases)}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest, predictor: PredictorDep) -> PredictResponse:
    """Rank the most likely diseases for the reported symptoms.

    Unrecognised symptoms do not fail the request: they come back in
    ``unknown_symptoms`` with spelling suggestions, so a client can correct them.
    """
    result = predictor.predict(
        request.symptoms,
        top_k=request.top_k,
        explain=request.explain,
        follow_up=request.follow_up,
    )
    return PredictResponse(**result.to_dict())
