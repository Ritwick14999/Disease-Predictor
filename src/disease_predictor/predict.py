"""Inference: symptoms in, a ranked and explained shortlist out.

:class:`DiseasePredictor` is the one object the API, the CLI and the Streamlit
UI all share, so behaviour cannot drift between them. It is deliberately opinionated
about three things a raw ``predict_proba`` call gets wrong:

* Unknown symptoms are **reported**, not silently dropped, so a typo cannot
  quietly change the answer.
* Thin evidence is **flagged**. Two symptoms out of a hundred-plus is not enough
  to separate diseases, and the response says so instead of implying confidence.
* Every prediction carries its **attribution** and the project disclaimer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from disease_predictor.config import (
    BUNDLE_FILENAME,
    BUNDLE_PATH_ENV_VAR,
    DEFAULT_ARTIFACT_DIR,
    DISCLAIMER,
)
from disease_predictor.explain import (
    SymptomContribution,
    counterfactual_symptoms,
    explain_prediction,
)
from disease_predictor.training import ModelBundle, load_bundle

LOGGER = logging.getLogger(__name__)

__all__ = ["DiseasePredictor", "PredictionResult", "DiseaseScore", "MIN_RECOMMENDED_SYMPTOMS"]

#: Below this many recognised symptoms the ranking is treated as under-determined.
MIN_RECOMMENDED_SYMPTOMS = 3

#: Below this top probability the model is not meaningfully committing to an answer.
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class DiseaseScore:
    """One entry of the ranked shortlist."""

    disease: str
    probability: float
    rank: int
    supporting_symptoms: tuple[SymptomContribution, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "probability": self.probability,
            "rank": self.rank,
            "supporting_symptoms": [
                {"symptom": c.symptom, "contribution": c.contribution, "share": c.share}
                for c in self.supporting_symptoms
            ],
        }


@dataclass(frozen=True)
class PredictionResult:
    """A complete, self-describing prediction response."""

    predictions: tuple[DiseaseScore, ...]
    recognised_symptoms: tuple[str, ...]
    unknown_symptoms: tuple[str, ...] = ()
    suggestions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    follow_up_symptoms: tuple[SymptomContribution, ...] = ()
    disclaimer: str = DISCLAIMER

    @property
    def top(self) -> DiseaseScore | None:
        """The highest-ranked disease, or ``None`` when nothing was recognised."""
        return self.predictions[0] if self.predictions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "recognised_symptoms": list(self.recognised_symptoms),
            "unknown_symptoms": list(self.unknown_symptoms),
            "suggestions": {k: list(v) for k, v in self.suggestions.items()},
            "warnings": list(self.warnings),
            "follow_up_symptoms": [
                {"symptom": c.symptom, "expected_gain": c.contribution}
                for c in self.follow_up_symptoms
            ],
            "disclaimer": self.disclaimer,
        }


class DiseasePredictor:
    """Loads a trained bundle and turns symptom names into a ranked shortlist.

    Args:
        bundle: A bundle produced by :func:`disease_predictor.training.train`.

    Example:
        >>> predictor = DiseasePredictor.load()               # doctest: +SKIP
        >>> result = predictor.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
        >>> result.top.disease                                # doctest: +SKIP
        'Fungal infection'
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle
        self._model = bundle.model
        self._encoder = bundle.encoder
        self._class_names = bundle.class_names

    @classmethod
    def load(cls, path: str | Path | None = None) -> DiseasePredictor:
        """Load a predictor from disk.

        Resolution order: the explicit ``path``, then the ``DISEASE_MODEL_PATH``
        environment variable, then ``artifacts/`` in the project root.

        Raises:
            FileNotFoundError: If no bundle is found.
        """
        candidate = (
            Path(path)
            if path is not None
            else Path(os.environ.get(BUNDLE_PATH_ENV_VAR, DEFAULT_ARTIFACT_DIR / BUNDLE_FILENAME))
        )
        return cls(load_bundle(candidate))

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    @property
    def symptoms(self) -> tuple[str, ...]:
        """The full symptom vocabulary, in model input order."""
        return self._encoder.feature_names

    @property
    def diseases(self) -> tuple[str, ...]:
        """Every disease the model can predict."""
        return self._class_names

    @property
    def metadata(self) -> dict[str, Any]:
        """Provenance recorded at training time: config, metrics, versions."""
        return dict(self._bundle.metadata)

    def search_symptoms(self, query: str, *, limit: int = 10) -> tuple[str, ...]:
        """Autocomplete over the symptom vocabulary."""
        return self._encoder.search(query, limit=limit)

    def predict(
        self,
        symptoms: Iterable[str],
        *,
        top_k: int = 3,
        explain: bool = True,
        follow_up: bool = False,
    ) -> PredictionResult:
        """Rank the most likely diseases for a set of reported symptoms.

        Args:
            symptoms: Symptom names in any casing or spacing.
            top_k: Length of the returned shortlist.
            explain: Attach per-symptom attribution to each entry.
            follow_up: Also return the unreported symptoms that would most
                sharpen the top prediction -- the "what to ask next" list.

        Returns:
            A :class:`PredictionResult`. When nothing is recognised the shortlist
            is empty and a warning explains why, rather than returning the
            model's prior over an all-zero vector.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {top_k}.")

        encoded = self._encoder.encode(symptoms)
        warnings: list[str] = []

        if encoded.unknown:
            warnings.append(
                f"{len(encoded.unknown)} symptom(s) not recognised and ignored: "
                f"{', '.join(encoded.unknown)}."
            )

        if encoded.n_recognised == 0:
            warnings.append("No known symptoms were provided, so no prediction was made.")
            return PredictionResult(
                predictions=(),
                recognised_symptoms=(),
                unknown_symptoms=encoded.unknown,
                suggestions=encoded.suggestions,
                warnings=tuple(warnings),
            )

        if encoded.n_recognised < MIN_RECOMMENDED_SYMPTOMS:
            warnings.append(
                f"Only {encoded.n_recognised} symptom(s) recognised. Diseases share symptoms, "
                f"so at least {MIN_RECOMMENDED_SYMPTOMS} are needed for a meaningful ranking."
            )

        probabilities = self._model.predict_proba(encoded.vector)[0]
        k = min(top_k, len(probabilities))
        ranked = np.argsort(probabilities)[::-1][:k]

        if float(probabilities[ranked[0]]) < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                "The model is not confident: the top probability is below "
                f"{LOW_CONFIDENCE_THRESHOLD:.0%}. Treat the whole shortlist as equally tentative."
            )

        predictions: list[DiseaseScore] = []
        for rank, class_index in enumerate(ranked, start=1):
            contributions: tuple[SymptomContribution, ...] = ()
            if explain:
                contributions = tuple(
                    explain_prediction(
                        self._model,
                        encoded.vector,
                        self._encoder.feature_names,
                        class_index=int(class_index),
                        top_n=5,
                    )
                )
            predictions.append(
                DiseaseScore(
                    disease=self._class_names[int(class_index)],
                    probability=round(float(probabilities[class_index]), 6),
                    rank=rank,
                    supporting_symptoms=contributions,
                )
            )

        follow_ups: tuple[SymptomContribution, ...] = ()
        if follow_up:
            follow_ups = tuple(
                counterfactual_symptoms(
                    self._model,
                    encoded.vector,
                    self._encoder.feature_names,
                    class_index=int(ranked[0]),
                    top_n=5,
                )
            )

        return PredictionResult(
            predictions=tuple(predictions),
            recognised_symptoms=encoded.recognised,
            unknown_symptoms=encoded.unknown,
            suggestions=encoded.suggestions,
            warnings=tuple(warnings),
            follow_up_symptoms=follow_ups,
        )

    def predict_batch(
        self, batches: Sequence[Iterable[str]], *, top_k: int = 3
    ) -> list[PredictionResult]:
        """Predict for several symptom lists. Explanations are skipped for speed."""
        return [self.predict(item, top_k=top_k, explain=False) for item in batches]
