"""Turning free-form symptom names into the exact vector the model expects.

The single most common way a working model produces garbage in production is a
feature-order mismatch between training and inference. :class:`SymptomEncoder`
owns the ordered feature list, so the vector handed to the model is correct by
construction and unknown input is reported instead of silently ignored.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from disease_predictor.data import canonical_symptom

__all__ = ["SymptomEncoder", "EncodedSymptoms", "UnknownSymptomError"]


class UnknownSymptomError(KeyError):
    """Raised in strict mode when a symptom is not part of the trained vocabulary."""

    def __init__(self, symptom: str, suggestions: Sequence[str]) -> None:
        self.symptom = symptom
        self.suggestions = list(suggestions)
        hint = f" Did you mean: {', '.join(self.suggestions)}?" if suggestions else ""
        super().__init__(f"Unknown symptom {symptom!r}.{hint}")


@dataclass(frozen=True)
class EncodedSymptoms:
    """Result of encoding a caller-supplied symptom list.

    Attributes:
        vector: Binary row vector of shape ``(1, n_features)``.
        recognised: Canonical names that matched the vocabulary.
        unknown: Inputs that could not be matched.
        suggestions: Nearest known symptom for each unknown input.
    """

    vector: np.ndarray
    recognised: tuple[str, ...]
    unknown: tuple[str, ...] = ()
    suggestions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def n_recognised(self) -> int:
        return len(self.recognised)


class SymptomEncoder:
    """Maps symptom names to a fixed-width binary vector.

    The encoder is stored inside the model bundle, so inference always uses the
    same vocabulary and ordering as training.

    Args:
        feature_names: Ordered symptom names as seen during training.

    Raises:
        ValueError: If the vocabulary is empty or contains duplicates.
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        canonical = [canonical_symptom(name) for name in feature_names]
        if not canonical:
            raise ValueError("SymptomEncoder needs at least one feature name.")
        duplicates = {name for name in canonical if canonical.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate symptom names after canonicalisation: {sorted(duplicates)}")

        self._feature_names: tuple[str, ...] = tuple(canonical)
        self._index: dict[str, int] = {name: i for i, name in enumerate(self._feature_names)}
        # Space-separated aliases so "skin rash" resolves to "skin_rash".
        self._aliases: dict[str, str] = {name.replace("_", " "): name for name in self._feature_names}

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered symptom vocabulary."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def __contains__(self, symptom: object) -> bool:
        return canonical_symptom(str(symptom)) in self._index

    def __len__(self) -> int:
        return len(self._feature_names)

    def suggest(self, symptom: str, *, limit: int = 3, cutoff: float = 0.6) -> tuple[str, ...]:
        """Return the closest known symptom names to a misspelled input."""
        key = canonical_symptom(symptom)
        matches = difflib.get_close_matches(key, self._feature_names, n=limit, cutoff=cutoff)
        if matches:
            return tuple(matches)
        # Fall back to substring containment, which catches partial words that
        # edit distance misses ("urine" -> "burning_micturition" is a miss, but
        # "chest" -> "chest_pain" is an obvious hit).
        contains = [name for name in self._feature_names if key and key in name]
        return tuple(contains[:limit])

    def encode(self, symptoms: Iterable[str], *, strict: bool = False) -> EncodedSymptoms:
        """Encode symptom names into a model-ready row vector.

        Args:
            symptoms: Names in any casing/spacing; ``"Skin Rash"`` and
                ``"skin_rash"`` are equivalent.
            strict: Raise on an unrecognised symptom instead of collecting it
                into :attr:`EncodedSymptoms.unknown`.

        Raises:
            UnknownSymptomError: In strict mode, on the first unknown symptom.
        """
        vector = np.zeros((1, self.n_features), dtype=np.int8)
        recognised: list[str] = []
        unknown: list[str] = []
        suggestions: dict[str, tuple[str, ...]] = {}

        for raw in symptoms:
            if not isinstance(raw, str) or not raw.strip():
                continue
            key = canonical_symptom(raw)
            resolved = key if key in self._index else self._aliases.get(raw.strip().lower())
            if resolved is None:
                if strict:
                    raise UnknownSymptomError(raw, self.suggest(raw))
                unknown.append(raw)
                close = self.suggest(raw)
                if close:
                    suggestions[raw] = close
                continue
            vector[0, self._index[resolved]] = 1
            if resolved not in recognised:
                recognised.append(resolved)

        return EncodedSymptoms(
            vector=vector,
            recognised=tuple(recognised),
            unknown=tuple(unknown),
            suggestions=suggestions,
        )

    def decode(self, vector: np.ndarray) -> tuple[str, ...]:
        """Return the symptom names that are set in a binary vector."""
        flat = np.asarray(vector).reshape(-1)
        if flat.size != self.n_features:
            raise ValueError(
                f"Expected a vector of length {self.n_features}; got {flat.size}."
            )
        return tuple(name for name, value in zip(self._feature_names, flat, strict=True) if value)

    def search(self, query: str, *, limit: int = 10) -> tuple[str, ...]:
        """Substring search over the vocabulary, for autocomplete in the UI."""
        key = canonical_symptom(query)
        if not key:
            return self._feature_names[:limit]
        starts = [n for n in self._feature_names if n.startswith(key)]
        contains = [n for n in self._feature_names if key in n and n not in starts]
        fuzzy = [n for n in self.suggest(query, limit=limit) if n not in starts and n not in contains]
        return tuple((starts + contains + fuzzy)[:limit])
