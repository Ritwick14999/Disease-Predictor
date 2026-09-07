"""The model registry, plus the baselines that keep the headline number honest.

A gradient-boosted ensemble is only a good answer if something simpler is a
worse one. This module therefore ships the strong model *and* the references it
has to beat:

``majority``    Predict the most frequent class. The floor.
``rules``       :class:`SymptomSignatureMatcher` -- no gradients, no training
                loop, just set overlap against each disease's symptom
                signature. If this ties the ensemble, the task is a lookup.
``tree``        A depth-limited decision tree: how many yes/no questions the
                problem actually needs.
``logreg``      A linear model, and the interpretability reference point.
``random_forest`` / ``xgboost``  The heavy hitters.

All estimators here expect **integer-encoded** labels. XGBoost requires it, and
keeping one convention avoids a class of bug where two models in the same
benchmark disagree about what class ``3`` means. The
:class:`~sklearn.preprocessing.LabelEncoder` that performs the mapping is owned
by :mod:`disease_predictor.training` and travels with the model in its bundle,
so inference can turn an index back into a disease name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

__all__ = [
    "SymptomSignatureMatcher",
    "MODEL_REGISTRY",
    "available_models",
    "build_estimator",
    "build_pipeline",
    "xgboost_available",
    "UnknownModelError",
]


class UnknownModelError(KeyError):
    """Raised when a model name is not in the registry."""


def xgboost_available() -> bool:
    """Return whether the optional XGBoost dependency can be imported."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


class SymptomSignatureMatcher(BaseEstimator, ClassifierMixin):
    """A transparent, non-learned baseline: match symptoms against class signatures.

    For every class it stores the mean prevalence of each symptom, thresholds
    that into a signature set, then scores a new patient by the Jaccard overlap
    between their reported symptoms and each signature. Probabilities are the
    normalised similarities.

    There is no gradient descent and no hyperparameter search here, which is the
    point: it is the number a boosted ensemble has to beat to justify itself.

    Args:
        prevalence_threshold: A symptom joins a class signature when it is
            present in at least this share of that class's training rows.
        smoothing: Added to every similarity before normalising so the output is
            a proper distribution even when nothing overlaps.
    """

    def __init__(self, prevalence_threshold: float = 0.5, smoothing: float = 1e-6) -> None:
        self.prevalence_threshold = prevalence_threshold
        self.smoothing = smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> SymptomSignatureMatcher:
        """Learn one binary signature vector per class."""
        X, y = check_X_y(X, y)
        if not 0.0 < self.prevalence_threshold <= 1.0:
            raise ValueError(
                f"prevalence_threshold must be in (0, 1]; got {self.prevalence_threshold}."
            )
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        signatures = np.zeros((len(self.classes_), X.shape[1]), dtype=np.float64)
        for index, label in enumerate(self.classes_):
            rows = X[y == label]
            prevalence = rows.mean(axis=0) if len(rows) else np.zeros(X.shape[1])
            signatures[index] = (prevalence >= self.prevalence_threshold).astype(np.float64)
        self.signatures_ = signatures
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised Jaccard similarity to each class signature."""
        check_is_fitted(self, "signatures_")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; got {X.shape[1]}."
            )
        intersection = X @ self.signatures_.T
        union = (
            X.sum(axis=1, keepdims=True)
            + self.signatures_.sum(axis=1)[np.newaxis, :]
            - intersection
        )
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        similarity = similarity + self.smoothing
        return similarity / similarity.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the best-matching class for each row."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _build_xgboost(random_state: int, **overrides: Any) -> Any:
    from xgboost import XGBClassifier

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def _build_rules(random_state: int = 42, **overrides: Any) -> SymptomSignatureMatcher:  # noqa: ARG001
    """Build the rule-based baseline.

    ``random_state`` is accepted so every registry factory shares one signature,
    and ignored because this baseline is fully deterministic.
    """
    return SymptomSignatureMatcher(**overrides)


MODEL_REGISTRY: dict[str, Callable[..., BaseEstimator]] = {
    "majority": lambda random_state=42, **kw: DummyClassifier(
        strategy="most_frequent", random_state=random_state, **kw
    ),
    "rules": _build_rules,
    "tree": lambda random_state=42, **kw: DecisionTreeClassifier(
        max_depth=kw.pop("max_depth", 8), random_state=random_state, **kw
    ),
    "logreg": lambda random_state=42, **kw: LogisticRegression(
        max_iter=kw.pop("max_iter", 2000),
        C=kw.pop("C", 1.0),
        random_state=random_state,
        **kw,
    ),
    "random_forest": lambda random_state=42, **kw: RandomForestClassifier(
        n_estimators=kw.pop("n_estimators", 300),
        max_depth=kw.pop("max_depth", None),
        random_state=random_state,
        n_jobs=-1,
        **kw,
    ),
    "xgboost": _build_xgboost,
}


def available_models(*, include_optional: bool = True) -> tuple[str, ...]:
    """List registry names that can actually be built in this environment."""
    names = []
    for name in MODEL_REGISTRY:
        if name == "xgboost" and not (include_optional and xgboost_available()):
            continue
        names.append(name)
    return tuple(names)


def build_estimator(name: str, *, random_state: int = 42, **overrides: Any) -> BaseEstimator:
    """Instantiate a model from the registry.

    Args:
        name: Registry key, e.g. ``"xgboost"`` or ``"rules"``.
        random_state: Seed forwarded to models that accept one.
        **overrides: Hyperparameters that replace the registry defaults.

    Raises:
        UnknownModelError: If ``name`` is not registered.
        ImportError: If the model needs an optional dependency that is missing.
    """
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise UnknownModelError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        ) from exc
    if name == "xgboost" and not xgboost_available():
        raise ImportError(
            "The 'xgboost' model needs the optional xgboost package: pip install 'disease-predictor[xgboost]'"
        )
    return factory(random_state=random_state, **overrides)


def build_sampler(sampler: str, *, random_state: int = 42) -> Any | None:
    """Build a resampler for class imbalance, or ``None`` for no resampling.

    Raises:
        ImportError: If imbalanced-learn is not installed but a sampler is asked for.
        ValueError: If ``sampler`` is not a known strategy.
    """
    if sampler == "none":
        return None
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Resampling needs imbalanced-learn: pip install imbalanced-learn"
        ) from exc
    if sampler == "smote":
        return SMOTE(random_state=random_state)
    if sampler == "random":
        return RandomOverSampler(random_state=random_state)
    raise ValueError(f"Unknown sampler {sampler!r}.")


def build_pipeline(
    name: str,
    *,
    sampler: str = "none",
    random_state: int = 42,
    **overrides: Any,
) -> SklearnPipeline:
    """Build the full estimator, optionally preceded by a resampling step.

    The resampler is *inside* the pipeline on purpose. Resampling before
    cross-validation is a classic leak: synthetic rows derived from a held-out
    fold end up in the training folds. Inside a pipeline, scikit-learn applies it
    per fold, and only to the training part.
    """
    estimator = build_estimator(name, random_state=random_state, **overrides)
    resampler = build_sampler(sampler, random_state=random_state)
    if resampler is None:
        return SklearnPipeline([("model", estimator)])
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline([("resample", resampler), ("model", estimator)])
