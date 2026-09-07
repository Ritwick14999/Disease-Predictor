"""Metrics, leakage-aware splitting and the robustness benchmark.

Two ideas drive this module.

**Split protocol decides the headline number.** ``random`` reproduces the
textbook stratified split; ``grouped`` keeps every copy of a repeated symptom
pattern on one side of the split. Reporting both, side by side, converts an
unexplained 100% into a measured statement about how much of the score is
memorisation.

**A single accuracy is not an evaluation.** A triage-style model is used under
noisy reporting -- patients forget symptoms and volunteer irrelevant ones -- so
:func:`robustness_curve` measures accuracy as a function of that noise, and
:func:`expected_calibration_error` asks whether the probabilities mean anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

__all__ = [
    "SplitIndices",
    "top_k_accuracy",
    "expected_calibration_error",
    "classification_metrics",
    "make_split",
    "iter_folds",
    "cross_validate_model",
    "compare_split_protocols",
    "perturb_symptoms",
    "robustness_curve",
    "benchmark_models",
    "per_class_report",
]


@dataclass(frozen=True)
class SplitIndices:
    """Row indices for one train/test division."""

    train: np.ndarray
    test: np.ndarray
    strategy: str

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Share of rows whose true label is among the ``k`` highest-scored classes.

    The realistic success criterion for a differential-diagnosis aid: a shortlist
    that contains the answer is useful even when the top-1 guess is wrong.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2-D; got shape {y_proba.shape}.")
    k = max(1, min(int(k), y_proba.shape[1]))
    top = np.argsort(y_proba, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(y_true, top, strict=True)]))


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, *, n_bins: int = 10
) -> float:
    """Gap between stated confidence and observed accuracy, averaged over bins.

    0.0 means "when the model says 90%, it is right 90% of the time". A model
    that is accurate but wildly over-confident is dangerous in exactly the
    settings where a shortlist gets shown to a person.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    confidence = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    n_classes: int | None = None,
    top_k: int = 3,
) -> dict[str, float]:
    """Compute the metric set reported everywhere in this project.

    Accuracy alone hides per-class failure on rare diseases, which is why
    balanced accuracy and macro-F1 are always reported next to it.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        labels = list(range(n_classes if n_classes is not None else y_proba.shape[1]))
        metrics[f"top_{top_k}_accuracy"] = top_k_accuracy(y_true, y_proba, k=top_k)
        metrics["expected_calibration_error"] = expected_calibration_error(y_true, y_proba)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        except ValueError:
            # Happens when a fold is missing a class entirely; not fatal.
            metrics["log_loss"] = float("nan")
    return {key: round(value, 6) for key, value in metrics.items()}


def make_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    strategy: str = "grouped",
    groups: np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitIndices:
    """Produce train/test indices under the requested protocol.

    Args:
        features: Feature matrix, used only for its length and for deriving
            groups when none are supplied.
        labels: Integer-encoded labels, used for stratification.
        strategy: ``"random"`` for a stratified shuffle, ``"grouped"`` to keep
            identical symptom patterns together.
        groups: Group ids; defaults to one group per distinct feature row.
        test_size: Approximate share of rows held out.
        random_state: Seed.

    Raises:
        ValueError: On an unknown strategy.
    """
    n_rows = len(labels)
    n_splits = max(2, int(round(1.0 / test_size)))

    if strategy == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(n_rows), labels))
    elif strategy == "grouped":
        if groups is None:
            from disease_predictor.data import pattern_groups

            groups = pattern_groups(features)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(n_rows), labels, groups=groups))
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}; expected 'random' or 'grouped'.")

    return SplitIndices(train=np.asarray(train_idx), test=np.asarray(test_idx), strategy=strategy)


def iter_folds(
    labels: np.ndarray,
    *,
    strategy: str = "grouped",
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield cross-validation folds under the requested protocol."""
    n_rows = len(labels)
    if strategy == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        yield from splitter.split(np.zeros(n_rows), labels)
    elif strategy == "grouped":
        if groups is None:
            raise ValueError("Grouped folds need group ids.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        yield from splitter.split(np.zeros(n_rows), labels, groups=groups)
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}.")


def cross_validate_model(
    estimator: BaseEstimator,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    strategy: str = "grouped",
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    top_k: int = 3,
) -> dict[str, float]:
    """Cross-validate a model and return mean/std for every metric.

    The estimator is cloned per fold, so any resampling step inside a pipeline is
    refitted on the training part only.
    """
    n_classes = int(len(np.unique(labels)))
    per_fold: list[dict[str, float]] = []

    for train_idx, test_idx in iter_folds(
        labels, strategy=strategy, groups=groups, n_splits=n_splits, random_state=random_state
    ):
        model = clone(estimator)
        model.fit(features[train_idx], labels[train_idx])
        y_pred = model.predict(features[test_idx])
        y_proba = model.predict_proba(features[test_idx]) if hasattr(model, "predict_proba") else None
        per_fold.append(
            classification_metrics(
                labels[test_idx], y_pred, y_proba, n_classes=n_classes, top_k=top_k
            )
        )

    summary: dict[str, float] = {"n_splits": float(len(per_fold))}
    for key in per_fold[0]:
        values = np.array([fold[key] for fold in per_fold], dtype=float)
        summary[f"{key}_mean"] = round(float(np.nanmean(values)), 6)
        summary[f"{key}_std"] = round(float(np.nanstd(values)), 6)
    return summary


def compare_split_protocols(
    estimator: BaseEstimator,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    top_k: int = 3,
) -> pd.DataFrame:
    """Score the same model under both protocols; the gap is the leakage estimate.

    Returns a two-row frame indexed by protocol. A large ``accuracy_mean`` drop
    from ``random`` to ``grouped`` means the random split was scoring the model's
    memory rather than its generalisation.
    """
    rows = []
    for strategy in ("random", "grouped"):
        summary = cross_validate_model(
            estimator,
            features,
            labels,
            strategy=strategy,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
            top_k=top_k,
        )
        rows.append({"protocol": strategy, **summary})
    frame = pd.DataFrame(rows).set_index("protocol")
    frame.loc["leakage_gap"] = frame.loc["random"] - frame.loc["grouped"]
    return frame


def perturb_symptoms(
    features: np.ndarray,
    *,
    dropout_rate: float = 0.0,
    false_positive_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate imperfect symptom reporting.

    Args:
        features: Binary matrix to perturb; never modified in place.
        dropout_rate: Probability that a present symptom is not reported.
        false_positive_rate: Probability that an absent symptom is reported.
        rng: Generator, for reproducible perturbations.

    Raises:
        ValueError: If either rate falls outside ``[0, 1]``.
    """
    for name, rate in (("dropout_rate", dropout_rate), ("false_positive_rate", false_positive_rate)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {rate}.")

    rng = rng or np.random.default_rng(0)
    perturbed = np.asarray(features).copy()
    present = perturbed == 1
    absent = ~present
    if dropout_rate:
        perturbed[present & (rng.random(perturbed.shape) < dropout_rate)] = 0
    if false_positive_rate:
        perturbed[absent & (rng.random(perturbed.shape) < false_positive_rate)] = 1
    return perturbed


def robustness_curve(
    model: BaseEstimator,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    dropout_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
    false_positive_rates: Sequence[float] = (0.0, 0.01, 0.03, 0.05),
    n_repeats: int = 5,
    random_state: int = 42,
    top_k: int = 3,
) -> pd.DataFrame:
    """Measure accuracy as reported symptoms get noisier.

    A clean-data score says how the model behaves on the benchmark; this says how
    it behaves on the input a real user would type. Each cell of the grid is
    averaged over ``n_repeats`` random perturbations.

    Returns:
        Long-format frame with one row per (dropout, false-positive) cell.
    """
    rows: list[dict[str, float]] = []
    n_classes = int(len(np.unique(labels)))
    has_proba = hasattr(model, "predict_proba")

    for dropout in dropout_rates:
        for false_positive in false_positive_rates:
            accuracies: list[float] = []
            topk: list[float] = []
            for repeat in range(n_repeats):
                rng = np.random.default_rng(random_state + repeat)
                noisy = perturb_symptoms(
                    features,
                    dropout_rate=dropout,
                    false_positive_rate=false_positive,
                    rng=rng,
                )
                y_pred = model.predict(noisy)
                accuracies.append(float(accuracy_score(labels, y_pred)))
                if has_proba:
                    topk.append(top_k_accuracy(labels, model.predict_proba(noisy), k=top_k))
            rows.append(
                {
                    "dropout_rate": dropout,
                    "false_positive_rate": false_positive,
                    "accuracy_mean": round(float(np.mean(accuracies)), 6),
                    "accuracy_std": round(float(np.std(accuracies)), 6),
                    f"top_{top_k}_accuracy_mean": round(float(np.mean(topk)), 6) if topk else float("nan"),
                    "n_repeats": n_repeats,
                    "n_classes": n_classes,
                }
            )
    return pd.DataFrame(rows)


def benchmark_models(
    model_names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    build: Callable[[str], BaseEstimator] | None = None,
    strategy: str = "grouped",
    n_splits: int = 5,
    random_state: int = 42,
    top_k: int = 3,
) -> pd.DataFrame:
    """Cross-validate several models under one protocol and rank them.

    Args:
        model_names: Registry keys to compare.
        build: Factory for an estimator given a name; defaults to
            :func:`disease_predictor.models.build_pipeline`.

    Returns:
        Frame indexed by model, sorted by mean accuracy, with a ``fit_seconds``
        column so the accuracy/cost trade-off is visible.
    """
    import time

    if build is None:
        from disease_predictor.models import build_pipeline

        def build(name: str) -> BaseEstimator:  # type: ignore[misc]
            return build_pipeline(name, random_state=random_state)

    rows = []
    for name in model_names:
        started = time.perf_counter()
        summary = cross_validate_model(
            build(name),
            features,
            labels,
            strategy=strategy,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
            top_k=top_k,
        )
        rows.append({"model": name, **summary, "fit_seconds": round(time.perf_counter() - started, 3)})

    return pd.DataFrame(rows).set_index("model").sort_values("accuracy_mean", ascending=False)


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Per-disease precision/recall/F1 as a dataframe, sorted by weakest recall."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows = {k: v for k, v in report.items() if isinstance(v, dict) and k in set(class_names)}
    frame = pd.DataFrame(rows).T
    frame.index.name = "disease"
    return frame.sort_values(["recall", "f1-score"])


def confusion_pairs(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], *, top_n: int = 10
) -> pd.DataFrame:
    """The most frequent (true -> predicted) mistakes, which is where insight lives."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(matrix, 0)
    rows = []
    for true_idx, pred_idx in zip(*np.nonzero(matrix), strict=True):
        rows.append(
            {
                "true_disease": class_names[true_idx],
                "predicted_disease": class_names[pred_idx],
                "count": int(matrix[true_idx, pred_idx]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["true_disease", "predicted_disease", "count"])
    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
