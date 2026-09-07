"""Dataset loading, schema validation and duplicate/leakage auditing.

The public symptom -> prognosis dataset this project was built on contains a
large number of *repeated rows*: the same binary symptom vector appears many
times under the same label. A plain random train/test split therefore places
byte-identical rows on both sides, and any model that can memorise scores a
perfect 100%.

Every helper in this module exists to make that fact measurable rather than
invisible: :func:`audit_duplicates` quantifies the repetition, and
:func:`pattern_groups` produces group ids that let the evaluation code keep
identical rows on one side of a split.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from disease_predictor.config import (
    DATA_PATH_ENV_VAR,
    DEFAULT_DATA_DIR,
    DEFAULT_DATASET_FILENAME,
    TARGET_COLUMN,
)

__all__ = [
    "SymptomDataset",
    "DatasetNotFoundError",
    "SchemaError",
    "canonical_symptom",
    "load_dataset",
    "read_symptom_frame",
    "audit_duplicates",
    "pattern_groups",
    "make_synthetic_dataset",
]

_DOWNLOAD_HINT = (
    "Place the symptom dataset at '{expected}', pass --data /path/to/Training.csv, "
    f"or set the {DATA_PATH_ENV_VAR} environment variable. "
    "The dataset used by this project is the public 'Disease Symptom Prediction' "
    "table (4920 rows x 132 binary symptoms x 41 prognoses) available on Kaggle. "
    "To explore the pipeline without it, use --synthetic."
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no dataset can be resolved from the given hints."""


class SchemaError(ValueError):
    """Raised when a dataset does not look like a symptom/prognosis table."""


def canonical_symptom(name: str) -> str:
    """Normalise a symptom name to a stable snake_case key.

    The published dataset has inconsistent headers -- stray spaces inside
    words (``dischromic _patches``), trailing spaces (``fluid_overload ``) and
    mixed separators. Normalising once, here, keeps every downstream component
    (encoder, API, UI) agreed on what a symptom is called.

    >>> canonical_symptom("dischromic _patches")
    'dischromic_patches'
    >>> canonical_symptom("  Foul smell of urine ")
    'foul_smell_of_urine'
    """
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def _dedupe_columns(columns: Sequence[str]) -> list[str]:
    """Make column names unique by suffixing repeats, preserving order."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for column in columns:
        if column in seen:
            seen[column] += 1
            out.append(f"{column}_{seen[column]}")
        else:
            seen[column] = 0
            out.append(column)
    return out


@dataclass(frozen=True)
class SymptomDataset:
    """A validated symptom/prognosis table plus its derived arrays.

    Attributes:
        frame: The cleaned dataframe, features plus the target column.
        feature_names: Canonical symptom names, in model input order.
        target_column: Name of the label column.
        source: Human-readable description of where the data came from.
    """

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_column: str = TARGET_COLUMN
    source: str = "unknown"

    @property
    def features(self) -> np.ndarray:
        """Binary feature matrix of shape ``(n_rows, n_symptoms)``."""
        return self.frame[list(self.feature_names)].to_numpy(dtype=np.int8)

    @property
    def labels(self) -> np.ndarray:
        """Raw (string) disease labels of shape ``(n_rows,)``."""
        return self.frame[self.target_column].to_numpy()

    @property
    def classes(self) -> tuple[str, ...]:
        """Sorted unique disease labels."""
        return tuple(sorted(pd.unique(self.frame[self.target_column]).tolist()))

    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def fingerprint(self) -> str:
        """Stable SHA-256 digest of the data, recorded in the model bundle.

        Two runs that report the same fingerprint were trained on byte-identical
        inputs, which is what makes a reported metric reproducible.
        """
        digest = hashlib.sha256()
        digest.update(",".join(self.feature_names).encode("utf-8"))
        digest.update(self.features.tobytes())
        digest.update("|".join(map(str, self.labels)).encode("utf-8"))
        return digest.hexdigest()

    def deduplicated(self) -> SymptomDataset:
        """Return a copy with exact duplicate rows collapsed to one.

        This is the honest training set: without it, repeated rows silently
        reweight the loss towards whichever patterns happen to be repeated most.
        """
        columns = [*self.feature_names, self.target_column]
        frame = self.frame.drop_duplicates(subset=columns).reset_index(drop=True)
        return SymptomDataset(
            frame=frame,
            feature_names=self.feature_names,
            target_column=self.target_column,
            source=f"{self.source} (deduplicated)",
        )

    def describe(self) -> dict[str, object]:
        """Summary suitable for logging or embedding in a report."""
        counts = self.frame[self.target_column].value_counts()
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "rows_per_class_min": int(counts.min()),
            "rows_per_class_max": int(counts.max()),
            "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
            "fingerprint": self.fingerprint()[:16],
        }


def _resolve_path(data_path: str | Path | None) -> Path:
    """Find the dataset from an explicit path, the env var, or the default dir."""
    candidates: list[Path] = []
    if data_path is not None:
        candidates.append(Path(data_path).expanduser())
    env_value = os.environ.get(DATA_PATH_ENV_VAR)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(DEFAULT_DATA_DIR / DEFAULT_DATASET_FILENAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected = candidates[0] if data_path is not None else DEFAULT_DATA_DIR / DEFAULT_DATASET_FILENAME
    raise DatasetNotFoundError(_DOWNLOAD_HINT.format(expected=expected))


def read_symptom_frame(frame: pd.DataFrame, *, target_column: str = TARGET_COLUMN) -> SymptomDataset:
    """Validate and clean an in-memory symptom table.

    Performs the fixes the raw file needs: drops the unnamed trailing column
    pandas invents from a stray comma, normalises headers, removes all-empty
    columns, and asserts that every feature really is binary.

    Raises:
        SchemaError: If the target column is missing, no usable features remain,
            or a feature column contains values other than 0/1.
    """
    if target_column not in frame.columns:
        raise SchemaError(
            f"Expected a {target_column!r} column; found {list(frame.columns)[:8]}..."
        )

    frame = frame.copy()
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    empty = [c for c in frame.columns if c != target_column and frame[c].isna().all()]
    frame = frame.drop(columns=list(dict.fromkeys([*unnamed, *empty])))

    labels = frame[target_column].astype(str).str.strip()
    feature_frame = frame.drop(columns=[target_column])
    feature_frame.columns = _dedupe_columns([canonical_symptom(c) for c in feature_frame.columns])

    if feature_frame.shape[1] == 0:
        raise SchemaError("No symptom columns left after cleaning; is this the right file?")

    non_numeric = [c for c in feature_frame.columns if not pd.api.types.is_numeric_dtype(feature_frame[c])]
    if non_numeric:
        raise SchemaError(f"Symptom columns must be numeric; offenders: {non_numeric[:5]}")

    feature_frame = feature_frame.fillna(0)
    values = feature_frame.to_numpy()
    if not np.isin(values, (0, 1)).all():
        bad = sorted(set(np.unique(values).tolist()) - {0, 1})[:5]
        raise SchemaError(f"Symptom columns must be binary 0/1; found values such as {bad}.")

    feature_frame = feature_frame.astype(np.int8)
    feature_names = tuple(feature_frame.columns)
    cleaned = feature_frame.copy()
    cleaned[target_column] = labels.to_numpy()

    return SymptomDataset(
        frame=cleaned.reset_index(drop=True),
        feature_names=feature_names,
        target_column=target_column,
    )


def load_dataset(
    data_path: str | Path | None = None,
    *,
    synthetic: bool = False,
    target_column: str = TARGET_COLUMN,
) -> SymptomDataset:
    """Load the symptom dataset.

    Args:
        data_path: Explicit path or URL. When ``None`` the loader falls back to
            the ``DISEASE_DATA_PATH`` environment variable and then to
            ``data/raw/Training.csv``.
        synthetic: When true, ignore ``data_path`` and generate a clearly
            labelled synthetic dataset instead. This keeps the test suite, CI
            and the demo runnable on a machine that has never seen the real file.
        target_column: Name of the label column.

    Raises:
        DatasetNotFoundError: If no dataset could be resolved.
        SchemaError: If the resolved file is not a symptom/prognosis table.
    """
    if synthetic:
        return make_synthetic_dataset()

    if data_path is not None and str(data_path).startswith(("http://", "https://")):
        frame = pd.read_csv(str(data_path))
        dataset = read_symptom_frame(frame, target_column=target_column)
        return SymptomDataset(
            frame=dataset.frame,
            feature_names=dataset.feature_names,
            target_column=target_column,
            source=str(data_path),
        )

    path = _resolve_path(data_path)
    frame = pd.read_csv(path)
    dataset = read_symptom_frame(frame, target_column=target_column)
    return SymptomDataset(
        frame=dataset.frame,
        feature_names=dataset.feature_names,
        target_column=target_column,
        source=str(path),
    )


def pattern_groups(features: np.ndarray) -> np.ndarray:
    """Assign an integer group id to every distinct symptom pattern.

    Rows that share a group are byte-identical in feature space. Passing these
    ids to a grouped cross-validator is what prevents a memorised row from being
    scored as a correct generalisation.

    Returns:
        Array of shape ``(n_rows,)`` with values in ``[0, n_unique_patterns)``.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix; got shape {features.shape}.")
    _, group_ids = np.unique(features, axis=0, return_inverse=True)
    return group_ids.astype(np.int64).ravel()


def audit_duplicates(dataset: SymptomDataset) -> dict[str, object]:
    """Quantify how much of the dataset is repetition rather than information.

    Returns a dictionary with:

    ``duplicate_row_fraction``
        Share of rows that are an exact repeat (features *and* label) of an
        earlier row. This is the number that explains a suspicious 100%.
    ``n_unique_patterns``
        Distinct symptom vectors, ignoring the label. The effective sample size
        of the problem.
    ``rows_per_unique_pattern``
        How many times the average pattern is repeated.
    ``ambiguous_patterns``
        Symptom vectors that appear under more than one disease -- the
        irreducible error floor. Zero here means the mapping is a lookup table.
    ``leakage_risk``
        Probability that a randomly chosen row has a twin elsewhere in the
        table, i.e. the share of a random test split that a memorising model
        gets for free.
    """
    features = dataset.features
    labels = dataset.labels
    n_rows = dataset.n_rows

    full = pd.DataFrame(features, columns=list(dataset.feature_names))
    full[dataset.target_column] = labels

    n_unique_rows = int(len(full.drop_duplicates()))
    duplicate_rows = n_rows - n_unique_rows

    groups = pattern_groups(features)
    n_unique_patterns = int(len(np.unique(groups)))

    pattern_frame = pd.DataFrame({"group": groups, "label": labels})
    labels_per_pattern = pattern_frame.groupby("group")["label"].nunique()
    ambiguous = labels_per_pattern[labels_per_pattern > 1]

    group_sizes = pattern_frame.groupby("group").size()
    repeated_rows = int(group_sizes[group_sizes > 1].sum())

    symptoms_per_row = features.sum(axis=1)

    return {
        "n_rows": n_rows,
        "n_unique_rows": n_unique_rows,
        "n_duplicate_rows": duplicate_rows,
        "duplicate_row_fraction": round(duplicate_rows / n_rows, 4) if n_rows else 0.0,
        "n_unique_patterns": n_unique_patterns,
        "rows_per_unique_pattern": round(n_rows / n_unique_patterns, 2) if n_unique_patterns else 0.0,
        "patterns_per_class": round(n_unique_patterns / dataset.n_classes, 2) if dataset.n_classes else 0.0,
        "n_ambiguous_patterns": int(len(ambiguous)),
        "ambiguous_pattern_fraction": round(len(ambiguous) / n_unique_patterns, 4)
        if n_unique_patterns
        else 0.0,
        "leakage_risk": round(repeated_rows / n_rows, 4) if n_rows else 0.0,
        "mean_symptoms_per_row": round(float(symptoms_per_row.mean()), 2),
        "min_symptoms_per_row": int(symptoms_per_row.min()) if n_rows else 0,
        "max_symptoms_per_row": int(symptoms_per_row.max()) if n_rows else 0,
    }


def make_synthetic_dataset(
    *,
    n_diseases: int = 12,
    n_symptoms: int = 48,
    rows_per_disease: int = 60,
    signature_size: int = 5,
    dropout_rate: float = 0.15,
    false_positive_rate: float = 0.02,
    duplicate_rate: float = 0.35,
    random_state: int = 7,
) -> SymptomDataset:
    """Generate a synthetic symptom table with the same failure mode as the real one.

    Each synthetic disease gets a signature set of symptoms; rows are noisy
    observations of that signature, and a configurable share of rows are exact
    duplicates so the leakage audit has something to find. Names are prefixed
    with ``synthetic_`` so that a synthetic run can never be mistaken for a real
    one in a report.

    Args:
        n_diseases: Number of distinct labels.
        n_symptoms: Number of binary feature columns.
        rows_per_disease: Rows generated per label before duplication.
        signature_size: Symptoms that characterise each disease.
        dropout_rate: Chance a signature symptom is missing from a row.
        false_positive_rate: Chance a non-signature symptom is present anyway.
        duplicate_rate: Share of rows replaced by copies of other rows.
        random_state: Seed for reproducibility.
    """
    if signature_size > n_symptoms:
        raise ValueError("signature_size cannot exceed n_symptoms.")
    rng = np.random.default_rng(random_state)

    feature_names = tuple(f"synthetic_symptom_{i:03d}" for i in range(n_symptoms))
    disease_names = [f"Synthetic Condition {i:02d}" for i in range(n_diseases)]

    signatures = [
        rng.choice(n_symptoms, size=signature_size, replace=False) for _ in range(n_diseases)
    ]

    rows: list[np.ndarray] = []
    labels: list[str] = []
    for disease_index, signature in enumerate(signatures):
        for _ in range(rows_per_disease):
            row = np.zeros(n_symptoms, dtype=np.int8)
            kept = signature[rng.random(signature_size) >= dropout_rate]
            if kept.size == 0:  # never emit an all-zero row
                kept = signature[:1]
            row[kept] = 1
            noise = rng.random(n_symptoms) < false_positive_rate
            row[noise] = 1
            rows.append(row)
            labels.append(disease_names[disease_index])

    features = np.vstack(rows)
    label_array = np.array(labels)

    n_duplicated = int(len(features) * duplicate_rate)
    if n_duplicated:
        targets = rng.choice(len(features), size=n_duplicated, replace=False)
        sources = rng.choice(len(features), size=n_duplicated, replace=True)
        features[targets] = features[sources]
        label_array[targets] = label_array[sources]

    frame = pd.DataFrame(features, columns=list(feature_names)).astype(np.int8)
    frame[TARGET_COLUMN] = label_array

    order = rng.permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)

    return SymptomDataset(
        frame=frame,
        feature_names=feature_names,
        target_column=TARGET_COLUMN,
        source="synthetic (generated, not real clinical data)",
    )


def summarise_class_signatures(
    dataset: SymptomDataset, *, min_prevalence: float = 0.5
) -> dict[str, list[str]]:
    """Return, per disease, the symptoms present in at least ``min_prevalence`` of rows.

    Useful both as a sanity check on the data and as the basis of the
    rule-based baseline in :mod:`disease_predictor.models`.
    """
    frame = dataset.frame
    signatures: dict[str, list[str]] = {}
    for label, group in frame.groupby(dataset.target_column):
        prevalence = group[list(dataset.feature_names)].mean()
        signatures[str(label)] = sorted(prevalence[prevalence >= min_prevalence].index.tolist())
    return signatures


def iter_symptom_names(names: Iterable[str]) -> list[str]:
    """Canonicalise an iterable of symptom names, dropping blanks."""
    return [canonical_symptom(n) for n in names if str(n).strip()]
