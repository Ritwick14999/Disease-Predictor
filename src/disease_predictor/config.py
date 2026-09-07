"""Typed configuration for training and evaluation runs.

Everything that changes the outcome of a run lives here, so a run can be
reproduced from a single serialisable object rather than from scattered
notebook cells.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Default locations are relative to the working directory rather than to this
# file. Deriving them from ``__file__`` breaks the moment the package is
# installed as a wheel instead of in editable mode, because the path then points
# inside site-packages. Every default here can be overridden by a CLI flag or an
# environment variable.
DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("reports")

#: Filename the loader looks for when no explicit dataset path is given.
DEFAULT_DATASET_FILENAME = "Training.csv"

#: Name of the label column in the symptom/prognosis dataset.
TARGET_COLUMN = "prognosis"

#: Filename of the serialised model bundle inside the artifact directory.
BUNDLE_FILENAME = "disease_predictor_bundle.joblib"

#: Environment variable that can point at the dataset instead of a CLI flag.
DATA_PATH_ENV_VAR = "DISEASE_DATA_PATH"

#: Environment variable used by the API to locate a trained bundle.
BUNDLE_PATH_ENV_VAR = "DISEASE_MODEL_PATH"

#: Resampling strategies available for class imbalance.
#:
#: ``none``
#:     No resampling. The right default here: the published dataset is already
#:     exactly balanced, so oversampling is a no-op that only costs runtime.
#: ``random``
#:     Random oversampling of minority classes. Safe for binary features because
#:     it copies real rows.
#: ``smote``
#:     Synthetic interpolation between neighbours. Kept for comparison, but see
#:     ``docs/MODEL_CARD.md``: interpolating binary symptom indicators invents
#:     fractional symptoms ("0.4 of a fever") that cannot occur at inference.
SAMPLERS = ("none", "random", "smote")

DISCLAIMER = (
    "Educational project only. These predictions are not medical advice and "
    "must not be used for diagnosis or treatment decisions."
)


@dataclass(frozen=True)
class SplitConfig:
    """How rows are divided into training and evaluation folds.

    ``strategy`` is the interesting knob:

    ``random``
        The textbook stratified split. On a dataset with repeated rows this
        leaks: an identical row can sit in both train and test.
    ``grouped``
        Rows sharing an identical symptom pattern form a group and are kept on
        the same side of the split, so the score measures generalisation to
        unseen symptom patterns rather than recall of memorised rows.
    """

    strategy: str = "grouped"
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.strategy not in {"random", "grouped"}:
            raise ValueError(
                f"Unknown split strategy {self.strategy!r}; expected 'random' or 'grouped'."
            )
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1); got {self.test_size}.")
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {self.n_splits}.")


@dataclass(frozen=True)
class RobustnessConfig:
    """Grid for the symptom-noise robustness benchmark.

    ``dropout_rates``
        Probability that a symptom the patient really has goes unreported.
    ``false_positive_rates``
        Probability that an absent symptom is reported anyway.
    """

    dropout_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    false_positive_rates: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)
    n_repeats: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Full description of a training run."""

    model_name: str = "xgboost"
    sampler: str = "none"
    deduplicate: bool = True
    top_k: int = 3
    split: SplitConfig = field(default_factory=SplitConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    data_path: str | None = None
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR)
    report_dir: str = str(DEFAULT_REPORT_DIR)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}.")
        if self.sampler not in SAMPLERS:
            raise ValueError(
                f"Unknown sampler {self.sampler!r}; expected one of {sorted(SAMPLERS)}."
            )

    @property
    def bundle_path(self) -> Path:
        """Where the trained bundle will be written."""
        return Path(self.artifact_dir) / BUNDLE_FILENAME

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        """Rebuild a configuration from :meth:`to_dict` output."""
        payload = dict(payload)
        if isinstance(payload.get("split"), dict):
            payload["split"] = SplitConfig(**payload["split"])
        if isinstance(payload.get("robustness"), dict):
            robustness = dict(payload["robustness"])
            for key in ("dropout_rates", "false_positive_rates"):
                if key in robustness:
                    robustness[key] = tuple(robustness[key])
            payload["robustness"] = RobustnessConfig(**robustness)
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Load a configuration from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
