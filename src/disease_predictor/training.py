"""The end-to-end training run.

One call to :func:`train` takes a configuration and produces everything needed
to defend the result: a versioned model bundle, a machine-readable metrics file,
a human-readable markdown report, and the leakage/robustness tables that explain
what the headline number does and does not mean.

Nothing here prints a score without also recording the protocol that produced
it, which is the whole reason this module exists as a script-free, importable
function rather than as notebook cells.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from disease_predictor.__about__ import __version__
from disease_predictor.config import BUNDLE_FILENAME, TrainingConfig
from disease_predictor.data import SymptomDataset, audit_duplicates, load_dataset, pattern_groups
from disease_predictor.evaluation import (
    benchmark_models,
    classification_metrics,
    compare_split_protocols,
    confusion_pairs,
    make_split,
    per_class_report,
    robustness_curve,
)
from disease_predictor.explain import global_importance
from disease_predictor.features import SymptomEncoder
from disease_predictor.models import available_models, build_pipeline

LOGGER = logging.getLogger(__name__)

__all__ = ["ModelBundle", "TrainingResult", "train", "save_bundle", "load_bundle"]


@dataclass
class ModelBundle:
    """Everything inference needs, versioned together.

    Saving the model without the label encoder and the feature order is the
    single most common way a working notebook becomes a broken service: at
    inference time the columns silently reorder and the predictions become
    noise. Bundling them makes that mismatch impossible.
    """

    model: BaseEstimator
    label_encoder: LabelEncoder
    encoder: SymptomEncoder
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.encoder.feature_names

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.label_encoder.classes_)


@dataclass
class TrainingResult:
    """Artifacts and tables produced by a training run."""

    bundle: ModelBundle
    metrics: dict[str, Any]
    leakage_audit: dict[str, Any]
    protocol_comparison: pd.DataFrame
    robustness: pd.DataFrame
    benchmark: pd.DataFrame
    importance: pd.DataFrame
    per_class: pd.DataFrame
    confusions: pd.DataFrame
    bundle_path: Path
    report_paths: dict[str, Path] = field(default_factory=dict)


def _environment() -> dict[str, str]:
    """Record the library versions a reported metric depends on."""
    versions = {
        "disease_predictor": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return versions


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Serialise a bundle to disk, creating parent directories as needed."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle written by :func:`save_bundle`.

    Raises:
        FileNotFoundError: If no bundle exists at ``path``.
        TypeError: If the file does not contain a :class:`ModelBundle`.
    """
    import joblib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No model bundle at {path}. Train one first: `dp train` (add --synthetic to try it without data)."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle (got {type(bundle).__name__}).")
    return bundle


def _write_report(result_tables: dict[str, pd.DataFrame], metrics: dict[str, Any], path: Path) -> Path:
    """Render the markdown run report."""
    lines: list[str] = []
    add = lines.append

    add("# Training report")
    add("")
    add(f"Generated {metrics['created_at']} by disease-predictor {metrics['environment']['disease_predictor']}.")
    add("")
    add("> Educational project. Not a medical device and not medical advice.")
    add("")

    add("## Run configuration")
    add("")
    config = metrics["config"]
    add(f"- Model: `{config['model_name']}`")
    add(f"- Sampler: `{config['sampler']}`")
    add(f"- Deduplicated: `{config['deduplicate']}`")
    add(f"- Split protocol: `{config['split']['strategy']}` (test_size={config['split']['test_size']})")
    add(f"- Seed: `{config['split']['random_state']}`")
    add("")

    add("## Dataset")
    add("")
    for key, value in metrics["dataset"].items():
        add(f"- {key}: `{value}`")
    add("")

    add("## Leakage audit")
    add("")
    add(
        "Repeated rows make a random split score memorisation. These numbers "
        "quantify how much of the dataset is repetition."
    )
    add("")
    for key, value in metrics["leakage_audit"].items():
        add(f"- {key}: `{value}`")
    add("")

    add("## Held-out performance")
    add("")
    add("| metric | value |")
    add("| --- | --- |")
    for key, value in metrics["holdout"].items():
        add(f"| {key} | {value} |")
    add("")

    for title, description, table in (
        (
            "Split-protocol comparison (raw data, duplicates included)",
            "The same model on the same raw rows, scored two ways. The "
            "`leakage_gap` row is how much score a random split hands over for "
            "free by putting identical rows on both sides.",
            result_tables["protocol_comparison"],
        ),
        (
            "Model benchmark",
            "Every model under the honest protocol. If the rule-based baseline "
            "keeps pace, the extra machinery is not earning its complexity.",
            result_tables["benchmark"],
        ),
        (
            "Robustness to imperfect symptom reporting",
            "Accuracy when reported symptoms are dropped or spuriously added, "
            "which is the input a real user produces.",
            result_tables["robustness"],
        ),
        (
            "Most influential symptoms",
            "Global attribution over the trained model.",
            result_tables["importance"],
        ),
        (
            "Weakest classes",
            "Diseases with the lowest recall: where the model actually fails.",
            result_tables["per_class"].head(10),
        ),
        (
            "Most common confusions",
            "Which diseases get mistaken for which.",
            result_tables["confusions"],
        ),
    ):
        add(f"## {title}")
        add("")
        add(description)
        add("")
        add(table.to_markdown() if not table.empty else "_No rows._")
        add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train(
    config: TrainingConfig | None = None,
    *,
    dataset: SymptomDataset | None = None,
    synthetic: bool = False,
    run_benchmark: bool = True,
    run_robustness: bool = True,
) -> TrainingResult:
    """Train a model and produce every artifact needed to justify its score.

    Args:
        config: Run configuration; defaults to :class:`TrainingConfig`.
        dataset: Pre-loaded dataset, mainly for tests. Loaded from
            ``config.data_path`` when omitted.
        synthetic: Train on generated data instead of the real file. Reports
            produced this way are clearly marked as synthetic.
        run_benchmark: Also cross-validate every other model in the registry.
        run_robustness: Also compute the symptom-noise robustness grid.

    Returns:
        A :class:`TrainingResult` holding the bundle, the metrics and the tables.

    Raises:
        DatasetNotFoundError: If no dataset can be resolved.
    """
    config = config or TrainingConfig()
    started = time.perf_counter()

    if dataset is None:
        dataset = load_dataset(config.data_path, synthetic=synthetic)
    LOGGER.info("Loaded dataset: %s", dataset.describe())

    leakage_audit = audit_duplicates(dataset)
    LOGGER.info(
        "Leakage audit: %.1f%% of rows are exact duplicates; %d unique symptom patterns.",
        100 * float(leakage_audit["duplicate_row_fraction"]),
        leakage_audit["n_unique_patterns"],
    )

    working = dataset.deduplicated() if config.deduplicate else dataset
    features = working.features
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(working.labels)
    groups = pattern_groups(features)
    class_names = tuple(str(c) for c in label_encoder.classes_)

    split = make_split(
        features,
        labels,
        strategy=config.split.strategy,
        groups=groups,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
    )
    LOGGER.info("Split (%s): %d train / %d test rows.", split.strategy, split.n_train, split.n_test)

    estimator = build_pipeline(
        config.model_name,
        sampler=config.sampler,
        random_state=config.split.random_state,
    )
    estimator.fit(features[split.train], labels[split.train])

    y_pred = estimator.predict(features[split.test])
    y_proba = estimator.predict_proba(features[split.test])
    holdout = classification_metrics(
        labels[split.test], y_pred, y_proba, n_classes=len(class_names), top_k=config.top_k
    )

    # The protocol comparison deliberately runs on the RAW dataset, duplicates
    # included. Comparing protocols after deduplication would measure nothing:
    # the leak being quantified is exactly what deduplication removes.
    raw_features = dataset.features
    raw_labels = LabelEncoder().fit_transform(dataset.labels)
    protocol_comparison = compare_split_protocols(
        build_pipeline(config.model_name, sampler=config.sampler, random_state=config.split.random_state),
        raw_features,
        raw_labels,
        pattern_groups(raw_features),
        n_splits=config.split.n_splits,
        random_state=config.split.random_state,
        top_k=config.top_k,
    )

    benchmark = pd.DataFrame()
    if run_benchmark:
        benchmark = benchmark_models(
            available_models(),
            features,
            labels,
            groups,
            strategy=config.split.strategy,
            n_splits=config.split.n_splits,
            random_state=config.split.random_state,
            top_k=config.top_k,
        )

    robustness = pd.DataFrame()
    if run_robustness:
        robustness = robustness_curve(
            estimator,
            features[split.test],
            labels[split.test],
            dropout_rates=config.robustness.dropout_rates,
            false_positive_rates=config.robustness.false_positive_rates,
            n_repeats=config.robustness.n_repeats,
            random_state=config.robustness.random_state,
            top_k=config.top_k,
        )

    importance = global_importance(
        estimator,
        working.feature_names,
        features=features[split.test],
        labels=labels[split.test],
        top_n=20,
        random_state=config.split.random_state,
    )
    per_class = per_class_report(labels[split.test], y_pred, class_names)
    confusions = confusion_pairs(labels[split.test], y_pred, class_names)

    bundle = ModelBundle(
        model=estimator,
        label_encoder=label_encoder,
        encoder=SymptomEncoder(working.feature_names),
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config.to_dict(),
            "dataset": working.describe(),
            "data_fingerprint": working.fingerprint(),
            "leakage_audit": leakage_audit,
            "holdout": holdout,
            "environment": _environment(),
            "synthetic": synthetic,
            "train_seconds": round(time.perf_counter() - started, 3),
        },
    )

    metrics: dict[str, Any] = {
        "created_at": bundle.metadata["created_at"],
        "config": config.to_dict(),
        "dataset": working.describe(),
        "synthetic": synthetic,
        "leakage_audit": leakage_audit,
        "holdout": holdout,
        "protocol_comparison": json.loads(protocol_comparison.to_json(orient="index")),
        "benchmark": json.loads(benchmark.to_json(orient="index")) if not benchmark.empty else {},
        "environment": _environment(),
        "train_seconds": bundle.metadata["train_seconds"],
    }

    bundle_path = save_bundle(bundle, Path(config.artifact_dir) / BUNDLE_FILENAME)

    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    protocol_comparison.to_csv(report_dir / "protocol_comparison.csv")
    if not robustness.empty:
        robustness.to_csv(report_dir / "robustness.csv", index=False)
    if not benchmark.empty:
        benchmark.to_csv(report_dir / "benchmark.csv")
    importance.to_csv(report_dir / "symptom_importance.csv", index=False)

    report_path = _write_report(
        {
            "protocol_comparison": protocol_comparison[
                [c for c in protocol_comparison.columns if c.endswith("_mean")]
            ].round(4),
            "benchmark": benchmark[
                [c for c in benchmark.columns if c.endswith("_mean") or c == "fit_seconds"]
            ].round(4)
            if not benchmark.empty
            else benchmark,
            "robustness": robustness,
            "importance": importance,
            "per_class": per_class.round(3),
            "confusions": confusions,
        },
        metrics,
        report_dir / "training_report.md",
    )

    LOGGER.info("Wrote bundle to %s and report to %s", bundle_path, report_path)

    return TrainingResult(
        bundle=bundle,
        metrics=metrics,
        leakage_audit=leakage_audit,
        protocol_comparison=protocol_comparison,
        robustness=robustness,
        benchmark=benchmark,
        importance=importance,
        per_class=per_class,
        confusions=confusions,
        bundle_path=bundle_path,
        report_paths={
            "metrics": report_dir / "metrics.json",
            "report": report_path,
        },
    )
