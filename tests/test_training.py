"""Tests for the end-to-end training run and its artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disease_predictor.config import RobustnessConfig, SplitConfig, TrainingConfig
from disease_predictor.training import train


def test_training_produces_a_usable_bundle(trained) -> None:
    bundle = trained.bundle
    assert bundle.feature_names
    assert bundle.class_names
    assert len(bundle.feature_names) == len(bundle.encoder)


def test_bundle_is_written_to_the_configured_directory(trained, training_config) -> None:
    assert trained.bundle_path == Path(training_config.artifact_dir) / trained.bundle_path.name
    assert trained.bundle_path.is_file()


def test_metrics_json_is_written_and_parseable(trained) -> None:
    payload = json.loads(trained.report_paths["metrics"].read_text(encoding="utf-8"))
    assert payload["holdout"]["accuracy"] == trained.metrics["holdout"]["accuracy"]
    assert payload["leakage_audit"]["n_rows"] > 0


def test_markdown_report_documents_the_protocol(trained) -> None:
    report = trained.report_paths["report"].read_text(encoding="utf-8")
    assert "Leakage audit" in report
    assert "Split-protocol comparison" in report
    assert "not medical advice" in report.lower()


def test_metadata_records_reproducibility_information(trained) -> None:
    metadata = trained.bundle.metadata
    assert metadata["data_fingerprint"]
    assert metadata["environment"]["scikit_learn"]
    assert metadata["config"]["split"]["strategy"] == "grouped"
    assert metadata["train_seconds"] >= 0


def test_holdout_metrics_are_in_range(trained) -> None:
    holdout = trained.metrics["holdout"]
    assert 0.0 <= holdout["accuracy"] <= 1.0
    assert 0.0 <= holdout["macro_f1"] <= 1.0
    assert holdout["top_3_accuracy"] >= holdout["accuracy"]


def test_protocol_comparison_is_computed_on_raw_rows(trained, dataset) -> None:
    """The gap can only be measured before duplicates are removed."""
    assert list(trained.protocol_comparison.index) == ["random", "grouped", "leakage_gap"]
    assert trained.metrics["leakage_audit"]["n_rows"] == dataset.n_rows


def test_deduplication_shrinks_the_modelling_set(trained, dataset) -> None:
    assert trained.metrics["dataset"]["n_rows"] < dataset.n_rows


def test_robustness_grid_covers_the_configured_rates(trained, training_config) -> None:
    expected = len(training_config.robustness.dropout_rates) * len(
        training_config.robustness.false_positive_rates
    )
    assert len(trained.robustness) == expected


def test_importance_and_per_class_tables_are_populated(trained) -> None:
    assert not trained.importance.empty
    assert not trained.per_class.empty
    assert trained.per_class.index.name == "disease"


@pytest.mark.slow
def test_benchmark_compares_several_models(dataset, tmp_path) -> None:
    config = TrainingConfig(
        model_name="logreg",
        split=SplitConfig(n_splits=3, random_state=0),
        artifact_dir=str(tmp_path / "artifacts"),
        report_dir=str(tmp_path / "reports"),
    )
    result = train(config, dataset=dataset, run_benchmark=True, run_robustness=False)
    assert len(result.benchmark) >= 4
    assert "majority" in result.benchmark.index
    assert result.robustness.empty


def test_training_on_synthetic_data_is_marked_as_such(tmp_path) -> None:
    config = TrainingConfig(
        model_name="rules",
        split=SplitConfig(n_splits=3),
        robustness=RobustnessConfig(dropout_rates=(0.0,), false_positive_rates=(0.0,), n_repeats=1),
        artifact_dir=str(tmp_path / "artifacts"),
        report_dir=str(tmp_path / "reports"),
    )
    result = train(config, synthetic=True, run_benchmark=False)
    assert result.metrics["synthetic"] is True
    assert result.bundle.metadata["synthetic"] is True


def test_keeping_duplicates_changes_the_modelling_set(dataset, tmp_path) -> None:
    config = TrainingConfig(
        model_name="rules",
        deduplicate=False,
        split=SplitConfig(n_splits=3),
        robustness=RobustnessConfig(dropout_rates=(0.0,), false_positive_rates=(0.0,), n_repeats=1),
        artifact_dir=str(tmp_path / "artifacts"),
        report_dir=str(tmp_path / "reports"),
    )
    result = train(config, dataset=dataset, run_benchmark=False)
    assert result.metrics["dataset"]["n_rows"] == dataset.n_rows
