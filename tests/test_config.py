"""Tests for configuration validation and round-tripping."""

from __future__ import annotations

import json

import pytest

from disease_predictor.config import RobustnessConfig, SplitConfig, TrainingConfig


def test_defaults_choose_the_honest_protocol() -> None:
    config = TrainingConfig()
    assert config.split.strategy == "grouped"
    assert config.deduplicate is True


def test_round_trips_through_a_dictionary() -> None:
    config = TrainingConfig(model_name="rules", sampler="random")
    assert TrainingConfig.from_dict(config.to_dict()) == config


def test_round_trips_through_json(tmp_path) -> None:
    config = TrainingConfig(model_name="tree", top_k=5)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert TrainingConfig.from_json(path) == config


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        TrainingConfig.from_dict({"model_name": "tree", "learning_rate": 0.1})


@pytest.mark.parametrize("strategy", ["stratified", "loo", ""])
def test_unknown_split_strategies_are_rejected(strategy: str) -> None:
    with pytest.raises(ValueError, match="Unknown split strategy"):
        SplitConfig(strategy=strategy)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_out_of_range_test_sizes_are_rejected(test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        SplitConfig(test_size=test_size)


def test_too_few_folds_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        SplitConfig(n_splits=1)


def test_unknown_samplers_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sampler"):
        TrainingConfig(sampler="adasyn")


def test_non_positive_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        TrainingConfig(top_k=0)


def test_bundle_path_lives_under_the_artifact_directory() -> None:
    config = TrainingConfig(artifact_dir="/tmp/example")
    assert str(config.bundle_path).startswith("/tmp/example")


def test_robustness_defaults_start_from_clean_input() -> None:
    assert RobustnessConfig().dropout_rates[0] == 0.0
