"""Shared fixtures.

Everything here runs on generated data so the suite is fast, deterministic and
does not need the real dataset -- which is exactly why the synthetic generator
is part of the package rather than a throwaway script.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from disease_predictor.config import RobustnessConfig, SplitConfig, TrainingConfig
from disease_predictor.data import SymptomDataset, make_synthetic_dataset, pattern_groups
from disease_predictor.predict import DiseasePredictor
from disease_predictor.training import TrainingResult, train


@pytest.fixture(scope="session")
def dataset() -> SymptomDataset:
    """A small synthetic symptom table with duplicate rows built in."""
    return make_synthetic_dataset(
        n_diseases=6, n_symptoms=20, rows_per_disease=40, random_state=11
    )


@pytest.fixture(scope="session")
def encoded(dataset: SymptomDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(features, integer_labels, group_ids)`` for the synthetic dataset."""
    features = dataset.features
    labels = LabelEncoder().fit_transform(dataset.labels)
    return features, labels, pattern_groups(features)


@pytest.fixture(scope="session")
def training_config(tmp_path_factory: pytest.TempPathFactory) -> TrainingConfig:
    """A fast configuration: few folds, tiny robustness grid, temp output dirs."""
    root = tmp_path_factory.mktemp("run")
    return TrainingConfig(
        model_name="logreg",
        split=SplitConfig(strategy="grouped", test_size=0.25, n_splits=3, random_state=0),
        robustness=RobustnessConfig(
            dropout_rates=(0.0, 0.2), false_positive_rates=(0.0,), n_repeats=2
        ),
        artifact_dir=str(root / "artifacts"),
        report_dir=str(root / "reports"),
    )


@pytest.fixture(scope="session")
def trained(dataset: SymptomDataset, training_config: TrainingConfig) -> TrainingResult:
    """One end-to-end training run, reused across the suite."""
    return train(training_config, dataset=dataset, run_benchmark=False)


@pytest.fixture(scope="session")
def predictor(trained: TrainingResult) -> DiseasePredictor:
    """A predictor backed by the session's trained bundle."""
    return DiseasePredictor(trained.bundle)
