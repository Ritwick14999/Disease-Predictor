"""Tests for metrics, split protocols and the robustness benchmark.

The leakage tests are the important ones: they assert that a grouped split
actually prevents an identical row from appearing on both sides, which is the
property the project's headline claim rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from disease_predictor.evaluation import (
    benchmark_models,
    classification_metrics,
    compare_split_protocols,
    confusion_pairs,
    cross_validate_model,
    expected_calibration_error,
    make_split,
    per_class_report,
    perturb_symptoms,
    robustness_curve,
    top_k_accuracy,
)
from disease_predictor.models import build_pipeline


class TestTopKAccuracy:
    def test_counts_a_hit_anywhere_in_the_top_k(self) -> None:
        proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        assert top_k_accuracy(np.array([1, 0]), proba, k=1) == 0.0
        assert top_k_accuracy(np.array([1, 0]), proba, k=2) == 0.5
        assert top_k_accuracy(np.array([1, 0]), proba, k=3) == 1.0

    def test_k_is_clamped_to_the_number_of_classes(self) -> None:
        proba = np.array([[0.6, 0.4]])
        assert top_k_accuracy(np.array([1]), proba, k=99) == 1.0

    def test_rejects_a_one_dimensional_probability_array(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            top_k_accuracy(np.array([0]), np.array([0.5, 0.5]))


class TestCalibration:
    def test_perfectly_confident_and_correct_scores_zero(self) -> None:
        proba = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert expected_calibration_error(np.array([0, 1]), proba) == pytest.approx(0.0)

    def test_confidently_wrong_scores_one(self) -> None:
        proba = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert expected_calibration_error(np.array([1, 1]), proba) == pytest.approx(1.0)


def test_classification_metrics_reports_the_full_set() -> None:
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 2])
    y_proba = np.eye(3)[y_pred] * 0.7 + 0.1
    metrics = classification_metrics(y_true, y_pred, y_proba, n_classes=3, top_k=2)
    assert set(metrics) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "top_2_accuracy",
        "expected_calibration_error",
        "log_loss",
    }
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_metrics_without_probabilities_omit_probability_metrics() -> None:
    metrics = classification_metrics(np.array([0, 1]), np.array([0, 1]))
    assert "log_loss" not in metrics
    assert metrics["accuracy"] == 1.0


class TestSplitProtocols:
    def test_grouped_split_never_puts_one_group_on_both_sides(self, encoded) -> None:
        features, labels, groups = encoded
        split = make_split(features, labels, strategy="grouped", groups=groups)
        assert not set(groups[split.train]) & set(groups[split.test])

    def test_random_split_leaks_identical_rows_across_the_boundary(self, encoded) -> None:
        """The failure mode this project exists to measure."""
        features, labels, groups = encoded
        split = make_split(features, labels, strategy="random")
        shared = set(groups[split.train]) & set(groups[split.test])
        assert shared, "expected duplicated symptom patterns to straddle a random split"

    def test_both_protocols_use_every_row_exactly_once(self, encoded) -> None:
        features, labels, groups = encoded
        for strategy in ("random", "grouped"):
            split = make_split(features, labels, strategy=strategy, groups=groups)
            assert split.n_train + split.n_test == len(labels)
            assert not set(split.train) & set(split.test)

    def test_unknown_strategy_is_rejected(self, encoded) -> None:
        features, labels, _ = encoded
        with pytest.raises(ValueError, match="Unknown split strategy"):
            make_split(features, labels, strategy="loo")

    def test_grouped_split_derives_groups_when_none_are_given(self, encoded) -> None:
        features, labels, groups = encoded
        split = make_split(features, labels, strategy="grouped")
        assert not set(groups[split.train]) & set(groups[split.test])


def test_cross_validation_reports_mean_and_std_per_metric(encoded) -> None:
    features, labels, groups = encoded
    summary = cross_validate_model(
        build_pipeline("logreg"), features, labels, groups=groups, n_splits=3
    )
    assert summary["n_splits"] == 3
    assert 0.0 <= summary["accuracy_mean"] <= 1.0
    assert summary["accuracy_std"] >= 0.0


def test_protocol_comparison_reports_the_leakage_gap(encoded) -> None:
    features, labels, groups = encoded
    table = compare_split_protocols(
        build_pipeline("logreg"), features, labels, groups, n_splits=3
    )
    assert list(table.index) == ["random", "grouped", "leakage_gap"]
    assert table.loc["leakage_gap", "accuracy_mean"] == pytest.approx(
        table.loc["random", "accuracy_mean"] - table.loc["grouped", "accuracy_mean"]
    )


class TestPerturbation:
    def test_dropout_only_removes_symptoms(self, encoded) -> None:
        features, _, _ = encoded
        noisy = perturb_symptoms(features, dropout_rate=1.0)
        assert noisy.sum() == 0

    def test_false_positives_only_add_symptoms(self, encoded) -> None:
        features, _, _ = encoded
        noisy = perturb_symptoms(features, false_positive_rate=1.0)
        assert noisy.all()

    def test_zero_noise_is_the_identity(self, encoded) -> None:
        features, _, _ = encoded
        np.testing.assert_array_equal(perturb_symptoms(features), features)

    def test_the_input_matrix_is_never_modified_in_place(self, encoded) -> None:
        features, _, _ = encoded
        before = features.copy()
        perturb_symptoms(features, dropout_rate=0.5, false_positive_rate=0.5)
        np.testing.assert_array_equal(features, before)

    def test_perturbation_is_reproducible_for_a_given_seed(self, encoded) -> None:
        features, _, _ = encoded
        first = perturb_symptoms(features, dropout_rate=0.3, rng=np.random.default_rng(1))
        second = perturb_symptoms(features, dropout_rate=0.3, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_rates_outside_the_unit_interval_are_rejected(self, encoded, rate: float) -> None:
        features, _, _ = encoded
        with pytest.raises(ValueError, match="must be in"):
            perturb_symptoms(features, dropout_rate=rate)


def test_robustness_accuracy_falls_as_symptoms_go_unreported(encoded) -> None:
    features, labels, groups = encoded
    split = make_split(features, labels, strategy="grouped", groups=groups)
    model = build_pipeline("logreg").fit(features[split.train], labels[split.train])
    curve = robustness_curve(
        model,
        features[split.test],
        labels[split.test],
        dropout_rates=(0.0, 0.8),
        false_positive_rates=(0.0,),
        n_repeats=2,
    )
    clean = curve.loc[curve.dropout_rate == 0.0, "accuracy_mean"].iloc[0]
    degraded = curve.loc[curve.dropout_rate == 0.8, "accuracy_mean"].iloc[0]
    assert degraded < clean


def test_benchmark_ranks_models_by_accuracy(encoded) -> None:
    features, labels, groups = encoded
    table = benchmark_models(["majority", "logreg", "rules"], features, labels, groups, n_splits=3)
    assert list(table.index) == list(table.sort_values("accuracy_mean", ascending=False).index)
    assert table.loc["majority", "accuracy_mean"] < table.loc["logreg", "accuracy_mean"]
    assert (table["fit_seconds"] >= 0).all()


def test_per_class_report_surfaces_the_weakest_class_first() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])
    report = per_class_report(y_true, y_pred, ["healthy", "sick"])
    assert report.index[0] == "sick"


def test_confusion_pairs_lists_mistakes_only() -> None:
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    pairs = confusion_pairs(y_true, y_pred, ["a", "b"])
    assert len(pairs) == 1
    assert pairs.iloc[0]["true_disease"] == "a"
    assert pairs.iloc[0]["predicted_disease"] == "b"


def test_confusion_pairs_is_empty_for_a_perfect_model() -> None:
    y = np.array([0, 1])
    assert confusion_pairs(y, y, ["a", "b"]).empty
