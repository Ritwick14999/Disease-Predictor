"""Tests for loading, validation and the duplicate/leakage audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disease_predictor.data import (
    DatasetNotFoundError,
    SchemaError,
    audit_duplicates,
    canonical_symptom,
    load_dataset,
    make_synthetic_dataset,
    pattern_groups,
    read_symptom_frame,
    summarise_class_signatures,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dischromic _patches", "dischromic_patches"),
        ("  Foul smell of urine ", "foul_smell_of_urine"),
        ("fluid_overload ", "fluid_overload"),
        ("toxic_look_(typhos)", "toxic_look_typhos"),
        ("Skin Rash", "skin_rash"),
        ("__itching__", "itching"),
    ],
)
def test_canonical_symptom_normalises_messy_headers(raw: str, expected: str) -> None:
    assert canonical_symptom(raw) == expected


def test_read_symptom_frame_drops_unnamed_and_empty_columns() -> None:
    frame = pd.DataFrame(
        {
            "itching": [1, 0],
            "Skin Rash": [0, 1],
            "all_empty": [None, None],
            "Unnamed: 133": [np.nan, np.nan],
            "prognosis": ["Fungal infection", "Allergy"],
        }
    )
    dataset = read_symptom_frame(frame)
    assert dataset.feature_names == ("itching", "skin_rash")
    assert dataset.n_rows == 2


def test_read_symptom_frame_requires_target_column() -> None:
    with pytest.raises(SchemaError, match="prognosis"):
        read_symptom_frame(pd.DataFrame({"itching": [1]}))


def test_read_symptom_frame_rejects_non_binary_features() -> None:
    frame = pd.DataFrame({"itching": [0, 5], "prognosis": ["a", "b"]})
    with pytest.raises(SchemaError, match="binary"):
        read_symptom_frame(frame)


def test_read_symptom_frame_rejects_non_numeric_features() -> None:
    frame = pd.DataFrame({"itching": ["yes", "no"], "prognosis": ["a", "b"]})
    with pytest.raises(SchemaError, match="numeric"):
        read_symptom_frame(frame)


def test_read_symptom_frame_disambiguates_colliding_headers() -> None:
    frame = pd.DataFrame(
        {"fluid overload": [1, 0], "fluid_overload": [0, 1], "prognosis": ["a", "b"]}
    )
    dataset = read_symptom_frame(frame)
    assert dataset.feature_names == ("fluid_overload", "fluid_overload_1")


def test_load_dataset_reports_a_helpful_error_when_missing(tmp_path) -> None:
    with pytest.raises(DatasetNotFoundError, match="--synthetic"):
        load_dataset(tmp_path / "nope.csv")


def test_load_dataset_reads_a_csv(tmp_path) -> None:
    path = tmp_path / "Training.csv"
    pd.DataFrame({"itching": [1, 0], "prognosis": ["a", "b"]}).to_csv(path, index=False)
    dataset = load_dataset(path)
    assert dataset.n_rows == 2
    assert str(path) in dataset.source


def test_pattern_groups_assigns_identical_rows_to_one_group() -> None:
    features = np.array([[1, 0], [1, 0], [0, 1]])
    groups = pattern_groups(features)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_pattern_groups_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pattern_groups(np.array([1, 0, 1]))


def test_audit_detects_planted_duplicates() -> None:
    dataset = make_synthetic_dataset(
        n_diseases=4, n_symptoms=12, rows_per_disease=25, duplicate_rate=0.5, random_state=3
    )
    report = audit_duplicates(dataset)
    assert report["n_duplicate_rows"] > 0
    assert 0.0 < report["duplicate_row_fraction"] <= 1.0
    assert report["n_unique_patterns"] <= report["n_rows"]
    assert report["rows_per_unique_pattern"] >= 1.0


def test_audit_reports_no_duplicates_for_a_unique_table() -> None:
    frame = pd.DataFrame(np.eye(6, dtype=int), columns=[f"s{i}" for i in range(6)])
    frame["prognosis"] = [f"disease_{i}" for i in range(6)]
    report = audit_duplicates(read_symptom_frame(frame))
    assert report["n_duplicate_rows"] == 0
    assert report["duplicate_row_fraction"] == 0.0
    assert report["leakage_risk"] == 0.0


def test_audit_flags_patterns_that_map_to_several_diseases() -> None:
    frame = pd.DataFrame({"fever": [1, 1], "cough": [1, 1], "prognosis": ["flu", "cold"]})
    report = audit_duplicates(read_symptom_frame(frame))
    assert report["n_ambiguous_patterns"] == 1


def test_deduplicated_removes_repeats_and_keeps_every_class(dataset) -> None:
    deduplicated = dataset.deduplicated()
    assert deduplicated.n_rows <= dataset.n_rows
    assert set(deduplicated.classes) == set(dataset.classes)
    assert audit_duplicates(deduplicated)["n_duplicate_rows"] == 0


def test_fingerprint_is_stable_and_content_sensitive(dataset) -> None:
    assert dataset.fingerprint() == dataset.fingerprint()
    mutated = dataset.frame.copy()
    mutated.iloc[0, 0] = 1 - mutated.iloc[0, 0]
    other = read_symptom_frame(mutated)
    assert other.fingerprint() != dataset.fingerprint()


def test_synthetic_dataset_is_labelled_as_synthetic(dataset) -> None:
    assert "synthetic" in dataset.source.lower()
    assert all(name.startswith("synthetic_") for name in dataset.feature_names)


def test_synthetic_generator_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="signature_size"):
        make_synthetic_dataset(n_symptoms=3, signature_size=10)


def test_class_signatures_cover_every_disease(dataset) -> None:
    signatures = summarise_class_signatures(dataset)
    assert set(signatures) == set(dataset.classes)
    assert any(symptoms for symptoms in signatures.values())
