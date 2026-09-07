"""Tests for the symptom encoder -- the component that prevents feature drift."""

from __future__ import annotations

import numpy as np
import pytest

from disease_predictor.features import SymptomEncoder, UnknownSymptomError

VOCABULARY = ["itching", "skin_rash", "chest_pain", "high_fever", "dischromic _patches"]


@pytest.fixture
def encoder() -> SymptomEncoder:
    return SymptomEncoder(VOCABULARY)


def test_encoder_canonicalises_its_vocabulary(encoder: SymptomEncoder) -> None:
    assert "dischromic_patches" in encoder.feature_names
    assert len(encoder) == len(VOCABULARY)


def test_encoder_rejects_an_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SymptomEncoder([])


def test_encoder_rejects_duplicate_names_after_canonicalisation() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        SymptomEncoder(["skin rash", "skin_rash"])


@pytest.mark.parametrize("spelling", ["skin_rash", "Skin Rash", "  SKIN   RASH  ", "skin-rash"])
def test_encoding_is_insensitive_to_casing_and_separators(
    encoder: SymptomEncoder, spelling: str
) -> None:
    result = encoder.encode([spelling])
    assert result.recognised == ("skin_rash",)


def test_encoding_produces_a_correctly_ordered_vector(encoder: SymptomEncoder) -> None:
    result = encoder.encode(["chest_pain", "itching"])
    expected = np.array([[1, 0, 1, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(result.vector, expected)


def test_repeated_symptoms_do_not_double_count(encoder: SymptomEncoder) -> None:
    result = encoder.encode(["itching", "itching", "Itching"])
    assert result.recognised == ("itching",)
    assert result.vector.sum() == 1


def test_unknown_symptoms_are_reported_with_suggestions(encoder: SymptomEncoder) -> None:
    result = encoder.encode(["itchng", "skin_rash"])
    assert result.unknown == ("itchng",)
    assert "itching" in result.suggestions["itchng"]
    assert result.recognised == ("skin_rash",)


def test_strict_mode_raises_on_unknown_symptoms(encoder: SymptomEncoder) -> None:
    with pytest.raises(UnknownSymptomError) as excinfo:
        encoder.encode(["not_a_symptom"], strict=True)
    assert excinfo.value.symptom == "not_a_symptom"


def test_blank_and_non_string_entries_are_skipped(encoder: SymptomEncoder) -> None:
    result = encoder.encode(["", "   ", None, 42, "itching"])  # type: ignore[list-item]
    assert result.recognised == ("itching",)
    assert result.unknown == ()


def test_decode_round_trips_an_encoded_vector(encoder: SymptomEncoder) -> None:
    result = encoder.encode(["itching", "high_fever"])
    assert set(encoder.decode(result.vector)) == {"itching", "high_fever"}


def test_decode_rejects_a_wrongly_sized_vector(encoder: SymptomEncoder) -> None:
    with pytest.raises(ValueError, match="Expected a vector of length"):
        encoder.decode(np.zeros(3))


def test_search_prefers_prefix_then_substring(encoder: SymptomEncoder) -> None:
    assert encoder.search("chest")[0] == "chest_pain"
    assert "high_fever" in encoder.search("fever")


def test_empty_search_returns_the_head_of_the_vocabulary(encoder: SymptomEncoder) -> None:
    assert encoder.search("", limit=2) == encoder.feature_names[:2]


def test_membership_check_uses_canonical_form(encoder: SymptomEncoder) -> None:
    assert "Skin Rash" in encoder
    assert "not_a_symptom" not in encoder
