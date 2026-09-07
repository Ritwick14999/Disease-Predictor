"""Tests for the command-line interface.

The CLI is how a reviewer reproduces every number in the README, so its exit
codes and output need to be part of the contract.
"""

from __future__ import annotations

import json

import pytest

from disease_predictor.cli import main
from disease_predictor.training import save_bundle


@pytest.fixture(scope="module")
def bundle_path(trained, tmp_path_factory) -> str:
    return str(save_bundle(trained.bundle, tmp_path_factory.mktemp("cli") / "bundle.joblib"))


def test_version_flag_exits_cleanly(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "disease-predictor" in capsys.readouterr().out


def test_audit_prints_the_leakage_summary(capsys) -> None:
    assert main(["audit", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "duplicate_row_fraction" in out
    assert "Leakage audit" in out


def test_audit_can_emit_json(capsys) -> None:
    assert main(["audit", "--synthetic", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["leakage_audit"]["n_rows"] > 0
    assert "synthetic" in payload["dataset"]["source"]


def test_audit_without_data_fails_with_guidance(caplog, tmp_path) -> None:
    assert main(["audit", "--data", str(tmp_path / "missing.csv")]) == 1
    assert "--synthetic" in caplog.text


def test_predict_reports_the_ranking_and_the_disclaimer(capsys, bundle_path, trained) -> None:
    symptoms = list(trained.bundle.feature_names[:3])
    assert main(["predict", "--model-path", bundle_path, *symptoms]) == 0
    out = capsys.readouterr().out
    assert "1." in out
    assert "not medical advice" in out


def test_predict_can_emit_json(capsys, bundle_path, trained) -> None:
    symptoms = list(trained.bundle.feature_names[:3])
    assert main(["predict", "--model-path", bundle_path, "--json", *symptoms]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["predictions"][0]["rank"] == 1


def test_predict_with_only_unknown_symptoms_exits_nonzero(capsys, bundle_path) -> None:
    assert main(["predict", "--model-path", bundle_path, "not_a_symptom"]) == 1


def test_predict_without_a_model_fails_with_guidance(caplog, tmp_path) -> None:
    assert main(["predict", "--model-path", str(tmp_path / "absent.joblib"), "itching"]) == 1
    assert "dp train" in caplog.text


def test_symptoms_lists_the_vocabulary(capsys, bundle_path) -> None:
    assert main(["symptoms", "--model-path", bundle_path, "--limit", "5"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 5


def test_symptoms_supports_a_query(capsys, bundle_path, trained) -> None:
    needle = trained.bundle.feature_names[0]
    assert main(["symptoms", "--model-path", bundle_path, needle]) == 0
    assert needle in capsys.readouterr().out


def test_info_prints_the_training_metadata(capsys, bundle_path) -> None:
    assert main(["info", "--model-path", bundle_path]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["model_name"] == "logreg"


@pytest.mark.slow
def test_train_writes_artifacts_and_prints_the_comparison(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "train",
            "--synthetic",
            "--model", "rules",
            "--folds", "3",
            "--no-benchmark",
            "--no-robustness",
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--report-dir", str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Held-out performance" in out
    assert "leakage_gap" in out
    assert (tmp_path / "artifacts").exists()
    assert (tmp_path / "reports" / "metrics.json").is_file()


@pytest.mark.slow
def test_benchmark_writes_a_csv(capsys, tmp_path) -> None:
    output = tmp_path / "benchmark.csv"
    assert main(["benchmark", "--synthetic", "--folds", "3", "--output", str(output)]) == 0
    assert output.is_file()
    assert "majority" in capsys.readouterr().out


def test_an_unknown_model_name_fails_cleanly(caplog, tmp_path) -> None:
    exit_code = main(
        [
            "train", "--synthetic", "--model", "not_a_model", "--no-benchmark",
            "--no-robustness", "--artifact-dir", str(tmp_path), "--report-dir", str(tmp_path),
        ]
    )
    assert exit_code == 1
    assert "Unknown model" in caplog.text
