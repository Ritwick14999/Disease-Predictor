"""Command-line interface: ``dp <command>``.

Every result in the README is reproducible from one of these commands, which is
the point -- a reviewer should be able to regenerate a claim rather than take it
on trust.

    dp audit                 Quantify duplicate rows and leakage risk.
    dp train                 Train, evaluate honestly, write bundle + report.
    dp benchmark             Compare every model under the honest protocol.
    dp predict               Rank diseases for symptoms given on the command line.
    dp symptoms              List or search the symptom vocabulary.
    dp info                  Show the serving model's provenance.
    dp serve                 Run the REST API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from disease_predictor.__about__ import __version__
from disease_predictor.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REPORT_DIR,
    RobustnessConfig,
    SplitConfig,
    TrainingConfig,
)

LOGGER = logging.getLogger("disease_predictor")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stderr,
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", dest="data_path", default=None, help="Path or URL to the dataset CSV.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated data instead of the real file. Lets the pipeline run anywhere.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp",
        description="Disease prediction from symptoms - training, evaluation and serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"disease-predictor {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Quantify duplicate rows and leakage risk in the dataset.")
    _add_data_arguments(audit)
    audit.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    train = sub.add_parser("train", help="Train a model and write the bundle plus reports.")
    _add_data_arguments(train)
    train.add_argument("--model", default="xgboost", help="Model registry name.")
    train.add_argument("--sampler", default="none", choices=("none", "random", "smote"))
    train.add_argument(
        "--split",
        default="grouped",
        choices=("grouped", "random"),
        help="'grouped' keeps identical symptom patterns on one side of the split.",
    )
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--top-k", type=int, default=3)
    train.add_argument("--keep-duplicates", action="store_true", help="Skip deduplication.")
    train.add_argument("--no-benchmark", action="store_true", help="Skip the model comparison.")
    train.add_argument("--no-robustness", action="store_true", help="Skip the noise benchmark.")
    train.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    train.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))

    benchmark = sub.add_parser("benchmark", help="Compare every model under the honest protocol.")
    _add_data_arguments(benchmark)
    benchmark.add_argument("--folds", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--split", default="grouped", choices=("grouped", "random"))
    benchmark.add_argument("--output", default=None, help="Optional CSV path for the results.")

    predict = sub.add_parser("predict", help="Rank diseases for a list of symptoms.")
    predict.add_argument("symptoms", nargs="+", help="Symptom names, e.g. itching skin_rash.")
    predict.add_argument("--model-path", default=None, help="Path to a trained bundle.")
    predict.add_argument("--top-k", type=int, default=3)
    predict.add_argument("--follow-up", action="store_true", help="Suggest what to ask about next.")
    predict.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    symptoms = sub.add_parser("symptoms", help="List or search the symptom vocabulary.")
    symptoms.add_argument("query", nargs="?", default=None)
    symptoms.add_argument("--model-path", default=None)
    symptoms.add_argument("--limit", type=int, default=50)

    info = sub.add_parser("info", help="Show the trained model's provenance and scores.")
    info.add_argument("--model-path", default=None)

    serve = sub.add_parser("serve", help="Run the REST API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def _command_audit(args: argparse.Namespace) -> int:
    from disease_predictor.data import audit_duplicates, load_dataset

    dataset = load_dataset(args.data_path, synthetic=args.synthetic)
    report = audit_duplicates(dataset)

    if args.json:
        print(json.dumps({"dataset": dataset.describe(), "leakage_audit": report}, indent=2))
        return 0

    print(f"Dataset: {dataset.source}")
    print(f"  {dataset.n_rows} rows x {dataset.n_features} symptoms x {dataset.n_classes} diseases\n")
    print("Leakage audit")
    for key, value in report.items():
        print(f"  {key:<32} {value}")

    duplicate_fraction = float(report["duplicate_row_fraction"])
    print()
    if duplicate_fraction > 0.5:
        print(
            f"  => {duplicate_fraction:.1%} of rows are exact duplicates. A random train/test "
            "split will place identical rows on both sides, so a reported accuracy from that "
            "protocol measures memorisation. Use the grouped split (the default)."
        )
    else:
        print(f"  => {duplicate_fraction:.1%} duplicate rows; random-split leakage risk is limited.")
    return 0


def _command_train(args: argparse.Namespace) -> int:
    from disease_predictor.training import train

    config = TrainingConfig(
        model_name=args.model,
        sampler=args.sampler,
        deduplicate=not args.keep_duplicates,
        top_k=args.top_k,
        split=SplitConfig(
            strategy=args.split,
            test_size=args.test_size,
            n_splits=args.folds,
            random_state=args.seed,
        ),
        robustness=RobustnessConfig(random_state=args.seed),
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
    )
    result = train(
        config,
        synthetic=args.synthetic,
        run_benchmark=not args.no_benchmark,
        run_robustness=not args.no_robustness,
    )

    print("\nHeld-out performance")
    for key, value in result.metrics["holdout"].items():
        print(f"  {key:<28} {value}")

    print("\nSplit-protocol comparison (raw data, duplicates included)")
    columns = [c for c in result.protocol_comparison.columns if c.endswith("_mean")]
    print(result.protocol_comparison[columns].round(4).to_string())

    if not result.benchmark.empty:
        print("\nModel benchmark (honest protocol)")
        print(
            result.benchmark[["accuracy_mean", "macro_f1_mean", "fit_seconds"]]
            .round(4)
            .to_string()
        )

    print(f"\nBundle:  {result.bundle_path}")
    for name, path in result.report_paths.items():
        print(f"{name + ':':<9}{path}")
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    from sklearn.preprocessing import LabelEncoder

    from disease_predictor.data import load_dataset, pattern_groups
    from disease_predictor.evaluation import benchmark_models
    from disease_predictor.models import available_models

    dataset = load_dataset(args.data_path, synthetic=args.synthetic).deduplicated()
    features = dataset.features
    labels = LabelEncoder().fit_transform(dataset.labels)
    table = benchmark_models(
        available_models(),
        features,
        labels,
        pattern_groups(features),
        strategy=args.split,
        n_splits=args.folds,
        random_state=args.seed,
    )
    columns = [c for c in table.columns if c.endswith("_mean")] + ["fit_seconds"]
    print(table[columns].round(4).to_string())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        print(f"\nWrote {args.output}")
    return 0


def _command_predict(args: argparse.Namespace) -> int:
    from disease_predictor.predict import DiseasePredictor

    predictor = DiseasePredictor.load(args.model_path)
    result = predictor.predict(args.symptoms, top_k=args.top_k, follow_up=args.follow_up)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    for warning in result.warnings:
        print(f"! {warning}")
    if not result.predictions:
        return 1

    print(f"\nRecognised symptoms: {', '.join(result.recognised_symptoms)}\n")
    for prediction in result.predictions:
        print(f"{prediction.rank}. {prediction.disease} - {prediction.probability:.1%}")
        for contribution in prediction.supporting_symptoms[:3]:
            if contribution.contribution > 0:
                print(f"     because {contribution.symptom} ({contribution.share:.0%} of the evidence)")
    if result.follow_up_symptoms:
        print("\nMost informative symptoms to check next:")
        for contribution in result.follow_up_symptoms:
            print(f"  - {contribution.symptom} (would shift the top probability by {contribution.contribution:+.1%})")
    print(f"\n{result.disclaimer}")
    return 0


def _command_symptoms(args: argparse.Namespace) -> int:
    from disease_predictor.predict import DiseasePredictor

    predictor = DiseasePredictor.load(args.model_path)
    names = (
        predictor.search_symptoms(args.query, limit=args.limit)
        if args.query
        else predictor.symptoms[: args.limit]
    )
    for name in names:
        print(name)
    print(f"\n{len(names)} of {len(predictor.symptoms)} symptoms.", file=sys.stderr)
    return 0


def _command_info(args: argparse.Namespace) -> int:
    from disease_predictor.predict import DiseasePredictor

    predictor = DiseasePredictor.load(args.model_path)
    metadata = predictor.metadata
    print(json.dumps(metadata, indent=2, default=str))
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Serving needs the api extra: pip install 'disease-predictor[api]'",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "disease_predictor.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


_COMMANDS = {
    "audit": _command_audit,
    "train": _command_train,
    "benchmark": _command_benchmark,
    "predict": _command_predict,
    "symptoms": _command_symptoms,
    "info": _command_info,
    "serve": _command_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``dp`` console script.

    Returns:
        A process exit code: 0 on success, 1 on an expected failure such as a
        missing dataset or an untrained model.
    """
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return _COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
