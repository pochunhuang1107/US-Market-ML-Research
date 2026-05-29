"""Create the QQQ-only RF research paper-tracking package."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data import read_local_market_data
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    build_qqq_only_feature_dataset,
    discover_reference_series,
    load_reference_series,
    run_qqq_only_spec,
)
from us_market_ml.qqq_only_rf_paper_tracking import (
    DEFAULT_OUTPUT_DIR,
    OUTPUT_PREFIX,
    REPORT_SECTIONS,
    build_qqq_only_rf_paper_tracking_artifacts,
    build_qqq_only_rf_paper_tracking_candidate_registry,
    build_qqq_only_rf_paper_tracking_specs,
    report_output_paths,
    write_qqq_only_rf_paper_tracking_artifacts,
)

DEFAULT_MARKET_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_REFERENCE_SERIES_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/reference_series/**/*.parquet"
)
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_MIN_TRAIN_ROWS = 252

DEFAULT_OUT_OF_TIME_REPORT_DIR = Path("artifacts/reports/qqq_only_rf_out_of_time_confirmation")
DEFAULT_OUT_OF_TIME_SWEEP_DIR = Path("artifacts/sweeps/qqq_only_rf_out_of_time_confirmation")
DEFAULT_OUT_OF_TIME_SUMMARY = (
    DEFAULT_OUT_OF_TIME_REPORT_DIR / "qqq_only_rf_out_of_time_confirmation_summary.csv"
)
DEFAULT_OUT_OF_TIME_DECISION_MATRIX = (
    DEFAULT_OUT_OF_TIME_REPORT_DIR / "qqq_only_rf_out_of_time_confirmation_decision_matrix.csv"
)
DEFAULT_OUT_OF_TIME_RESULTS = (
    DEFAULT_OUT_OF_TIME_SWEEP_DIR / "qqq_only_rf_out_of_time_confirmation_results.csv"
)
DEFAULT_OUT_OF_TIME_EXPOSURE_LOG = (
    DEFAULT_OUT_OF_TIME_SWEEP_DIR / "qqq_only_rf_out_of_time_confirmation_exposure_log.csv"
)


def create_qqq_only_rf_paper_tracking_package_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create the local research paper-tracking package."""

    args = _parse_args(argv)
    for flag_name, path in (
        ("--out-of-time-summary", args.out_of_time_summary),
        ("--out-of-time-decision-matrix", args.out_of_time_decision_matrix),
        ("--out-of-time-results", args.out_of_time_results),
        ("--out-of-time-exposure-log", args.out_of_time_exposure_log),
    ):
        _ensure_local_file(path, flag_name)
    _ensure_local_only_path(args.market_dataset_glob, "--market-dataset-glob")
    _ensure_local_only_path(args.reference_series_glob, "--reference-series-glob")
    _ensure_local_only_path(str(args.output_dir), "--output-dir")
    _ensure_not_model_signals_path(args.output_dir)

    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    discovery = discover_reference_series(args.reference_series_glob)
    _print_plan(args=args, registry=registry, matched_reference_count=len(discovery.matched_paths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / f"{OUTPUT_PREFIX}_preflight.md"
    preflight_path.write_text(
        _render_preflight(
            args=args,
            registry=registry,
            matched_reference_count=len(discovery.matched_paths),
        ),
        encoding="utf-8",
    )
    print(f"preflight Markdown path: {preflight_path}")
    if args.dry_run:
        return 0

    if discovery.missing_symbols:
        raise FileNotFoundError(
            "Missing required local reference series for QQQ-only RF paper tracking: "
            + ", ".join(discovery.missing_symbols)
        )
    raw_market = read_local_market_data(args.market_dataset_glob)
    reference = load_reference_series(discovery.selected_glob)
    dataset = build_qqq_only_feature_dataset(
        raw_market,
        reference,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    replay_outputs = {}
    prediction_cache: dict[tuple[object, ...], pd.DataFrame] = {}
    specs = build_qqq_only_rf_paper_tracking_specs()
    for candidate_id, spec in zip(registry["candidate_id"].astype(str), specs, strict=True):
        replay_outputs[candidate_id] = run_qqq_only_spec(
            dataset,
            spec,
            start_date=args.start_date,
            end_date=args.end_date,
            prediction_cache=prediction_cache,
            min_train_rows=args.min_train_rows,
        )

    artifacts = build_qqq_only_rf_paper_tracking_artifacts(
        out_of_time_summary=pd.read_csv(args.out_of_time_summary),
        out_of_time_decision_matrix=pd.read_csv(args.out_of_time_decision_matrix),
        out_of_time_results=pd.read_csv(args.out_of_time_results),
        out_of_time_exposure_log=pd.read_csv(args.out_of_time_exposure_log),
        replay_outputs=replay_outputs,
    )
    paths = write_qqq_only_rf_paper_tracking_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print()
    print("QQQ-only RF paper-tracking package artifacts written.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print()
    print(artifacts.initial_signal.to_string(index=False))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the QQQ-only RF research paper-tracking package."
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_MARKET_DATASET_GLOB),
    )
    parser.add_argument(
        "--reference-series-glob",
        default=os.environ.get("REFERENCE_SERIES_GLOB", DEFAULT_REFERENCE_SERIES_GLOB),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", DEFAULT_END_DATE))
    parser.add_argument(
        "--min-train-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(os.environ.get("MIN_TRAIN_ROWS", str(DEFAULT_MIN_TRAIN_ROWS))),
    )
    parser.add_argument(
        "--out-of-time-summary",
        type=Path,
        default=Path(os.environ.get("OUT_OF_TIME_SUMMARY", str(DEFAULT_OUT_OF_TIME_SUMMARY))),
    )
    parser.add_argument(
        "--out-of-time-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get(
                "OUT_OF_TIME_DECISION_MATRIX",
                str(DEFAULT_OUT_OF_TIME_DECISION_MATRIX),
            )
        ),
    )
    parser.add_argument(
        "--out-of-time-results",
        type=Path,
        default=Path(os.environ.get("OUT_OF_TIME_RESULTS", str(DEFAULT_OUT_OF_TIME_RESULTS))),
    )
    parser.add_argument(
        "--out-of-time-exposure-log",
        type=Path,
        default=Path(
            os.environ.get("OUT_OF_TIME_EXPOSURE_LOG", str(DEFAULT_OUT_OF_TIME_EXPOSURE_LOG))
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    args = parser.parse_args(argv)
    _parse_date(args.start_date, "--start-date")
    _parse_date(args.end_date, "--end-date")
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    return args


def _print_plan(
    *,
    args: argparse.Namespace,
    registry: pd.DataFrame,
    matched_reference_count: int,
) -> None:
    print("US Market ML QQQ-only RF paper-tracking package")
    print("local-only: true")
    print("reads prior out-of-time artifacts: true")
    print("locked candidate registry: true")
    print("new broad model zoo: false")
    print("production signal export: false")
    print("model card: false")
    print("production_ready always false")
    print("tradable holdings: QQQ/cash only")
    print("SPY core/sleeve: false")
    print("ETF rotation holdings: false")
    print(f"dry_run: {args.dry_run}")
    print(f"market dataset glob: {args.market_dataset_glob}")
    print(f"reference series glob: {args.reference_series_glob}")
    print(f"reference matched file count: {matched_reference_count}")
    print(f"output directory: {args.output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"locked registry rows: {len(registry)}")


def _render_preflight(
    *,
    args: argparse.Namespace,
    registry: pd.DataFrame,
    matched_reference_count: int,
) -> str:
    return "\n".join(
        [
            "# QQQ-Only RF Paper-Tracking Package Preflight",
            "",
            f"Local-only: `{True}`",
            f"Reads prior out-of-time artifacts: `{True}`",
            f"Locked candidate registry: `{True}`",
            f"New broad model zoo: `{False}`",
            f"Production signal export: `{False}`",
            f"Model card: `{False}`",
            f"Production-ready marking: `{False}`",
            "Tradable holdings: `QQQ,cash`",
            f"SPY core/sleeve: `{False}`",
            f"ETF rotation holdings: `{False}`",
            f"Dry run: `{args.dry_run}`",
            "",
            "## Inputs",
            "",
            f"- MARKET_DATASET_GLOB: `{args.market_dataset_glob}`",
            f"- REFERENCE_SERIES_GLOB: `{args.reference_series_glob}`",
            f"- Matched reference files: `{matched_reference_count}`",
            f"- OUT_OF_TIME_SUMMARY: `{args.out_of_time_summary}`",
            f"- OUT_OF_TIME_DECISION_MATRIX: `{args.out_of_time_decision_matrix}`",
            f"- OUT_OF_TIME_RESULTS: `{args.out_of_time_results}`",
            f"- OUT_OF_TIME_EXPOSURE_LOG: `{args.out_of_time_exposure_log}`",
            f"- OUTPUT_DIR: `{args.output_dir}`",
            f"- START_DATE: `{args.start_date}`",
            f"- END_DATE: `{args.end_date}`",
            "",
            "## Locked Registry",
            "",
            _markdown_table(
                registry,
                [
                    "candidate_id",
                    "role",
                    "threshold",
                    "min_holding_days",
                    "production_ready",
                ],
            ),
            "",
            "## Required Report Sections",
            "",
            *[f"- {section}" for section in REPORT_SECTIONS],
            "",
            "## Output Paths",
            "",
            *[f"- {name}: `{path}`" for name, path in report_output_paths(args.output_dir).items()],
            "",
            "Generated artifacts remain uncommitted.",
            "",
        ]
    )


def _ensure_local_only_path(path: str, flag_name: str) -> None:
    if str(path).lower().startswith(
        ("s3://", "s3:/", "gs://", "gs:/", "http://", "http:/", "https://", "https:/")
    ):
        raise ValueError(f"{flag_name} must be a local path, not {path!r}.")


def _ensure_local_file(path: Path, flag_name: str) -> None:
    _ensure_local_only_path(str(path), flag_name)
    if not path.exists():
        raise FileNotFoundError(f"{flag_name} does not exist: {path}")


def _ensure_not_model_signals_path(path: Path) -> None:
    parts = tuple(part.lower() for part in Path(path).parts)
    for index, part in enumerate(parts[:-1]):
        if part == "artifacts" and parts[index + 1] == "model_signals":
            raise ValueError(
                "--output-dir must not be under artifacts/model_signals for paper tracking."
            )


def _parse_positive_int(value: str | int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_date(value: str, flag_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be a valid date: {value!r}") from exc


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(create_qqq_only_rf_paper_tracking_package_from_args())
