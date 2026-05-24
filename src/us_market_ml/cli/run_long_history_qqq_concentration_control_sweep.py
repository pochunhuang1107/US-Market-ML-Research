"""Run the targeted long-history QQQ concentration-control sweep."""

from __future__ import annotations

import argparse
import glob
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data import read_local_market_data
from us_market_ml.long_history_concentration_control import (
    CANDIDATES_NAME,
    DATASET_SOURCE_LABEL,
    DEFAULT_CANDIDATE_IDS,
    DEFAULT_COST_SENSITIVITY_BPS,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_CSV_NAMES,
    PREFLIGHT_MD_NAME,
    build_concentration_control_candidates,
    build_long_history_concentration_control_artifacts,
    output_paths,
    render_preflight_markdown,
    select_concentration_control_source_candidates,
    write_long_history_concentration_control_artifacts,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

DEFAULT_MODEL_ZOO_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_relative_model_zoo/"
    "long_history_qqq_relative_model_zoo_results.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_SUMMARY = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_summary.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_DECISION_MATRIX = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_decision_matrix.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_selected_symbols.csv"
)
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_concentration_control_sweep")
DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS = 1000


def run_long_history_qqq_concentration_control_sweep_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the targeted concentration-control sweep."""

    args = _parse_args(argv)
    _ensure_local_file(args.model_zoo_results, "--model-zoo-results")
    _ensure_local_file(args.candidate_deep_dive_summary, "--candidate-deep-dive-summary")
    _ensure_local_file(
        args.candidate_deep_dive_decision_matrix,
        "--candidate-deep-dive-decision-matrix",
    )
    _ensure_local_file(
        args.candidate_deep_dive_selected_symbols,
        "--candidate-deep-dive-selected-symbols",
    )
    _ensure_local_only_path(args.market_dataset_glob, "--market-dataset-glob")
    _ensure_local_only_path(str(args.output_dir), "--output-dir")
    _ensure_local_only_path(str(args.report_output_dir), "--report-output-dir")
    _ensure_dataset_available(args.market_dataset_glob)

    model_zoo_results = pd.read_csv(args.model_zoo_results)
    deep_dive_summary = pd.read_csv(args.candidate_deep_dive_summary)
    deep_dive_decision_matrix = pd.read_csv(args.candidate_deep_dive_decision_matrix)
    deep_dive_selected_symbols = pd.read_csv(args.candidate_deep_dive_selected_symbols)
    source_candidates = select_concentration_control_source_candidates(
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=tuple(args.candidate_id),
    )
    control_candidates = build_concentration_control_candidates(source_candidates)

    outputs = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_candidates.to_csv(outputs[CANDIDATES_NAME], index=False)
    outputs[PREFLIGHT_MD_NAME].write_text(
        render_preflight_markdown(
            source_candidates=source_candidates,
            control_candidates=control_candidates,
            market_dataset_glob=args.market_dataset_glob,
            model_zoo_results=args.model_zoo_results,
            candidate_deep_dive_summary=args.candidate_deep_dive_summary,
            candidate_deep_dive_decision_matrix=args.candidate_deep_dive_decision_matrix,
            candidate_deep_dive_selected_symbols=args.candidate_deep_dive_selected_symbols,
            model_zoo_result_count=len(model_zoo_results),
            output_dir=args.output_dir,
            report_output_dir=args.report_output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            candidate_ids=tuple(args.candidate_id),
            dry_run=args.dry_run,
        ),
        encoding="utf-8",
    )
    _print_preflight(
        args=args,
        source_candidates=source_candidates,
        control_candidates=control_candidates,
        model_zoo_result_count=len(model_zoo_results),
        deep_dive_selected_symbol_count=len(deep_dive_selected_symbols),
        outputs=outputs,
    )
    if args.dry_run:
        return 0

    market_data = read_local_market_data(args.market_dataset_glob)
    artifacts = build_long_history_concentration_control_artifacts(
        raw_df=market_data,
        source_candidates=source_candidates,
        control_candidates=control_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        cost_sensitivity_bps=tuple(args.cost_sensitivity_bps),
    )
    write_long_history_concentration_control_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    from us_market_ml.cli.create_long_history_qqq_concentration_control_report import (
        run_long_history_qqq_concentration_control_report_from_args,
    )

    run_long_history_qqq_concentration_control_report_from_args(
        [
            "--sweep-output-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.report_output_dir),
        ]
    )

    print()
    print("Long-history QQQ concentration-control sweep artifacts written.")
    for output_name in (*OUTPUT_CSV_NAMES, PREFLIGHT_MD_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    print(f"report output directory: {args.report_output_dir}")
    print()
    preview_cols = [
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "strategy_vs_qqq",
        "positive_vs_qqq_window_rate",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "cost_30bps_strategy_vs_qqq",
        "decision",
        "production_ready",
    ]
    print(artifacts.results.loc[:, preview_cols].to_string(index=False))
    return 0


def _print_preflight(
    *,
    args: argparse.Namespace,
    source_candidates: pd.DataFrame,
    control_candidates: pd.DataFrame,
    model_zoo_result_count: int,
    deep_dive_selected_symbol_count: int,
    outputs: dict[str, Path],
) -> None:
    resolved_display = args.candidate_id if args.candidate_id else list(DEFAULT_CANDIDATE_IDS)
    print("US Market ML long-history QQQ concentration-control sweep")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("broad sweep: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"dry_run: {args.dry_run}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"market dataset glob: {args.market_dataset_glob}")
    print(f"model zoo results: {args.model_zoo_results}")
    print(f"candidate deep-dive summary: {args.candidate_deep_dive_summary}")
    print(f"candidate deep-dive decision matrix: {args.candidate_deep_dive_decision_matrix}")
    print(f"candidate deep-dive selected symbols: {args.candidate_deep_dive_selected_symbols}")
    print(f"deep-dive selected symbol rows: {deep_dive_selected_symbol_count}")
    print(f"output directory: {args.output_dir}")
    print(f"report output directory: {args.report_output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"candidate_id: {resolved_display}")
    print(f"selected source candidate count: {len(source_candidates)}")
    print(f"generated concentration-control config count: {len(control_candidates)}")
    print(f"model zoo result count: {model_zoo_result_count}")
    print(f"smaller than model zoo: {len(control_candidates) < model_zoo_result_count}")
    print(f"candidates CSV path: {outputs[CANDIDATES_NAME]}")
    print(f"preflight Markdown path: {outputs[PREFLIGHT_MD_NAME]}")
    print()
    print("Selected source candidates:")
    source_cols = [
        "candidate_id",
        "candidate_label",
        "model_type",
        "training_mode",
        "target_variant",
        "strategy_vs_qqq",
        "top_symbol",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "source_decision",
    ]
    print(source_candidates.loc[:, source_cols].to_string(index=False))
    print()
    print("Generated controls:")
    control_cols = [
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "allocation_rule",
        "excluded_symbols",
        "diagnostic_control",
    ]
    print(control_candidates.loc[:, control_cols].to_string(index=False))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local targeted long-history QQQ concentration-control sweep."
    )
    parser.add_argument(
        "--model-zoo-results",
        type=Path,
        default=Path(os.environ.get("MODEL_ZOO_RESULTS", str(DEFAULT_MODEL_ZOO_RESULTS))),
    )
    parser.add_argument(
        "--candidate-deep-dive-summary",
        type=Path,
        default=Path(
            os.environ.get(
                "CANDIDATE_DEEP_DIVE_SUMMARY",
                str(DEFAULT_CANDIDATE_DEEP_DIVE_SUMMARY),
            )
        ),
    )
    parser.add_argument(
        "--candidate-deep-dive-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get(
                "CANDIDATE_DEEP_DIVE_DECISION_MATRIX",
                str(DEFAULT_CANDIDATE_DEEP_DIVE_DECISION_MATRIX),
            )
        ),
    )
    parser.add_argument(
        "--candidate-deep-dive-selected-symbols",
        type=Path,
        default=Path(
            os.environ.get(
                "CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS",
                str(DEFAULT_CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS),
            )
        ),
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument(
        "--report-output-dir",
        type=Path,
        default=Path(os.environ.get("REPORT_OUTPUT_DIR", str(DEFAULT_REPORT_OUTPUT_DIR))),
    )
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", END_DATE))
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=_split_env_list(os.environ.get("CANDIDATE_IDS", "")),
        help="Source candidate ID to include; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    parser.add_argument(
        "--min-history-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(
            os.environ.get("MIN_HISTORY_ROWS", str(DEFAULT_MIN_HISTORY_ROWS))
        ),
    )
    parser.add_argument(
        "--min-train-eligible-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(
            os.environ.get("MIN_TRAIN_ELIGIBLE_ROWS", str(DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS))
        ),
    )
    parser.add_argument(
        "--cost-sensitivity-bps",
        type=float,
        nargs="+",
        default=list(DEFAULT_COST_SENSITIVITY_BPS),
    )
    args = parser.parse_args(argv)
    _parse_date(args.start_date, "--start-date")
    _parse_date(args.end_date, "--end-date")
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    return args


def _split_env_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _parse_date(value: str, flag_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be a valid date: {value}") from exc


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _ensure_local_only_path(path: str, flag_name: str) -> None:
    if str(path).lower().startswith(("s3://", "gs://", "http://", "https://")):
        raise ValueError(f"{flag_name} must be a local path/glob, not {path!r}.")


def _ensure_local_file(path: Path, flag_name: str) -> None:
    _ensure_local_only_path(str(path), flag_name)
    if not path.exists():
        raise FileNotFoundError(f"{flag_name} does not exist: {path}")


def _ensure_dataset_available(pattern: str) -> None:
    if glob.has_magic(pattern):
        if glob.glob(pattern, recursive=True):
            return
        raise FileNotFoundError(f"Market dataset glob matched no files: {pattern}")
    if not Path(pattern).exists():
        raise FileNotFoundError(f"Market dataset path does not exist: {pattern}")


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_concentration_control_sweep_from_args())
