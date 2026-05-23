"""Run the focused long-history QQQ shortlist validation bundle."""

from __future__ import annotations

import argparse
import glob
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data import read_local_market_data
from us_market_ml.long_history_shortlist_validation import (
    CANDIDATES_NAME,
    DATASET_SOURCE_LABEL,
    DEFAULT_COST_SENSITIVITY_BPS,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_CSV_NAMES,
    PREFLIGHT_MD_NAME,
    REPORT_NAME,
    build_long_history_shortlist_validation_artifacts,
    output_paths,
    render_preflight_markdown,
    select_shortlist_candidates,
    write_long_history_shortlist_validation_artifacts,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

DEFAULT_MODEL_ZOO_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_relative_model_zoo/"
    "long_history_qqq_relative_model_zoo_results.csv"
)
DEFAULT_MODEL_ZOO_DECISION_MATRIX = Path(
    "artifacts/reports/long_history_qqq_relative_model_zoo/"
    "long_history_qqq_relative_model_zoo_decision_matrix.csv"
)
DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_TOP_N = 10
DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS = 1000


def run_long_history_qqq_shortlist_validation_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the focused shortlist validation bundle."""

    args = _parse_args(argv)
    _ensure_local_only_path(args.market_dataset_glob, "--market-dataset-glob")
    _ensure_local_file(args.model_zoo_results, "--model-zoo-results")
    _ensure_local_file(args.model_zoo_decision_matrix, "--model-zoo-decision-matrix")
    _ensure_dataset_available(args.market_dataset_glob)

    results = pd.read_csv(args.model_zoo_results)
    decision_matrix = pd.read_csv(args.model_zoo_decision_matrix)
    candidates = select_shortlist_candidates(
        results=results,
        decision_matrix=decision_matrix,
        top_n=args.top_n,
        candidate_ids=tuple(args.candidate_id),
        include_decisions=tuple(args.include_decision),
    )

    outputs = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(outputs[CANDIDATES_NAME], index=False)
    outputs[PREFLIGHT_MD_NAME].write_text(
        render_preflight_markdown(
            candidates=candidates,
            market_dataset_glob=args.market_dataset_glob,
            model_zoo_results=args.model_zoo_results,
            model_zoo_decision_matrix=args.model_zoo_decision_matrix,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=args.top_n,
            include_decisions=tuple(args.include_decision),
            candidate_ids=tuple(args.candidate_id),
            dry_run=args.dry_run,
        ),
        encoding="utf-8",
    )
    _print_preflight(args=args, candidates=candidates, outputs=outputs)
    if args.dry_run:
        return 0

    market_data = read_local_market_data(args.market_dataset_glob)
    artifacts = build_long_history_shortlist_validation_artifacts(
        raw_df=market_data,
        candidates=candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        cost_sensitivity_bps=tuple(args.cost_sensitivity_bps),
    )
    write_long_history_shortlist_validation_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    from us_market_ml.cli.create_long_history_qqq_shortlist_validation_report import (
        read_long_history_qqq_shortlist_validation_artifacts,
        render_long_history_qqq_shortlist_validation_markdown,
    )

    report_data = read_long_history_qqq_shortlist_validation_artifacts(args.output_dir)
    outputs[REPORT_NAME].write_text(
        render_long_history_qqq_shortlist_validation_markdown(report_data),
        encoding="utf-8",
    )

    print()
    print("Long-history QQQ shortlist validation artifacts written.")
    for output_name in (*OUTPUT_CSV_NAMES, PREFLIGHT_MD_NAME, REPORT_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    print()
    preview_cols = [
        "candidate_id",
        "total_return",
        "qqq_buy_hold_total_return",
        "strategy_vs_qqq",
        "return_to_drawdown",
        "decision",
    ]
    print(artifacts.decision_matrix.loc[:, preview_cols].to_string(index=False))
    return 0


def _print_preflight(
    *,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    print("US Market ML long-history QQQ shortlist validation")
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
    print(f"model zoo decision matrix: {args.model_zoo_decision_matrix}")
    print(f"output directory: {args.output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"top_n: {args.top_n}")
    print(f"include_decision: {args.include_decision or 'all'}")
    print(f"candidate_id: {args.candidate_id or 'none'}")
    print(f"selected candidate count: {len(candidates)}")
    print(f"candidates CSV path: {outputs[CANDIDATES_NAME]}")
    print(f"preflight Markdown path: {outputs[PREFLIGHT_MD_NAME]}")
    print()
    print("Selected candidates:")
    display_cols = [
        "candidate_id",
        "candidate_label",
        "switch_margin",
        "total_return",
        "strategy_vs_qqq",
        "return_to_drawdown",
        "decision",
        "selection_reason",
    ]
    print(candidates.loc[:, display_cols].to_string(index=False))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local focused long-history QQQ shortlist validation bundle."
    )
    parser.add_argument(
        "--model-zoo-results",
        type=Path,
        default=Path(os.environ.get("MODEL_ZOO_RESULTS", str(DEFAULT_MODEL_ZOO_RESULTS))),
    )
    parser.add_argument(
        "--model-zoo-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get(
                "MODEL_ZOO_DECISION_MATRIX",
                str(DEFAULT_MODEL_ZOO_DECISION_MATRIX),
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
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", END_DATE))
    parser.add_argument(
        "--top-n",
        type=_parse_positive_int,
        default=_parse_positive_int(os.environ.get("TOP_N", str(DEFAULT_TOP_N))),
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=_split_env_list(os.environ.get("CANDIDATE_ID", "")),
        help="Candidate ID to force-include; may be repeated.",
    )
    parser.add_argument(
        "--include-decision",
        action="append",
        default=_split_env_list(os.environ.get("INCLUDE_DECISION", "")),
        help="Model-zoo decision label to include in the top-N pool; may be repeated.",
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
    raise SystemExit(run_long_history_qqq_shortlist_validation_from_args())
