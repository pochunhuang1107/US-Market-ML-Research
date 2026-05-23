"""Run the rich long-history QQQ candidate deep-dive bundle."""

from __future__ import annotations

import argparse
import glob
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data import read_local_market_data
from us_market_ml.long_history_candidate_deep_dive import (
    DATASET_SOURCE_LABEL,
    DEFAULT_CANDIDATE_IDS,
    DEFAULT_COST_SENSITIVITY_BPS,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_CSV_NAMES,
    PREFLIGHT_MD_NAME,
    REPORT_NAME,
    build_long_history_candidate_deep_dive_artifacts,
    output_paths,
    render_preflight_markdown,
    select_deep_dive_candidates,
    write_long_history_candidate_deep_dive_artifacts,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

DEFAULT_MODEL_ZOO_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_relative_model_zoo/"
    "long_history_qqq_relative_model_zoo_results.csv"
)
DEFAULT_SHORTLIST_CANDIDATES = Path(
    "artifacts/validation/long_history_qqq_shortlist_validation/"
    "long_history_qqq_shortlist_validation_candidates.csv"
)
DEFAULT_SHORTLIST_DECISION_MATRIX = Path(
    "artifacts/validation/long_history_qqq_shortlist_validation/"
    "long_history_qqq_shortlist_validation_decision_matrix.csv"
)
DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS = 1000


def run_long_history_qqq_candidate_deep_dive_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the focused candidate deep-dive bundle."""

    args = _parse_args(argv)
    _ensure_local_file(args.model_zoo_results, "--model-zoo-results")
    _ensure_local_file(args.shortlist_candidates, "--shortlist-candidates")
    _ensure_local_file(args.shortlist_decision_matrix, "--shortlist-decision-matrix")
    _ensure_local_only_path(args.market_dataset_glob, "--market-dataset-glob")
    _ensure_local_only_path(str(args.output_dir), "--output-dir")
    _ensure_dataset_available(args.market_dataset_glob)

    model_zoo_results = pd.read_csv(args.model_zoo_results)
    shortlist_candidates = pd.read_csv(args.shortlist_candidates)
    shortlist_decision_matrix = pd.read_csv(args.shortlist_decision_matrix)
    selected_candidates = select_deep_dive_candidates(
        shortlist_candidates=shortlist_candidates,
        shortlist_decision_matrix=shortlist_decision_matrix,
        candidate_ids=tuple(args.candidate_id),
        include_default_candidates=args.include_default_candidates,
    )

    outputs = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[PREFLIGHT_MD_NAME].write_text(
        render_preflight_markdown(
            selected_candidates=selected_candidates,
            market_dataset_glob=args.market_dataset_glob,
            model_zoo_results=args.model_zoo_results,
            shortlist_candidates=args.shortlist_candidates,
            shortlist_decision_matrix=args.shortlist_decision_matrix,
            model_zoo_result_count=len(model_zoo_results),
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            candidate_ids=tuple(args.candidate_id),
            include_default_candidates=args.include_default_candidates,
            dry_run=args.dry_run,
        ),
        encoding="utf-8",
    )
    _print_preflight(args=args, selected_candidates=selected_candidates, outputs=outputs)
    if args.dry_run:
        return 0

    market_data = read_local_market_data(args.market_dataset_glob)
    artifacts = build_long_history_candidate_deep_dive_artifacts(
        raw_df=market_data,
        selected_candidates=selected_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        cost_sensitivity_bps=tuple(args.cost_sensitivity_bps),
    )
    write_long_history_candidate_deep_dive_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    from us_market_ml.cli.create_long_history_qqq_candidate_deep_dive_report import (
        read_long_history_qqq_candidate_deep_dive_artifacts,
        render_long_history_qqq_candidate_deep_dive_markdown,
    )

    report_data = read_long_history_qqq_candidate_deep_dive_artifacts(args.output_dir)
    outputs[REPORT_NAME].write_text(
        render_long_history_qqq_candidate_deep_dive_markdown(report_data),
        encoding="utf-8",
    )

    print()
    print("Long-history QQQ candidate deep-dive artifacts written.")
    for output_name in (*OUTPUT_CSV_NAMES, PREFLIGHT_MD_NAME, REPORT_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    print()
    preview_cols = [
        "candidate_id",
        "decision",
        "total_return",
        "strategy_vs_qqq",
        "positive_vs_qqq_window_rate",
        "top_symbol",
        "top_symbol_exposure_pct",
        "cost_30bps_strategy_vs_qqq",
        "production_ready",
    ]
    print(artifacts.decision_matrix.loc[:, preview_cols].to_string(index=False))
    return 0


def _print_preflight(
    *,
    args: argparse.Namespace,
    selected_candidates: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    resolved_display = (
        args.candidate_id
        if args.candidate_id
        else list(DEFAULT_CANDIDATE_IDS)
    )
    print("US Market ML long-history QQQ candidate deep dive")
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
    print(f"shortlist candidates: {args.shortlist_candidates}")
    print(f"shortlist decision matrix: {args.shortlist_decision_matrix}")
    print(f"output directory: {args.output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"candidate_id: {resolved_display}")
    print(f"include_default_candidates: {args.include_default_candidates}")
    print(f"selected candidate count: {len(selected_candidates)}")
    print(f"preflight Markdown path: {outputs[PREFLIGHT_MD_NAME]}")
    print()
    print("Selected candidates:")
    display_cols = [
        "candidate_id",
        "candidate_label",
        "model_type",
        "training_mode",
        "target_variant",
        "transaction_cost_bps",
        "shortlist_decision",
    ]
    print(selected_candidates.loc[:, display_cols].to_string(index=False))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local rich long-history QQQ candidate deep-dive bundle."
    )
    parser.add_argument(
        "--model-zoo-results",
        type=Path,
        default=Path(os.environ.get("MODEL_ZOO_RESULTS", str(DEFAULT_MODEL_ZOO_RESULTS))),
    )
    parser.add_argument(
        "--shortlist-candidates",
        type=Path,
        default=Path(
            os.environ.get("SHORTLIST_CANDIDATES", str(DEFAULT_SHORTLIST_CANDIDATES))
        ),
    )
    parser.add_argument(
        "--shortlist-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get(
                "SHORTLIST_DECISION_MATRIX",
                str(DEFAULT_SHORTLIST_DECISION_MATRIX),
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
        "--candidate-id",
        action="append",
        default=_split_env_list(os.environ.get("CANDIDATE_IDS", "")),
        help="Candidate ID to deep-dive; may be repeated.",
    )
    parser.add_argument(
        "--include-default-candidates",
        action="store_true",
        default=os.environ.get("INCLUDE_DEFAULT_CANDIDATES", "0") == "1",
        help="Include candidate_0001 and candidate_0003 in addition to explicit IDs.",
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
    raise SystemExit(run_long_history_qqq_candidate_deep_dive_from_args())
