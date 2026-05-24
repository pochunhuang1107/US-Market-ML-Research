"""Run the focused long-history QQQ SPY core/candidate sleeve study."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.long_history_spy_core_sleeve import (
    DATASET_SOURCE_LABEL,
    DEFAULT_CANDIDATE_IDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPY_CORE_WEIGHTS,
    OUTPUT_CSV_NAMES,
    PREFLIGHT_MD_NAME,
    build_long_history_spy_core_sleeve_artifacts,
    generate_core_sleeve_weight_grid,
    output_paths,
    render_preflight_markdown,
    select_source_candidates,
    write_long_history_spy_core_sleeve_artifacts,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

DEFAULT_CANDIDATE_DEEP_DIVE_DAILY_EQUITY = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_daily_equity.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_daily_drawdown.csv"
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
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_spy_core_sleeve")


def run_long_history_qqq_spy_core_sleeve_from_args(argv: list[str] | None = None) -> int:
    """Run or dry-run the focused SPY core/candidate sleeve study."""

    args = _parse_args(argv)
    _ensure_local_file(args.candidate_deep_dive_daily_equity, "--candidate-deep-dive-daily-equity")
    _ensure_local_file(
        args.candidate_deep_dive_daily_drawdown,
        "--candidate-deep-dive-daily-drawdown",
    )
    _ensure_local_file(args.candidate_deep_dive_summary, "--candidate-deep-dive-summary")
    _ensure_local_file(
        args.candidate_deep_dive_decision_matrix,
        "--candidate-deep-dive-decision-matrix",
    )
    _ensure_local_file(
        args.candidate_deep_dive_selected_symbols,
        "--candidate-deep-dive-selected-symbols",
    )
    _ensure_local_only_path(str(args.output_dir), "--output-dir")
    _ensure_local_only_path(str(args.report_output_dir), "--report-output-dir")

    daily_equity = pd.read_csv(args.candidate_deep_dive_daily_equity)
    daily_drawdown = pd.read_csv(args.candidate_deep_dive_daily_drawdown)
    deep_dive_summary = pd.read_csv(args.candidate_deep_dive_summary)
    deep_dive_decision_matrix = pd.read_csv(args.candidate_deep_dive_decision_matrix)
    selected_symbols = pd.read_csv(args.candidate_deep_dive_selected_symbols)
    source_candidates = select_source_candidates(
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=tuple(args.candidate_id),
    )
    weight_grid = generate_core_sleeve_weight_grid(tuple(args.spy_core_weight))

    outputs = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[PREFLIGHT_MD_NAME].write_text(
        render_preflight_markdown(
            source_candidates=source_candidates,
            weight_grid=weight_grid,
            candidate_deep_dive_daily_equity=args.candidate_deep_dive_daily_equity,
            candidate_deep_dive_daily_drawdown=args.candidate_deep_dive_daily_drawdown,
            candidate_deep_dive_summary=args.candidate_deep_dive_summary,
            candidate_deep_dive_decision_matrix=args.candidate_deep_dive_decision_matrix,
            candidate_deep_dive_selected_symbols=args.candidate_deep_dive_selected_symbols,
            selected_symbol_rows=len(selected_symbols),
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
        weight_grid=weight_grid,
        selected_symbol_rows=len(selected_symbols),
        outputs=outputs,
    )
    if args.dry_run:
        return 0

    artifacts = build_long_history_spy_core_sleeve_artifacts(
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=tuple(args.candidate_id),
        spy_core_weights=tuple(args.spy_core_weight),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_long_history_spy_core_sleeve_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    from us_market_ml.cli.create_long_history_qqq_spy_core_sleeve_report import (
        run_long_history_qqq_spy_core_sleeve_report_from_args,
    )

    run_long_history_qqq_spy_core_sleeve_report_from_args(
        [
            "--sweep-output-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.report_output_dir),
        ]
    )

    print()
    print("Long-history QQQ SPY core-sleeve artifacts written.")
    for output_name in (*OUTPUT_CSV_NAMES, PREFLIGHT_MD_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    print(f"report output directory: {args.report_output_dir}")
    print()
    preview_cols = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "decision",
        "production_ready",
    ]
    print(artifacts.decision_matrix.loc[:, preview_cols].to_string(index=False))
    return 0


def _print_preflight(
    *,
    args: argparse.Namespace,
    source_candidates: pd.DataFrame,
    weight_grid: pd.DataFrame,
    selected_symbol_rows: int,
    outputs: dict[str, Path],
) -> None:
    resolved_display = args.candidate_id if args.candidate_id else list(DEFAULT_CANDIDATE_IDS)
    print("US Market ML long-history QQQ SPY core-sleeve study")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("broad sweep: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"dry_run: {args.dry_run}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"candidate deep-dive daily equity: {args.candidate_deep_dive_daily_equity}")
    print(f"candidate deep-dive daily drawdown: {args.candidate_deep_dive_daily_drawdown}")
    print(f"candidate deep-dive summary: {args.candidate_deep_dive_summary}")
    print(f"candidate deep-dive decision matrix: {args.candidate_deep_dive_decision_matrix}")
    print(f"candidate deep-dive selected symbols: {args.candidate_deep_dive_selected_symbols}")
    print(f"deep-dive selected symbol rows: {selected_symbol_rows}")
    print(f"output directory: {args.output_dir}")
    print(f"report output directory: {args.report_output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"candidate_id: {resolved_display}")
    print("portfolio accounting: daily rebalanced fixed SPY/candidate weights")
    print(f"selected source candidate count: {len(source_candidates)}")
    print(f"portfolio config count: {len(source_candidates) * len(weight_grid)}")
    print(f"preflight Markdown path: {outputs[PREFLIGHT_MD_NAME]}")
    print()
    print("Candidate IDs:")
    print(", ".join(source_candidates["candidate_id"].astype(str).tolist()))
    print()
    print("Core weight grid:")
    print(weight_grid.to_string(index=False))
    print()
    print("Selected source candidates:")
    display_cols = [
        "candidate_id",
        "candidate_label",
        "model_type",
        "training_mode",
        "target_variant",
        "strategy_vs_qqq",
        "max_drawdown",
        "return_to_drawdown",
        "top_symbol",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "source_decision",
    ]
    print(source_candidates.loc[:, display_cols].to_string(index=False))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local focused long-history QQQ SPY core-sleeve study."
    )
    parser.add_argument(
        "--candidate-deep-dive-daily-equity",
        type=Path,
        default=Path(
            os.environ.get(
                "CANDIDATE_DEEP_DIVE_DAILY_EQUITY",
                str(DEFAULT_CANDIDATE_DEEP_DIVE_DAILY_EQUITY),
            )
        ),
    )
    parser.add_argument(
        "--candidate-deep-dive-daily-drawdown",
        type=Path,
        default=Path(
            os.environ.get(
                "CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN",
                str(DEFAULT_CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN),
            )
        ),
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
        help="Candidate ID to include; may be repeated.",
    )
    parser.add_argument(
        "--spy-core-weight",
        action="append",
        type=float,
        default=_split_float_env_list(
            os.environ.get("SPY_CORE_WEIGHTS", ""),
            DEFAULT_SPY_CORE_WEIGHTS,
        ),
        help="SPY core weight to include; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    return parser.parse_args(argv)


def _split_env_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _split_float_env_list(value: str, default: tuple[float, ...]) -> list[float]:
    if not value:
        return list(default)
    return [float(item) for item in value.replace(",", " ").split() if item.strip()]


def _ensure_local_only_path(path: str, flag_name: str) -> None:
    if str(path).lower().startswith(
        ("s3://", "s3:/", "gs://", "gs:/", "http://", "http:/", "https://", "https:/")
    ):
        raise ValueError(f"{flag_name} must be a local path/glob, not {path!r}.")


def _ensure_local_file(path: Path, flag_name: str) -> None:
    _ensure_local_only_path(str(path), flag_name)
    if not path.exists():
        raise FileNotFoundError(f"{flag_name} does not exist: {path}")


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_spy_core_sleeve_from_args())
