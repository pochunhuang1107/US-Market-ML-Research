"""Create the long-history QQQ core-sleeve rich replay report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.long_history_core_sleeve_rich_replay import (
    DAILY_DRAWDOWN_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_CANDIDATE_SELECTED_SYMBOLS,
    DEFAULT_CANDIDATE_SYMBOL_EXPOSURE,
    DEFAULT_CANDIDATE_SYMBOL_PERIODS,
    DEFAULT_DECISION_MATRIX,
    DEFAULT_DECISION_RECOMMENDATIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINEMENT_DAILY_DRAWDOWN,
    DEFAULT_REFINEMENT_DAILY_EQUITY,
    DEFAULT_REFINEMENT_RESULTS,
    DEFAULT_REFINEMENT_ROLLING_WINDOWS,
    DEFAULT_REFINEMENT_SUBPERIODS,
    REPORT_NAME,
    ROLLING_WINDOWS_NAME,
    SELECTED_SYMBOLS_NAME,
    SLEEVE_CONTRIBUTION_NAME,
    SUBPERIODS_NAME,
    SUMMARY_NAME,
    SYMBOL_EXPOSURE_NAME,
    build_long_history_core_sleeve_rich_replay_artifacts,
    read_long_history_core_sleeve_rich_replay_inputs,
    write_long_history_core_sleeve_rich_replay_artifacts,
)


def run_long_history_qqq_core_sleeve_rich_replay_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the rich replay report from existing local artifacts."""

    args = _parse_args(argv)
    for label, value in [
        ("--core-sleeve-decision-recommendations", args.core_sleeve_decision_recommendations),
        ("--core-sleeve-decision-matrix", args.core_sleeve_decision_matrix),
        ("--core-sleeve-refinement-results", args.core_sleeve_refinement_results),
        ("--core-sleeve-refinement-daily-equity", args.core_sleeve_refinement_daily_equity),
        (
            "--core-sleeve-refinement-daily-drawdown",
            args.core_sleeve_refinement_daily_drawdown,
        ),
        ("--core-sleeve-refinement-subperiods", args.core_sleeve_refinement_subperiods),
        (
            "--core-sleeve-refinement-rolling-windows",
            args.core_sleeve_refinement_rolling_windows,
        ),
        ("--candidate-selected-symbols", args.candidate_selected_symbols),
        ("--candidate-symbol-exposure", args.candidate_symbol_exposure),
        ("--candidate-symbol-periods", args.candidate_symbol_periods),
        ("--output-dir", args.output_dir),
    ]:
        _ensure_local_only_path(str(value), label)

    inputs = read_long_history_core_sleeve_rich_replay_inputs(
        decision_recommendations=args.core_sleeve_decision_recommendations,
        decision_matrix=args.core_sleeve_decision_matrix,
        refinement_results=args.core_sleeve_refinement_results,
        refinement_daily_equity=args.core_sleeve_refinement_daily_equity,
        refinement_daily_drawdown=args.core_sleeve_refinement_daily_drawdown,
        refinement_subperiods=args.core_sleeve_refinement_subperiods,
        refinement_rolling_windows=args.core_sleeve_refinement_rolling_windows,
        candidate_selected_symbols=args.candidate_selected_symbols,
        candidate_symbol_exposure=args.candidate_symbol_exposure,
        candidate_symbol_periods=args.candidate_symbol_periods,
    )
    artifacts = build_long_history_core_sleeve_rich_replay_artifacts(inputs)
    paths = write_long_history_core_sleeve_rich_replay_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print("US Market ML long-history QQQ core-sleeve rich replay report")
    print("report/deep-dive only: true")
    print("local-only: true")
    print("broad sweep: false")
    print("model training: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"output dir: {args.output_dir}")
    print(f"summary path: {paths[SUMMARY_NAME]}")
    print(f"daily equity path: {paths[DAILY_EQUITY_NAME]}")
    print(f"daily drawdown path: {paths[DAILY_DRAWDOWN_NAME]}")
    print(f"subperiods path: {paths[SUBPERIODS_NAME]}")
    print(f"rolling windows path: {paths[ROLLING_WINDOWS_NAME]}")
    print(f"sleeve contribution path: {paths[SLEEVE_CONTRIBUTION_NAME]}")
    print(f"selected symbols path: {paths[SELECTED_SYMBOLS_NAME]}")
    print(f"symbol exposure path: {paths[SYMBOL_EXPOSURE_NAME]}")
    print(f"decision matrix path: {paths[DECISION_MATRIX_NAME]}")
    print(f"report path: {paths[REPORT_NAME]}")
    print(f"summary rows: {len(artifacts.summary)}")
    print(f"selected-symbol rows: {len(artifacts.selected_symbols)}")
    print(f"production-ready rows: {int(artifacts.summary['production_ready'].sum())}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the long-history QQQ core-sleeve rich replay report from completed "
            "decision, refinement, and candidate deep-dive artifacts."
        )
    )
    parser.add_argument(
        "--core-sleeve-decision-recommendations",
        type=Path,
        default=Path(
            os.environ.get(
                "CORE_SLEEVE_DECISION_RECOMMENDATIONS",
                str(DEFAULT_DECISION_RECOMMENDATIONS),
            )
        ),
    )
    parser.add_argument(
        "--core-sleeve-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get("CORE_SLEEVE_DECISION_MATRIX", str(DEFAULT_DECISION_MATRIX))
        ),
    )
    parser.add_argument(
        "--core-sleeve-refinement-results",
        type=Path,
        default=Path(
            os.environ.get("CORE_SLEEVE_REFINEMENT_RESULTS", str(DEFAULT_REFINEMENT_RESULTS))
        ),
    )
    parser.add_argument(
        "--core-sleeve-refinement-daily-equity",
        type=Path,
        default=Path(
            os.environ.get(
                "CORE_SLEEVE_REFINEMENT_DAILY_EQUITY",
                str(DEFAULT_REFINEMENT_DAILY_EQUITY),
            )
        ),
    )
    parser.add_argument(
        "--core-sleeve-refinement-daily-drawdown",
        type=Path,
        default=Path(
            os.environ.get(
                "CORE_SLEEVE_REFINEMENT_DAILY_DRAWDOWN",
                str(DEFAULT_REFINEMENT_DAILY_DRAWDOWN),
            )
        ),
    )
    parser.add_argument(
        "--core-sleeve-refinement-subperiods",
        type=Path,
        default=Path(
            os.environ.get(
                "CORE_SLEEVE_REFINEMENT_SUBPERIODS",
                str(DEFAULT_REFINEMENT_SUBPERIODS),
            )
        ),
    )
    parser.add_argument(
        "--core-sleeve-refinement-rolling-windows",
        type=Path,
        default=Path(
            os.environ.get(
                "CORE_SLEEVE_REFINEMENT_ROLLING_WINDOWS",
                str(DEFAULT_REFINEMENT_ROLLING_WINDOWS),
            )
        ),
    )
    parser.add_argument(
        "--candidate-selected-symbols",
        type=Path,
        default=Path(
            os.environ.get("CANDIDATE_SELECTED_SYMBOLS", str(DEFAULT_CANDIDATE_SELECTED_SYMBOLS))
        ),
    )
    parser.add_argument(
        "--candidate-symbol-exposure",
        type=Path,
        default=Path(
            os.environ.get("CANDIDATE_SYMBOL_EXPOSURE", str(DEFAULT_CANDIDATE_SYMBOL_EXPOSURE))
        ),
    )
    parser.add_argument(
        "--candidate-symbol-periods",
        type=Path,
        default=Path(
            os.environ.get("CANDIDATE_SYMBOL_PERIODS", str(DEFAULT_CANDIDATE_SYMBOL_PERIODS))
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    return parser.parse_args(argv)


def _ensure_local_only_path(value: str, label: str) -> None:
    lowered = value.lower()
    blocked_prefixes = ("s3://", "gs://", "az://", "abfs://", "http://", "https://")
    if lowered.startswith(blocked_prefixes):
        raise ValueError(f"{label} must be a local filesystem path, got: {value}")


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_core_sleeve_rich_replay_report_from_args())
