"""Run the focused long-history QQQ core-sleeve refinement study."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_refinement import (
    DATASET_SOURCE_LABEL,
    DEFAULT_CANDIDATE_IDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PORTFOLIO_ACCOUNTING_MODES,
    OUTPUT_CSV_NAMES,
    PREFLIGHT_MD_NAME,
    accounting_mode_availability,
    build_long_history_core_sleeve_refinement_artifacts,
    generate_refinement_weight_grid,
    output_paths,
    render_preflight_markdown,
    resolve_portfolio_accounting_modes,
    select_source_candidates,
    write_long_history_core_sleeve_refinement_artifacts,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

DEFAULT_SPY_CORE_SLEEVE_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_spy_core_sleeve/"
    "long_history_qqq_spy_core_sleeve_results.csv"
)
DEFAULT_SPY_CORE_SLEEVE_DAILY_EQUITY = Path(
    "artifacts/sweeps/long_history_qqq_spy_core_sleeve/"
    "long_history_qqq_spy_core_sleeve_daily_equity.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_DAILY_EQUITY = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_daily_equity.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_SUMMARY = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_summary.csv"
)
DEFAULT_CANDIDATE_DEEP_DIVE_DECISION_MATRIX = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_decision_matrix.csv"
)
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_refinement")


def run_long_history_qqq_core_sleeve_refinement_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the focused core-sleeve refinement study."""

    args = _parse_args(argv)
    _ensure_local_file(args.spy_core_sleeve_results, "--spy-core-sleeve-results")
    _ensure_local_file(args.spy_core_sleeve_daily_equity, "--spy-core-sleeve-daily-equity")
    _ensure_local_file(args.candidate_deep_dive_daily_equity, "--candidate-deep-dive-daily-equity")
    _ensure_local_file(args.candidate_deep_dive_summary, "--candidate-deep-dive-summary")
    _ensure_local_file(
        args.candidate_deep_dive_decision_matrix,
        "--candidate-deep-dive-decision-matrix",
    )
    _ensure_local_only_path(str(args.output_dir), "--output-dir")
    _ensure_local_only_path(str(args.report_output_dir), "--report-output-dir")

    previous_results = pd.read_csv(args.spy_core_sleeve_results)
    previous_daily_equity = pd.read_csv(args.spy_core_sleeve_daily_equity)
    daily_equity = pd.read_csv(args.candidate_deep_dive_daily_equity)
    deep_dive_summary = pd.read_csv(args.candidate_deep_dive_summary)
    deep_dive_decision_matrix = pd.read_csv(args.candidate_deep_dive_decision_matrix)
    source_candidates = select_source_candidates(
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=tuple(args.candidate_id),
    )
    candidate_ids = tuple(source_candidates["candidate_id"].astype(str))
    weight_grid = generate_refinement_weight_grid(candidate_ids)
    accounting_modes = resolve_portfolio_accounting_modes(tuple(args.portfolio_accounting_mode))
    availability = accounting_mode_availability(
        daily_equity=daily_equity,
        candidate_ids=candidate_ids,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    outputs = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[PREFLIGHT_MD_NAME].write_text(
        render_preflight_markdown(
            source_candidates=source_candidates,
            weight_grid=weight_grid,
            accounting_modes=accounting_modes,
            accounting_availability=availability,
            spy_core_sleeve_results=args.spy_core_sleeve_results,
            spy_core_sleeve_daily_equity=args.spy_core_sleeve_daily_equity,
            candidate_deep_dive_daily_equity=args.candidate_deep_dive_daily_equity,
            candidate_deep_dive_summary=args.candidate_deep_dive_summary,
            candidate_deep_dive_decision_matrix=args.candidate_deep_dive_decision_matrix,
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
        accounting_modes=accounting_modes,
        availability=availability,
        previous_results=previous_results,
        previous_daily_equity=previous_daily_equity,
        outputs=outputs,
    )
    if args.dry_run:
        return 0

    artifacts = build_long_history_core_sleeve_refinement_artifacts(
        daily_equity=daily_equity,
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=candidate_ids,
        portfolio_accounting_modes=accounting_modes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_long_history_core_sleeve_refinement_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    from us_market_ml.cli.create_long_history_qqq_core_sleeve_refinement_report import (
        run_long_history_qqq_core_sleeve_refinement_report_from_args,
    )

    run_long_history_qqq_core_sleeve_refinement_report_from_args(
        [
            "--sweep-output-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.report_output_dir),
            "--previous-spy-core-sleeve-results",
            str(args.spy_core_sleeve_results),
        ]
    )

    print()
    print("Long-history QQQ core-sleeve refinement artifacts written.")
    for output_name in (*OUTPUT_CSV_NAMES, PREFLIGHT_MD_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    print(f"report output directory: {args.report_output_dir}")
    print()
    preview_cols = [
        "candidate_id",
        "portfolio_accounting_mode",
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
    accounting_modes: tuple[str, ...],
    availability: pd.DataFrame,
    previous_results: pd.DataFrame,
    previous_daily_equity: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    resolved_display = args.candidate_id if args.candidate_id else list(DEFAULT_CANDIDATE_IDS)
    print("US Market ML long-history QQQ core-sleeve refinement")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("broad model zoo: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print("Market Data repo modification: false")
    print(f"dry_run: {args.dry_run}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"previous SPY core-sleeve results: {args.spy_core_sleeve_results}")
    print(f"previous SPY core-sleeve daily equity: {args.spy_core_sleeve_daily_equity}")
    print(f"previous result rows: {len(previous_results)}")
    print(f"previous daily equity rows: {len(previous_daily_equity)}")
    print(f"candidate deep-dive daily equity: {args.candidate_deep_dive_daily_equity}")
    print(f"candidate deep-dive summary: {args.candidate_deep_dive_summary}")
    print(f"candidate deep-dive decision matrix: {args.candidate_deep_dive_decision_matrix}")
    print(f"output directory: {args.output_dir}")
    print(f"report output directory: {args.report_output_dir}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"candidate_id: {resolved_display}")
    print(f"selected source candidate count: {len(source_candidates)}")
    print(f"portfolio accounting mode count: {len(accounting_modes)}")
    print(f"portfolio config count before candidate-date skips: {len(weight_grid) * len(accounting_modes)}")
    print(f"preflight Markdown path: {outputs[PREFLIGHT_MD_NAME]}")
    print()
    print("Candidate IDs:")
    print(", ".join(source_candidates["candidate_id"].astype(str).tolist()))
    print()
    print("Accounting modes:")
    print("\n".join(accounting_modes))
    print()
    print("Candidate rebalance-aligned availability:")
    print(availability.to_string(index=False))
    print()
    print("Fine core weight grid:")
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
        description="Run the local focused long-history QQQ core-sleeve refinement."
    )
    parser.add_argument(
        "--spy-core-sleeve-results",
        type=Path,
        default=Path(
            os.environ.get(
                "SPY_CORE_SLEEVE_RESULTS",
                str(DEFAULT_SPY_CORE_SLEEVE_RESULTS),
            )
        ),
    )
    parser.add_argument(
        "--spy-core-sleeve-daily-equity",
        type=Path,
        default=Path(
            os.environ.get(
                "SPY_CORE_SLEEVE_DAILY_EQUITY",
                str(DEFAULT_SPY_CORE_SLEEVE_DAILY_EQUITY),
            )
        ),
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
        "--portfolio-accounting-mode",
        action="append",
        default=_split_env_list(
            os.environ.get("PORTFOLIO_ACCOUNTING_MODES", "")
        )
        or list(DEFAULT_PORTFOLIO_ACCOUNTING_MODES),
        help="Portfolio accounting mode to include; may be repeated.",
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
    raise SystemExit(run_long_history_qqq_core_sleeve_refinement_from_args())
