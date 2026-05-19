"""Run the compact Tiingo-only QQQ timing baseline study."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data.local_reader import read_local_market_data
from us_market_ml.qqq_timing import (
    OPTIONAL_RISK_OFF_SYMBOLS,
    REQUIRED_SYMBOLS,
    available_symbol_summary,
    default_qqq_timing_strategy_specs,
    prepare_qqq_timing_prices,
    run_qqq_timing_study,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_qqq_timing_baseline_study")
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRANSACTION_COST_BPS = 10.0

OUTPUT_PREFIX = "tiingo_qqq_timing_baseline"
RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"

DATASET_SOURCE_LABEL = "tiingo_only"
STUDY_STAGE = "qqq_timing_baseline"


def run_tiingo_qqq_timing_baseline_study_from_args(argv: list[str] | None = None) -> int:
    """Run or dry-run the Tiingo-only QQQ timing baseline study."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.market_dataset_glob)
    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    available_symbols = set(prices.columns[prices.notna().any()].astype(str))
    strategy_specs, strategy_warnings = default_qqq_timing_strategy_specs(available_symbols)
    outputs = _output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_preflight_markdown(
        path=outputs[PREFLIGHT_MD_NAME],
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        strategy_count=len(strategy_specs),
        strategy_names=[spec.strategy_name for spec in strategy_specs],
        strategy_warnings=strategy_warnings,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )
    _print_preflight(
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        strategy_count=len(strategy_specs),
        strategy_names=[spec.strategy_name for spec in strategy_specs],
        strategy_warnings=strategy_warnings,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    study = run_qqq_timing_study(
        market_data,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    study.results.to_csv(outputs[RESULTS_NAME], index=False)
    study.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    study.trades.to_csv(outputs[TRADES_NAME], index=False)

    print()
    print("Tiingo QQQ timing baseline study complete.")
    for output_name in (RESULTS_NAME, DAILY_EQUITY_NAME, TRADES_NAME, PREFLIGHT_MD_NAME):
        print(f"{output_name}: {outputs[output_name]}")
    if study.warnings:
        print("warnings:")
        for warning in study.warnings:
            print(f"  - {warning}")
    if not study.results.empty:
        preview_cols = [
            "strategy_name",
            "total_return",
            "strategy_vs_qqq",
            "max_drawdown",
            "exposure_rate_to_qqq",
            "transaction_count",
            "estimated_transaction_cost",
        ]
        print()
        print(study.results.loc[:, preview_cols].to_string(index=False))
    return 0


def _write_preflight_markdown(
    *,
    path: Path,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    strategy_count: int,
    strategy_names: Sequence[str],
    strategy_warnings: Sequence[str],
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    lines = [
        "# Tiingo QQQ Timing Baseline Study Preflight",
        "",
        "## Scope",
        "",
        "This is a QQQ timing/risk-on-risk-off study, not a QQQ-core overlay and "
        "not a production signal export.",
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Study stage: `{STUDY_STAGE}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Requested date range: `{start_date}` to `{end_date}`",
        f"- Actual price range: `{prices.index.min().date()}` to `{prices.index.max().date()}`",
        f"- Trading days: `{len(prices)}`",
        f"- Transaction cost: `{transaction_cost_bps:.4f}` bps per unit turnover",
        f"- Dry run: `{str(dry_run).lower()}`",
        "",
        "## Symbols",
        "",
        f"- Required symbols: `{', '.join(REQUIRED_SYMBOLS)}`",
        f"- Required present: `{_join_or_none(required_present)}`",
        f"- Required missing: `{_join_or_none(required_missing)}`",
        f"- Optional defensive symbols: `{', '.join(OPTIONAL_RISK_OFF_SYMBOLS)}`",
        f"- Optional present: `{_join_or_none(optional_present)}`",
        f"- Optional missing: `{_join_or_none(optional_missing)}`",
        "",
        "## Strategies",
        "",
        f"- Strategy count: `{strategy_count}`",
        "",
        *[f"- `{name}`" for name in strategy_names],
        "",
        "## Timing Convention",
        "",
        (
            "Signals are computed from close data available on date T. The selected "
            "position applies on the next trading day, and transaction costs are "
            "charged when the new active asset takes effect."
        ),
        "",
        "## Guardrails",
        "",
        "- Local-only Tiingo research input.",
        "- No S3/cloud/GPU.",
        "- No tree_screen or tree models.",
        "- No MLP.",
        "- No benchmark_gap_features_v4.",
        "- No fixed QQQ core portfolio.",
        "- No production signal export.",
        "- No production-ready label.",
    ]
    if strategy_warnings:
        lines.extend(["", "## Skipped Optional Variants", ""])
        lines.extend(f"- {warning}" for warning in strategy_warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_preflight(
    *,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    strategy_count: int,
    strategy_names: Sequence[str],
    strategy_warnings: Sequence[str],
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    print("US Market ML Tiingo QQQ timing baseline study")
    print(f"dataset source: {DATASET_SOURCE_LABEL}")
    print(f"market dataset glob: {market_dataset_glob}")
    print(f"output directory: {output_dir}")
    print(f"price range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"trading days: {len(prices)}")
    print(f"transaction cost bps: {transaction_cost_bps:g}")
    print(f"strategy count: {strategy_count}")
    print(f"strategy names: {', '.join(strategy_names)}")
    print(f"required symbols: {', '.join(REQUIRED_SYMBOLS)}")
    print(f"required symbols present: {_join_or_none(required_present)}")
    print(f"required symbols missing: {_join_or_none(required_missing)}")
    print(f"optional symbols present: {_join_or_none(optional_present)}")
    print(f"missing optional symbols: {_join_or_none(optional_missing)}")
    for warning in strategy_warnings:
        print(f"warning: {warning}")
    print("timing convention: close T signal applies from the next trading day")
    print("No S3/cloud/GPU, no tree_screen, no MLP, no feature v4, no production export.")
    print("Fixed QQQ core portfolio: false")
    print("Production-ready status: false")
    if dry_run:
        print("DRY_RUN: preflight only; study outputs were not generated.")


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        RESULTS_NAME: output_dir / RESULTS_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
    }


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the compact Tiingo-only QQQ timing baseline study."
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", DEFAULT_END_DATE))
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=float(os.environ.get("TRANSACTION_COST_BPS", DEFAULT_TRANSACTION_COST_BPS)),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


run_tiingo_qqq_timing_baseline_from_args = run_tiingo_qqq_timing_baseline_study_from_args


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_timing_baseline_study_from_args())
