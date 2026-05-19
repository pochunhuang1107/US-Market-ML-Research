"""Run the narrow Tiingo-only QQQ defensive timing robustness sweep."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.data.local_reader import read_local_market_data
from us_market_ml.qqq_timing import (
    DEFENSIVE_TIMING_SUBPERIODS,
    NARROW_DEFENSIVE_TIMING_FAMILIES,
    OPTIONAL_RISK_OFF_SYMBOLS,
    REQUIRED_SYMBOLS,
    QQQDefensiveTimingStrategySpec,
    available_symbol_summary,
    default_qqq_defensive_timing_narrow_sweep_specs,
    prepare_qqq_timing_prices,
    run_qqq_defensive_timing_narrow_sweep,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_qqq_defensive_timing_narrow_sweep")
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRANSACTION_COST_BPS = 10.0

OUTPUT_PREFIX = "tiingo_qqq_defensive_timing_narrow"
RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
SUBPERIOD_RESULTS_NAME = f"{OUTPUT_PREFIX}_subperiod_results.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"

DATASET_SOURCE_LABEL = "tiingo_only"
STUDY_STAGE = "qqq_defensive_timing_narrow_sweep"


def run_tiingo_qqq_defensive_timing_narrow_sweep_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the narrow Tiingo-only QQQ defensive timing sweep."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.market_dataset_glob)
    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    available_symbols = set(prices.columns[prices.notna().any()].astype(str))
    strategy_specs, strategy_warnings = default_qqq_defensive_timing_narrow_sweep_specs(
        available_symbols,
    )
    timing_config_count = sum(1 for spec in strategy_specs if not spec.is_benchmark)
    total_strategy_count = len(strategy_specs)
    outputs = _output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_preflight_markdown(
        path=outputs[PREFLIGHT_MD_NAME],
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        strategy_specs=strategy_specs,
        strategy_warnings=strategy_warnings,
        timing_config_count=timing_config_count,
        total_strategy_count=total_strategy_count,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )
    _print_preflight(
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        strategy_specs=strategy_specs,
        strategy_warnings=strategy_warnings,
        timing_config_count=timing_config_count,
        total_strategy_count=total_strategy_count,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    sweep = run_qqq_defensive_timing_narrow_sweep(
        market_data,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    sweep.results.to_csv(outputs[RESULTS_NAME], index=False)
    sweep.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    sweep.trades.to_csv(outputs[TRADES_NAME], index=False)
    sweep.subperiod_results.to_csv(outputs[SUBPERIOD_RESULTS_NAME], index=False)

    print()
    print("Tiingo QQQ defensive timing narrow sweep complete.")
    for output_name in (
        RESULTS_NAME,
        DAILY_EQUITY_NAME,
        TRADES_NAME,
        SUBPERIOD_RESULTS_NAME,
        PREFLIGHT_MD_NAME,
    ):
        print(f"{output_name}: {outputs[output_name]}")
    if sweep.warnings:
        print("warnings:")
        for warning in sweep.warnings:
            print(f"  - {warning}")
    if not sweep.results.empty:
        preview_cols = [
            "strategy_name",
            "family",
            "total_return",
            "strategy_vs_qqq",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
            "transaction_count",
        ]
        print()
        print(
            sweep.results.sort_values(
                ["total_return", "return_to_drawdown"],
                ascending=[False, False],
                kind="mergesort",
            )
            .loc[:, preview_cols]
            .head(25)
            .to_string(index=False)
        )
    return 0


def _write_preflight_markdown(
    *,
    path: Path,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    strategy_specs: Sequence[QQQDefensiveTimingStrategySpec],
    strategy_warnings: Sequence[str],
    timing_config_count: int,
    total_strategy_count: int,
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    family_counts = _family_counts(strategy_specs)
    lines = [
        "# Tiingo QQQ Defensive Timing Narrow Sweep Preflight",
        "",
        "## Scope",
        "",
        (
            "This is a narrow defensive QQQ timing sweep around the best prior "
            "cash/SPY candidates. It is research-only, Tiingo-only, and not a "
            "production signal export."
        ),
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
        "## Strategy Counts",
        "",
        f"- Timing config count: `{timing_config_count}`",
        f"- Benchmark reference count: `{total_strategy_count - timing_config_count}`",
        f"- Total result row count: `{total_strategy_count}`",
        (
            "- Scale note: the exact requested grid expands to 426 timing configs; "
            "this is still a narrow, local-only run."
        ),
        "",
        "## Family Counts",
        "",
        *[
            f"- `{family}`: `{family_counts.get(family, 0)}`"
            for family in NARROW_DEFENSIVE_TIMING_FAMILIES
        ],
        "",
        "## Parameter Ranges",
        "",
        (
            "- Total-return SPY defensive: slow SMA 165/170/175/180/185/190/"
            "195/200/205, buffer -0.5%/0%/+0.5%/+1%, min hold 0/3/5/10, "
            "risk-off SPY."
        ),
        (
            "- Drawdown-adjusted cash defensive: slow SMA 180/190/200/210/220, "
            "buffer -2%/-1%/0%/+0.5%/+1%/+2%, min hold 5/10/15, risk-off cash."
        ),
        (
            "- Low-turnover crossover cash: fast SMA 60/75/90/100, slow SMA "
            "180/200/220/240 with fast < slow, buffer -1.5%/-1%/-0.5%/0%, "
            "min hold 0/5/10, risk-off cash."
        ),
        "",
        "## Subperiods",
        "",
        *[
            f"- `{name}`: `{start}` to `{end}`"
            for name, start, end in DEFENSIVE_TIMING_SUBPERIODS
        ],
        "",
        "## Timing Convention",
        "",
        (
            "Signals are computed from close data available on date T. The selected "
            "position applies starting the next trading day, and transaction costs "
            "are charged when that next-day active asset changes."
        ),
        "",
        "## Guardrails",
        "",
        "- Local-only Tiingo research input.",
        "- No S3/cloud/GPU.",
        "- No tree_screen or tree models.",
        "- No MLP.",
        "- No benchmark_gap_features_v4.",
        "- No production signal export.",
        "- No production-ready label.",
        "- Do not modify the Market Data repo.",
    ]
    if strategy_warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in strategy_warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_preflight(
    *,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    strategy_specs: Sequence[QQQDefensiveTimingStrategySpec],
    strategy_warnings: Sequence[str],
    timing_config_count: int,
    total_strategy_count: int,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    family_counts = _family_counts(strategy_specs)
    print("US Market ML Tiingo QQQ defensive timing narrow sweep")
    print(f"dataset source: {DATASET_SOURCE_LABEL}")
    print(f"market dataset glob: {market_dataset_glob}")
    print(f"output directory: {output_dir}")
    print(f"price range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"trading days: {len(prices)}")
    print(f"transaction cost bps: {transaction_cost_bps:g}")
    print(f"timing config count: {timing_config_count}")
    print(f"benchmark reference count: {total_strategy_count - timing_config_count}")
    print(f"total result row count: {total_strategy_count}")
    for family in NARROW_DEFENSIVE_TIMING_FAMILIES:
        print(f"family count {family}: {family_counts.get(family, 0)}")
    print(f"required symbols: {', '.join(REQUIRED_SYMBOLS)}")
    print(f"required symbols present: {_join_or_none(required_present)}")
    print(f"required symbols missing: {_join_or_none(required_missing)}")
    print(f"optional symbols: {', '.join(OPTIONAL_RISK_OFF_SYMBOLS)}")
    print(f"optional symbols present: {_join_or_none(optional_present)}")
    print(f"missing optional symbols: {_join_or_none(optional_missing)}")
    print("benchmark references: qqq_buy_hold, spy_buy_hold, cash")
    for warning in strategy_warnings:
        print(f"warning: {warning}")
    print("scale control: exact requested narrow grid is 426 timing configs.")
    print("timing convention: close T signal applies from the next trading day")
    print("No S3/cloud/GPU, no tree_screen, no tree models, no MLP, no feature v4.")
    print("No production signal export. Production-ready status: false")
    print("Market Data repo mutation: false")
    if dry_run:
        print("DRY_RUN: preflight only; sweep outputs were not generated.")


def _family_counts(
    strategy_specs: Sequence[QQQDefensiveTimingStrategySpec],
) -> dict[str, int]:
    counts = Counter(spec.family for spec in strategy_specs if not spec.is_benchmark)
    return dict(counts)


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        RESULTS_NAME: output_dir / RESULTS_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        SUBPERIOD_RESULTS_NAME: output_dir / SUBPERIOD_RESULTS_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
    }


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the narrow Tiingo-only QQQ defensive timing sweep."
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    return parser.parse_args(argv)


run_tiingo_qqq_defensive_narrow_sweep_from_args = (
    run_tiingo_qqq_defensive_timing_narrow_sweep_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_narrow_sweep_from_args())
