"""Create a local-only extended-history start-year viability report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    drop_rows_with_missing_model_values,
    get_feature_columns,
    read_local_market_data,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_extended_tiingo_split/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/extended_history_start_year_viability")
DEFAULT_CANDIDATE_START_YEARS: tuple[int, ...] = (1999, 2002, 2005, 2007, 2010, 2013, 2016)
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TEST_START = "2024-01-01"
FEATURE_SET_VERSION = "default_features_v1"
TARGET_COL = "excess_return_vs_spy_20d"
LABEL_HORIZON_DAYS = 20
MIN_LOOKBACK_TRADING_DAYS = 252
TOP_K_VALUES: tuple[int, ...] = (1, 3, 5)

CSV_NAME = "start_year_viability.csv"
MARKDOWN_NAME = "start_year_viability_report.md"
SUMMARY_NAME = "start_year_viability_summary.json"


@dataclass(frozen=True)
class StartYearViabilityRow:
    """One report row for a candidate extended-history start year."""

    start_year: int
    start_date: str
    end_date: str
    available_symbol_count: int
    available_symbols: str
    symbols_available_at_start_count: int
    symbols_available_at_start: str
    symbols_with_enough_lookback_count: int
    symbols_with_enough_lookback: str
    usable_feature_label_rows_after_dropna: int
    usable_train_rows: int
    usable_test_rows: int
    train_symbol_count: int
    test_symbol_count: int
    earliest_symbol_date: str
    latest_symbol_date: str
    early_3y_train_end: str
    early_3y_usable_train_rows: int
    early_3y_train_symbol_count: int
    early_3y_supports_top1: bool
    early_3y_supports_top3: bool
    early_3y_supports_top5: bool
    early_5y_train_end: str
    early_5y_usable_train_rows: int
    early_5y_train_symbol_count: int
    early_5y_supports_top1: bool
    early_5y_supports_top3: bool
    early_5y_supports_top5: bool
    supports_top1: bool
    supports_top3: bool
    supports_top5: bool
    recommended_for_full_universe: bool
    recommended_for_dynamic_universe: bool
    warning_notes: str


def run_extended_history_start_year_viability_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build local start-year viability CSV, Markdown, and JSON artifacts."""

    args = _parse_args(argv)
    print("Extended-history start-year viability report")
    print(f"dataset glob: {args.market_dataset_glob}")
    print(f"output dir: {args.output_dir}")
    print(f"candidate start years: {', '.join(str(year) for year in args.start_years)}")
    print(f"end date: {args.end_date}")
    print()

    print("Reading local market dataset...")
    raw_df = read_local_market_data(args.market_dataset_glob)
    print(
        "Loaded "
        f"{len(raw_df):,} rows across {raw_df['symbol'].nunique():,} symbols "
        f"from {raw_df['date'].min().date()} to {raw_df['date'].max().date()}."
    )

    print("Evaluating candidate start years...")
    rows = evaluate_start_years(
        raw_df,
        start_years=args.start_years,
        end_date=args.end_date,
        test_start=args.test_start,
        min_lookback_trading_days=args.min_lookback_trading_days,
    )

    outputs = write_start_year_viability_artifacts(rows, output_dir=args.output_dir)
    print()
    print("Generated artifacts")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def evaluate_start_years(
    raw_df: pd.DataFrame,
    *,
    start_years: Sequence[int] = DEFAULT_CANDIDATE_START_YEARS,
    end_date: str = DEFAULT_END_DATE,
    test_start: str = DEFAULT_TEST_START,
    min_lookback_trading_days: int = MIN_LOOKBACK_TRADING_DAYS,
) -> list[StartYearViabilityRow]:
    """Evaluate all candidate start years against a local OHLCV frame."""

    all_symbols = _sorted_symbols(raw_df)
    if not all_symbols:
        raise ValueError("Input market data contains no symbols.")

    rows: list[StartYearViabilityRow] = []
    for start_year in start_years:
        print(f"  - {start_year}: building supervised dataset and viability counts")
        rows.append(
            evaluate_start_year(
                raw_df,
                start_year=start_year,
                end_date=end_date,
                test_start=test_start,
                all_symbols=all_symbols,
                min_lookback_trading_days=min_lookback_trading_days,
            )
        )
    return rows


def evaluate_start_year(
    raw_df: pd.DataFrame,
    *,
    start_year: int,
    end_date: str,
    test_start: str,
    all_symbols: Sequence[str] | None = None,
    min_lookback_trading_days: int = MIN_LOOKBACK_TRADING_DAYS,
) -> StartYearViabilityRow:
    """Evaluate one candidate start year."""

    start_ts = pd.Timestamp(f"{start_year}-01-01")
    end_ts = pd.Timestamp(end_date)
    test_start_ts = pd.Timestamp(test_start)
    train_end_ts = test_start_ts - pd.Timedelta(days=1)
    all_symbol_list = list(all_symbols) if all_symbols is not None else _sorted_symbols(raw_df)

    period_raw = raw_df.loc[(raw_df["date"] >= start_ts) & (raw_df["date"] <= end_ts)].copy()
    if period_raw.empty:
        return _empty_row(
            start_year=start_year,
            start_ts=start_ts,
            end_ts=end_ts,
            test_start_ts=test_start_ts,
            all_symbols=all_symbol_list,
        )

    symbol_bounds = _symbol_bounds(period_raw)
    available_symbols = _sorted_symbols(period_raw)
    first_period_date = pd.Timestamp(period_raw["date"].min())
    available_at_start = [
        symbol
        for symbol in available_symbols
        if symbol_bounds.loc[symbol, "first_date"] <= first_period_date
    ]
    enough_lookback_symbols = [
        symbol
        for symbol in available_symbols
        if int(symbol_bounds.loc[symbol, "row_count"]) >= min_lookback_trading_days
    ]

    dataset = build_supervised_dataset(
        period_raw,
        SupervisedDatasetConfig(
            forward_horizons=(5, 10, LABEL_HORIZON_DAYS),
            benchmark_symbol="SPY",
            excess_return_horizons=(LABEL_HORIZON_DAYS,),
            rank_target_col=TARGET_COL,
            rank_col=f"rank_{TARGET_COL}",
            feature_set_version=FEATURE_SET_VERSION,
        ),
    )
    feature_cols = get_feature_columns(dataset, feature_set_version=FEATURE_SET_VERSION)
    usable = drop_rows_with_missing_model_values(dataset, feature_cols, TARGET_COL)
    train_usable = _date_slice(usable, start_ts, train_end_ts)
    test_usable = _date_slice(usable, test_start_ts, end_ts)
    early_3y_end = start_ts + pd.DateOffset(years=3) - pd.Timedelta(days=1)
    early_5y_end = start_ts + pd.DateOffset(years=5) - pd.Timedelta(days=1)
    early_3y_usable = _date_slice(usable, start_ts, min(early_3y_end, end_ts))
    early_5y_usable = _date_slice(usable, start_ts, min(early_5y_end, end_ts))

    train_symbol_count = _symbol_count(train_usable)
    test_symbol_count = _symbol_count(test_usable)
    early_3y_symbol_count = _symbol_count(early_3y_usable)
    early_5y_symbol_count = _symbol_count(early_5y_usable)
    full_count = len(all_symbol_list)

    supports = {
        top_k: train_symbol_count >= top_k and test_symbol_count >= top_k
        for top_k in TOP_K_VALUES
    }
    early_3y_supports = {top_k: early_3y_symbol_count >= top_k for top_k in TOP_K_VALUES}
    early_5y_supports = {top_k: early_5y_symbol_count >= top_k for top_k in TOP_K_VALUES}

    missing_at_start = sorted(set(all_symbol_list).difference(available_at_start))
    insufficient_lookback = sorted(set(all_symbol_list).difference(enough_lookback_symbols))
    train_missing = sorted(set(all_symbol_list).difference(_sorted_symbols(train_usable)))
    test_missing = sorted(set(all_symbol_list).difference(_sorted_symbols(test_usable)))

    recommended_full = (
        len(available_at_start) == full_count
        and len(enough_lookback_symbols) == full_count
        and train_symbol_count == full_count
        and test_symbol_count == full_count
        and early_3y_symbol_count == full_count
        and early_5y_symbol_count == full_count
    )
    recommended_dynamic = supports[5] and (early_3y_supports[5] or early_5y_supports[5])

    warnings = _warning_notes(
        missing_at_start=missing_at_start,
        insufficient_lookback=insufficient_lookback,
        train_missing=train_missing,
        test_missing=test_missing,
        supports=supports,
        early_3y_supports=early_3y_supports,
        early_5y_supports=early_5y_supports,
        min_lookback_trading_days=min_lookback_trading_days,
        recommended_full=recommended_full,
    )

    return StartYearViabilityRow(
        start_year=start_year,
        start_date=_date_text(start_ts),
        end_date=_date_text(end_ts),
        available_symbol_count=len(available_symbols),
        available_symbols=_join_symbols(available_symbols),
        symbols_available_at_start_count=len(available_at_start),
        symbols_available_at_start=_join_symbols(available_at_start),
        symbols_with_enough_lookback_count=len(enough_lookback_symbols),
        symbols_with_enough_lookback=_join_symbols(enough_lookback_symbols),
        usable_feature_label_rows_after_dropna=int(len(usable)),
        usable_train_rows=int(len(train_usable)),
        usable_test_rows=int(len(test_usable)),
        train_symbol_count=train_symbol_count,
        test_symbol_count=test_symbol_count,
        earliest_symbol_date=_date_text(period_raw["date"].min()),
        latest_symbol_date=_date_text(period_raw["date"].max()),
        early_3y_train_end=_date_text(min(early_3y_end, end_ts)),
        early_3y_usable_train_rows=int(len(early_3y_usable)),
        early_3y_train_symbol_count=early_3y_symbol_count,
        early_3y_supports_top1=early_3y_supports[1],
        early_3y_supports_top3=early_3y_supports[3],
        early_3y_supports_top5=early_3y_supports[5],
        early_5y_train_end=_date_text(min(early_5y_end, end_ts)),
        early_5y_usable_train_rows=int(len(early_5y_usable)),
        early_5y_train_symbol_count=early_5y_symbol_count,
        early_5y_supports_top1=early_5y_supports[1],
        early_5y_supports_top3=early_5y_supports[3],
        early_5y_supports_top5=early_5y_supports[5],
        supports_top1=supports[1],
        supports_top3=supports[3],
        supports_top5=supports[5],
        recommended_for_full_universe=recommended_full,
        recommended_for_dynamic_universe=recommended_dynamic,
        warning_notes=warnings,
    )


def write_start_year_viability_artifacts(
    rows: Sequence[StartYearViabilityRow],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write CSV, Markdown, and JSON report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME
    markdown_path = output_dir / MARKDOWN_NAME
    summary_path = output_dir / SUMMARY_NAME

    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(build_markdown_report(frame), encoding="utf-8")
    summary_path.write_text(
        json.dumps(build_summary(frame), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "CSV report": csv_path,
        "Markdown report": markdown_path,
        "JSON summary": summary_path,
    }


def build_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Build a compact machine-readable summary."""

    return {
        "candidate_start_years": [int(value) for value in frame["start_year"].tolist()],
        "recommended_full_universe_start_year": _recommended_start_year(
            frame,
            "recommended_for_full_universe",
        ),
        "recommended_dynamic_universe_start_year": _recommended_start_year(
            frame,
            "recommended_for_dynamic_universe",
        ),
        "full_universe_usable_start_years": _flagged_start_years(
            frame,
            "recommended_for_full_universe",
        ),
        "dynamic_universe_usable_start_years": _flagged_start_years(
            frame,
            "recommended_for_dynamic_universe",
        ),
        "feature_set_version": FEATURE_SET_VERSION,
        "target_col": TARGET_COL,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "min_lookback_trading_days": MIN_LOOKBACK_TRADING_DAYS,
    }


def build_markdown_report(frame: pd.DataFrame) -> str:
    """Render the human-readable start-year viability report."""

    full_year = _recommended_start_year(frame, "recommended_for_full_universe")
    dynamic_year = _recommended_start_year(frame, "recommended_for_dynamic_universe")
    too_early = frame.loc[~frame["recommended_for_full_universe"], "start_year"].tolist()

    recommendation = frame[
        [
            "start_year",
            "available_symbol_count",
            "symbols_available_at_start_count",
            "symbols_with_enough_lookback_count",
            "train_symbol_count",
            "test_symbol_count",
            "early_3y_train_symbol_count",
            "early_5y_train_symbol_count",
            "supports_top1",
            "supports_top3",
            "supports_top5",
            "recommended_for_full_universe",
            "recommended_for_dynamic_universe",
        ]
    ]
    decision = frame[
        [
            "start_year",
            "recommended_for_full_universe",
            "recommended_for_dynamic_universe",
            "warning_notes",
        ]
    ]

    lines = [
        "# Extended-History Start-Year Viability Report",
        "",
        "Local-only diagnostic for the Tiingo/Massive extended ETF dataset.",
        "",
        "## Configuration",
        "",
        f"- Feature set: `{FEATURE_SET_VERSION}`",
        f"- Target: `{TARGET_COL}`",
        f"- Label horizon: `{LABEL_HORIZON_DAYS}` trading days",
        f"- Minimum lookback threshold: `{MIN_LOOKBACK_TRADING_DAYS}` trading rows",
        "- Modern test sanity period: `2024-01-01` through report end date",
        "",
        "## Recommendation Table",
        "",
        _markdown_table(recommendation),
        "",
        "## Use Or Avoid",
        "",
        _markdown_table(decision),
        "",
        "## Conclusion",
        "",
        f"- Recommended start year for full-universe studies: {_year_or_none(full_year)}.",
        f"- Recommended start year for dynamic-universe studies: {_year_or_none(dynamic_year)}.",
        "- Start years too early for the full modern universe: "
        f"{_year_list_or_none(too_early)}.",
        "- Full-universe means every dataset symbol has start-year availability, "
        "enough lookback, and usable feature/label rows in early and modern checks.",
        "- Dynamic-universe means unavailable symbols are not forced and the usable "
        "symbol set supports top1/top3/top5 selection in both early and modern checks.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local-only extended-history start-year viability report."
    )
    parser.add_argument("--market-dataset-glob", default=DEFAULT_DATASET_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--start-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_START_YEARS),
    )
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument(
        "--min-lookback-trading-days",
        type=int,
        default=MIN_LOOKBACK_TRADING_DAYS,
    )
    return parser.parse_args(argv)


def _empty_row(
    *,
    start_year: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    all_symbols: Sequence[str],
) -> StartYearViabilityRow:
    early_3y_end = start_ts + pd.DateOffset(years=3) - pd.Timedelta(days=1)
    early_5y_end = start_ts + pd.DateOffset(years=5) - pd.Timedelta(days=1)
    missing = _join_symbols(all_symbols)
    return StartYearViabilityRow(
        start_year=start_year,
        start_date=_date_text(start_ts),
        end_date=_date_text(end_ts),
        available_symbol_count=0,
        available_symbols="",
        symbols_available_at_start_count=0,
        symbols_available_at_start="",
        symbols_with_enough_lookback_count=0,
        symbols_with_enough_lookback="",
        usable_feature_label_rows_after_dropna=0,
        usable_train_rows=0,
        usable_test_rows=0,
        train_symbol_count=0,
        test_symbol_count=0,
        earliest_symbol_date="",
        latest_symbol_date="",
        early_3y_train_end=_date_text(min(early_3y_end, end_ts)),
        early_3y_usable_train_rows=0,
        early_3y_train_symbol_count=0,
        early_3y_supports_top1=False,
        early_3y_supports_top3=False,
        early_3y_supports_top5=False,
        early_5y_train_end=_date_text(min(early_5y_end, end_ts)),
        early_5y_usable_train_rows=0,
        early_5y_train_symbol_count=0,
        early_5y_supports_top1=False,
        early_5y_supports_top3=False,
        early_5y_supports_top5=False,
        supports_top1=False,
        supports_top3=False,
        supports_top5=False,
        recommended_for_full_universe=False,
        recommended_for_dynamic_universe=False,
        warning_notes=(
            f"no data in period; all symbols missing: {missing}; "
            f"test_start={_date_text(test_start_ts)}"
        ),
    )


def _symbol_bounds(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("symbol", sort=True)
        .agg(first_date=("date", "min"), last_date=("date", "max"), row_count=("date", "size"))
        .sort_index()
    )


def _date_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] >= start) & (df["date"] <= end)].copy()


def _symbol_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["symbol"].nunique())


def _sorted_symbols(df: pd.DataFrame) -> list[str]:
    if "symbol" not in df.columns or df.empty:
        return []
    return sorted(str(symbol) for symbol in df["symbol"].dropna().unique())


def _join_symbols(symbols: Sequence[str]) -> str:
    return ",".join(sorted(str(symbol) for symbol in symbols))


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _warning_notes(
    *,
    missing_at_start: Sequence[str],
    insufficient_lookback: Sequence[str],
    train_missing: Sequence[str],
    test_missing: Sequence[str],
    supports: dict[int, bool],
    early_3y_supports: dict[int, bool],
    early_5y_supports: dict[int, bool],
    min_lookback_trading_days: int,
    recommended_full: bool,
) -> str:
    notes: list[str] = []
    if missing_at_start:
        notes.append(f"symbols unavailable at start: {_join_symbols(missing_at_start)}")
    if insufficient_lookback:
        notes.append(
            "symbols below "
            f"{min_lookback_trading_days}-row lookback: {_join_symbols(insufficient_lookback)}"
        )
    if train_missing:
        notes.append(f"symbols without usable train rows: {_join_symbols(train_missing)}")
    if test_missing:
        notes.append(f"symbols without usable test rows: {_join_symbols(test_missing)}")
    if not supports[5]:
        notes.append("modern train/test usable set does not support top5")
    if not early_3y_supports[5]:
        notes.append("first 3-year training window does not support top5")
    if not early_5y_supports[5]:
        notes.append("first 5-year training window does not support top5")
    if not recommended_full:
        notes.append("avoid for full modern universe")
    return "; ".join(notes) if notes else "ok"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(_format_markdown_value)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def _format_markdown_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def _recommended_start_year(frame: pd.DataFrame, flag_col: str) -> int | None:
    recommended = frame.loc[frame[flag_col], "start_year"]
    if recommended.empty:
        return None
    return int(recommended.min())


def _flagged_start_years(frame: pd.DataFrame, flag_col: str) -> list[int]:
    return [int(value) for value in frame.loc[frame[flag_col], "start_year"].tolist()]


def _year_or_none(year: int | None) -> str:
    return str(year) if year is not None else "none"


def _year_list_or_none(years: Sequence[int]) -> str:
    if not years:
        return "none"
    return ", ".join(str(year) for year in years)


if __name__ == "__main__":
    raise SystemExit(run_extended_history_start_year_viability_report_from_args())
