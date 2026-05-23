"""Rich replay artifacts for selected long-history QQQ-relative candidates.

This module is local research infrastructure. It replays a very small set of
already-shortlisted candidates, writes diagnostic tables, and deliberately
avoids broad sweeps, MLP batches, model cards, production signal exports, and
cloud access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.long_history_shortlist_validation import (
    DATASET_SOURCE_LABEL,
    PRICE_MODE,
    SUBPERIODS,
    replay_shortlist_candidate,
    summarize_rolling_windows,
)
from us_market_ml.long_history_training import END_DATE, START_DATE

OUTPUT_PREFIX = "long_history_qqq_candidate_deep_dive"
DEFAULT_OUTPUT_DIR = Path("artifacts/deep_dives/long_history_qqq_candidate_deep_dive")
DEFAULT_CANDIDATE_IDS: tuple[str, ...] = ("candidate_0001", "candidate_0003")
DEFAULT_COST_SENSITIVITY_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
DAILY_DRAWDOWN_NAME = f"{OUTPUT_PREFIX}_daily_drawdown.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
SELECTED_SYMBOLS_NAME = f"{OUTPUT_PREFIX}_selected_symbols.csv"
SYMBOL_EXPOSURE_NAME = f"{OUTPUT_PREFIX}_symbol_exposure.csv"
SYMBOL_PERIODS_NAME = f"{OUTPUT_PREFIX}_symbol_periods.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
COST_SENSITIVITY_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

OUTPUT_CSV_NAMES = (
    SUMMARY_NAME,
    DAILY_EQUITY_NAME,
    DAILY_DRAWDOWN_NAME,
    TRADES_NAME,
    SELECTED_SYMBOLS_NAME,
    SYMBOL_EXPOSURE_NAME,
    SYMBOL_PERIODS_NAME,
    SUBPERIODS_NAME,
    ROLLING_WINDOWS_NAME,
    COST_SENSITIVITY_NAME,
    DECISION_MATRIX_NAME,
)

CONFIG_COLUMNS = [
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "recency_decay_profile",
    "base_weight",
    "recent_window_years",
    "finetune_window_years",
    "finetune_lr_scale",
]

SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    *CONFIG_COLUMNS,
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_like",
    "return_to_drawdown",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "drawdown_improvement_vs_qqq",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "average_strategy_vs_qqq",
    "worst_strategy_vs_qqq",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "unique_symbol_count",
    "shortlist_decision",
]

DAILY_EQUITY_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "strategy_name",
    "active_symbol",
    "active_symbols",
    "daily_return",
    "equity",
    "drawdown",
    "gross_return",
    "transaction_cost",
    "turnover",
]

DAILY_DRAWDOWN_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "strategy_name",
    "equity",
    "drawdown",
]

TRADE_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "selected_symbol",
    "previous_symbol",
    "new_symbol",
    "predicted_score",
    "rank",
    "turnover",
    "estimated_transaction_cost",
    "reason",
]

SELECTED_SYMBOL_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "symbol",
    "target_weight",
    "predicted_score",
    "rank",
    "allocation_rule",
    "rebalance_frequency",
]

SYMBOL_EXPOSURE_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "symbol",
    "rank_by_exposure",
    "selected_rebalance_count",
    "exposure_days",
    "exposure_pct",
    "total_return_contribution",
    "is_top_symbol",
    "is_top3_symbol",
]

SYMBOL_PERIOD_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "symbol",
    "start_date",
    "end_date",
    "trading_days",
    "segment_return",
    "qqq_segment_return",
    "segment_vs_qqq",
    "max_drawdown_in_segment",
]

SUBPERIOD_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "period_name",
    "period_start",
    "period_end",
    "total_return",
    "qqq_total_return",
    "spy_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "transaction_count",
    "top_symbol",
]

ROLLING_WINDOW_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "window_label",
    "window_start",
    "window_end",
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "underperformed_qqq",
    "materially_improved_drawdown",
]

COST_SENSITIVITY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "transaction_cost_bps",
    "total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "return_to_drawdown",
    "decision_at_cost",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    *CONFIG_COLUMNS,
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "return_to_drawdown",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "unique_symbol_count",
    "cost_20bps_strategy_vs_qqq",
    "cost_30bps_strategy_vs_qqq",
    "covid_crash_recovery_strategy_vs_qqq",
    "ai_bull_2023_2024_strategy_vs_qqq",
    "recent_2025_2026_strategy_vs_qqq",
    "shortlist_decision",
    "decision",
    "rationale",
    "production_ready",
]


@dataclass(frozen=True)
class LongHistoryCandidateDeepDiveArtifacts:
    """CSV-ready rich deep-dive outputs."""

    selected_candidates: pd.DataFrame
    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    trades: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    symbol_periods: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    decision_matrix: pd.DataFrame


def resolve_deep_dive_candidate_ids(
    candidate_ids: tuple[str, ...] = (),
    *,
    include_default_candidates: bool = False,
) -> tuple[str, ...]:
    """Resolve explicit/default candidate IDs while preserving order."""

    if candidate_ids:
        ordered = list(candidate_ids)
        if include_default_candidates:
            ordered = [*DEFAULT_CANDIDATE_IDS, *ordered]
    else:
        ordered = list(DEFAULT_CANDIDATE_IDS)
    return tuple(dict.fromkeys(str(candidate_id) for candidate_id in ordered))


def select_deep_dive_candidates(
    *,
    shortlist_candidates: pd.DataFrame,
    shortlist_decision_matrix: pd.DataFrame,
    candidate_ids: tuple[str, ...] = (),
    include_default_candidates: bool = False,
) -> pd.DataFrame:
    """Select candidate rows from the shortlist validation outputs."""

    if shortlist_candidates.empty:
        raise ValueError("shortlist candidates input is empty.")
    _validate_columns(
        shortlist_candidates,
        ["candidate_id", *CONFIG_COLUMNS],
        "shortlist candidates",
    )
    _validate_columns(
        shortlist_decision_matrix,
        ["candidate_id", "decision"],
        "shortlist decision matrix",
    )

    resolved_ids = resolve_deep_dive_candidate_ids(
        candidate_ids,
        include_default_candidates=include_default_candidates,
    )
    available = set(shortlist_candidates["candidate_id"].astype(str))
    unknown = [candidate_id for candidate_id in resolved_ids if candidate_id not in available]
    if unknown:
        raise ValueError(f"Unknown candidate_id values: {', '.join(unknown)}")

    selected = (
        shortlist_candidates.assign(candidate_id=lambda df: df["candidate_id"].astype(str))
        .set_index("candidate_id", drop=False)
        .loc[list(resolved_ids)]
        .reset_index(drop=True)
        .copy()
    )
    if "candidate_label" not in selected.columns:
        selected["candidate_label"] = selected.apply(_candidate_label, axis=1)
    if "decision" in selected.columns:
        selected = selected.rename(columns={"decision": "model_zoo_decision"})

    decision_cols = [
        column
        for column in [
            "candidate_id",
            "decision",
            "rationale",
            "positive_vs_qqq_window_rate",
            "top_symbol",
            "top_symbol_exposure_pct",
            "top3_symbol_exposure_pct",
            "unique_symbol_count",
        ]
        if column in shortlist_decision_matrix.columns
    ]
    decisions = shortlist_decision_matrix.loc[:, decision_cols].copy()
    decisions["candidate_id"] = decisions["candidate_id"].astype(str)
    decisions = decisions.rename(
        columns={
            "decision": "shortlist_decision",
            "rationale": "shortlist_rationale",
            "positive_vs_qqq_window_rate": "shortlist_positive_vs_qqq_window_rate",
            "top_symbol": "shortlist_top_symbol",
            "top_symbol_exposure_pct": "shortlist_top_symbol_exposure_pct",
            "top3_symbol_exposure_pct": "shortlist_top3_symbol_exposure_pct",
            "unique_symbol_count": "shortlist_unique_symbol_count",
        }
    )
    selected = selected.merge(decisions, on="candidate_id", how="left")
    for column in ["shortlist_decision", "shortlist_rationale", "model_zoo_decision"]:
        if column not in selected.columns:
            selected[column] = pd.NA
    return selected


def build_long_history_candidate_deep_dive_artifacts(
    *,
    raw_df: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_history_rows: int = 252,
    min_train_eligible_rows: int = 1000,
    cost_sensitivity_bps: tuple[float, ...] = DEFAULT_COST_SENSITIVITY_BPS,
) -> LongHistoryCandidateDeepDiveArtifacts:
    """Replay selected candidates and build rich deep-dive artifact tables."""

    if selected_candidates.empty:
        raise ValueError("At least one selected candidate is required.")

    replays = [
        replay_shortlist_candidate(
            raw_df=raw_df,
            candidate=candidate,
            start_date=start_date,
            end_date=end_date,
            min_history_rows=min_history_rows,
            min_train_eligible_rows=min_train_eligible_rows,
            cost_sensitivity_bps=cost_sensitivity_bps,
        )
        for _, candidate in selected_candidates.iterrows()
    ]

    summary = _concat_or_empty(
        [_summary_row_from_replay(replay) for replay in replays],
        SUMMARY_COLUMNS,
    )
    daily_equity = _concat_or_empty(
        [build_daily_equity_rows(replay.daily_equity) for replay in replays],
        DAILY_EQUITY_COLUMNS,
    )
    daily_drawdown = _concat_or_empty(
        [build_daily_drawdown_rows(replay.daily_equity) for replay in replays],
        DAILY_DRAWDOWN_COLUMNS,
    )
    trades = _concat_or_empty(
        [build_trade_rebalance_log(replay) for replay in replays],
        TRADE_COLUMNS,
    )
    selected_symbols = _concat_or_empty(
        [build_selected_symbols_log(replay) for replay in replays],
        SELECTED_SYMBOL_COLUMNS,
    )
    symbol_exposure = _concat_or_empty(
        [build_symbol_exposure_rows(replay.selected_symbols) for replay in replays],
        SYMBOL_EXPOSURE_COLUMNS,
    )
    symbol_periods = _concat_or_empty(
        [build_symbol_periods(replay.daily_equity) for replay in replays],
        SYMBOL_PERIOD_COLUMNS,
    )
    subperiods = _concat_or_empty(
        [_with_candidate_label(replay.subperiods, replay.candidate) for replay in replays],
        SUBPERIOD_COLUMNS,
    )
    rolling_windows = _concat_or_empty(
        [_with_candidate_label(replay.rolling_windows, replay.candidate) for replay in replays],
        ROLLING_WINDOW_COLUMNS,
    )
    cost_sensitivity = _concat_or_empty(
        [_with_candidate_label(replay.cost_sensitivity, replay.candidate) for replay in replays],
        COST_SENSITIVITY_COLUMNS,
    )
    decision_matrix = build_deep_dive_decision_matrix(
        summary=summary,
        subperiods=subperiods,
        cost_sensitivity=cost_sensitivity,
    )

    return LongHistoryCandidateDeepDiveArtifacts(
        selected_candidates=selected_candidates,
        summary=summary,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        trades=trades,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        symbol_periods=symbol_periods,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        decision_matrix=decision_matrix,
    )


def build_daily_equity_rows(wide_daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Convert wide candidate/benchmark equity into long replay rows."""

    if wide_daily_equity.empty:
        return pd.DataFrame(columns=DAILY_EQUITY_COLUMNS)
    source = wide_daily_equity.copy()
    source["date"] = pd.to_datetime(source["date"])
    rows: list[dict[str, object]] = []
    series_specs = (
        (
            "candidate_strategy",
            "daily_return",
            "equity",
            "drawdown",
            "gross_return",
            "transaction_cost",
            "turnover",
        ),
        ("QQQ buy-and-hold", "qqq_daily_return", "qqq_equity", "qqq_drawdown", None, None, None),
        ("SPY buy-and-hold", "spy_daily_return", "spy_equity", "spy_drawdown", None, None, None),
        ("cash", "cash_daily_return", "cash_equity", "cash_drawdown", None, None, None),
    )
    for _, row in source.iterrows():
        for (
            strategy_name,
            return_col,
            equity_col,
            drawdown_col,
            gross_col,
            cost_col,
            turnover_col,
        ) in series_specs:
            rows.append(
                {
                    "date": row["date"],
                    "candidate_id": row["candidate_id"],
                    "candidate_label": row["candidate_label"],
                    "strategy_name": strategy_name,
                    "active_symbol": row.get("active_symbol", "")
                    if strategy_name == "candidate_strategy"
                    else "",
                    "active_symbols": row.get("active_symbols", "")
                    if strategy_name == "candidate_strategy"
                    else "",
                    "daily_return": row.get(return_col, np.nan),
                    "equity": row.get(equity_col, np.nan),
                    "drawdown": row.get(drawdown_col, np.nan),
                    "gross_return": row.get(gross_col, row.get(return_col, np.nan))
                    if gross_col
                    else row.get(return_col, np.nan),
                    "transaction_cost": row.get(cost_col, 0.0) if cost_col else 0.0,
                    "turnover": row.get(turnover_col, 0.0) if turnover_col else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=DAILY_EQUITY_COLUMNS)


def build_daily_drawdown_rows(wide_daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Build long drawdown rows for the strategy, QQQ, and SPY."""

    if wide_daily_equity.empty:
        return pd.DataFrame(columns=DAILY_DRAWDOWN_COLUMNS)
    source = wide_daily_equity.copy()
    source["date"] = pd.to_datetime(source["date"])
    rows: list[dict[str, object]] = []
    series_specs = (
        ("candidate_strategy", "equity", "drawdown"),
        ("QQQ buy-and-hold", "qqq_equity", "qqq_drawdown"),
        ("SPY buy-and-hold", "spy_equity", "spy_drawdown"),
    )
    for _, row in source.iterrows():
        for strategy_name, equity_col, drawdown_col in series_specs:
            rows.append(
                {
                    "date": row["date"],
                    "candidate_id": row["candidate_id"],
                    "candidate_label": row["candidate_label"],
                    "strategy_name": strategy_name,
                    "equity": row.get(equity_col, np.nan),
                    "drawdown": row.get(drawdown_col, np.nan),
                }
            )
    return pd.DataFrame(rows, columns=DAILY_DRAWDOWN_COLUMNS)


def build_selected_symbols_log(replay: object) -> pd.DataFrame:
    """Return selected symbols at each rebalance with rank and score."""

    weights = replay.rebalance_weights.copy()
    if weights.empty:
        return pd.DataFrame(columns=SELECTED_SYMBOL_COLUMNS)
    weights["date"] = pd.to_datetime(weights["date"])
    rankings = _prediction_rankings(replay.predictions)
    out = weights.merge(rankings, on=["date", "symbol"], how="left")
    out["candidate_id"] = replay.candidate["candidate_id"]
    out["candidate_label"] = replay.candidate["candidate_label"]
    out["allocation_rule"] = replay.candidate["allocation_rule"]
    out["rebalance_frequency"] = replay.candidate["rebalance_frequency"]
    out = out.rename(columns={"prediction": "predicted_score"})
    for column in SELECTED_SYMBOL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, SELECTED_SYMBOL_COLUMNS].sort_values(
        ["date", "rank", "symbol"],
        kind="mergesort",
    )


def build_trade_rebalance_log(replay: object) -> pd.DataFrame:
    """Build rebalance rows with previous/new symbol, rank, score, and reason."""

    trades = replay.trades.copy()
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    trades["date"] = pd.to_datetime(trades["date"])
    selected = build_selected_symbols_log(replay)
    selected_best = (
        selected.sort_values(["date", "rank"], kind="mergesort")
        .groupby("date", sort=False)
        .head(1)
        .loc[:, ["date", "predicted_score", "rank"]]
    )
    out = trades.merge(selected_best, on="date", how="left")
    out["candidate_label"] = replay.candidate["candidate_label"]
    out["previous_symbol"] = out.get("previous_symbols", "").fillna("")
    out["new_symbol"] = out.get("new_symbols", "").fillna("")
    out["reason"] = [
        _trade_reason(previous_symbol, new_symbol, turnover)
        for previous_symbol, new_symbol, turnover in zip(
            out["previous_symbol"],
            out["new_symbol"],
            pd.to_numeric(out["turnover"], errors="coerce").fillna(0.0),
            strict=True,
        )
    ]
    for column in TRADE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, TRADE_COLUMNS]


def build_symbol_exposure_rows(selected_symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-symbol exposure/concentration rows."""

    if selected_symbol_summary.empty:
        return pd.DataFrame(columns=SYMBOL_EXPOSURE_COLUMNS)
    out = selected_symbol_summary.copy()
    out = out.sort_values(
        ["candidate_id", "exposure_pct", "selected_rebalance_count", "symbol"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    out["rank_by_exposure"] = out.groupby("candidate_id", sort=False).cumcount() + 1
    out["is_top_symbol"] = out["rank_by_exposure"].eq(1)
    out["is_top3_symbol"] = out["rank_by_exposure"].le(3)
    for column in SYMBOL_EXPOSURE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, SYMBOL_EXPOSURE_COLUMNS]


def build_symbol_periods(wide_daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Build continuous selected-symbol holding segments."""

    if wide_daily_equity.empty:
        return pd.DataFrame(columns=SYMBOL_PERIOD_COLUMNS)
    source = wide_daily_equity.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["active_symbols"] = source["active_symbols"].fillna("").astype(str)
    source = source.loc[source["active_symbols"].ne("")].sort_values("date", kind="mergesort")
    if source.empty:
        return pd.DataFrame(columns=SYMBOL_PERIOD_COLUMNS)

    rows: list[dict[str, object]] = []
    group_id = source["active_symbols"].ne(source["active_symbols"].shift()).cumsum()
    for _, segment in source.groupby(group_id, sort=False):
        symbol = str(segment["active_symbols"].iloc[0])
        segment_return = _compound_return(segment["daily_return"])
        qqq_return = _compound_return(segment["qqq_daily_return"])
        rows.append(
            {
                "candidate_id": segment["candidate_id"].iloc[0],
                "candidate_label": segment["candidate_label"].iloc[0],
                "symbol": symbol,
                "start_date": segment["date"].iloc[0],
                "end_date": segment["date"].iloc[-1],
                "trading_days": int(len(segment)),
                "segment_return": segment_return,
                "qqq_segment_return": qqq_return,
                "segment_vs_qqq": segment_return - qqq_return,
                "max_drawdown_in_segment": _max_drawdown_from_returns(segment["daily_return"]),
            }
        )
    return pd.DataFrame(rows, columns=SYMBOL_PERIOD_COLUMNS)


def build_deep_dive_decision_matrix(
    *,
    summary: pd.DataFrame,
    subperiods: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final research-only decision matrix."""

    if summary.empty:
        return pd.DataFrame(columns=DECISION_MATRIX_COLUMNS)
    matrix = summary.copy()
    cost_edges = cost_sensitivity.pivot_table(
        index="candidate_id",
        columns="transaction_cost_bps",
        values="strategy_vs_qqq",
        aggfunc="first",
    )
    for cost_bps in (20.0, 30.0):
        column_name = f"cost_{int(cost_bps)}bps_strategy_vs_qqq"
        matrix = matrix.merge(
            cost_edges.get(cost_bps, pd.Series(dtype="float64")).rename(column_name),
            left_on="candidate_id",
            right_index=True,
            how="left",
        )
    subperiod_edges = subperiods.pivot_table(
        index="candidate_id",
        columns="period_name",
        values="strategy_vs_qqq",
        aggfunc="first",
    )
    for period_name in ("covid_crash_recovery", "ai_bull_2023_2024", "recent_2025_2026"):
        matrix = matrix.merge(
            subperiod_edges.get(period_name, pd.Series(dtype="float64")).rename(
                f"{period_name}_strategy_vs_qqq"
            ),
            left_on="candidate_id",
            right_index=True,
            how="left",
        )

    candidate_0001_positive_rate = _candidate_metric(
        matrix,
        "candidate_0001",
        "positive_vs_qqq_window_rate",
    )
    decisions = matrix.apply(
        lambda row: _deep_dive_decision_and_rationale(
            row,
            candidate_0001_positive_rate=candidate_0001_positive_rate,
        ),
        axis=1,
    )
    matrix["decision"] = [item[0] for item in decisions]
    matrix["rationale"] = [item[1] for item in decisions]
    matrix["production_ready"] = False
    for column in DECISION_MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = pd.NA
    return matrix.loc[:, DECISION_MATRIX_COLUMNS]


def write_long_history_candidate_deep_dive_artifacts(
    *,
    artifacts: LongHistoryCandidateDeepDiveArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all deep-dive CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir)
    artifacts.summary.to_csv(outputs[SUMMARY_NAME], index=False)
    artifacts.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    artifacts.daily_drawdown.to_csv(outputs[DAILY_DRAWDOWN_NAME], index=False)
    artifacts.trades.to_csv(outputs[TRADES_NAME], index=False)
    artifacts.selected_symbols.to_csv(outputs[SELECTED_SYMBOLS_NAME], index=False)
    artifacts.symbol_exposure.to_csv(outputs[SYMBOL_EXPOSURE_NAME], index=False)
    artifacts.symbol_periods.to_csv(outputs[SYMBOL_PERIODS_NAME], index=False)
    artifacts.subperiods.to_csv(outputs[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(outputs[ROLLING_WINDOWS_NAME], index=False)
    artifacts.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_NAME], index=False)
    artifacts.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    return outputs


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for a deep-dive directory."""

    return {
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        DAILY_DRAWDOWN_NAME: output_dir / DAILY_DRAWDOWN_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        SELECTED_SYMBOLS_NAME: output_dir / SELECTED_SYMBOLS_NAME,
        SYMBOL_EXPOSURE_NAME: output_dir / SYMBOL_EXPOSURE_NAME,
        SYMBOL_PERIODS_NAME: output_dir / SYMBOL_PERIODS_NAME,
        SUBPERIODS_NAME: output_dir / SUBPERIODS_NAME,
        ROLLING_WINDOWS_NAME: output_dir / ROLLING_WINDOWS_NAME,
        COST_SENSITIVITY_NAME: output_dir / COST_SENSITIVITY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }


def render_preflight_markdown(
    *,
    selected_candidates: pd.DataFrame,
    market_dataset_glob: str,
    model_zoo_results: Path,
    shortlist_candidates: Path,
    shortlist_decision_matrix: Path,
    model_zoo_result_count: int,
    output_dir: Path,
    start_date: str,
    end_date: str,
    candidate_ids: tuple[str, ...],
    include_default_candidates: bool,
    dry_run: bool,
) -> str:
    """Render research-only preflight markdown for the candidate deep dive."""

    lines = [
        "# Long-History QQQ Candidate Deep Dive Preflight",
        "",
        "Rich replay/deep-dive bundle for selected long-history QQQ-relative candidates.",
        "",
        "## Controls",
        "",
        "- Local-only: `true`",
        "- S3/cloud access: `false`",
        "- Broad sweep: `false`",
        "- MLP batch: `false`",
        "- Production signal export: `false`",
        "- Model card: `false`",
        "- Production-ready marking: `false`",
        "- Primary benchmark: `QQQ buy-and-hold`",
        "- Secondary references: `SPY buy-and-hold`, `cash`",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Price mode: `{PRICE_MODE}`",
        "",
        "## Inputs",
        "",
        f"- Model-zoo results: `{model_zoo_results}`",
        f"- Model-zoo result rows: `{model_zoo_result_count}`",
        f"- Shortlist candidates: `{shortlist_candidates}`",
        f"- Shortlist decision matrix: `{shortlist_decision_matrix}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Start date: `{start_date}`",
        f"- End date: `{end_date}`",
        f"- Requested candidate IDs: `{list(candidate_ids) if candidate_ids else 'defaults'}`",
        f"- Include default candidates: `{include_default_candidates}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Selected Candidates",
        "",
        _markdown_table(
            selected_candidates,
            [
                "candidate_id",
                "candidate_label",
                "model_type",
                "training_mode",
                "target_variant",
                "transaction_cost_bps",
                "shortlist_decision",
                "shortlist_rationale",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _summary_row_from_replay(replay: object) -> pd.DataFrame:
    metrics = replay.metrics
    candidate = replay.candidate
    rolling_summary = summarize_rolling_windows(replay.rolling_windows)
    rolling_row = rolling_summary.iloc[0] if not rolling_summary.empty else pd.Series(dtype=object)
    concentration = (
        replay.symbol_exposure.iloc[0]
        if not replay.symbol_exposure.empty
        else pd.Series(dtype=object)
    )
    row = {
        "candidate_id": candidate["candidate_id"],
        "candidate_label": candidate["candidate_label"],
        "model_type": candidate["model_type"],
        "training_mode": candidate["training_mode"],
        "target_variant": candidate["target_variant"],
        "feature_set_version": candidate["feature_set_version"],
        "allocation_rule": candidate["allocation_rule"],
        "rebalance_frequency": candidate["rebalance_frequency"],
        "min_holding_days": candidate["min_holding_days"],
        "switch_margin": candidate["switch_margin"],
        "transaction_cost_bps": candidate["transaction_cost_bps"],
        "recency_decay_profile": candidate.get("recency_decay_profile", pd.NA),
        "base_weight": candidate.get("base_weight", pd.NA),
        "recent_window_years": candidate.get("recent_window_years", pd.NA),
        "finetune_window_years": candidate.get("finetune_window_years", pd.NA),
        "finetune_lr_scale": candidate.get("finetune_lr_scale", pd.NA),
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe_like": metrics.sharpe_like,
        "return_to_drawdown": metrics.return_to_drawdown,
        "transaction_count": metrics.transaction_count,
        "average_turnover": metrics.average_turnover,
        "estimated_transaction_cost": metrics.estimated_transaction_cost,
        "qqq_buy_hold_total_return": metrics.qqq_buy_hold_total_return,
        "spy_buy_hold_total_return": metrics.spy_buy_hold_total_return,
        "strategy_vs_qqq": metrics.strategy_vs_qqq,
        "strategy_vs_spy": metrics.strategy_vs_spy,
        "qqq_return_capture": metrics.qqq_return_capture,
        "beats_qqq_return": metrics.beats_qqq_return,
        "beats_qqq_return_to_drawdown": metrics.beats_qqq_return_to_drawdown,
        "drawdown_improvement_vs_qqq": metrics.drawdown_improvement_vs_qqq,
        "rolling_window_count": rolling_row.get("rolling_window_count", pd.NA),
        "positive_vs_qqq_window_rate": rolling_row.get("positive_vs_qqq_window_rate", pd.NA),
        "qqq_underperformance_rate": rolling_row.get("qqq_underperformance_rate", pd.NA),
        "average_strategy_vs_qqq": rolling_row.get("average_strategy_vs_qqq", pd.NA),
        "worst_strategy_vs_qqq": rolling_row.get("worst_strategy_vs_qqq", pd.NA),
        "top_symbol": concentration.get("top_symbol", pd.NA),
        "top_symbol_exposure_pct": concentration.get("top_symbol_exposure_pct", pd.NA),
        "top3_symbol_exposure_pct": concentration.get("top3_symbol_exposure_pct", pd.NA),
        "unique_symbol_count": concentration.get("unique_symbol_count", pd.NA),
        "shortlist_decision": candidate.get("shortlist_decision", pd.NA),
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def _with_candidate_label(frame: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["candidate_label"] = candidate["candidate_label"]
    return out


def _prediction_rankings(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["date", "symbol", "prediction", "rank"])
    out = predictions.loc[:, ["date", "symbol", "prediction"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out = out.sort_values(["date", "prediction", "symbol"], ascending=[True, False, True])
    out["rank"] = out.groupby("date", sort=False).cumcount() + 1
    return out


def _trade_reason(previous_symbol: object, new_symbol: object, turnover: object) -> str:
    previous_text = str(previous_symbol or "")
    new_text = str(new_symbol or "")
    turnover_value = _float(turnover)
    if not previous_text and new_text:
        return "initial_selection"
    if previous_text != new_text and turnover_value > 0.0:
        return "symbol_change"
    if turnover_value > 0.0:
        return "weight_change"
    return "scheduled_rebalance_no_change"


def _deep_dive_decision_and_rationale(
    row: pd.Series,
    *,
    candidate_0001_positive_rate: float,
) -> tuple[str, str]:
    candidate_id = str(row.get("candidate_id", ""))
    beats_return = _bool_value(row.get("beats_qqq_return"))
    beats_rtd = _bool_value(row.get("beats_qqq_return_to_drawdown"))
    positive_rate = _float(row.get("positive_vs_qqq_window_rate"))
    top_symbol = str(row.get("top_symbol", ""))
    top_exposure = _float(row.get("top_symbol_exposure_pct"))
    top3_exposure = _float(row.get("top3_symbol_exposure_pct"))
    unique_count = _float(row.get("unique_symbol_count"))
    cost20 = _float(row.get("cost_20bps_strategy_vs_qqq"))
    cost30 = _float(row.get("cost_30bps_strategy_vs_qqq"))
    covid_edge = _float(row.get("covid_crash_recovery_strategy_vs_qqq"))
    ai_edge = _float(row.get("ai_bull_2023_2024_strategy_vs_qqq"))
    recent_edge = _float(row.get("recent_2025_2026_strategy_vs_qqq"))
    worst_edge = _float(row.get("worst_strategy_vs_qqq"))

    high_cost_ok = cost20 > 0.0 and cost30 > 0.0
    rolling_acceptable = positive_rate >= 0.50
    rolling_strong = positive_rate >= 0.60
    severe_concentration = top_exposure >= 0.75 or unique_count <= 1
    concentration_watch = top_exposure >= 0.45 or top3_exposure >= 0.80
    arkk_watch = top_symbol.upper() == "ARKK" and top_exposure >= 0.50
    weak_subperiods = sum(edge < 0.0 for edge in (covid_edge, ai_edge, recent_edge))

    parts = [
        "beats QQQ return" if beats_return else "does not beat QQQ return",
        "beats QQQ return-to-drawdown" if beats_rtd else "does not beat QQQ return-to-drawdown",
    ]
    if np.isfinite(cost20):
        parts.append(f"20bps edge {cost20:.4f}")
    if np.isfinite(cost30):
        parts.append(f"30bps edge {cost30:.4f}")
    if np.isfinite(positive_rate):
        parts.append(f"rolling positive rate {positive_rate:.2f}")
    if np.isfinite(worst_edge):
        parts.append(f"worst rolling QQQ-relative edge {worst_edge:.4f}")
    if concentration_watch:
        parts.append("concentration risk acknowledged")
    if arkk_watch:
        parts.append("ARKK concentration watch")
    if covid_edge < 0.0:
        parts.append("covid crash/recovery lagged QQQ")
    if ai_edge < 0.0:
        parts.append("AI bull period lagged QQQ")
    if recent_edge < 0.0:
        parts.append("recent period lagged QQQ")

    rationale = "; ".join(parts)
    if not beats_return or not high_cost_ok:
        return "reject_deprioritize", rationale
    if severe_concentration:
        return "mixed_candidate", rationale
    if weak_subperiods >= 3 or positive_rate < 0.45:
        return "reject_deprioritize", rationale
    if candidate_id == "candidate_0001":
        if rolling_acceptable:
            return "keep_for_targeted_sweep", rationale
        return "mixed_candidate", rationale
    if candidate_id == "candidate_0003":
        stronger_than_0001 = (
            not np.isfinite(candidate_0001_positive_rate)
            or positive_rate >= candidate_0001_positive_rate
        )
        if rolling_strong and stronger_than_0001 and not arkk_watch:
            return "keep_for_signal_review", rationale
        if rolling_acceptable:
            return "keep_for_targeted_sweep", rationale
        return "mixed_candidate", rationale
    if rolling_strong and not concentration_watch:
        return "keep_for_signal_review", rationale
    if rolling_acceptable:
        return "keep_for_targeted_sweep", rationale
    return "mixed_candidate", rationale


def _candidate_metric(frame: pd.DataFrame, candidate_id: str, column: str) -> float:
    match = frame.loc[frame["candidate_id"].astype(str).eq(candidate_id), column]
    if match.empty:
        return np.nan
    return _float(match.iloc[0])


def _candidate_label(row: pd.Series) -> str:
    return (
        f"{row['model_type']} | {row['training_mode']} | {row['target_variant']} | "
        f"{row['allocation_rule']} | {row['rebalance_frequency']}"
    )


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    out = pd.concat(non_empty, ignore_index=True)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, columns]


def _compound_return(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    return calculate_max_drawdown(equity)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(_markdown_cell(value) for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


__all__ = [
    "COST_SENSITIVITY_COLUMNS",
    "COST_SENSITIVITY_NAME",
    "DAILY_DRAWDOWN_COLUMNS",
    "DAILY_DRAWDOWN_NAME",
    "DAILY_EQUITY_COLUMNS",
    "DAILY_EQUITY_NAME",
    "DECISION_MATRIX_COLUMNS",
    "DECISION_MATRIX_NAME",
    "DEFAULT_CANDIDATE_IDS",
    "DEFAULT_COST_SENSITIVITY_BPS",
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_CSV_NAMES",
    "PREFLIGHT_MD_NAME",
    "REPORT_NAME",
    "ROLLING_WINDOW_COLUMNS",
    "ROLLING_WINDOWS_NAME",
    "SELECTED_SYMBOL_COLUMNS",
    "SELECTED_SYMBOLS_NAME",
    "SUMMARY_COLUMNS",
    "SUMMARY_NAME",
    "SYMBOL_EXPOSURE_COLUMNS",
    "SYMBOL_EXPOSURE_NAME",
    "SYMBOL_PERIOD_COLUMNS",
    "SYMBOL_PERIODS_NAME",
    "SUBPERIOD_COLUMNS",
    "SUBPERIODS",
    "SUBPERIODS_NAME",
    "TRADE_COLUMNS",
    "TRADES_NAME",
    "LongHistoryCandidateDeepDiveArtifacts",
    "build_daily_drawdown_rows",
    "build_daily_equity_rows",
    "build_deep_dive_decision_matrix",
    "build_long_history_candidate_deep_dive_artifacts",
    "build_selected_symbols_log",
    "build_symbol_exposure_rows",
    "build_symbol_periods",
    "build_trade_rebalance_log",
    "output_paths",
    "render_preflight_markdown",
    "resolve_deep_dive_candidate_ids",
    "select_deep_dive_candidates",
    "write_long_history_candidate_deep_dive_artifacts",
]
