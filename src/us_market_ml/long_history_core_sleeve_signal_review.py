"""Research-only signal-path review for long-history QQQ core-sleeve rows.

This module reads completed local tracking, rich replay, and candidate
deep-dive artifacts. It does not run a broad sweep, train models, write model
cards, create production signal exports, or mark anything production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_signal_review"
TRACKING_PREFIX = "long_history_qqq_core_sleeve_tracking"
RICH_REPLAY_PREFIX = "long_history_qqq_core_sleeve_rich_replay"
DEEP_DIVE_PREFIX = "long_history_qqq_candidate_deep_dive"

DEFAULT_TRACKING_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_tracking")
DEFAULT_RICH_REPLAY_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_rich_replay")
DEFAULT_CANDIDATE_DEEP_DIVE_DIR = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_signal_review")

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
SYMBOL_PATH_NAME = f"{OUTPUT_PREFIX}_symbol_path.csv"
SYMBOL_REGIME_BREAKDOWN_NAME = f"{OUTPUT_PREFIX}_symbol_regime_breakdown.csv"
DRAWDOWN_EVENTS_NAME = f"{OUTPUT_PREFIX}_drawdown_events.csv"
RECENT_WEAKNESS_NAME = f"{OUTPUT_PREFIX}_recent_weakness.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

OUTPUT_NAMES = (
    SUMMARY_NAME,
    SYMBOL_PATH_NAME,
    SYMBOL_REGIME_BREAKDOWN_NAME,
    DRAWDOWN_EVENTS_NAME,
    RECENT_WEAKNESS_NAME,
    DECISION_MATRIX_NAME,
    REPORT_NAME,
)

KEY_COLUMNS = [
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
]

TRACKING_ROLES = {
    "primary return-oriented tracking candidate",
    "primary balanced/risk-controlled tracking candidate",
    "secondary tracking reference",
}

SUMMARY_COLUMNS = [
    "candidate_id",
    "practical_role",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "selected_symbols_in_order",
    "unique_symbol_count",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "sleeve_weighted_top_symbol_exposure_pct",
    "sleeve_weighted_top3_symbol_exposure_pct",
    "concentration_watch_level",
    "signal_path_interpretation",
    "production_ready",
]

SYMBOL_PATH_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "date",
    "symbol",
    "target_weight",
    "sleeve_weighted_target_weight",
    "predicted_score",
    "rank",
    "allocation_rule",
    "rebalance_frequency",
    "production_ready",
]

REGIME_COLUMNS = [
    "candidate_id",
    "practical_role",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "symbol",
    "start_date",
    "end_date",
    "trading_days",
    "exposure_pct",
    "sleeve_weighted_exposure_pct",
    "period_name",
    "segment_return",
    "qqq_segment_return",
    "segment_vs_qqq",
    "max_drawdown_in_segment",
    "regime_interpretation",
    "production_ready",
]

DRAWDOWN_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "full_period_max_drawdown",
    "recent_max_drawdown",
    "worst_rolling_drawdown_gap_vs_qqq",
    "drawdown_watch_level",
    "major_drawdown_start",
    "major_drawdown_end",
    "selected_symbol_during_drawdown",
    "likely_cause",
    "drawdown_interpretation",
    "production_ready",
]

RECENT_WEAKNESS_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "window_label",
    "window_start",
    "window_end",
    "strategy_vs_qqq",
    "drawdown_improvement_vs_qqq",
    "underperformed_qqq",
    "materially_improved_drawdown",
    "recent_behavior_label",
    "production_ready",
]

DECISION_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "review_status",
    "next_action",
    "production_ready",
    "rationale",
]


@dataclass(frozen=True)
class LongHistoryCoreSleeveSignalReviewInputs:
    """CSV-backed inputs for the signal-path review."""

    tracking_summary: pd.DataFrame
    tracking_status: pd.DataFrame
    tracking_recent_windows: pd.DataFrame
    tracking_drawdown_watch: pd.DataFrame
    tracking_symbol_watch: pd.DataFrame
    tracking_decision_matrix: pd.DataFrame
    rich_replay_summary: pd.DataFrame
    rich_replay_selected_symbols: pd.DataFrame
    rich_replay_symbol_exposure: pd.DataFrame
    rich_replay_subperiods: pd.DataFrame
    rich_replay_rolling_windows: pd.DataFrame
    rich_replay_decision_matrix: pd.DataFrame
    candidate_symbol_periods: pd.DataFrame
    candidate_selected_symbols: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryCoreSleeveSignalReviewArtifacts:
    """CSV and Markdown outputs for the signal-path review."""

    summary: pd.DataFrame
    symbol_path: pd.DataFrame
    symbol_regime_breakdown: pd.DataFrame
    drawdown_events: pd.DataFrame
    recent_weakness: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def read_long_history_core_sleeve_signal_review_inputs(
    *,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    rich_replay_dir: Path = DEFAULT_RICH_REPLAY_DIR,
    candidate_deep_dive_dir: Path = DEFAULT_CANDIDATE_DEEP_DIVE_DIR,
) -> LongHistoryCoreSleeveSignalReviewInputs:
    """Read completed local artifacts required for the signal-path review."""

    return LongHistoryCoreSleeveSignalReviewInputs(
        tracking_summary=_read_required_csv(tracking_dir / f"{TRACKING_PREFIX}_summary.csv"),
        tracking_status=_read_required_csv(tracking_dir / f"{TRACKING_PREFIX}_status.csv"),
        tracking_recent_windows=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_recent_windows.csv"
        ),
        tracking_drawdown_watch=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_drawdown_watch.csv"
        ),
        tracking_symbol_watch=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_symbol_watch.csv"
        ),
        tracking_decision_matrix=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_decision_matrix.csv"
        ),
        rich_replay_summary=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_summary.csv"
        ),
        rich_replay_selected_symbols=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_selected_symbols.csv"
        ),
        rich_replay_symbol_exposure=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_symbol_exposure.csv"
        ),
        rich_replay_subperiods=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv"
        ),
        rich_replay_rolling_windows=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv"
        ),
        rich_replay_decision_matrix=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_decision_matrix.csv"
        ),
        candidate_symbol_periods=_read_required_csv(
            candidate_deep_dive_dir / f"{DEEP_DIVE_PREFIX}_symbol_periods.csv"
        ),
        candidate_selected_symbols=_read_required_csv(
            candidate_deep_dive_dir / f"{DEEP_DIVE_PREFIX}_selected_symbols.csv"
        ),
    )


def build_long_history_core_sleeve_signal_review_artifacts(
    inputs: LongHistoryCoreSleeveSignalReviewInputs,
) -> LongHistoryCoreSleeveSignalReviewArtifacts:
    """Build all signal-path review artifacts from completed local inputs."""

    selected = select_signal_review_tracking_candidates(inputs.tracking_summary)
    symbol_path = build_signal_symbol_path(
        selected=selected,
        selected_symbols=inputs.rich_replay_selected_symbols,
    )
    symbol_regimes = build_symbol_regime_breakdown(
        selected=selected,
        symbol_periods=inputs.candidate_symbol_periods,
        subperiods=inputs.rich_replay_subperiods,
    )
    summary = build_signal_path_summary(
        selected=selected,
        symbol_regimes=symbol_regimes,
        symbol_watch=inputs.tracking_symbol_watch,
    )
    drawdown_events = build_drawdown_event_review(
        selected=selected,
        rich_replay_summary=inputs.rich_replay_summary,
        drawdown_watch=inputs.tracking_drawdown_watch,
        symbol_regimes=symbol_regimes,
    )
    recent_weakness = build_recent_weakness_review(
        selected=selected,
        recent_windows=inputs.tracking_recent_windows,
    )
    decision_matrix = build_signal_review_decision_matrix(
        selected=selected,
        summary=summary,
        drawdown_events=drawdown_events,
        recent_weakness=recent_weakness,
    )
    report = render_long_history_core_sleeve_signal_review_markdown(
        summary=summary,
        symbol_regime_breakdown=symbol_regimes,
        drawdown_events=drawdown_events,
        recent_weakness=recent_weakness,
        decision_matrix=decision_matrix,
    )
    return LongHistoryCoreSleeveSignalReviewArtifacts(
        summary=summary,
        symbol_path=symbol_path,
        symbol_regime_breakdown=symbol_regimes,
        drawdown_events=drawdown_events,
        recent_weakness=recent_weakness,
        decision_matrix=decision_matrix,
        report_markdown=report,
    )


def select_signal_review_tracking_candidates(tracking_summary: pd.DataFrame) -> pd.DataFrame:
    """Select primary and secondary tracking rows only."""

    if tracking_summary.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "practical_role", "production_ready"])
    _validate_columns(tracking_summary, [*KEY_COLUMNS, "practical_role"], "tracking summary")
    selected = tracking_summary.loc[
        tracking_summary["practical_role"].astype(str).isin(TRACKING_ROLES)
    ].copy()
    selected = selected.loc[selected["candidate_sleeve_weight"].astype(float).gt(0.0)].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["spy_core_weight"] = pd.to_numeric(selected["spy_core_weight"], errors="coerce")
    selected["candidate_sleeve_weight"] = pd.to_numeric(
        selected["candidate_sleeve_weight"],
        errors="coerce",
    )
    selected["portfolio_accounting_mode"] = selected["portfolio_accounting_mode"].astype(str)
    selected["production_ready"] = False
    return _rank_selected(selected).reset_index(drop=True)


def build_signal_symbol_path(
    *,
    selected: pd.DataFrame,
    selected_symbols: pd.DataFrame,
) -> pd.DataFrame:
    """Join selected tracking rows to the underlying candidate ETF path."""

    if selected.empty or selected_symbols.empty:
        return pd.DataFrame(columns=SYMBOL_PATH_COLUMNS)
    _validate_columns(selected_symbols, ["candidate_id", "date", "symbol"], "selected symbols")
    source = selected_symbols.assign(candidate_id=lambda df: df["candidate_id"].astype(str))
    if set(KEY_COLUMNS).issubset(source.columns):
        rows = _filter_by_selected_keys(source, selected).merge(
            selected.loc[:, [*KEY_COLUMNS, "practical_role"]],
            on=KEY_COLUMNS,
            how="left",
            suffixes=("_source", ""),
        )
    else:
        rows = selected.loc[:, [*KEY_COLUMNS, "practical_role"]].merge(
            source,
            on="candidate_id",
            how="left",
            suffixes=("", "_source"),
        )
    rows["sleeve_weighted_target_weight"] = pd.to_numeric(
        rows["candidate_sleeve_weight"],
        errors="coerce",
    ) * pd.to_numeric(rows.get("target_weight"), errors="coerce")
    rows["production_ready"] = False
    rows = _with_missing_columns(rows, SYMBOL_PATH_COLUMNS)
    rows["_date_sort"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.sort_values([*KEY_COLUMNS, "_date_sort"], kind="mergesort")
    return rows.loc[:, SYMBOL_PATH_COLUMNS].reset_index(drop=True)


def build_symbol_regime_breakdown(
    *,
    selected: pd.DataFrame,
    symbol_periods: pd.DataFrame,
    subperiods: pd.DataFrame,
) -> pd.DataFrame:
    """Build continuous symbol regimes for each tracking row."""

    if selected.empty or symbol_periods.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    _validate_columns(
        symbol_periods,
        ["candidate_id", "symbol", "start_date", "end_date", "trading_days"],
        "symbol periods",
    )
    period_lookup = _known_periods(subperiods)
    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        periods = symbol_periods.loc[
            symbol_periods["candidate_id"].astype(str).eq(str(selected_row["candidate_id"]))
        ].copy()
        if periods.empty:
            continue
        total_days = pd.to_numeric(periods["trading_days"], errors="coerce").sum()
        periods["_start_sort"] = pd.to_datetime(periods["start_date"], errors="coerce")
        for _, period in periods.sort_values("_start_sort", kind="mergesort").iterrows():
            exposure = _to_float(period.get("trading_days")) / total_days if total_days else np.nan
            rows.append(
                {
                    **_key_dict(selected_row),
                    "practical_role": selected_row.get("practical_role"),
                    "symbol": period.get("symbol"),
                    "start_date": period.get("start_date"),
                    "end_date": period.get("end_date"),
                    "trading_days": period.get("trading_days"),
                    "exposure_pct": exposure,
                    "sleeve_weighted_exposure_pct": _to_float(
                        selected_row.get("candidate_sleeve_weight")
                    )
                    * exposure,
                    "period_name": _overlapping_period_names(
                        period.get("start_date"),
                        period.get("end_date"),
                        period_lookup,
                    ),
                    "segment_return": period.get("segment_return"),
                    "qqq_segment_return": period.get("qqq_segment_return"),
                    "segment_vs_qqq": period.get("segment_vs_qqq"),
                    "max_drawdown_in_segment": period.get("max_drawdown_in_segment"),
                    "regime_interpretation": regime_interpretation(
                        str(selected_row["candidate_id"]),
                        str(period.get("symbol")),
                    ),
                    "production_ready": False,
                }
            )
    out = pd.DataFrame(rows)
    out = _with_missing_columns(out, REGIME_COLUMNS)
    return out.loc[:, REGIME_COLUMNS].reset_index(drop=True)


def build_signal_path_summary(
    *,
    selected: pd.DataFrame,
    symbol_regimes: pd.DataFrame,
    symbol_watch: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize signal path and concentration for each tracking row."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        regimes = _matching_rows(symbol_regimes, selected_row)
        watch = _matching_rows(symbol_watch, selected_row)
        selected_symbols = regimes["symbol"].dropna().astype(str).tolist()
        watch_row = watch.iloc[0] if not watch.empty else pd.Series(dtype=object)
        rows.append(
            {
                **_key_dict(selected_row),
                "practical_role": selected_row.get("practical_role"),
                "selected_symbols_in_order": " -> ".join(selected_symbols),
                "unique_symbol_count": len(set(selected_symbols)),
                "top_symbol": watch_row.get("top_symbol", pd.NA),
                "top_symbol_exposure_pct": watch_row.get("top_symbol_exposure_pct", pd.NA),
                "top3_symbol_exposure_pct": watch_row.get("top3_symbol_exposure_pct", pd.NA),
                "sleeve_weighted_top_symbol_exposure_pct": watch_row.get(
                    "sleeve_weighted_top_symbol_exposure_pct",
                    pd.NA,
                ),
                "sleeve_weighted_top3_symbol_exposure_pct": watch_row.get(
                    "sleeve_weighted_top3_symbol_exposure_pct",
                    pd.NA,
                ),
                "concentration_watch_level": watch_row.get(
                    "concentration_watch_level",
                    "normal",
                ),
                "signal_path_interpretation": signal_path_interpretation(
                    str(selected_row["candidate_id"]),
                    selected_symbols,
                    str(watch_row.get("concentration_watch_level", "normal")),
                ),
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_drawdown_event_review(
    *,
    selected: pd.DataFrame,
    rich_replay_summary: pd.DataFrame,
    drawdown_watch: pd.DataFrame,
    symbol_regimes: pd.DataFrame,
) -> pd.DataFrame:
    """Review full-period, recent, and rolling drawdown risks."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        rich = _matching_rows(rich_replay_summary, selected_row)
        watch = _matching_rows(drawdown_watch, selected_row)
        rich_row = rich.iloc[0] if not rich.empty else pd.Series(dtype=object)
        watch_row = watch.iloc[0] if not watch.empty else pd.Series(dtype=object)
        start = rich_row.get("portfolio_max_drawdown_start", pd.NA)
        end = rich_row.get("portfolio_max_drawdown_end", pd.NA)
        symbol = _symbol_for_range(_matching_rows(symbol_regimes, selected_row), start, end)
        drawdown_level = str(watch_row.get("drawdown_watch_level", "normal"))
        rows.append(
            {
                **_key_dict(selected_row),
                "practical_role": selected_row.get("practical_role"),
                "full_period_max_drawdown": watch_row.get(
                    "full_period_max_drawdown",
                    rich_row.get("max_drawdown", pd.NA),
                ),
                "recent_max_drawdown": watch_row.get("recent_max_drawdown", pd.NA),
                "worst_rolling_drawdown_gap_vs_qqq": watch_row.get(
                    "worst_rolling_drawdown_gap_vs_qqq",
                    rich_row.get("worst_rolling_drawdown_improvement_vs_qqq", pd.NA),
                ),
                "drawdown_watch_level": drawdown_level,
                "major_drawdown_start": start,
                "major_drawdown_end": end,
                "selected_symbol_during_drawdown": symbol,
                "likely_cause": likely_drawdown_cause(str(selected_row["candidate_id"]), symbol),
                "drawdown_interpretation": drawdown_interpretation(
                    str(selected_row["candidate_id"]),
                    drawdown_level,
                    symbol,
                ),
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=DRAWDOWN_COLUMNS)


def build_recent_weakness_review(
    *,
    selected: pd.DataFrame,
    recent_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Return latest 1y, 2y, and 3y windows with behavior labels."""

    if selected.empty or recent_windows.empty:
        return pd.DataFrame(columns=RECENT_WEAKNESS_COLUMNS)
    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        windows = _matching_rows(recent_windows, selected_row).copy()
        if windows.empty:
            continue
        windows["window_years"] = windows["window_label"].astype(str).str.extract(r"^(\d+)y_")
        windows["window_years"] = pd.to_numeric(windows["window_years"], errors="coerce")
        windows["_end_sort"] = pd.to_datetime(windows["window_end"], errors="coerce")
        for years in (1, 2, 3):
            group = windows.loc[windows["window_years"].eq(years)].copy()
            if group.empty:
                continue
            latest = group.sort_values("_end_sort", ascending=False, kind="mergesort").iloc[0]
            rows.append(
                {
                    **_key_dict(selected_row),
                    "practical_role": selected_row.get("practical_role"),
                    "window_label": latest.get("window_label"),
                    "window_start": latest.get("window_start"),
                    "window_end": latest.get("window_end"),
                    "strategy_vs_qqq": latest.get("strategy_vs_qqq"),
                    "drawdown_improvement_vs_qqq": latest.get("drawdown_improvement_vs_qqq"),
                    "underperformed_qqq": _as_bool(
                        latest.get("underperformed_qqq"),
                        default=_to_float(latest.get("strategy_vs_qqq")) < 0.0,
                    ),
                    "materially_improved_drawdown": _as_bool(
                        latest.get("materially_improved_drawdown"),
                        default=_to_float(latest.get("drawdown_improvement_vs_qqq")) >= 0.02,
                    ),
                    "recent_behavior_label": classify_recent_behavior(latest),
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows, columns=RECENT_WEAKNESS_COLUMNS)


def classify_recent_behavior(window: pd.Series) -> str:
    """Classify one recent rolling window."""

    strategy_gap = _to_float(window.get("strategy_vs_qqq"))
    drawdown_gap = _to_float(window.get("drawdown_improvement_vs_qqq"))
    underperformed = _as_bool(window.get("underperformed_qqq"), default=strategy_gap < 0.0)
    improved_drawdown = _as_bool(
        window.get("materially_improved_drawdown"),
        default=drawdown_gap >= 0.02,
    )
    if pd.notna(drawdown_gap) and drawdown_gap <= -0.05:
        return "drawdown_concern"
    if underperformed and not improved_drawdown:
        return "weak_vs_qqq"
    if underperformed and improved_drawdown:
        return "mixed"
    if not underperformed and improved_drawdown:
        return "improving"
    return "mixed"


def build_signal_review_decision_matrix(
    *,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    drawdown_events: pd.DataFrame,
    recent_weakness: pd.DataFrame,
) -> pd.DataFrame:
    """Build final research-only signal review decisions."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        summary_row = _first_or_empty(_matching_rows(summary, selected_row))
        drawdown_row = _first_or_empty(_matching_rows(drawdown_events, selected_row))
        recent = _matching_rows(recent_weakness, selected_row)
        status, action, rationale = classify_signal_review_decision(
            selected_row,
            concentration_watch_level=str(
                summary_row.get("concentration_watch_level", "normal")
            ),
            drawdown_watch_level=str(drawdown_row.get("drawdown_watch_level", "normal")),
            recent_labels=recent["recent_behavior_label"].astype(str).tolist()
            if not recent.empty
            else [],
        )
        rows.append(
            {
                **_key_dict(selected_row),
                "practical_role": selected_row.get("practical_role"),
                "review_status": status,
                "next_action": action,
                "production_ready": False,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def classify_signal_review_decision(
    row: pd.Series,
    *,
    concentration_watch_level: str,
    drawdown_watch_level: str,
    recent_labels: list[str],
) -> tuple[str, str, str]:
    """Classify one signal review row."""

    role = str(row.get("practical_role", ""))
    candidate_id = str(row.get("candidate_id", ""))
    if role == "secondary tracking reference":
        return (
            "watchlist_only",
            "consider_lower_sleeve_weight",
            "secondary 40/60 row remains useful for sizing comparison, but it is not a "
            "promotion candidate",
        )
    if candidate_id == "candidate_0001" or concentration_watch_level == "high":
        return (
            "keep_tracking_with_concentration_caution",
            "review_concentration",
            "strong full-period QQQ-relative result, but sleeve exposure is dominated by a "
            "small XLE/XLU/VUG/XLRE path",
        )
    if drawdown_watch_level == "severe" or "drawdown_concern" in recent_labels:
        return (
            "keep_tracking_with_drawdown_caution",
            "review_signal_path",
            "more diversified than candidate_0001, but recent drawdown behavior remains severe",
        )
    if concentration_watch_level == "caution":
        return (
            "keep_tracking_with_concentration_caution",
            "review_concentration",
            "tracking remains reasonable, with concentration still a live sizing concern",
        )
    return (
        "keep_tracking",
        "continue_tracking",
        "tracking row has no elevated signal-path label in this diagnostic pass",
    )


def render_long_history_core_sleeve_signal_review_markdown(
    *,
    summary: pd.DataFrame,
    symbol_regime_breakdown: pd.DataFrame,
    drawdown_events: pd.DataFrame,
    recent_weakness: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the research-only signal-path review Markdown report."""

    lines = [
        "# Long-History QQQ Core-Sleeve Signal-Path Review",
        "",
        "This is a research-only signal-path and concentration review over completed "
        "local artifacts. It does not run a broad sweep, train models, export "
        "production signals, create a model card, or mark any candidate "
        "production-ready.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(decision_matrix),
        "",
        "## 2. Why signal-path review matters",
        "",
        (
            "ETF concentration is not an automatic rejection. It is a risk label, an "
            "interpretability clue, a live-sizing concern, and the main target for "
            "ongoing signal-path review. QQQ buy-and-hold remains the primary benchmark; "
            "SPY is the core sleeve asset and a secondary reference."
        ),
        "",
        "## 3. Candidate_0001 signal path",
        "",
        _candidate_signal_path_text(summary, symbol_regime_breakdown, "candidate_0001"),
        "",
        "## 4. Candidate_0001 concentration interpretation",
        "",
        _candidate_concentration_text(summary, "candidate_0001"),
        "",
        "## 5. Candidate_0001 drawdown/recent weakness review",
        "",
        _candidate_drawdown_recent_text(drawdown_events, recent_weakness, "candidate_0001"),
        "",
        "## 6. Candidate_0003 signal path",
        "",
        _candidate_signal_path_text(summary, symbol_regime_breakdown, "candidate_0003"),
        "",
        "## 7. Candidate_0003 concentration interpretation",
        "",
        _candidate_concentration_text(summary, "candidate_0003"),
        "",
        "## 8. Candidate_0003 drawdown/recent weakness review",
        "",
        _candidate_drawdown_recent_text(drawdown_events, recent_weakness, "candidate_0003"),
        "",
        "## 9. Primary vs secondary tracking candidate comparison",
        "",
        _markdown_table(_format_for_markdown(decision_matrix)),
        "",
        "## 10. What to track going forward",
        "",
        (
            "- Continue tracking the two primary rows against QQQ first and SPY second.\n"
            "- Review whether candidate_0001 keeps rotating coherently or remains too "
            "dependent on the XLE/XLU/VUG/XLRE cluster.\n"
            "- Review whether candidate_0003's ARKK, XLK, and MTUM behavior is repeatable "
            "and whether IEF/SPY/XLF fallback periods help enough during stress.\n"
            "- Use the 40/60 secondary rows as sizing references, not promotion targets."
        ),
        "",
        "## 11. What not to do next",
        "",
        (
            "- Do not run a broad sweep from this report.\n"
            "- Do not train models.\n"
            "- Do not create production signal exports.\n"
            "- Do not create a model card.\n"
            "- Do not mark any row production-ready."
        ),
        "",
        "## 12. Caveats",
        "",
        (
            "- Research-only diagnostic report.\n"
            "- Tiingo-only historical artifact lineage.\n"
            "- No production signal export.\n"
            "- No model card.\n"
            "- No production-ready marking; `production_ready = false` throughout.\n"
            "- QQQ remains the primary benchmark.\n"
            "- SPY is the core sleeve asset, not the primary benchmark."
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_long_history_core_sleeve_signal_review_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveSignalReviewArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write signal-path review artifacts to the report output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.symbol_path.to_csv(paths[SYMBOL_PATH_NAME], index=False)
    artifacts.symbol_regime_breakdown.to_csv(paths[SYMBOL_REGIME_BREAKDOWN_NAME], index=False)
    artifacts.drawdown_events.to_csv(paths[DRAWDOWN_EVENTS_NAME], index=False)
    artifacts.recent_weakness.to_csv(paths[RECENT_WEAKNESS_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    return paths


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all signal review output paths."""

    return {name: output_dir / name for name in OUTPUT_NAMES}


def signal_path_interpretation(
    candidate_id: str,
    symbols_in_order: list[str],
    concentration_watch_level: str,
) -> str:
    """Return a compact candidate-level signal-path interpretation."""

    unique = set(symbols_in_order)
    if candidate_id == "candidate_0001":
        if {"XLU", "XLE", "XLRE", "VUG"}.issubset(unique):
            return (
                "coherent_regime_rotation_with_concentration_caution: XLU, XLE, XLRE, "
                "and VUG form an interpretable defensive-to-energy-to-real-estate-to-growth "
                "path, but the high XLE/top3 concentration means it should be reviewed as a "
                "cluster-driven result rather than treated as production-ready"
            )
        return (
            "concentrated_symbol_cluster: candidate_0001 needs review before the signal "
            "path can be treated as coherent"
        )
    if candidate_id == "candidate_0003":
        if {"ARKK", "XLK", "MTUM"}.issubset(unique):
            return (
                "broader_rotation_with_arkk_driven_risk: ARKK, XLK, MTUM, and fallback "
                "symbols show broader rotation than candidate_0001, but ARKK is still a "
                "major driver of both return and drawdown behavior"
            )
        return (
            "arkk_driven_behavior: candidate_0003 needs review for ARKK dependence and "
            "recent drawdown behavior"
        )
    return f"tracking_reference_with_{concentration_watch_level}_concentration_label"


def regime_interpretation(candidate_id: str, symbol: str) -> str:
    """Return a symbol-regime interpretation label."""

    if candidate_id == "candidate_0001":
        mapping = {
            "XLU": "early defensive utilities period",
            "XLE": "energy-cycle period and dominant concentration driver",
            "XLRE": "real-estate period",
            "VUG": "recent growth tilt",
        }
        return mapping.get(symbol, "other candidate_0001 sleeve regime")
    if candidate_id == "candidate_0003":
        mapping = {
            "ARKK": "innovation growth exposure and ARKK-driven risk regime",
            "XLK": "technology exposure",
            "MTUM": "momentum exposure",
            "IEF": "Treasury defensive fallback",
            "SPY": "broad-market fallback",
            "XLF": "financials fallback or cyclical sleeve",
            "XLRE": "real-estate sleeve",
            "XLE": "energy sleeve",
        }
        return mapping.get(symbol, "other candidate_0003 rotation regime")
    return "tracking reference regime"


def likely_drawdown_cause(candidate_id: str, symbol: object) -> str:
    """Infer a drawdown cause from candidate and selected symbol path."""

    symbol_text = str(symbol)
    if candidate_id == "candidate_0001":
        if symbol_text == "XLU":
            return "COVID-era defensive XLU sleeve still participated in the market shock"
        if symbol_text == "XLE":
            return "energy sleeve concentration exposed the portfolio to cyclical drawdowns"
        if symbol_text == "VUG":
            return "recent growth tilt lagged QQQ during part of the rebound window"
        return "drawdown aligns with concentrated candidate_0001 sleeve path"
    if candidate_id == "candidate_0003":
        if symbol_text == "ARKK":
            return "ARKK sleeve exposure created a severe recent growth drawdown"
        if symbol_text == "XLK":
            return "technology sleeve carried QQQ-like downside during stress"
        return "drawdown aligns with ARKK-led and growth/momentum sleeve rotation"
    return "drawdown cause inferred from selected sleeve symbol"


def drawdown_interpretation(candidate_id: str, drawdown_watch_level: str, symbol: object) -> str:
    """Return a candidate-level drawdown interpretation."""

    if candidate_id == "candidate_0001":
        return (
            "full-period drawdown is roughly QQQ-like, but concentration means drawdown "
            f"events should be tied back to the active sleeve symbol ({symbol})"
        )
    if candidate_id == "candidate_0003":
        return (
            f"{drawdown_watch_level} drawdown watch: diversified symbol path does not remove "
            f"the ARKK/growth drawdown concern around {symbol}"
        )
    return f"{drawdown_watch_level} drawdown watch for tracking reference"


def _candidate_signal_path_text(
    summary: pd.DataFrame,
    symbol_regime_breakdown: pd.DataFrame,
    candidate_id: str,
) -> str:
    rows = summary.loc[summary["candidate_id"].astype(str).eq(candidate_id)].copy()
    regimes = symbol_regime_breakdown.loc[
        symbol_regime_breakdown["candidate_id"].astype(str).eq(candidate_id)
    ].copy()
    if rows.empty:
        return "No tracking rows available."
    return "\n\n".join(
        [
            _markdown_table(
                _format_for_markdown(
                    rows.loc[
                        :,
                        [
                            "practical_role",
                            "spy_core_weight",
                            "candidate_sleeve_weight",
                            "selected_symbols_in_order",
                            "unique_symbol_count",
                            "signal_path_interpretation",
                        ],
                    ]
                )
            ),
            _markdown_table(
                _format_for_markdown(
                    regimes.loc[
                        :,
                        [
                            "practical_role",
                            "symbol",
                            "start_date",
                            "end_date",
                            "trading_days",
                            "period_name",
                            "regime_interpretation",
                        ],
                    ].head(18)
                )
            ),
        ]
    )


def _candidate_concentration_text(summary: pd.DataFrame, candidate_id: str) -> str:
    rows = summary.loc[summary["candidate_id"].astype(str).eq(candidate_id)].copy()
    if rows.empty:
        return "No concentration rows available."
    lead = rows.iloc[0]
    if candidate_id == "candidate_0001":
        text = (
            "candidate_0001 remains a tracking candidate with concentration caution. The "
            "signal path looks economically interpretable, not just a single-symbol accident, "
            "but the XLE/top3 sleeve dominance makes concentration a live sizing concern."
        )
    else:
        text = (
            "candidate_0003 is broader than candidate_0001, but ARKK remains a large return "
            "and drawdown driver. XLK and MTUM add rotation breadth, while fallback symbols "
            "help interpretation more than they eliminate risk."
        )
    return "\n\n".join(
        [
            text,
            _markdown_table(
                _format_for_markdown(
                    rows.loc[
                        :,
                        [
                            "practical_role",
                            "top_symbol",
                            "top_symbol_exposure_pct",
                            "top3_symbol_exposure_pct",
                            "sleeve_weighted_top_symbol_exposure_pct",
                            "sleeve_weighted_top3_symbol_exposure_pct",
                            "concentration_watch_level",
                        ],
                    ]
                )
            ),
            f"Primary concentration read: `{lead['concentration_watch_level']}`.",
        ]
    )


def _candidate_drawdown_recent_text(
    drawdown_events: pd.DataFrame,
    recent_weakness: pd.DataFrame,
    candidate_id: str,
) -> str:
    drawdown = drawdown_events.loc[drawdown_events["candidate_id"].astype(str).eq(candidate_id)]
    recent = recent_weakness.loc[recent_weakness["candidate_id"].astype(str).eq(candidate_id)]
    if drawdown.empty and recent.empty:
        return "No drawdown or recent-window rows available."
    return "\n\n".join(
        [
            _markdown_table(_format_for_markdown(drawdown)),
            _markdown_table(_format_for_markdown(recent)),
        ]
    )


def _executive_summary(decision_matrix: pd.DataFrame) -> str:
    if decision_matrix.empty:
        return "No tracking candidates were available for review."
    primary = decision_matrix.loc[
        decision_matrix["practical_role"].astype(str).str.startswith("primary")
    ]
    status_text = "; ".join(
        f"{row.candidate_id} {row.review_status}" for row in primary.itertuples()
    )
    return (
        f"{status_text}. No row is production-ready. The review supports continued tracking "
        "with concentration caution for candidate_0001 and drawdown caution for candidate_0003; "
        "it does not support a broad sweep, model training, production export, or model card."
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_markdown_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].map(lambda value: str(bool(value)).lower())
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(_format_number)
    return out


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) < 10:
        return f"{number:.4f}"
    return f"{number:.1f}"


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _known_periods(subperiods: pd.DataFrame) -> pd.DataFrame:
    if subperiods.empty or not {"period_name", "period_start", "period_end"}.issubset(
        subperiods.columns
    ):
        return pd.DataFrame(columns=["period_name", "period_start", "period_end"])
    periods = subperiods.loc[
        ~subperiods["period_name"].astype(str).eq("full"),
        ["period_name", "period_start", "period_end"],
    ].drop_duplicates()
    periods["period_start"] = pd.to_datetime(periods["period_start"], errors="coerce")
    periods["period_end"] = pd.to_datetime(periods["period_end"], errors="coerce")
    return periods.dropna(subset=["period_start", "period_end"]).reset_index(drop=True)


def _overlapping_period_names(
    start: object,
    end: object,
    period_lookup: pd.DataFrame,
) -> str:
    if period_lookup.empty:
        return ""
    start_date = pd.to_datetime(start, errors="coerce")
    end_date = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date):
        return ""
    overlaps = period_lookup.loc[
        period_lookup["period_start"].le(end_date) & period_lookup["period_end"].ge(start_date)
    ]
    return "; ".join(overlaps["period_name"].astype(str).tolist())


def _symbol_for_range(regimes: pd.DataFrame, start: object, end: object) -> str:
    if regimes.empty:
        return ""
    start_date = pd.to_datetime(start, errors="coerce")
    end_date = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date):
        return ""
    rows = regimes.copy()
    rows["_start"] = pd.to_datetime(rows["start_date"], errors="coerce")
    rows["_end"] = pd.to_datetime(rows["end_date"], errors="coerce")
    overlap = rows.loc[rows["_start"].le(end_date) & rows["_end"].ge(start_date)]
    if overlap.empty:
        return ""
    return "; ".join(overlap["symbol"].dropna().astype(str).drop_duplicates().tolist())


def _rank_selected(selected: pd.DataFrame) -> pd.DataFrame:
    role_order = {
        "primary return-oriented tracking candidate": 0,
        "primary balanced/risk-controlled tracking candidate": 1,
        "secondary tracking reference": 2,
    }
    out = selected.copy()
    out["_role_order"] = out["practical_role"].map(role_order).fillna(99)
    return out.sort_values(
        ["_role_order", "candidate_id", "spy_core_weight"],
        kind="mergesort",
    ).drop(columns=["_role_order"])


def _filter_by_selected_keys(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or selected.empty:
        return frame.iloc[0:0].copy()
    rows = frame.copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["spy_core_weight"] = pd.to_numeric(rows["spy_core_weight"], errors="coerce").round(10)
    rows["candidate_sleeve_weight"] = pd.to_numeric(
        rows["candidate_sleeve_weight"],
        errors="coerce",
    ).round(10)
    rows["portfolio_accounting_mode"] = rows["portfolio_accounting_mode"].astype(str)
    keys = selected.loc[:, KEY_COLUMNS].copy()
    keys["candidate_id"] = keys["candidate_id"].astype(str)
    keys["spy_core_weight"] = pd.to_numeric(keys["spy_core_weight"], errors="coerce").round(10)
    keys["candidate_sleeve_weight"] = pd.to_numeric(
        keys["candidate_sleeve_weight"],
        errors="coerce",
    ).round(10)
    keys["portfolio_accounting_mode"] = keys["portfolio_accounting_mode"].astype(str)
    selected_keys = set(map(tuple, keys.loc[:, KEY_COLUMNS].to_numpy()))
    mask = rows.loc[:, KEY_COLUMNS].apply(tuple, axis=1).isin(selected_keys)
    return rows.loc[mask].copy()


def _matching_rows(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy()
    mask = frame["candidate_id"].astype(str).eq(str(row["candidate_id"]))
    mask &= pd.to_numeric(frame["spy_core_weight"], errors="coerce").round(10).eq(
        round(float(row["spy_core_weight"]), 10)
    )
    mask &= pd.to_numeric(frame["candidate_sleeve_weight"], errors="coerce").round(10).eq(
        round(float(row["candidate_sleeve_weight"]), 10)
    )
    mask &= frame["portfolio_accounting_mode"].astype(str).eq(str(row["portfolio_accounting_mode"]))
    return frame.loc[mask].copy()


def _key_dict(row: pd.Series) -> dict[str, object]:
    return {column: row.get(column) for column in KEY_COLUMNS}


def _first_or_empty(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _as_bool(value: object, *, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return default


def _with_missing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
