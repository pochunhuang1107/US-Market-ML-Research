"""Research-only tracking report for selected long-history QQQ core-sleeve rows.

This module reads completed local rich replay artifacts only. It does not run a
broad sweep, train models, write model cards, create production signal exports,
or mark anything production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_tracking"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_tracking")
DEFAULT_RICH_REPLAY_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_rich_replay")

RICH_REPLAY_PREFIX = "long_history_qqq_core_sleeve_rich_replay"

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
STATUS_NAME = f"{OUTPUT_PREFIX}_status.csv"
RECENT_WINDOWS_NAME = f"{OUTPUT_PREFIX}_recent_windows.csv"
DRAWDOWN_WATCH_NAME = f"{OUTPUT_PREFIX}_drawdown_watch.csv"
SYMBOL_WATCH_NAME = f"{OUTPUT_PREFIX}_symbol_watch.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

OUTPUT_NAMES = (
    SUMMARY_NAME,
    STATUS_NAME,
    RECENT_WINDOWS_NAME,
    DRAWDOWN_WATCH_NAME,
    SYMBOL_WATCH_NAME,
    DECISION_MATRIX_NAME,
    REPORT_NAME,
)

KEY_COLUMNS = [
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
]

TRACKING_SUMMARY_COLUMNS = [
    "candidate_id",
    "practical_role",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "positive_vs_qqq_window_rate",
    "worst_strategy_vs_qqq",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "production_ready",
]

RECENT_WINDOW_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
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
    "production_ready",
]

DRAWDOWN_WATCH_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "full_period_max_drawdown",
    "recent_max_drawdown",
    "worst_rolling_drawdown_gap_vs_qqq",
    "recent_drawdown_worse_than_qqq",
    "drawdown_watch_level",
    "production_ready",
]

SYMBOL_WATCH_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "sleeve_weighted_top_symbol_exposure_pct",
    "sleeve_weighted_top3_symbol_exposure_pct",
    "concentration_watch_level",
    "production_ready",
]

STATUS_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "tracking_status",
    "production_ready",
    "rationale",
]

DECISION_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "tracking_status",
    "track_active",
    "track_with_caution",
    "watchlist_only",
    "production_ready",
    "next_action",
    "rationale",
]

PRIMARY_ROLES = {
    "primary return-oriented tracking candidate",
    "primary balanced/risk-controlled tracking candidate",
}


@dataclass(frozen=True)
class LongHistoryCoreSleeveTrackingInputs:
    """CSV-backed rich replay inputs for the tracking report."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    symbol_exposure: pd.DataFrame
    selected_symbols: pd.DataFrame
    decision_matrix: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryCoreSleeveTrackingArtifacts:
    """CSV and Markdown outputs for the tracking report."""

    summary: pd.DataFrame
    status: pd.DataFrame
    recent_windows: pd.DataFrame
    drawdown_watch: pd.DataFrame
    symbol_watch: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def read_long_history_core_sleeve_tracking_inputs(
    *,
    rich_replay_dir: Path = DEFAULT_RICH_REPLAY_DIR,
) -> LongHistoryCoreSleeveTrackingInputs:
    """Read completed local rich replay artifacts required for tracking."""

    return LongHistoryCoreSleeveTrackingInputs(
        summary=_read_required_csv(rich_replay_dir / f"{RICH_REPLAY_PREFIX}_summary.csv"),
        daily_equity=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_daily_equity.csv"
        ),
        daily_drawdown=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_daily_drawdown.csv"
        ),
        subperiods=_read_required_csv(rich_replay_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv"),
        rolling_windows=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv"
        ),
        symbol_exposure=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_symbol_exposure.csv"
        ),
        selected_symbols=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_selected_symbols.csv"
        ),
        decision_matrix=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_decision_matrix.csv"
        ),
    )


def build_long_history_core_sleeve_tracking_artifacts(
    inputs: LongHistoryCoreSleeveTrackingInputs,
) -> LongHistoryCoreSleeveTrackingArtifacts:
    """Build all tracking artifacts from completed rich replay inputs."""

    selected = select_tracking_candidates(inputs.summary)
    summary = build_tracking_summary(selected)
    recent_windows = build_recent_tracking_windows(
        selected=selected,
        rolling_windows=inputs.rolling_windows,
    )
    drawdown_watch = build_drawdown_watch(
        selected=selected,
        recent_windows=recent_windows,
        rolling_windows=inputs.rolling_windows,
    )
    symbol_watch = build_symbol_watch(
        selected=selected,
        symbol_exposure=inputs.symbol_exposure,
    )
    status = build_tracking_status(
        summary=summary,
        drawdown_watch=drawdown_watch,
        symbol_watch=symbol_watch,
    )
    decision_matrix = build_tracking_decision_matrix(status=status)
    report = render_long_history_core_sleeve_tracking_markdown(
        summary=summary,
        status=status,
        recent_windows=recent_windows,
        drawdown_watch=drawdown_watch,
        symbol_watch=symbol_watch,
        decision_matrix=decision_matrix,
    )
    return LongHistoryCoreSleeveTrackingArtifacts(
        summary=summary,
        status=status,
        recent_windows=recent_windows,
        drawdown_watch=drawdown_watch,
        symbol_watch=symbol_watch,
        decision_matrix=decision_matrix,
        report_markdown=report,
    )


def select_tracking_candidates(rich_replay_summary: pd.DataFrame) -> pd.DataFrame:
    """Return primary, secondary, and standalone reference rows for tracking review."""

    if rich_replay_summary.empty:
        return pd.DataFrame(columns=[*TRACKING_SUMMARY_COLUMNS, "recommendation_action"])
    _validate_columns(
        rich_replay_summary,
        [
            "candidate_id",
            "practical_role",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "portfolio_accounting_mode",
            "production_ready",
        ],
        "rich replay summary",
    )
    if "recommendation_action" in rich_replay_summary.columns:
        actions = {"rich_replay_next", "track_next", "reference_only"}
        selected = rich_replay_summary.loc[
            rich_replay_summary["recommendation_action"].astype(str).isin(actions)
        ].copy()
    else:
        selected = rich_replay_summary.copy()
    selected = selected.loc[~selected["candidate_id"].astype(str).eq("broad_sweep")].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["spy_core_weight"] = pd.to_numeric(selected["spy_core_weight"], errors="coerce")
    selected["candidate_sleeve_weight"] = pd.to_numeric(
        selected["candidate_sleeve_weight"],
        errors="coerce",
    )
    selected["portfolio_accounting_mode"] = selected["portfolio_accounting_mode"].astype(str)
    selected["production_ready"] = False
    return _rank_selected(selected).reset_index(drop=True)


def build_tracking_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """Build the compact selected candidate tracking summary."""

    summary = _with_missing_columns(selected.copy(), TRACKING_SUMMARY_COLUMNS)
    summary = summary.loc[:, TRACKING_SUMMARY_COLUMNS].copy()
    summary["production_ready"] = False
    return summary.reset_index(drop=True)


def build_recent_tracking_windows(
    *,
    selected: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    windows_per_candidate: int = 5,
) -> pd.DataFrame:
    """Return the most recent rolling windows for each selected portfolio row."""

    if selected.empty or rolling_windows.empty:
        return pd.DataFrame(columns=RECENT_WINDOW_COLUMNS)
    _validate_columns(rolling_windows, KEY_COLUMNS, "rolling windows")
    rows: list[pd.DataFrame] = []
    for _, selected_row in selected.iterrows():
        group = _matching_rows(rolling_windows, selected_row)
        if group.empty:
            continue
        group = group.copy()
        group["_window_end_sort"] = pd.to_datetime(group.get("window_end"), errors="coerce")
        group["_window_start_sort"] = pd.to_datetime(group.get("window_start"), errors="coerce")
        recent = group.sort_values(
            ["_window_end_sort", "_window_start_sort", "window_label"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(windows_per_candidate)
        recent = recent.sort_values(
            ["_window_end_sort", "_window_start_sort", "window_label"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        recent["practical_role"] = selected_row.get("practical_role")
        rows.append(recent)
    if not rows:
        return pd.DataFrame(columns=RECENT_WINDOW_COLUMNS)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["underperformed_qqq"] = _as_bool_or_numeric_negative(
        out.get("underperformed_qqq"),
        out.get("strategy_vs_qqq"),
    )
    out["materially_improved_drawdown"] = _as_bool_or_numeric_threshold(
        out.get("materially_improved_drawdown"),
        out.get("drawdown_improvement_vs_qqq"),
        threshold=0.02,
    )
    out["production_ready"] = False
    out = _with_missing_columns(out, RECENT_WINDOW_COLUMNS)
    return out.loc[:, RECENT_WINDOW_COLUMNS].reset_index(drop=True)


def build_drawdown_watch(
    *,
    selected: pd.DataFrame,
    recent_windows: pd.DataFrame,
    rolling_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize current and rolling drawdown risks for selected rows."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        recent = _matching_rows(recent_windows, selected_row)
        rolling = _matching_rows(rolling_windows, selected_row)
        recent_gap = pd.to_numeric(recent.get("drawdown_improvement_vs_qqq"), errors="coerce")
        all_gap = pd.to_numeric(rolling.get("drawdown_improvement_vs_qqq"), errors="coerce")
        recent_dd = pd.to_numeric(recent.get("strategy_max_drawdown"), errors="coerce")
        recent_worse = bool(recent_gap.lt(0.0).any()) if recent_gap.notna().any() else False
        worst_gap = float(all_gap.min()) if all_gap.notna().any() else np.nan
        recent_max_drawdown = float(recent_dd.min()) if recent_dd.notna().any() else np.nan
        level = assign_drawdown_watch_level(
            recent_drawdown_worse_than_qqq=recent_worse,
            recent_worst_gap=float(recent_gap.min()) if recent_gap.notna().any() else np.nan,
            worst_rolling_gap=worst_gap,
        )
        rows.append(
            {
                **_key_dict(selected_row),
                "practical_role": selected_row.get("practical_role"),
                "full_period_max_drawdown": selected_row.get("max_drawdown"),
                "recent_max_drawdown": recent_max_drawdown,
                "worst_rolling_drawdown_gap_vs_qqq": worst_gap,
                "recent_drawdown_worse_than_qqq": recent_worse,
                "drawdown_watch_level": level,
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=DRAWDOWN_WATCH_COLUMNS)


def assign_drawdown_watch_level(
    *,
    recent_drawdown_worse_than_qqq: bool,
    recent_worst_gap: float,
    worst_rolling_gap: float,
) -> str:
    """Classify drawdown watch severity from recent and full rolling gaps."""

    if recent_drawdown_worse_than_qqq and pd.notna(recent_worst_gap) and recent_worst_gap <= -0.05:
        return "severe"
    if pd.notna(worst_rolling_gap) and worst_rolling_gap <= -0.10:
        return "severe"
    if recent_drawdown_worse_than_qqq:
        return "caution"
    if pd.notna(worst_rolling_gap) and worst_rolling_gap < 0.0:
        return "caution"
    return "normal"


def build_symbol_watch(
    *,
    selected: pd.DataFrame,
    symbol_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize symbol concentration as a monitoring label, not a rejection rule."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        exposure = _matching_rows(symbol_exposure, selected_row)
        top_symbol = selected_row.get("top_symbol")
        top_exposure = _to_float(selected_row.get("top_symbol_exposure_pct"))
        top3_exposure = _to_float(selected_row.get("top3_symbol_exposure_pct"))
        sleeve_weighted_top = (
            float(selected_row["candidate_sleeve_weight"]) * top_exposure
            if pd.notna(top_exposure)
            else np.nan
        )
        sleeve_weighted_top3 = (
            float(selected_row["candidate_sleeve_weight"]) * top3_exposure
            if pd.notna(top3_exposure)
            else np.nan
        )
        if not exposure.empty:
            ranked = exposure.sort_values("rank_by_exposure", kind="mergesort")
            top = ranked.iloc[0]
            top_symbol = top.get("symbol")
            top_exposure = _to_float(top.get("exposure_pct"))
            sleeve_weighted_top = _to_float(top.get("sleeve_weighted_exposure_pct"))
            if "is_top3_symbol" in ranked.columns:
                top3_mask = _as_bool_series(ranked["is_top3_symbol"])
            else:
                top3_mask = pd.to_numeric(ranked["rank_by_exposure"], errors="coerce").le(3)
            top3_exposure = float(
                pd.to_numeric(ranked.loc[top3_mask, "exposure_pct"], errors="coerce").sum()
            )
            if "sleeve_weighted_exposure_pct" in ranked.columns:
                sleeve_weighted_top3 = float(
                    pd.to_numeric(
                        ranked.loc[top3_mask, "sleeve_weighted_exposure_pct"],
                        errors="coerce",
                    ).sum()
                )
        rows.append(
            {
                **_key_dict(selected_row),
                "practical_role": selected_row.get("practical_role"),
                "top_symbol": top_symbol,
                "top_symbol_exposure_pct": top_exposure,
                "top3_symbol_exposure_pct": top3_exposure,
                "sleeve_weighted_top_symbol_exposure_pct": sleeve_weighted_top,
                "sleeve_weighted_top3_symbol_exposure_pct": sleeve_weighted_top3,
                "concentration_watch_level": assign_concentration_watch_level(
                    top_symbol_exposure_pct=top_exposure,
                    top3_symbol_exposure_pct=top3_exposure,
                ),
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=SYMBOL_WATCH_COLUMNS)


def assign_concentration_watch_level(
    *,
    top_symbol_exposure_pct: float,
    top3_symbol_exposure_pct: float,
) -> str:
    """Classify concentration severity for interpretability and monitoring."""

    if (
        pd.notna(top_symbol_exposure_pct)
        and top_symbol_exposure_pct >= 0.40
        or pd.notna(top3_symbol_exposure_pct)
        and top3_symbol_exposure_pct >= 0.80
    ):
        return "high"
    if (
        pd.notna(top_symbol_exposure_pct)
        and top_symbol_exposure_pct >= 0.30
        or pd.notna(top3_symbol_exposure_pct)
        and top3_symbol_exposure_pct >= 0.65
    ):
        return "caution"
    return "normal"


def build_tracking_status(
    *,
    summary: pd.DataFrame,
    drawdown_watch: pd.DataFrame,
    symbol_watch: pd.DataFrame,
) -> pd.DataFrame:
    """Assign one tracking status to each selected row."""

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        drawdown = _matching_rows(drawdown_watch, row)
        symbol = _matching_rows(symbol_watch, row)
        drawdown_level = (
            str(drawdown["drawdown_watch_level"].iloc[0]) if not drawdown.empty else "normal"
        )
        concentration_level = (
            str(symbol["concentration_watch_level"].iloc[0]) if not symbol.empty else "normal"
        )
        status, rationale = classify_tracking_status(
            row,
            drawdown_watch_level=drawdown_level,
            concentration_watch_level=concentration_level,
        )
        rows.append(
            {
                **_key_dict(row),
                "practical_role": row.get("practical_role"),
                "tracking_status": status,
                "production_ready": False,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def classify_tracking_status(
    row: pd.Series,
    *,
    drawdown_watch_level: str = "normal",
    concentration_watch_level: str = "normal",
) -> tuple[str, str]:
    """Classify one selected row into the tracking status vocabulary."""

    role = str(row.get("practical_role", ""))
    if role in PRIMARY_ROLES:
        beats_qqq = _to_float(row.get("strategy_vs_qqq")) > 0.0
        nearly_matches_drawdown = _to_float(row.get("drawdown_improvement_vs_qqq")) >= -0.002
        rtd_beats = _to_float(row.get("return_to_drawdown")) > _qqq_return_to_drawdown(row)
        weak_rolling = _to_float(row.get("positive_vs_qqq_window_rate")) < 0.50 or _to_float(
            row.get("worst_strategy_vs_qqq")
        ) <= -0.50
        concentration_caveat = concentration_watch_level in {"caution", "high"}
        if beats_qqq and nearly_matches_drawdown and rtd_beats:
            elevated_watch = (
                weak_rolling
                or concentration_watch_level == "high"
                or drawdown_watch_level == "severe"
            )
            if elevated_watch:
                caveats: list[str] = []
                if weak_rolling:
                    caveats.append("rolling-window behavior needs monitoring")
                if concentration_caveat:
                    caveats.append(f"{concentration_watch_level} concentration label")
                if drawdown_watch_level == "severe":
                    caveats.append("drawdown watch is severe")
                return "track_with_caution", "; ".join(
                    ["good full-period QQQ-relative result", *caveats]
                )
            return (
                "track_active",
                "primary row beats QQQ full-period return, nearly matches or improves QQQ "
                "drawdown, and beats QQQ return-to-drawdown",
            )
        return (
            "track_with_caution",
            "primary row is selected for monitoring, but one of the return, drawdown, or "
            "return-to-drawdown checks is weak",
        )
    if role == "secondary tracking reference":
        return "watchlist_only", "secondary 40/60 row retained as a reference beside primary rows"
    return (
        "reject_deprioritize",
        "standalone/reference row is useful for interpretation only and must not be promoted",
    )


def build_tracking_decision_matrix(status: pd.DataFrame) -> pd.DataFrame:
    """Build the final tracking decision matrix."""

    rows: list[dict[str, object]] = []
    for _, row in status.iterrows():
        tracking_status = str(row.get("tracking_status", "reject_deprioritize"))
        rows.append(
            {
                **_key_dict(row),
                "practical_role": row.get("practical_role"),
                "tracking_status": tracking_status,
                "track_active": tracking_status == "track_active",
                "track_with_caution": tracking_status == "track_with_caution",
                "watchlist_only": tracking_status == "watchlist_only",
                "production_ready": False,
                "next_action": _next_action(row),
                "rationale": row.get("rationale"),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def render_long_history_core_sleeve_tracking_markdown(
    *,
    summary: pd.DataFrame,
    status: pd.DataFrame,
    recent_windows: pd.DataFrame,
    drawdown_watch: pd.DataFrame,
    symbol_watch: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the research-only tracking Markdown report."""

    lines = [
        "# Long-History QQQ Core-Sleeve Tracking Report",
        "",
        "This is a research-only tracking and forward-validation framework over "
        "completed local rich replay artifacts. It does not run a broad sweep, "
        "train models, export production signals, create a model card, or mark "
        "any candidate production-ready.",
        "",
        "## 1. Executive summary",
        "",
        _tracking_executive_summary(status),
        "",
        "## 2. Tracking candidates",
        "",
        _markdown_table(_format_for_markdown(summary)),
        "",
        "## 3. Current tracking status",
        "",
        _markdown_table(_format_for_markdown(status)),
        "",
        "## 4. Recent rolling-window behavior",
        "",
        _recent_window_text(recent_windows),
        "",
        "## 5. Drawdown watch",
        "",
        _markdown_table(_format_for_markdown(drawdown_watch)),
        "",
        "## 6. Symbol/concentration watch",
        "",
        _symbol_watch_text(symbol_watch),
        "",
        "## 7. Candidate_0001 interpretation",
        "",
        _candidate_interpretation(summary, status, symbol_watch, "candidate_0001"),
        "",
        "## 8. Candidate_0003 interpretation",
        "",
        _candidate_interpretation(summary, status, symbol_watch, "candidate_0003"),
        "",
        "## 9. What would be required before production promotion",
        "",
        _production_promotion_requirements(),
        "",
        "## 10. Recommended next action",
        "",
        _markdown_table(_format_for_markdown(decision_matrix)),
        "",
        "## 11. Caveats",
        "",
        _tracking_caveats(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_long_history_core_sleeve_tracking_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveTrackingArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write tracking report artifacts to the report output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.status.to_csv(paths[STATUS_NAME], index=False)
    artifacts.recent_windows.to_csv(paths[RECENT_WINDOWS_NAME], index=False)
    artifacts.drawdown_watch.to_csv(paths[DRAWDOWN_WATCH_NAME], index=False)
    artifacts.symbol_watch.to_csv(paths[SYMBOL_WATCH_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    return paths


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for the tracking report."""

    return {name: output_dir / name for name in OUTPUT_NAMES}


def _next_action(row: pd.Series) -> str:
    status = str(row.get("tracking_status", ""))
    rationale = str(row.get("rationale", "")).lower()
    role = str(row.get("practical_role", "")).lower()
    if status == "track_active":
        return "prepare_forward_validation_template"
    if status == "track_with_caution":
        if "concentration" in rationale:
            return "review_concentration"
        return "review_signal_path"
    if status == "watchlist_only":
        return "continue_tracking"
    if "signal-review" in role:
        return "review_signal_path"
    return "do_not_promote"


def _tracking_executive_summary(status: pd.DataFrame) -> str:
    if status.empty:
        return "No selected rich replay rows were available for tracking."
    counts = status["tracking_status"].value_counts().to_dict()
    active = int(counts.get("track_active", 0))
    caution = int(counts.get("track_with_caution", 0))
    watchlist = int(counts.get("watchlist_only", 0))
    reject = int(counts.get("reject_deprioritize", 0))
    prod_ready = int(_as_bool_series(status["production_ready"]).sum())
    return (
        f"Tracking framework covers `{active}` active rows, `{caution}` caution rows, "
        f"`{watchlist}` watchlist rows, and `{reject}` deprioritized reference rows. "
        "QQQ buy-and-hold remains the primary benchmark; SPY buy-and-hold is retained "
        "as a secondary reference and SPY remains the core sleeve asset. "
        f"Production-ready rows: `{prod_ready}`."
    )


def _recent_window_text(recent_windows: pd.DataFrame) -> str:
    if recent_windows.empty:
        return "No rolling-window rows were available."
    columns = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "practical_role",
        "window_label",
        "window_start",
        "window_end",
        "strategy_vs_qqq",
        "drawdown_improvement_vs_qqq",
        "underperformed_qqq",
        "materially_improved_drawdown",
    ]
    return _markdown_table(_format_for_markdown(recent_windows.loc[:, columns]))


def _symbol_watch_text(symbol_watch: pd.DataFrame) -> str:
    if symbol_watch.empty:
        return "No symbol exposure rows were available."
    table = _markdown_table(_format_for_markdown(symbol_watch))
    return (
        f"{table}\n\nETF concentration is not an automatic rejection in this report. "
        "It is treated as a risk label and interpretability clue for ongoing tracking."
    )


def _candidate_interpretation(
    summary: pd.DataFrame,
    status: pd.DataFrame,
    symbol_watch: pd.DataFrame,
    candidate_id: str,
) -> str:
    rows = summary.loc[summary["candidate_id"].astype(str).eq(candidate_id)]
    if rows.empty:
        return f"No tracking rows were available for `{candidate_id}`."
    primary = rows.loc[rows["practical_role"].astype(str).isin(PRIMARY_ROLES)]
    selected = primary.iloc[0] if not primary.empty else rows.iloc[0]
    status_row = _matching_rows(status, selected)
    symbol_row = _matching_rows(symbol_watch, selected)
    tracking_status = (
        str(status_row["tracking_status"].iloc[0]) if not status_row.empty else "unavailable"
    )
    concentration = (
        str(symbol_row["concentration_watch_level"].iloc[0])
        if not symbol_row.empty
        else "unavailable"
    )
    role = str(selected.get("practical_role"))
    return (
        f"`{candidate_id}` remains a `{role}` with status `{tracking_status}`. "
        f"Full-period strategy_vs_qqq is `{_format_value(selected.get('strategy_vs_qqq'))}`, "
        f"max_drawdown is `{_format_value(selected.get('max_drawdown'))}`, and "
        f"return_to_drawdown is `{_format_value(selected.get('return_to_drawdown'))}`. "
        f"The top sleeve symbol is `{selected.get('top_symbol')}` and the concentration "
        f"watch label is `{concentration}`."
    )


def _production_promotion_requirements() -> str:
    return (
        "Production promotion would require a separate forward-validation process, "
        "fresh out-of-sample behavior that still beats QQQ on the intended objective, "
        "acceptable drawdown behavior, reviewed signal paths, reviewed concentration "
        "risk, operational export design, health checks, and a separate approval step. "
        "This report explicitly does not provide that approval."
    )


def _tracking_caveats() -> str:
    return "\n".join(
        [
            "- Research-only tracking report.",
            "- Tiingo-only historical context inherited from the source artifacts.",
            "- No production signal export is created.",
            "- No model card is created.",
            "- No candidate is marked production-ready; `production_ready` is always false.",
            "- QQQ remains the primary benchmark.",
            "- SPY is the core sleeve asset and secondary benchmark, not the primary benchmark.",
            "- Generated artifacts should remain uncommitted unless explicitly reviewed.",
        ]
    )


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required tracking input is missing: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _rank_selected(frame: pd.DataFrame) -> pd.DataFrame:
    role_rank = {
        "primary return-oriented tracking candidate": 0,
        "primary balanced/risk-controlled tracking candidate": 1,
        "secondary tracking reference": 2,
        "raw return reference only": 3,
        "signal-review reference only": 4,
    }
    return (
        frame.assign(_role_rank=frame["practical_role"].map(role_rank).fillna(9))
        .sort_values(
            ["_role_rank", "candidate_id", "spy_core_weight"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .drop(columns=["_role_rank"])
        .reset_index(drop=True)
    )


def _matching_rows(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy()
    spy_weight = _to_float(row.get("spy_core_weight"))
    sleeve_weight = _to_float(row.get("candidate_sleeve_weight"))
    mask = (
        frame["candidate_id"].astype(str).eq(str(row.get("candidate_id")))
        & pd.to_numeric(frame["spy_core_weight"], errors="coerce")
        .round(10)
        .eq(round(spy_weight, 10))
        & pd.to_numeric(frame["candidate_sleeve_weight"], errors="coerce")
        .round(10)
        .eq(round(sleeve_weight, 10))
        & frame["portfolio_accounting_mode"]
        .astype(str)
        .eq(str(row.get("portfolio_accounting_mode")))
    )
    return frame.loc[mask].copy()


def _key_dict(row: pd.Series) -> dict[str, object]:
    return {column: row.get(column) for column in KEY_COLUMNS}


def _with_missing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _qqq_return_to_drawdown(row: pd.Series) -> float:
    explicit_delta = _to_float(row.get("return_to_drawdown_vs_qqq"))
    own_rtd = _to_float(row.get("return_to_drawdown"))
    if pd.notna(explicit_delta) and pd.notna(own_rtd):
        return own_rtd - explicit_delta
    qqq_return = _to_float(row.get("qqq_buy_hold_total_return"))
    qqq_drawdown = abs(_to_float(row.get("qqq_buy_hold_max_drawdown")))
    if pd.notna(qqq_return) and qqq_drawdown:
        return qqq_return / qqq_drawdown
    return 0.0


def _as_bool_or_numeric_negative(values: object, fallback: object) -> pd.Series:
    if isinstance(values, pd.Series) and values.notna().any():
        return _as_bool_series(values)
    return pd.to_numeric(fallback, errors="coerce").lt(0.0)


def _as_bool_or_numeric_threshold(
    values: object,
    fallback: object,
    *,
    threshold: float,
) -> pd.Series:
    if isinstance(values, pd.Series) and values.notna().any():
        return _as_bool_series(values)
    return pd.to_numeric(fallback, errors="coerce").ge(threshold)


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].map(lambda value: "true" if bool(value) else "false")
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(_format_value)
        else:
            out[column] = out[column].fillna("")
    return out


def _format_value(value: object) -> str:
    numeric = _to_float(value)
    if pd.isna(numeric):
        return ""
    return f"{numeric:.4f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
