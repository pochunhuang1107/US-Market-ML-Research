"""Visual deep-dive report for selected long-history QQQ core-sleeve rows.

This module reads completed local tracking, signal-review, and rich-replay
artifacts. It does not run a broad sweep, train models, write model cards,
create production signal exports, or mark anything production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_visual_deep_dive"
RICH_REPLAY_PREFIX = "long_history_qqq_core_sleeve_rich_replay"
SIGNAL_REVIEW_PREFIX = "long_history_qqq_core_sleeve_signal_review"
TRACKING_PREFIX = "long_history_qqq_core_sleeve_tracking"

DEFAULT_RICH_REPLAY_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_rich_replay")
DEFAULT_SIGNAL_REVIEW_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_signal_review")
DEFAULT_TRACKING_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_tracking")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_visual_deep_dive")

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_PRIMARY_CHART = "equity_curve_primary_candidates.png"
DRAWDOWN_PRIMARY_CHART = "drawdown_curve_primary_candidates.png"
EQUITY_WITH_SECONDARY_CHART = "equity_curve_with_secondary_references.png"
DRAWDOWN_WITH_SECONDARY_CHART = "drawdown_curve_with_secondary_references.png"
ROLLING_STRATEGY_VS_QQQ_CHART = "rolling_strategy_vs_qqq_primary_candidates.png"
ROLLING_DRAWDOWN_GAP_CHART = "rolling_drawdown_gap_vs_qqq_primary_candidates.png"
SUBPERIOD_RETURNS_CHART = "subperiod_returns_primary_candidates.png"
SUBPERIOD_DRAWDOWNS_CHART = "subperiod_drawdowns_primary_candidates.png"
CANDIDATE_0001_TIMELINE_CHART = "candidate_0001_symbol_timeline.png"
CANDIDATE_0003_TIMELINE_CHART = "candidate_0003_symbol_timeline.png"
CANDIDATE_0001_EXPOSURE_CHART = "candidate_0001_symbol_exposure.png"
CANDIDATE_0003_EXPOSURE_CHART = "candidate_0003_symbol_exposure.png"

CHART_NAMES = (
    EQUITY_PRIMARY_CHART,
    DRAWDOWN_PRIMARY_CHART,
    EQUITY_WITH_SECONDARY_CHART,
    DRAWDOWN_WITH_SECONDARY_CHART,
    ROLLING_STRATEGY_VS_QQQ_CHART,
    ROLLING_DRAWDOWN_GAP_CHART,
    SUBPERIOD_RETURNS_CHART,
    SUBPERIOD_DRAWDOWNS_CHART,
    CANDIDATE_0001_TIMELINE_CHART,
    CANDIDATE_0003_TIMELINE_CHART,
    CANDIDATE_0001_EXPOSURE_CHART,
    CANDIDATE_0003_EXPOSURE_CHART,
)

OUTPUT_NAMES = (SUMMARY_NAME, REPORT_NAME, *CHART_NAMES)

KEY_COLUMNS = [
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
]

PRIMARY_ROLES = {
    "primary return-oriented tracking candidate",
    "primary balanced/risk-controlled tracking candidate",
}

SUMMARY_COLUMNS = [
    *KEY_COLUMNS,
    "practical_role",
    "visual_role",
    "review_status",
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "positive_vs_qqq_window_rate",
    "worst_strategy_vs_qqq",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "concentration_watch_level",
    "drawdown_watch_level",
    "selected_symbols_in_order",
    "production_ready",
]


@dataclass(frozen=True)
class LongHistoryCoreSleeveVisualDeepDiveInputs:
    """CSV-backed inputs for the visual deep-dive report."""

    rich_replay_daily_equity: pd.DataFrame
    rich_replay_daily_drawdown: pd.DataFrame
    rich_replay_summary: pd.DataFrame
    rich_replay_subperiods: pd.DataFrame
    rich_replay_rolling_windows: pd.DataFrame
    signal_review_symbol_regime_breakdown: pd.DataFrame
    signal_review_symbol_path: pd.DataFrame
    signal_review_summary: pd.DataFrame
    tracking_summary: pd.DataFrame
    tracking_recent_windows: pd.DataFrame
    tracking_drawdown_watch: pd.DataFrame
    tracking_symbol_watch: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryCoreSleeveVisualDeepDiveArtifacts:
    """Summary, Markdown, selected rows, and chart names for the visual report."""

    summary: pd.DataFrame
    selected_rows: pd.DataFrame
    report_markdown: str
    chart_names: tuple[str, ...] = CHART_NAMES


def read_long_history_core_sleeve_visual_deep_dive_inputs(
    *,
    rich_replay_dir: Path = DEFAULT_RICH_REPLAY_DIR,
    signal_review_dir: Path = DEFAULT_SIGNAL_REVIEW_DIR,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
) -> LongHistoryCoreSleeveVisualDeepDiveInputs:
    """Read completed local artifacts required for the visual deep dive."""

    return LongHistoryCoreSleeveVisualDeepDiveInputs(
        rich_replay_daily_equity=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_daily_equity.csv"
        ),
        rich_replay_daily_drawdown=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_daily_drawdown.csv"
        ),
        rich_replay_summary=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_summary.csv"
        ),
        rich_replay_subperiods=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv"
        ),
        rich_replay_rolling_windows=_read_required_csv(
            rich_replay_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv"
        ),
        signal_review_symbol_regime_breakdown=_read_required_csv(
            signal_review_dir / f"{SIGNAL_REVIEW_PREFIX}_symbol_regime_breakdown.csv"
        ),
        signal_review_symbol_path=_read_required_csv(
            signal_review_dir / f"{SIGNAL_REVIEW_PREFIX}_symbol_path.csv"
        ),
        signal_review_summary=_read_required_csv(
            signal_review_dir / f"{SIGNAL_REVIEW_PREFIX}_summary.csv"
        ),
        tracking_summary=_read_required_csv(tracking_dir / f"{TRACKING_PREFIX}_summary.csv"),
        tracking_recent_windows=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_recent_windows.csv"
        ),
        tracking_drawdown_watch=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_drawdown_watch.csv"
        ),
        tracking_symbol_watch=_read_required_csv(
            tracking_dir / f"{TRACKING_PREFIX}_symbol_watch.csv"
        ),
    )


def build_long_history_core_sleeve_visual_deep_dive_artifacts(
    inputs: LongHistoryCoreSleeveVisualDeepDiveInputs,
) -> LongHistoryCoreSleeveVisualDeepDiveArtifacts:
    """Build visual-report metadata from completed local inputs."""

    selected = select_visual_deep_dive_rows(inputs.tracking_summary)
    summary = build_visual_deep_dive_summary(
        selected=selected,
        rich_replay_summary=inputs.rich_replay_summary,
        signal_review_summary=inputs.signal_review_summary,
        tracking_drawdown_watch=inputs.tracking_drawdown_watch,
        tracking_symbol_watch=inputs.tracking_symbol_watch,
    )
    report = render_long_history_core_sleeve_visual_deep_dive_markdown(summary)
    return LongHistoryCoreSleeveVisualDeepDiveArtifacts(
        summary=summary,
        selected_rows=selected,
        report_markdown=report,
    )


def select_visual_deep_dive_rows(tracking_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the two primary rows and the two 40/60 sizing references."""

    if tracking_summary.empty:
        return pd.DataFrame(
            columns=[*KEY_COLUMNS, "practical_role", "visual_role", "production_ready"]
        )
    _validate_columns(tracking_summary, [*KEY_COLUMNS, "practical_role"], "tracking summary")
    rows = tracking_summary.copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["spy_core_weight"] = pd.to_numeric(rows["spy_core_weight"], errors="coerce")
    rows["candidate_sleeve_weight"] = pd.to_numeric(
        rows["candidate_sleeve_weight"], errors="coerce"
    )
    rows["portfolio_accounting_mode"] = rows["portfolio_accounting_mode"].astype(str)

    primary_mask = rows["practical_role"].astype(str).isin(PRIMARY_ROLES) & rows[
        "candidate_id"
    ].isin(["candidate_0001", "candidate_0003"])
    secondary_mask = (
        rows["practical_role"].astype(str).eq("secondary tracking reference")
        & rows["candidate_id"].isin(["candidate_0001", "candidate_0003"])
        & rows["spy_core_weight"].round(10).eq(0.40)
        & rows["candidate_sleeve_weight"].round(10).eq(0.60)
    )
    selected = rows.loc[primary_mask | secondary_mask].copy()
    selected["visual_role"] = np.where(
        primary_mask.loc[selected.index], "primary_candidate", "secondary_40_60_reference"
    )
    selected["production_ready"] = False
    return _rank_selected(selected).reset_index(drop=True)


def primary_visual_rows(selected: pd.DataFrame) -> pd.DataFrame:
    """Return the current primary visual rows only."""

    if selected.empty:
        return selected.copy()
    return selected.loc[selected["visual_role"].astype(str).eq("primary_candidate")].copy()


def secondary_visual_rows(selected: pd.DataFrame) -> pd.DataFrame:
    """Return the retained 40/60 sizing-reference rows only."""

    if selected.empty:
        return selected.copy()
    return selected.loc[selected["visual_role"].astype(str).eq("secondary_40_60_reference")].copy()


def build_visual_deep_dive_summary(
    *,
    selected: pd.DataFrame,
    rich_replay_summary: pd.DataFrame,
    signal_review_summary: pd.DataFrame,
    tracking_drawdown_watch: pd.DataFrame,
    tracking_symbol_watch: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact visual report summary table."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        rich = _first_or_empty(_matching_rows(rich_replay_summary, selected_row))
        signal = _first_or_empty(_matching_rows(signal_review_summary, selected_row))
        drawdown = _first_or_empty(_matching_rows(tracking_drawdown_watch, selected_row))
        symbol = _first_or_empty(_matching_rows(tracking_symbol_watch, selected_row))
        out = {
            column: rich.get(column, selected_row.get(column, pd.NA)) for column in SUMMARY_COLUMNS
        }
        out.update(_key_dict(selected_row))
        out["practical_role"] = selected_row.get("practical_role")
        out["visual_role"] = selected_row.get("visual_role")
        out["review_status"] = visual_review_status(selected_row)
        out["concentration_watch_level"] = symbol.get(
            "concentration_watch_level",
            signal.get("concentration_watch_level", pd.NA),
        )
        out["drawdown_watch_level"] = drawdown.get("drawdown_watch_level", pd.NA)
        out["top_symbol"] = symbol.get(
            "top_symbol", signal.get("top_symbol", out.get("top_symbol"))
        )
        out["top_symbol_exposure_pct"] = symbol.get(
            "top_symbol_exposure_pct",
            signal.get("top_symbol_exposure_pct", out.get("top_symbol_exposure_pct")),
        )
        out["top3_symbol_exposure_pct"] = symbol.get(
            "top3_symbol_exposure_pct",
            signal.get("top3_symbol_exposure_pct", out.get("top3_symbol_exposure_pct")),
        )
        out["selected_symbols_in_order"] = signal.get("selected_symbols_in_order", pd.NA)
        out["production_ready"] = False
        rows.append(out)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary["production_ready"] = False
    return summary.reset_index(drop=True)


def visual_review_status(row: pd.Series) -> str:
    """Return the fixed current review status for visualized rows."""

    candidate_id = str(row.get("candidate_id"))
    visual_role = str(row.get("visual_role", ""))
    if visual_role == "secondary_40_60_reference":
        return "watchlist_sizing_reference"
    if candidate_id == "candidate_0001":
        return "keep_tracking_with_concentration_caution"
    if candidate_id == "candidate_0003":
        return "keep_tracking_with_drawdown_caution"
    return "research_reference_only"


def write_long_history_core_sleeve_visual_deep_dive_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveVisualDeepDiveArtifacts,
    inputs: LongHistoryCoreSleeveVisualDeepDiveInputs,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the visual report CSV, Markdown, and PNG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    write_visual_deep_dive_charts(
        inputs=inputs,
        selected=artifacts.selected_rows,
        output_dir=output_dir,
    )
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    return paths


def write_visual_deep_dive_charts(
    *,
    inputs: LongHistoryCoreSleeveVisualDeepDiveInputs,
    selected: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write all required Matplotlib PNG charts to the output directory."""

    primary = primary_visual_rows(selected)
    _plot_equity_curves(
        inputs.rich_replay_daily_equity,
        primary,
        output_dir / EQUITY_PRIMARY_CHART,
        "Primary QQQ-relative core-sleeve equity curves",
    )
    _plot_drawdown_curves(
        inputs.rich_replay_daily_drawdown,
        primary,
        output_dir / DRAWDOWN_PRIMARY_CHART,
        "Primary QQQ-relative core-sleeve drawdowns",
    )
    _plot_equity_curves(
        inputs.rich_replay_daily_equity,
        selected,
        output_dir / EQUITY_WITH_SECONDARY_CHART,
        "Primary candidates with 40/60 sizing references",
    )
    _plot_drawdown_curves(
        inputs.rich_replay_daily_drawdown,
        selected,
        output_dir / DRAWDOWN_WITH_SECONDARY_CHART,
        "Drawdowns with 40/60 sizing references",
    )
    _plot_rolling_metric(
        inputs.rich_replay_rolling_windows,
        primary,
        output_dir / ROLLING_STRATEGY_VS_QQQ_CHART,
        metric="strategy_vs_qqq",
        title="Rolling strategy return gap versus QQQ",
        ylabel="Strategy minus QQQ total return",
    )
    _plot_rolling_metric(
        inputs.rich_replay_rolling_windows,
        primary,
        output_dir / ROLLING_DRAWDOWN_GAP_CHART,
        metric="drawdown_improvement_vs_qqq",
        title="Rolling drawdown gap versus QQQ",
        ylabel="Strategy drawdown minus QQQ drawdown",
    )
    _plot_subperiod_metric(
        inputs.rich_replay_subperiods,
        primary,
        output_dir / SUBPERIOD_RETURNS_CHART,
        metric="total_return",
        benchmark_metric="qqq_total_return",
        title="Primary candidate subperiod returns",
        ylabel="Total return",
    )
    _plot_subperiod_metric(
        inputs.rich_replay_subperiods,
        primary,
        output_dir / SUBPERIOD_DRAWDOWNS_CHART,
        metric="max_drawdown",
        benchmark_metric="qqq_max_drawdown",
        title="Primary candidate subperiod drawdowns",
        ylabel="Max drawdown",
    )
    _plot_symbol_timeline(
        inputs.signal_review_symbol_regime_breakdown,
        "candidate_0001",
        output_dir / CANDIDATE_0001_TIMELINE_CHART,
    )
    _plot_symbol_timeline(
        inputs.signal_review_symbol_regime_breakdown,
        "candidate_0003",
        output_dir / CANDIDATE_0003_TIMELINE_CHART,
    )
    _plot_symbol_exposure(
        inputs.signal_review_symbol_regime_breakdown,
        "candidate_0001",
        output_dir / CANDIDATE_0001_EXPOSURE_CHART,
    )
    _plot_symbol_exposure(
        inputs.signal_review_symbol_regime_breakdown,
        "candidate_0003",
        output_dir / CANDIDATE_0003_EXPOSURE_CHART,
    )


def render_long_history_core_sleeve_visual_deep_dive_markdown(summary: pd.DataFrame) -> str:
    """Render the Markdown visual report with relative PNG links."""

    lines = [
        "# Long-History QQQ Core-Sleeve Visual Deep-Dive",
        "",
        "This is a research-only visual report over completed local artifacts. It does "
        "not run a broad sweep, train models, export production signals, create a "
        "model card, or mark any candidate production-ready.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(summary),
        "",
        "## 2. Current strategies being visualized",
        "",
        _markdown_table(_format_for_markdown(summary)),
        "",
        "## 3. Equity curve comparison",
        "",
        f"![Primary candidate equity curves]({EQUITY_PRIMARY_CHART})",
        "",
        f"![Primary candidates with 40/60 references]({EQUITY_WITH_SECONDARY_CHART})",
        "",
        "## 4. Drawdown comparison",
        "",
        f"![Primary candidate drawdowns]({DRAWDOWN_PRIMARY_CHART})",
        "",
        f"![Drawdowns with 40/60 references]({DRAWDOWN_WITH_SECONDARY_CHART})",
        "",
        "## 5. Rolling QQQ-relative behavior",
        "",
        f"![Rolling strategy return gap versus QQQ]({ROLLING_STRATEGY_VS_QQQ_CHART})",
        "",
        f"![Rolling drawdown gap versus QQQ]({ROLLING_DRAWDOWN_GAP_CHART})",
        "",
        "## 6. Subperiod performance",
        "",
        f"![Subperiod returns]({SUBPERIOD_RETURNS_CHART})",
        "",
        f"![Subperiod drawdowns]({SUBPERIOD_DRAWDOWNS_CHART})",
        "",
        "## 7. candidate_0001 ETF sleeve timeline and exposure",
        "",
        f"![candidate_0001 ETF sleeve timeline]({CANDIDATE_0001_TIMELINE_CHART})",
        "",
        f"![candidate_0001 ETF sleeve exposure]({CANDIDATE_0001_EXPOSURE_CHART})",
        "",
        "## 8. candidate_0003 ETF sleeve timeline and exposure",
        "",
        f"![candidate_0003 ETF sleeve timeline]({CANDIDATE_0003_TIMELINE_CHART})",
        "",
        f"![candidate_0003 ETF sleeve exposure]({CANDIDATE_0003_EXPOSURE_CHART})",
        "",
        "## 9. What the charts reveal",
        "",
        _chart_takeaways(summary),
        "",
        "## 10. Current decision",
        "",
        (
            "- Keep tracking candidate_0001 with active sleeve exposure review.\n"
            "- Keep tracking candidate_0003 with drawdown caution.\n"
            "- Keep the 40/60 rows as sizing references and watchlist rows.\n"
            "- Do not promote any row to production."
        ),
        "",
        "## 11. Caveats",
        "",
        (
            "- Research-only visual report.\n"
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


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected visual-report output paths."""

    return {name: output_dir / name for name in OUTPUT_NAMES}


def _plot_equity_curves(
    frame: pd.DataFrame, selected: pd.DataFrame, path: Path, title: str
) -> None:
    _plot_curve_family(
        frame=frame,
        selected=selected,
        path=path,
        title=title,
        value_column="equity",
        qqq_column="qqq_equity",
        spy_column="spy_equity",
        ylabel="Growth of $1",
        as_percent=False,
    )


def _plot_drawdown_curves(
    frame: pd.DataFrame, selected: pd.DataFrame, path: Path, title: str
) -> None:
    _plot_curve_family(
        frame=frame,
        selected=selected,
        path=path,
        title=title,
        value_column="drawdown",
        qqq_column="qqq_drawdown",
        spy_column="spy_drawdown",
        ylabel="Drawdown",
        as_percent=True,
    )


def _plot_curve_family(
    *,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
    title: str,
    value_column: str,
    qqq_column: str,
    spy_column: str,
    ylabel: str,
    as_percent: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    if frame.empty or selected.empty:
        _empty_chart(ax, "No daily curve rows available.")
    else:
        plotted_benchmarks = False
        for _, row in selected.iterrows():
            group = _matching_rows(frame, row).copy()
            if group.empty:
                continue
            group["date"] = pd.to_datetime(group["date"], errors="coerce")
            group = group.sort_values("date", kind="mergesort")
            ax.plot(
                group["date"],
                pd.to_numeric(group[value_column], errors="coerce"),
                linewidth=1.8,
                label=_row_label(row),
            )
            if not plotted_benchmarks:
                ax.plot(
                    group["date"],
                    pd.to_numeric(group[qqq_column], errors="coerce"),
                    linewidth=2.2,
                    color="black",
                    label="QQQ buy-and-hold",
                )
                ax.plot(
                    group["date"],
                    pd.to_numeric(group[spy_column], errors="coerce"),
                    linewidth=1.8,
                    color="dimgray",
                    linestyle="--",
                    label="SPY buy-and-hold",
                )
                plotted_benchmarks = True
        if not ax.lines:
            _empty_chart(ax, "No matching daily curve rows available.")
    _finish_date_chart(ax, title, ylabel, as_percent=as_percent)
    _save_figure(fig, path)


def _plot_rolling_metric(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    if frame.empty or selected.empty:
        _empty_chart(ax, "No rolling-window rows available.")
    else:
        for _, row in selected.iterrows():
            group = _matching_rows(frame, row).copy()
            if group.empty:
                continue
            group["window_end"] = pd.to_datetime(group["window_end"], errors="coerce")
            group["window_years"] = pd.to_numeric(group.get("window_years"), errors="coerce")
            for years in sorted(group["window_years"].dropna().unique()):
                subset = group.loc[group["window_years"].eq(years)].sort_values(
                    "window_end", kind="mergesort"
                )
                label = f"{_short_row_label(row)} {int(years)}y"
                ax.plot(
                    subset["window_end"],
                    pd.to_numeric(subset[metric], errors="coerce"),
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    label=label,
                )
        ax.axhline(0.0, color="black", linewidth=1.0)
        if not ax.lines:
            _empty_chart(ax, "No matching rolling-window rows available.")
    _finish_date_chart(ax, title, ylabel, as_percent=True)
    _save_figure(fig, path)


def _plot_subperiod_metric(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    benchmark_metric: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    rows: list[pd.DataFrame] = []
    if not frame.empty:
        for _, row in selected.iterrows():
            group = _matching_rows(frame, row).copy()
            if group.empty:
                continue
            group["series_label"] = _short_row_label(row)
            rows.append(group)
    if not rows:
        _empty_chart(ax, "No subperiod rows available.")
    else:
        data = pd.concat(rows, ignore_index=True, sort=False)
        periods = data["period_name"].dropna().astype(str).unique().tolist()
        labels = data["series_label"].dropna().astype(str).unique().tolist()
        x = np.arange(len(periods))
        width = 0.22
        offsets = np.linspace(-width, width, num=max(len(labels), 1))
        for offset, label in zip(offsets, labels, strict=False):
            values = [
                pd.to_numeric(
                    data.loc[
                        data["period_name"].astype(str).eq(period) & data["series_label"].eq(label),
                        metric,
                    ],
                    errors="coerce",
                ).mean()
                for period in periods
            ]
            ax.bar(x + offset, values, width=width, label=label)
        benchmark_values = [
            pd.to_numeric(
                data.loc[data["period_name"].astype(str).eq(period), benchmark_metric],
                errors="coerce",
            ).mean()
            for period in periods
        ]
        ax.plot(
            x, benchmark_values, color="black", marker="o", linewidth=2.0, label="QQQ buy-and-hold"
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Subperiod")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save_figure(fig, path)


def _plot_symbol_timeline(frame: pd.DataFrame, candidate_id: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    rows = frame.loc[
        frame.get("candidate_id", pd.Series(dtype=object)).astype(str).eq(candidate_id)
    ].copy()
    if "practical_role" in rows.columns:
        primary_rows = rows.loc[rows["practical_role"].astype(str).isin(PRIMARY_ROLES)].copy()
        if not primary_rows.empty:
            rows = primary_rows
    if rows.empty:
        _empty_chart(ax, f"No symbol timeline rows available for {candidate_id}.")
    else:
        rows["start_date"] = pd.to_datetime(rows["start_date"], errors="coerce")
        rows["end_date"] = pd.to_datetime(rows["end_date"], errors="coerce")
        rows = rows.dropna(subset=["start_date", "end_date"]).sort_values(
            "start_date", kind="mergesort"
        )
        symbols = rows["symbol"].dropna().astype(str).unique().tolist()
        positions = {symbol: index for index, symbol in enumerate(symbols)}
        cmap = plt.get_cmap("tab20")
        for index, row in rows.iterrows():
            symbol = str(row["symbol"])
            start = mdates.date2num(row["start_date"])
            end = mdates.date2num(row["end_date"])
            ax.broken_barh(
                [(start, max(end - start, 1.0))],
                (positions[symbol] - 0.38, 0.76),
                facecolors=cmap(index % 20),
                edgecolors="white",
            )
            midpoint = mdates.num2date(start + max(end - start, 1.0) / 2.0)
            ax.text(midpoint, positions[symbol], symbol, ha="center", va="center", fontsize=8)
        ax.set_yticks(list(positions.values()))
        ax.set_yticklabels(list(positions.keys()))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"{candidate_id} ETF sleeve timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Selected ETF")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, path)


def _plot_symbol_exposure(frame: pd.DataFrame, candidate_id: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    rows = frame.loc[
        frame.get("candidate_id", pd.Series(dtype=object)).astype(str).eq(candidate_id)
    ].copy()
    if "practical_role" in rows.columns:
        primary_rows = rows.loc[rows["practical_role"].astype(str).isin(PRIMARY_ROLES)].copy()
        if not primary_rows.empty:
            rows = primary_rows
    if rows.empty:
        _empty_chart(ax, f"No symbol exposure rows available for {candidate_id}.")
    else:
        exposure = (
            rows.assign(exposure_pct=lambda df: pd.to_numeric(df["exposure_pct"], errors="coerce"))
            .groupby("symbol", as_index=False)["exposure_pct"]
            .sum()
            .sort_values("exposure_pct", ascending=False, kind="mergesort")
        )
        ax.bar(exposure["symbol"].astype(str), exposure["exposure_pct"], color="#4472c4")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_ylim(bottom=0.0)
    ax.set_title(f"{candidate_id} ETF sleeve exposure")
    ax.set_xlabel("ETF")
    ax.set_ylabel("Exposure share")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _finish_date_chart(ax: plt.Axes, title: str, ylabel: str, *, as_percent: bool) -> None:
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if as_percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _empty_chart(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _executive_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No visual deep-dive rows were available."
    primary = int(summary["visual_role"].astype(str).eq("primary_candidate").sum())
    secondary = int(summary["visual_role"].astype(str).eq("secondary_40_60_reference").sum())
    prod_ready = int(_as_bool_series(summary["production_ready"]).sum())
    return (
        f"Visual deep dive covers `{primary}` primary tracking candidates and `{secondary}` "
        "40/60 sizing-reference rows. candidate_0001 remains a return-oriented tracking "
        "candidate with XLE-led active sleeve exposure to monitor; candidate_0003 remains "
        "a balanced/risk-controlled tracking candidate with drawdown caution. QQQ buy-and-hold "
        "is the primary benchmark; SPY buy-and-hold is shown as the secondary benchmark "
        "and core sleeve asset. "
        f"Production-ready rows: `{prod_ready}`."
    )


def _chart_takeaways(summary: pd.DataFrame) -> str:
    candidate_0001 = _first_by_candidate(summary, "candidate_0001")
    candidate_0003 = _first_by_candidate(summary, "candidate_0003")
    lines = [
        "- The equity and drawdown charts should be read against QQQ first, with "
        "SPY included to show the core sleeve anchor.",
        "- The 40/60 rows are sizing references only; they are not promotion candidates.",
    ]
    if not candidate_0001.empty:
        lines.append(
            "- candidate_0001 shows the return-oriented case, with an XLE-led regime "
            "sleeve that should be monitored as the strategy's active bet."
        )
    if not candidate_0003.empty:
        lines.append(
            "- candidate_0003 shows broader rotation, but ARKK/growth-linked "
            "drawdown behavior remains the main visual risk to keep reviewing."
        )
    return "\n".join(lines)


def _first_by_candidate(frame: pd.DataFrame, candidate_id: str) -> pd.Series:
    rows = frame.loc[
        frame["candidate_id"].astype(str).eq(candidate_id)
        & frame["visual_role"].astype(str).eq("primary_candidate")
    ]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _row_label(row: pd.Series) -> str:
    return (
        f"{row.get('candidate_id')} "
        f"{_pct(row.get('spy_core_weight'))}/{_pct(row.get('candidate_sleeve_weight'))}"
    )


def _short_row_label(row: pd.Series) -> str:
    return (
        f"{row.get('candidate_id')} "
        f"{_pct(row.get('spy_core_weight'))}/{_pct(row.get('candidate_sleeve_weight'))}"
    )


def _pct(value: object) -> str:
    numeric = _to_float(value)
    if pd.isna(numeric):
        return "NA"
    return f"{numeric:.0%}"


def _rank_selected(frame: pd.DataFrame) -> pd.DataFrame:
    role_rank = {
        "primary return-oriented tracking candidate": 0,
        "primary balanced/risk-controlled tracking candidate": 1,
        "secondary tracking reference": 2,
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
        return frame.copy()
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy()
    mask = frame["candidate_id"].astype(str).eq(str(row.get("candidate_id")))
    mask &= (
        pd.to_numeric(frame["spy_core_weight"], errors="coerce")
        .round(10)
        .eq(round(_to_float(row.get("spy_core_weight")), 10))
    )
    mask &= (
        pd.to_numeric(frame["candidate_sleeve_weight"], errors="coerce")
        .round(10)
        .eq(round(_to_float(row.get("candidate_sleeve_weight")), 10))
    )
    mask &= (
        frame["portfolio_accounting_mode"].astype(str).eq(str(row.get("portfolio_accounting_mode")))
    )
    return frame.loc[mask].copy()


def _key_dict(row: pd.Series) -> dict[str, object]:
    return {column: row.get(column) for column in KEY_COLUMNS}


def _first_or_empty(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)


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


def _to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required visual deep-dive input is missing: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
