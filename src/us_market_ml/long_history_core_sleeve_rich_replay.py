"""Rich replay report for selected long-history QQQ core-sleeve candidates.

This module only reads completed local decision, refinement, and candidate
deep-dive artifacts. It does not run sweeps, train models, write model cards,
create production signal exports, or mark anything production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_rich_replay"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_rich_replay")

DEFAULT_DECISION_RECOMMENDATIONS = Path(
    "artifacts/reports/long_history_qqq_core_sleeve_decision/"
    "long_history_qqq_core_sleeve_decision_recommendations.csv"
)
DEFAULT_DECISION_MATRIX = Path(
    "artifacts/reports/long_history_qqq_core_sleeve_decision/"
    "long_history_qqq_core_sleeve_decision_matrix.csv"
)
DEFAULT_REFINEMENT_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_core_sleeve_refinement/"
    "long_history_qqq_core_sleeve_refinement_results.csv"
)
DEFAULT_REFINEMENT_DAILY_EQUITY = Path(
    "artifacts/sweeps/long_history_qqq_core_sleeve_refinement/"
    "long_history_qqq_core_sleeve_refinement_daily_equity.csv"
)
DEFAULT_REFINEMENT_DAILY_DRAWDOWN = Path(
    "artifacts/sweeps/long_history_qqq_core_sleeve_refinement/"
    "long_history_qqq_core_sleeve_refinement_daily_drawdown.csv"
)
DEFAULT_REFINEMENT_SUBPERIODS = Path(
    "artifacts/sweeps/long_history_qqq_core_sleeve_refinement/"
    "long_history_qqq_core_sleeve_refinement_subperiods.csv"
)
DEFAULT_REFINEMENT_ROLLING_WINDOWS = Path(
    "artifacts/sweeps/long_history_qqq_core_sleeve_refinement/"
    "long_history_qqq_core_sleeve_refinement_rolling_windows.csv"
)
DEFAULT_CANDIDATE_SELECTED_SYMBOLS = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_selected_symbols.csv"
)
DEFAULT_CANDIDATE_SYMBOL_EXPOSURE = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_symbol_exposure.csv"
)
DEFAULT_CANDIDATE_SYMBOL_PERIODS = Path(
    "artifacts/deep_dives/long_history_qqq_candidate_deep_dive/"
    "long_history_qqq_candidate_deep_dive_symbol_periods.csv"
)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
DAILY_DRAWDOWN_NAME = f"{OUTPUT_PREFIX}_daily_drawdown.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
SLEEVE_CONTRIBUTION_NAME = f"{OUTPUT_PREFIX}_sleeve_contribution.csv"
SELECTED_SYMBOLS_NAME = f"{OUTPUT_PREFIX}_selected_symbols.csv"
SYMBOL_EXPOSURE_NAME = f"{OUTPUT_PREFIX}_symbol_exposure.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

OUTPUT_NAMES = (
    SUMMARY_NAME,
    DAILY_EQUITY_NAME,
    DAILY_DRAWDOWN_NAME,
    SUBPERIODS_NAME,
    ROLLING_WINDOWS_NAME,
    SLEEVE_CONTRIBUTION_NAME,
    SELECTED_SYMBOLS_NAME,
    SYMBOL_EXPOSURE_NAME,
    DECISION_MATRIX_NAME,
    REPORT_NAME,
)

REPLAY_PERIODS = (
    "full",
    "covid_crash_recovery",
    "bear_2022",
    "ai_bull_2023_2024",
    "recent_2025_2026",
)

KEY_COLUMNS = [
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
]

SUMMARY_COLUMNS = [
    "recommendation_action",
    "practical_role",
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "standalone_candidate_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "standalone_candidate_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "drawdown_improvement_vs_candidate",
    "return_to_drawdown",
    "return_to_drawdown_vs_qqq",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "best_strategy_vs_qqq_window_label",
    "best_strategy_vs_qqq",
    "worst_strategy_vs_qqq_window_label",
    "rolling_drawdown_improvement_vs_qqq_rate",
    "best_rolling_drawdown_improvement_vs_qqq",
    "worst_rolling_drawdown_improvement_vs_qqq",
    "portfolio_max_drawdown_start",
    "portfolio_max_drawdown_end",
    "qqq_max_drawdown_start",
    "qqq_max_drawdown_end",
    "spy_max_drawdown_start",
    "spy_max_drawdown_end",
    "standalone_candidate_max_drawdown_start",
    "standalone_candidate_max_drawdown_end",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "top_contributors",
    "decision_category",
    "final_decision",
    "production_ready",
]

SLEEVE_CONTRIBUTION_COLUMNS = [
    "recommendation_action",
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "total_portfolio_return",
    "estimated_spy_core_contribution",
    "estimated_candidate_sleeve_contribution",
    "estimated_unexplained_compounding",
    "sleeve_contribution_share",
    "core_contribution_share",
    "production_ready",
]


@dataclass(frozen=True)
class LongHistoryCoreSleeveRichReplayInputs:
    """CSV-backed inputs for the rich replay report."""

    decision_recommendations: pd.DataFrame
    decision_matrix: pd.DataFrame
    refinement_results: pd.DataFrame
    refinement_daily_equity: pd.DataFrame
    refinement_daily_drawdown: pd.DataFrame
    refinement_subperiods: pd.DataFrame
    refinement_rolling_windows: pd.DataFrame
    candidate_selected_symbols: pd.DataFrame
    candidate_symbol_exposure: pd.DataFrame
    candidate_symbol_periods: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryCoreSleeveRichReplayArtifacts:
    """CSV and Markdown outputs for the rich replay report."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    sleeve_contribution: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def read_long_history_core_sleeve_rich_replay_inputs(
    *,
    decision_recommendations: Path = DEFAULT_DECISION_RECOMMENDATIONS,
    decision_matrix: Path = DEFAULT_DECISION_MATRIX,
    refinement_results: Path = DEFAULT_REFINEMENT_RESULTS,
    refinement_daily_equity: Path = DEFAULT_REFINEMENT_DAILY_EQUITY,
    refinement_daily_drawdown: Path = DEFAULT_REFINEMENT_DAILY_DRAWDOWN,
    refinement_subperiods: Path = DEFAULT_REFINEMENT_SUBPERIODS,
    refinement_rolling_windows: Path = DEFAULT_REFINEMENT_ROLLING_WINDOWS,
    candidate_selected_symbols: Path = DEFAULT_CANDIDATE_SELECTED_SYMBOLS,
    candidate_symbol_exposure: Path = DEFAULT_CANDIDATE_SYMBOL_EXPOSURE,
    candidate_symbol_periods: Path = DEFAULT_CANDIDATE_SYMBOL_PERIODS,
) -> LongHistoryCoreSleeveRichReplayInputs:
    """Read all completed local artifacts required for the rich replay report."""

    return LongHistoryCoreSleeveRichReplayInputs(
        decision_recommendations=_read_required_csv(decision_recommendations),
        decision_matrix=_read_required_csv(decision_matrix),
        refinement_results=_read_required_csv(refinement_results),
        refinement_daily_equity=_read_required_csv(refinement_daily_equity),
        refinement_daily_drawdown=_read_required_csv(refinement_daily_drawdown),
        refinement_subperiods=_read_required_csv(refinement_subperiods),
        refinement_rolling_windows=_read_required_csv(refinement_rolling_windows),
        candidate_selected_symbols=_read_required_csv(candidate_selected_symbols),
        candidate_symbol_exposure=_read_required_csv(candidate_symbol_exposure),
        candidate_symbol_periods=_read_required_csv(candidate_symbol_periods),
    )


def build_long_history_core_sleeve_rich_replay_artifacts(
    inputs: LongHistoryCoreSleeveRichReplayInputs,
) -> LongHistoryCoreSleeveRichReplayArtifacts:
    """Build all rich replay artifacts from completed inputs."""

    selected = select_rich_replay_portfolios(inputs.decision_recommendations)
    decision_matrix = build_rich_replay_decision_matrix(
        selected=selected,
        decision_matrix=inputs.decision_matrix,
    )
    daily_equity = build_rich_replay_daily_equity(
        selected=selected,
        daily_equity=inputs.refinement_daily_equity,
    )
    daily_drawdown = build_rich_replay_daily_drawdown(daily_equity)
    rolling_windows = build_rich_replay_rolling_windows(
        selected=selected,
        rolling_windows=inputs.refinement_rolling_windows,
    )
    subperiods = build_rich_replay_subperiods(
        selected=selected,
        subperiods=inputs.refinement_subperiods,
    )
    sleeve_contribution = build_sleeve_contribution(selected=selected, daily_equity=daily_equity)
    selected_symbols = build_selected_symbol_path(
        selected=selected,
        candidate_selected_symbols=inputs.candidate_selected_symbols,
    )
    symbol_exposure = build_selected_symbol_exposure(
        selected=selected,
        candidate_symbol_exposure=inputs.candidate_symbol_exposure,
    )
    summary = build_rich_replay_summary(
        selected=selected,
        decision_matrix=decision_matrix,
        daily_equity=daily_equity,
        rolling_windows=rolling_windows,
        sleeve_contribution=sleeve_contribution,
        symbol_exposure=symbol_exposure,
    )
    report = render_long_history_core_sleeve_rich_replay_markdown(
        inputs=inputs,
        summary=summary,
        decision_matrix=decision_matrix,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        sleeve_contribution=sleeve_contribution,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
    )
    return LongHistoryCoreSleeveRichReplayArtifacts(
        summary=summary,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        sleeve_contribution=sleeve_contribution,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        decision_matrix=decision_matrix,
        report_markdown=report,
    )


def select_rich_replay_portfolios(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Return rich replay, tracking, and standalone reference rows."""

    if recommendations.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "recommendation_action", "practical_role"])
    _validate_columns(
        recommendations,
        ["recommendation_action", *KEY_COLUMNS, "production_ready"],
        "decision recommendations",
    )
    actions = {"rich_replay_next", "track_next", "reference_only"}
    selected = recommendations.loc[
        recommendations["recommendation_action"].astype(str).isin(actions)
    ].copy()
    selected = selected.loc[~selected["candidate_id"].astype(str).eq("broad_sweep")].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["spy_core_weight"] = pd.to_numeric(selected["spy_core_weight"], errors="coerce")
    selected["candidate_sleeve_weight"] = pd.to_numeric(
        selected["candidate_sleeve_weight"],
        errors="coerce",
    )
    selected["production_ready"] = False
    selected["practical_role"] = selected.apply(_practical_role_from_recommendation, axis=1)
    return selected.reset_index(drop=True)


def build_rich_replay_decision_matrix(
    *,
    selected: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact rich replay decision matrix for selected/reference rows."""

    rows = _filter_by_selected_keys(decision_matrix, selected)
    if rows.empty:
        rows = selected.copy()
    rows = rows.merge(
        selected.loc[
            :,
            [
                "candidate_id",
                "spy_core_weight",
                "candidate_sleeve_weight",
                "portfolio_accounting_mode",
                "recommendation_action",
                "practical_role",
            ],
        ],
        on=KEY_COLUMNS,
        how="left",
    )
    rows["final_decision"] = rows.apply(classify_rich_replay_decision, axis=1)
    rows["production_ready"] = False

    stop_rows = _stop_spending_time_rows(decision_matrix=decision_matrix, selected=selected)
    out = pd.concat([rows, stop_rows], ignore_index=True, sort=False)
    out["production_ready"] = False
    return out.reset_index(drop=True)


def classify_rich_replay_decision(row: pd.Series) -> str:
    """Classify one rich replay decision row."""

    action = str(row.get("recommendation_action", ""))
    if action == "rich_replay_next":
        beats = _as_bool_scalar(row.get("beats_qqq_return"))
        dd = _as_bool_scalar(row.get("improves_drawdown_vs_qqq"))
        rtd = _as_bool_scalar(row.get("improves_return_to_drawdown_vs_qqq"))
        if beats and dd and rtd:
            return "primary_tracking_candidate"
        return "reject_deprioritize"
    if action == "track_next":
        return "secondary_tracking_candidate"
    if action == "reference_only":
        return "reference_only"
    return "reject_deprioritize"


def build_rich_replay_daily_equity(
    *,
    selected: pd.DataFrame,
    daily_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Return daily equity rows for selected portfolio/reference configurations."""

    rows = _filter_by_selected_keys(daily_equity, selected)
    if rows.empty:
        return rows
    rows = rows.merge(_selected_labels(selected), on=KEY_COLUMNS, how="left")
    rows["production_ready"] = False
    return rows.reset_index(drop=True)


def build_rich_replay_daily_drawdown(daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Return drawdown replay rows with QQQ/SPY/standalone references."""

    if daily_equity.empty:
        return pd.DataFrame()
    columns = [
        "date",
        *KEY_COLUMNS,
        "recommendation_action",
        "practical_role",
        "portfolio_name",
        "equity",
        "drawdown",
        "candidate_equity",
        "candidate_drawdown",
        "qqq_equity",
        "qqq_drawdown",
        "spy_equity",
        "spy_drawdown",
        "production_ready",
    ]
    return _with_missing_columns(daily_equity, columns).loc[:, columns].reset_index(drop=True)


def build_rich_replay_subperiods(
    *,
    selected: pd.DataFrame,
    subperiods: pd.DataFrame,
) -> pd.DataFrame:
    """Return selected rows for the required replay subperiods."""

    rows = _filter_by_selected_keys(subperiods, selected)
    if rows.empty:
        return rows
    rows = rows.loc[rows["period_name"].astype(str).isin(REPLAY_PERIODS)].copy()
    rows = rows.merge(_selected_labels(selected), on=KEY_COLUMNS, how="left")
    rows["production_ready"] = False
    return rows.reset_index(drop=True)


def build_rich_replay_rolling_windows(
    *,
    selected: pd.DataFrame,
    rolling_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Return selected rolling-window rows with replay labels."""

    rows = _filter_by_selected_keys(rolling_windows, selected)
    if rows.empty:
        return rows
    rows = rows.merge(_selected_labels(selected), on=KEY_COLUMNS, how="left")
    rows["rolling_drawdown_improvement_vs_qqq"] = pd.to_numeric(
        rows.get("drawdown_improvement_vs_qqq"),
        errors="coerce",
    )
    rows["production_ready"] = False
    return rows.reset_index(drop=True)


def build_sleeve_contribution(
    *,
    selected: pd.DataFrame,
    daily_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate SPY core and candidate sleeve contribution from daily weights."""

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        group = _matching_rows(daily_equity, selected_row)
        if group.empty:
            continue
        group = group.sort_values("date", kind="mergesort").copy()
        strategy_return = _period_return(group["equity"])
        spy_daily = pd.to_numeric(group.get("spy_daily_return"), errors="coerce").fillna(0.0)
        candidate_daily = pd.to_numeric(
            group.get("candidate_daily_return"),
            errors="coerce",
        ).fillna(0.0)
        effective_spy_weight = pd.to_numeric(
            group.get("effective_spy_weight"),
            errors="coerce",
        ).fillna(float(selected_row["spy_core_weight"]))
        effective_candidate_weight = pd.to_numeric(
            group.get("effective_candidate_weight"),
            errors="coerce",
        ).fillna(float(selected_row["candidate_sleeve_weight"]))
        spy_contribution = float((effective_spy_weight * spy_daily).sum())
        candidate_contribution = float((effective_candidate_weight * candidate_daily).sum())
        explained = spy_contribution + candidate_contribution
        sleeve_share = candidate_contribution / explained if explained else np.nan
        core_share = spy_contribution / explained if explained else np.nan
        rows.append(
            {
                "recommendation_action": selected_row.get("recommendation_action"),
                "candidate_id": selected_row["candidate_id"],
                "spy_core_weight": selected_row["spy_core_weight"],
                "candidate_sleeve_weight": selected_row["candidate_sleeve_weight"],
                "portfolio_accounting_mode": selected_row["portfolio_accounting_mode"],
                "portfolio_name": group["portfolio_name"].iloc[0],
                "total_portfolio_return": strategy_return,
                "estimated_spy_core_contribution": spy_contribution,
                "estimated_candidate_sleeve_contribution": candidate_contribution,
                "estimated_unexplained_compounding": strategy_return - explained,
                "sleeve_contribution_share": sleeve_share,
                "core_contribution_share": core_share,
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=SLEEVE_CONTRIBUTION_COLUMNS)


def build_selected_symbol_path(
    *,
    selected: pd.DataFrame,
    candidate_selected_symbols: pd.DataFrame,
) -> pd.DataFrame:
    """Join selected core-sleeve rows to the underlying candidate symbol timeline."""

    selected_paths = selected.loc[
        selected["recommendation_action"].isin(["rich_replay_next", "track_next"])
    ].copy()
    if selected_paths.empty:
        return pd.DataFrame()
    rows = selected_paths.merge(
        candidate_selected_symbols.assign(
            candidate_id=lambda df: df["candidate_id"].astype(str)
        ),
        on="candidate_id",
        how="left",
        suffixes=("", "_candidate"),
    )
    rows["production_ready"] = False
    return rows.reset_index(drop=True)


def build_selected_symbol_exposure(
    *,
    selected: pd.DataFrame,
    candidate_symbol_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Join selected rows to candidate symbol exposure and sleeve-weight it."""

    selected_paths = selected.loc[
        selected["recommendation_action"].isin(["rich_replay_next", "track_next"])
    ].copy()
    if selected_paths.empty:
        return pd.DataFrame()
    rows = selected_paths.merge(
        candidate_symbol_exposure.assign(
            candidate_id=lambda df: df["candidate_id"].astype(str)
        ),
        on="candidate_id",
        how="left",
        suffixes=("", "_candidate"),
    )
    rows["sleeve_weighted_exposure_pct"] = (
        pd.to_numeric(rows["candidate_sleeve_weight"], errors="coerce")
        * pd.to_numeric(rows.get("exposure_pct"), errors="coerce")
    )
    rows["production_ready"] = False
    return rows.reset_index(drop=True)


def build_rich_replay_summary(
    *,
    selected: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    daily_equity: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    sleeve_contribution: pd.DataFrame,
    symbol_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact full-period and robustness summary table."""

    rows = decision_matrix.loc[
        decision_matrix["recommendation_action"].isin(
            ["rich_replay_next", "track_next", "reference_only"]
        )
    ].copy()
    if rows.empty:
        rows = selected.copy()
    summary_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        out = {column: row.get(column, pd.NA) for column in SUMMARY_COLUMNS}
        group = _matching_rows(daily_equity, row)
        rolling = _matching_rows(rolling_windows, row)
        exposure = _matching_rows(symbol_exposure, row)
        out.update(_drawdown_summary(group))
        out.update(_rolling_summary(rolling))
        out.update(_symbol_summary(exposure, row))
        if "final_decision" not in out or pd.isna(out["final_decision"]):
            out["final_decision"] = classify_rich_replay_decision(row)
        out["production_ready"] = False
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    contribution_cols = [
        *KEY_COLUMNS,
        "estimated_spy_core_contribution",
        "estimated_candidate_sleeve_contribution",
        "sleeve_contribution_share",
    ]
    if not sleeve_contribution.empty:
        summary = summary.merge(
            sleeve_contribution.loc[:, contribution_cols],
            on=KEY_COLUMNS,
            how="left",
        )
    summary["production_ready"] = False
    contribution_summary_cols = [column for column in contribution_cols[4:] if column in summary]
    return _rank_summary(summary.loc[:, [*SUMMARY_COLUMNS, *contribution_summary_cols]])


def render_long_history_core_sleeve_rich_replay_markdown(
    *,
    inputs: LongHistoryCoreSleeveRichReplayInputs,
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    subperiods: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    sleeve_contribution: pd.DataFrame,
    selected_symbols: pd.DataFrame,
    symbol_exposure: pd.DataFrame,
) -> str:
    """Render the rich replay Markdown report."""

    lines = [
        "# Long-History QQQ Core-Sleeve Rich Replay Report",
        "",
        "This is a research-only deep-dive over completed local artifacts. It does "
        "not run a broad sweep, train models, export production signals, create a "
        "model card, or mark any candidate production-ready.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(summary),
        "",
        "## 2. Selected core-sleeve candidates",
        "",
        _selected_candidates_text(summary),
        "",
        "## 3. Full-period performance",
        "",
        _markdown_table(_format_for_markdown(_full_period_table(summary))),
        "",
        "## 4. Equity and drawdown replay",
        "",
        _equity_drawdown_text(summary),
        "",
        "## 5. SPY core versus candidate sleeve contribution",
        "",
        _markdown_table(_format_for_markdown(sleeve_contribution)),
        "",
        "## 6. Underlying candidate ETF selection path",
        "",
        _selected_symbol_path_text(selected_symbols),
        "",
        "## 7. Symbol exposure and concentration",
        "",
        _symbol_exposure_text(symbol_exposure),
        "",
        "## 8. Subperiod robustness",
        "",
        _subperiod_text(subperiods),
        "",
        "## 9. Rolling-window robustness",
        "",
        _rolling_window_text(summary, rolling_windows),
        "",
        "## 10. Candidate_0001 core-sleeve interpretation",
        "",
        _candidate_interpretation(summary, "candidate_0001"),
        "",
        "## 11. Candidate_0003 core-sleeve interpretation",
        "",
        _candidate_interpretation(summary, "candidate_0003"),
        "",
        "## 12. Final recommendation",
        "",
        _final_recommendation(summary, decision_matrix),
        "",
        "## 13. Caveats",
        "",
        _caveats(inputs),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_long_history_core_sleeve_rich_replay_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveRichReplayArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write rich replay artifacts to the report output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.daily_equity.to_csv(paths[DAILY_EQUITY_NAME], index=False)
    artifacts.daily_drawdown.to_csv(paths[DAILY_DRAWDOWN_NAME], index=False)
    artifacts.subperiods.to_csv(paths[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(paths[ROLLING_WINDOWS_NAME], index=False)
    artifacts.sleeve_contribution.to_csv(paths[SLEEVE_CONTRIBUTION_NAME], index=False)
    artifacts.selected_symbols.to_csv(paths[SELECTED_SYMBOLS_NAME], index=False)
    artifacts.symbol_exposure.to_csv(paths[SYMBOL_EXPOSURE_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    return paths


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all rich replay output paths."""

    return {name: output_dir / name for name in OUTPUT_NAMES}


def _practical_role_from_recommendation(row: pd.Series) -> str:
    action = str(row.get("recommendation_action", ""))
    candidate_id = str(row.get("candidate_id", ""))
    if action == "rich_replay_next" and candidate_id == "candidate_0001":
        return "primary return-oriented tracking candidate"
    if action == "rich_replay_next" and candidate_id == "candidate_0003":
        return "primary balanced/risk-controlled tracking candidate"
    if action == "track_next":
        return "secondary tracking reference"
    if action == "reference_only" and candidate_id == "candidate_0001":
        return "raw return reference only"
    if action == "reference_only" and candidate_id == "candidate_0003":
        return "signal-review reference only"
    return "reference only"


def _filter_by_selected_keys(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or selected.empty:
        return frame.iloc[0:0].copy()
    _validate_columns(frame, KEY_COLUMNS, "artifact")
    keys = _normalized_keys(selected)
    rows = frame.copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["spy_core_weight"] = pd.to_numeric(rows["spy_core_weight"], errors="coerce").round(10)
    rows["candidate_sleeve_weight"] = pd.to_numeric(
        rows["candidate_sleeve_weight"],
        errors="coerce",
    ).round(10)
    rows["portfolio_accounting_mode"] = rows["portfolio_accounting_mode"].astype(str)
    row_keys = set(map(tuple, keys.loc[:, KEY_COLUMNS].to_numpy()))
    mask = rows.loc[:, KEY_COLUMNS].apply(tuple, axis=1).isin(row_keys)
    return rows.loc[mask].copy()


def _normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame.loc[:, KEY_COLUMNS].copy()
    keys["candidate_id"] = keys["candidate_id"].astype(str)
    keys["spy_core_weight"] = pd.to_numeric(keys["spy_core_weight"], errors="coerce").round(10)
    keys["candidate_sleeve_weight"] = pd.to_numeric(
        keys["candidate_sleeve_weight"],
        errors="coerce",
    ).round(10)
    keys["portfolio_accounting_mode"] = keys["portfolio_accounting_mode"].astype(str)
    return keys


def _selected_labels(selected: pd.DataFrame) -> pd.DataFrame:
    return selected.loc[
        :,
        [
            *KEY_COLUMNS,
            "recommendation_action",
            "practical_role",
        ],
    ].copy()


def _matching_rows(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        frame["candidate_id"].astype(str).eq(str(row.get("candidate_id")))
        & pd.to_numeric(frame["spy_core_weight"], errors="coerce")
        .round(10)
        .eq(round(float(row.get("spy_core_weight", np.nan)), 10))
        & pd.to_numeric(frame["candidate_sleeve_weight"], errors="coerce")
        .round(10)
        .eq(round(float(row.get("candidate_sleeve_weight", np.nan)), 10))
        & frame["portfolio_accounting_mode"]
        .astype(str)
        .eq(str(row.get("portfolio_accounting_mode")))
    )
    return frame.loc[mask].copy()


def _period_return(equity: pd.Series) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def _drawdown_summary(group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return {}
    sorted_group = group.sort_values("date", kind="mergesort").copy()
    return {
        **_drawdown_range(
            sorted_group,
            equity_column="equity",
            drawdown_column="drawdown",
            prefix="portfolio",
        ),
        **_drawdown_range(
            sorted_group,
            equity_column="qqq_equity",
            drawdown_column="qqq_drawdown",
            prefix="qqq",
        ),
        **_drawdown_range(
            sorted_group,
            equity_column="spy_equity",
            drawdown_column="spy_drawdown",
            prefix="spy",
        ),
        **_drawdown_range(
            sorted_group,
            equity_column="candidate_equity",
            drawdown_column="candidate_drawdown",
            prefix="standalone_candidate",
        ),
    }


def _drawdown_range(
    frame: pd.DataFrame,
    *,
    equity_column: str,
    drawdown_column: str,
    prefix: str,
) -> dict[str, object]:
    if equity_column not in frame.columns or drawdown_column not in frame.columns:
        return {
            f"{prefix}_max_drawdown_start": pd.NA,
            f"{prefix}_max_drawdown_end": pd.NA,
        }
    drawdown = pd.to_numeric(frame[drawdown_column], errors="coerce")
    if drawdown.dropna().empty:
        return {
            f"{prefix}_max_drawdown_start": pd.NA,
            f"{prefix}_max_drawdown_end": pd.NA,
        }
    trough_idx = drawdown.idxmin()
    history = frame.loc[:trough_idx].copy()
    equity = pd.to_numeric(history[equity_column], errors="coerce")
    if equity.dropna().empty:
        start = pd.NA
    else:
        start = history.loc[equity.idxmax(), "date"]
    return {
        f"{prefix}_max_drawdown_start": _date_string(start),
        f"{prefix}_max_drawdown_end": _date_string(frame.loc[trough_idx, "date"]),
    }


def _rolling_summary(rolling: pd.DataFrame) -> dict[str, object]:
    if rolling.empty:
        return {}
    strategy_vs_qqq = pd.to_numeric(rolling.get("strategy_vs_qqq"), errors="coerce")
    drawdown_improvement = pd.to_numeric(
        rolling.get("drawdown_improvement_vs_qqq"),
        errors="coerce",
    )
    best_idx = strategy_vs_qqq.idxmax() if strategy_vs_qqq.notna().any() else None
    worst_idx = strategy_vs_qqq.idxmin() if strategy_vs_qqq.notna().any() else None
    return {
        "best_strategy_vs_qqq_window_label": rolling.loc[best_idx, "window_label"]
        if best_idx is not None
        else pd.NA,
        "best_strategy_vs_qqq": float(strategy_vs_qqq.loc[best_idx])
        if best_idx is not None
        else np.nan,
        "worst_strategy_vs_qqq_window_label": rolling.loc[worst_idx, "window_label"]
        if worst_idx is not None
        else pd.NA,
        "rolling_drawdown_improvement_vs_qqq_rate": float(drawdown_improvement.gt(0.0).mean())
        if drawdown_improvement.notna().any()
        else np.nan,
        "best_rolling_drawdown_improvement_vs_qqq": float(drawdown_improvement.max()),
        "worst_rolling_drawdown_improvement_vs_qqq": float(drawdown_improvement.min()),
    }


def _symbol_summary(exposure: pd.DataFrame, row: pd.Series) -> dict[str, object]:
    if exposure.empty:
        return {
            "top_symbol": row.get("source_top_symbol", pd.NA),
            "top_symbol_exposure_pct": row.get("source_top_symbol_exposure_pct", pd.NA),
            "top3_symbol_exposure_pct": row.get("source_top3_symbol_exposure_pct", pd.NA),
            "top_contributors": pd.NA,
        }
    ranked = exposure.sort_values("rank_by_exposure", kind="mergesort")
    top = ranked.iloc[0]
    contributors = ranked.sort_values(
        "total_return_contribution",
        ascending=False,
        kind="mergesort",
    ).head(3)
    if "is_top3_symbol" in ranked.columns:
        top3_mask = ranked["is_top3_symbol"].astype(str).str.lower().eq("true")
    else:
        top3_mask = pd.to_numeric(ranked["rank_by_exposure"], errors="coerce").le(3)
    return {
        "top_symbol": top.get("symbol"),
        "top_symbol_exposure_pct": top.get("exposure_pct"),
        "top3_symbol_exposure_pct": pd.to_numeric(
            ranked.loc[top3_mask, "exposure_pct"],
            errors="coerce",
        ).sum(),
        "top_contributors": "; ".join(
            f"{item.symbol}:{float(item.total_return_contribution):.4f}"
            for item in contributors.itertuples(index=False)
            if pd.notna(item.total_return_contribution)
        ),
    }


def _stop_spending_time_rows(
    *,
    decision_matrix: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if decision_matrix.empty:
        return pd.DataFrame(
            [
                {
                    "candidate_id": "broad_sweep",
                    "recommendation_action": "stop_spending_time",
                    "practical_role": "do not run another broad sweep",
                    "final_decision": "reject_deprioritize",
                    "production_ready": False,
                }
            ]
        )
    selected_keys = set(map(tuple, _normalized_keys(selected).loc[:, KEY_COLUMNS].to_numpy()))
    matrix = decision_matrix.copy()
    matrix["candidate_id"] = matrix["candidate_id"].astype(str)
    matrix["spy_core_weight"] = pd.to_numeric(
        matrix["spy_core_weight"],
        errors="coerce",
    ).round(10)
    matrix["candidate_sleeve_weight"] = pd.to_numeric(
        matrix["candidate_sleeve_weight"],
        errors="coerce",
    ).round(10)
    matrix["portfolio_accounting_mode"] = matrix["portfolio_accounting_mode"].astype(str)
    matrix["_key"] = matrix.loc[:, KEY_COLUMNS].apply(tuple, axis=1)
    rejected = matrix.loc[
        ~matrix["_key"].isin(selected_keys)
        & matrix.get("decision_category", "").astype(str).eq("reject_deprioritize")
    ].head(8)
    if rejected.empty:
        rejected = pd.DataFrame(
            [
                {
                    "candidate_id": "broad_sweep",
                    "recommendation_action": "stop_spending_time",
                    "practical_role": "do not run another broad sweep",
                    "final_decision": "reject_deprioritize",
                    "production_ready": False,
                }
            ]
        )
    else:
        rejected = rejected.drop(columns=["_key"], errors="ignore").copy()
        rejected["recommendation_action"] = "stop_spending_time"
        rejected["practical_role"] = "deprioritized non-selected row"
        rejected["final_decision"] = "reject_deprioritize"
        rejected["production_ready"] = False
    return rejected


def _rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rank = {
        "primary_tracking_candidate": 0,
        "secondary_tracking_candidate": 1,
        "reference_only": 2,
        "reject_deprioritize": 3,
    }
    return (
        summary.assign(_rank=summary["final_decision"].map(rank).fillna(9))
        .sort_values(
            ["_rank", "candidate_id", "spy_core_weight"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def _executive_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No selected rich replay rows were available."
    primary = summary.loc[summary["final_decision"].eq("primary_tracking_candidate")]
    secondary = summary.loc[summary["final_decision"].eq("secondary_tracking_candidate")]
    prod_ready = int(_as_bool_series(summary["production_ready"]).sum())
    primary_text = _portfolio_list(primary)
    secondary_text = _portfolio_list(secondary)
    return (
        f"Rich replay covers `{len(primary)}` primary rows and `{len(secondary)}` "
        "secondary tracking rows, with standalone candidate sleeves, QQQ buy-and-hold, "
        "and SPY buy-and-hold retained as references. Primary tracking candidates: "
        f"{primary_text or '`none`'}. Secondary tracking references: "
        f"{secondary_text or '`none`'}. Production-ready rows: `{prod_ready}`."
    )


def _selected_candidates_text(summary: pd.DataFrame) -> str:
    columns = [
        "candidate_id",
        "recommendation_action",
        "practical_role",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_accounting_mode",
        "strategy_vs_qqq",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown_vs_qqq",
        "final_decision",
    ]
    return _markdown_table(_format_for_markdown(summary.loc[:, columns]))


def _full_period_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "standalone_candidate_total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "qqq_buy_hold_max_drawdown",
        "spy_buy_hold_max_drawdown",
        "standalone_candidate_max_drawdown",
        "return_to_drawdown",
        "final_decision",
    ]
    return summary.loc[:, columns]


def _equity_drawdown_text(summary: pd.DataFrame) -> str:
    columns = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_max_drawdown_start",
        "portfolio_max_drawdown_end",
        "max_drawdown",
        "qqq_buy_hold_max_drawdown",
        "spy_buy_hold_max_drawdown",
        "standalone_candidate_max_drawdown",
        "drawdown_improvement_vs_qqq",
        "drawdown_improvement_vs_candidate",
    ]
    return _markdown_table(_format_for_markdown(summary.loc[:, columns]))


def _selected_symbol_path_text(selected_symbols: pd.DataFrame) -> str:
    if selected_symbols.empty:
        return "_No selected-symbol timeline rows._"
    columns = [
        "candidate_id",
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "date",
        "symbol",
        "target_weight",
        "predicted_score",
        "rank",
        "rebalance_frequency",
    ]
    sample = selected_symbols.sort_values(
        ["candidate_id", "spy_core_weight", "date"],
        kind="mergesort",
    ).head(40)
    return _markdown_table(_format_for_markdown(sample.loc[:, columns]))


def _symbol_exposure_text(symbol_exposure: pd.DataFrame) -> str:
    if symbol_exposure.empty:
        return "_No symbol exposure rows._"
    columns = [
        "candidate_id",
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "symbol",
        "rank_by_exposure",
        "exposure_pct",
        "sleeve_weighted_exposure_pct",
        "total_return_contribution",
        "is_top_symbol",
        "is_top3_symbol",
    ]
    display = symbol_exposure.sort_values(
        ["candidate_id", "spy_core_weight", "rank_by_exposure"],
        kind="mergesort",
    ).head(30)
    return _markdown_table(_format_for_markdown(display.loc[:, columns]))


def _subperiod_text(subperiods: pd.DataFrame) -> str:
    if subperiods.empty:
        return "_No subperiod rows._"
    columns = [
        "candidate_id",
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "period_name",
        "total_return",
        "qqq_total_return",
        "spy_total_return",
        "standalone_candidate_total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "drawdown_improvement_vs_candidate",
    ]
    return _markdown_table(_format_for_markdown(subperiods.loc[:, columns]))


def _rolling_window_text(summary: pd.DataFrame, rolling_windows: pd.DataFrame) -> str:
    columns = [
        "candidate_id",
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "best_strategy_vs_qqq_window_label",
        "best_strategy_vs_qqq",
        "worst_strategy_vs_qqq_window_label",
        "rolling_drawdown_improvement_vs_qqq_rate",
        "worst_rolling_drawdown_improvement_vs_qqq",
    ]
    text = _markdown_table(_format_for_markdown(summary.loc[:, columns]))
    if rolling_windows.empty:
        return text
    worst = rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(8)
    best = rolling_windows.sort_values(
        "strategy_vs_qqq",
        ascending=False,
        kind="mergesort",
    ).head(8)
    detail_columns = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "window_label",
        "strategy_vs_qqq",
        "strategy_max_drawdown",
        "qqq_max_drawdown",
        "drawdown_improvement_vs_qqq",
        "underperformed_qqq",
    ]
    return (
        text
        + "\n\nBest rolling windows:\n\n"
        + _markdown_table(_format_for_markdown(best.loc[:, detail_columns]))
        + "\n\nWorst rolling windows:\n\n"
        + _markdown_table(_format_for_markdown(worst.loc[:, detail_columns]))
    )


def _candidate_interpretation(summary: pd.DataFrame, candidate_id: str) -> str:
    family = summary.loc[summary["candidate_id"].astype(str).eq(candidate_id)]
    if family.empty:
        return "_No rows._"
    if candidate_id == "candidate_0001":
        lead = (
            "Candidate_0001 is the main return-oriented core-sleeve direction. "
            "The 35%-40% SPY core zone is useful because it keeps the raw sleeve's "
            "return profile while reducing QQQ-relative drawdown risk."
        )
    else:
        lead = (
            "Candidate_0003 is the more balanced/risk-controlled direction. The "
            "25%-40% SPY core zone is a tracking range, while standalone remains "
            "a signal-review reference rather than a promotion candidate."
        )
    columns = [
        "recommendation_action",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "strategy_vs_qqq",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown_vs_qqq",
        "top_symbol",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "final_decision",
    ]
    return lead + "\n\n" + _markdown_table(_format_for_markdown(family.loc[:, columns]))


def _final_recommendation(summary: pd.DataFrame, decision_matrix: pd.DataFrame) -> str:
    primary = summary.loc[summary["final_decision"].eq("primary_tracking_candidate")]
    secondary = summary.loc[summary["final_decision"].eq("secondary_tracking_candidate")]
    raw = summary.loc[
        summary["candidate_id"].eq("candidate_0001")
        & summary["final_decision"].eq("reference_only")
    ]
    signal = summary.loc[
        summary["candidate_id"].eq("candidate_0003")
        & summary["final_decision"].eq("reference_only")
    ]
    stop_count = int(decision_matrix["final_decision"].eq("reject_deprioritize").sum())
    return (
        f"- Primary tracking candidate: {_portfolio_list(primary) or '`none`'}\n"
        f"- Secondary tracking candidate: {_portfolio_list(secondary) or '`none`'}\n"
        f"- Raw return reference: {_portfolio_list(raw) or '`standalone candidate_0001`'}\n"
        f"- Signal-review reference: {_portfolio_list(signal) or '`standalone candidate_0003`'}\n"
        f"- Candidates to stop spending time on: `{stop_count}` deprioritized/non-selected rows "
        "plus broad-sweep expansion.\n"
        "- Another sweep needed: `no`.\n"
        "- Production promotion allowed: `no`.\n"
        "- production_ready: `false` for every output row."
    )


def _caveats(inputs: LongHistoryCoreSleeveRichReplayInputs) -> str:
    return (
        "- Research-only; this report does not mark anything production-ready.\n"
        "- Tiingo-only source artifacts.\n"
        "- No production signal export.\n"
        "- No model card.\n"
        "- QQQ buy-and-hold remains the primary benchmark; SPY buy-and-hold is a "
        "secondary reference.\n"
        "- Core-sleeve accounting depends on the selected rebalance mode, especially "
        "`candidate_rebalance_aligned_weights`.\n"
        "- Input symbol periods were read for context but no symbol-level return path "
        "was recomputed.\n"
        f"- Candidate symbol-period rows available: `{len(inputs.candidate_symbol_periods)}`."
    )


def _portfolio_list(frame: pd.DataFrame) -> str:
    labels: list[str] = []
    for row in frame.itertuples(index=False):
        spy_weight = getattr(row, "spy_core_weight", np.nan)
        sleeve_weight = getattr(row, "candidate_sleeve_weight", np.nan)
        mode = getattr(row, "portfolio_accounting_mode", "")
        if pd.isna(spy_weight):
            labels.append(f"`{row.candidate_id}`")
        else:
            labels.append(
                f"`{row.candidate_id}` {float(spy_weight):.0%} SPY / "
                f"{float(sleeve_weight):.0%} sleeve (`{mode}`)"
            )
    return ", ".join(labels)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return out


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    if columns is None:
        columns = table.columns.tolist()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _with_missing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _date_string(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(pd.Timestamp(value).date())


def _as_bool_scalar(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return str(value).lower() in {"true", "1", "yes"}


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)
