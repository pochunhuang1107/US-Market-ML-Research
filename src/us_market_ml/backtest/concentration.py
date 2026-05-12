"""Concentration diagnostics for selected holdings and position contributions."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd

SELECTION_CONCENTRATION_COLUMNS = [
    "total_selected_days",
    "unique_selected_symbols",
    "top_symbol_by_selection",
    "top_symbol_selected_days",
    "top_symbol_selected_share",
    "top_3_selected_share",
    "selection_hhi",
    "effective_selected_symbols",
    "max_avg_target_weight",
]
CONTRIBUTION_CONCENTRATION_COLUMNS = [
    "total_positive_contribution",
    "total_absolute_contribution",
    "top_symbol_by_contribution",
    "top_symbol_contribution",
    "top_symbol_positive_contribution_share",
    "top_3_positive_contribution_share",
    "contribution_hhi",
    "effective_contributing_symbols",
]
CONCENTRATION_REPORT_COLUMNS = [
    *SELECTION_CONCENTRATION_COLUMNS,
    *CONTRIBUTION_CONCENTRATION_COLUMNS,
    "concentration_warning",
    "contribution_warning",
]


def calculate_hhi(weights: pd.Series) -> float:
    """Return Herfindahl-Hirschman concentration over absolute normalized weights."""

    shares = pd.to_numeric(weights, errors="coerce").dropna().abs()
    total = float(shares.sum())
    if shares.empty or total == 0.0:
        return math.nan

    normalized = shares / total
    return float((normalized**2).sum())


def effective_number_from_hhi(hhi: float) -> float:
    """Return the effective number of equally sized contributors implied by HHI."""

    if pd.isna(hhi) or hhi <= 0.0:
        return math.nan
    return float(1.0 / hhi)


def summarize_selection_concentration(
    holdings_summary: pd.DataFrame,
) -> dict[str, float | str]:
    """Summarize how concentrated selected holdings are by symbol."""

    _validate_required_columns(
        holdings_summary,
        ["symbol", "selected_days", "avg_target_weight"],
        "holdings_summary",
    )
    if holdings_summary.empty:
        return {
            "total_selected_days": 0.0,
            "unique_selected_symbols": 0.0,
            "top_symbol_by_selection": "",
            "top_symbol_selected_days": 0.0,
            "top_symbol_selected_share": math.nan,
            "top_3_selected_share": math.nan,
            "selection_hhi": math.nan,
            "effective_selected_symbols": math.nan,
            "max_avg_target_weight": math.nan,
        }

    summary = holdings_summary.copy()
    summary["selected_days"] = pd.to_numeric(summary["selected_days"], errors="coerce").fillna(0.0)
    summary["avg_target_weight"] = pd.to_numeric(
        summary["avg_target_weight"], errors="coerce"
    )

    total_selected_days = float(summary["selected_days"].sum())
    unique_selected_symbols = float((summary["selected_days"] > 0.0).sum())
    max_avg_target_weight = float(summary["avg_target_weight"].abs().max())
    if pd.isna(max_avg_target_weight):
        max_avg_target_weight = math.nan

    if total_selected_days == 0.0:
        return {
            "total_selected_days": total_selected_days,
            "unique_selected_symbols": unique_selected_symbols,
            "top_symbol_by_selection": "",
            "top_symbol_selected_days": 0.0,
            "top_symbol_selected_share": math.nan,
            "top_3_selected_share": math.nan,
            "selection_hhi": math.nan,
            "effective_selected_symbols": math.nan,
            "max_avg_target_weight": max_avg_target_weight,
        }

    ranked = summary.sort_values(["selected_days", "symbol"], ascending=[False, True])
    top = ranked.iloc[0]
    selection_shares = ranked["selected_days"] / total_selected_days
    selection_hhi = calculate_hhi(ranked["selected_days"])

    return {
        "total_selected_days": total_selected_days,
        "unique_selected_symbols": unique_selected_symbols,
        "top_symbol_by_selection": str(top["symbol"]),
        "top_symbol_selected_days": float(top["selected_days"]),
        "top_symbol_selected_share": float(selection_shares.iloc[0]),
        "top_3_selected_share": float(selection_shares.head(3).sum()),
        "selection_hhi": selection_hhi,
        "effective_selected_symbols": effective_number_from_hhi(selection_hhi),
        "max_avg_target_weight": max_avg_target_weight,
    }


def summarize_contribution_concentration(
    contribution_summary: pd.DataFrame,
) -> dict[str, float | str]:
    """Summarize how concentrated position return contributions are by symbol."""

    _validate_required_columns(
        contribution_summary,
        ["symbol", "total_position_return"],
        "contribution_summary",
    )
    if contribution_summary.empty:
        return {
            "total_positive_contribution": 0.0,
            "total_absolute_contribution": 0.0,
            "top_symbol_by_contribution": "",
            "top_symbol_contribution": 0.0,
            "top_symbol_positive_contribution_share": math.nan,
            "top_3_positive_contribution_share": math.nan,
            "contribution_hhi": math.nan,
            "effective_contributing_symbols": math.nan,
        }

    summary = contribution_summary.copy()
    summary["total_position_return"] = pd.to_numeric(
        summary["total_position_return"], errors="coerce"
    ).fillna(0.0)

    positive = summary.loc[summary["total_position_return"] > 0.0].copy()
    total_positive_contribution = float(positive["total_position_return"].sum())
    total_absolute_contribution = float(summary["total_position_return"].abs().sum())
    contribution_hhi = calculate_hhi(summary["total_position_return"])

    if total_positive_contribution == 0.0:
        ranked_abs = summary.assign(
            absolute_contribution=summary["total_position_return"].abs()
        ).sort_values(["absolute_contribution", "symbol"], ascending=[False, True])
        top = ranked_abs.iloc[0]
        return {
            "total_positive_contribution": total_positive_contribution,
            "total_absolute_contribution": total_absolute_contribution,
            "top_symbol_by_contribution": str(top["symbol"]),
            "top_symbol_contribution": float(top["total_position_return"]),
            "top_symbol_positive_contribution_share": math.nan,
            "top_3_positive_contribution_share": math.nan,
            "contribution_hhi": contribution_hhi,
            "effective_contributing_symbols": effective_number_from_hhi(contribution_hhi),
        }

    ranked_positive = positive.sort_values(
        ["total_position_return", "symbol"], ascending=[False, True]
    )
    top = ranked_positive.iloc[0]
    positive_shares = ranked_positive["total_position_return"] / total_positive_contribution

    return {
        "total_positive_contribution": total_positive_contribution,
        "total_absolute_contribution": total_absolute_contribution,
        "top_symbol_by_contribution": str(top["symbol"]),
        "top_symbol_contribution": float(top["total_position_return"]),
        "top_symbol_positive_contribution_share": float(positive_shares.iloc[0]),
        "top_3_positive_contribution_share": float(positive_shares.head(3).sum()),
        "contribution_hhi": contribution_hhi,
        "effective_contributing_symbols": effective_number_from_hhi(contribution_hhi),
    }


def build_concentration_report(
    holdings_summary: pd.DataFrame,
    contribution_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return a one-row concentration report from holdings and contribution summaries."""

    selection_metrics = summarize_selection_concentration(holdings_summary)
    contribution_metrics = summarize_contribution_concentration(contribution_summary)
    row: dict[str, object] = {**selection_metrics, **contribution_metrics}
    row["concentration_warning"] = bool(
        _metric_float(row["top_symbol_selected_share"]) >= 0.30
        or _metric_float(row["effective_selected_symbols"]) < 5.0
    )
    row["contribution_warning"] = bool(
        _metric_float(row["top_symbol_positive_contribution_share"]) >= 0.35
        or _metric_float(row["effective_contributing_symbols"]) < 4.0
    )

    return pd.DataFrame([row], columns=CONCENTRATION_REPORT_COLUMNS)


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required concentration columns: {missing}")


def _metric_float(value: object) -> float:
    if isinstance(value, Real):
        return float(value)
    return math.nan


__all__ = [
    "CONCENTRATION_REPORT_COLUMNS",
    "calculate_hhi",
    "effective_number_from_hhi",
    "summarize_selection_concentration",
    "summarize_contribution_concentration",
    "build_concentration_report",
]
