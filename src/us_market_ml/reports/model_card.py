"""Markdown model cards for research strategy candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import pandas as pd

VERDICT_PROMISING = "promising_for_further_research"
VERDICT_NEEDS_VALIDATION = "needs_more_validation"
VERDICT_NOT_PROMISING = "not_currently_promising"


def build_candidate_model_card(
    candidate_name: str,
    candidate_family: str,
    model_type: str,
    config: dict[str, object],
    metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    concentration_report: pd.DataFrame,
    holding_period_summary: pd.DataFrame,
    holdings_summary: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame | None = None,
) -> str:
    """Return a deterministic Markdown model card for one strategy candidate."""

    candidate_metrics = _series_metrics(metrics, candidate_name)
    spy_metrics = _series_metrics(metrics, "SPY")
    concentration = _first_row(concentration_report)
    holding_period = _first_row(holding_period_summary)
    verdict = _research_verdict(candidate_metrics, spy_metrics, concentration)

    sections = [
        _title_section(candidate_name),
        _candidate_overview_section(
            candidate_name,
            candidate_family,
            model_type,
            verdict,
        ),
        _strategy_configuration_section(config),
        _out_of_sample_performance_section(candidate_metrics),
        _benchmark_comparison_section(candidate_metrics, spy_metrics),
        _fold_by_fold_performance_section(fold_summary),
        _cost_sensitivity_section(candidate_name, candidate_metrics, cost_sensitivity),
        _holdings_and_contributors_section(holdings_summary, contribution_summary),
        _concentration_risk_section(concentration),
        _holding_period_section(candidate_metrics, holding_period),
        _strengths_section(candidate_metrics, spy_metrics, concentration, holding_period),
        _weaknesses_section(candidate_metrics, spy_metrics, concentration, holding_period),
        _research_verdict_section(verdict),
        _disclaimer_section(),
    ]
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _title_section(candidate_name: str) -> str:
    return f"# Candidate Model Card: {candidate_name}"


def _candidate_overview_section(
    candidate_name: str,
    candidate_family: str,
    model_type: str,
    verdict: str,
) -> str:
    rows = [
        ("Candidate", candidate_name),
        ("Family", candidate_family),
        ("Model type", model_type),
        ("Research triage verdict", verdict),
    ]
    return "## Candidate Overview\n\n" + _two_column_table("Field", "Value", rows)


def _strategy_configuration_section(config: dict[str, object]) -> str:
    if not config:
        return "## Strategy Configuration\n\nNo strategy configuration metadata was provided."

    rows = [(key, _format_object(config[key])) for key in sorted(config)]
    return "## Strategy Configuration\n\n" + _two_column_table("Parameter", "Value", rows)


def _out_of_sample_performance_section(candidate_metrics: pd.Series) -> str:
    rows = [
        ("Total return", _format_percent(candidate_metrics.get("total_return"))),
        ("Sharpe-like", _format_number(candidate_metrics.get("sharpe_like"))),
        ("Max drawdown", _format_percent(candidate_metrics.get("max_drawdown"))),
        ("Final equity", _format_number(candidate_metrics.get("final_equity"))),
        (
            "Total transaction cost",
            _format_number(candidate_metrics.get("total_transaction_cost")),
        ),
        (
            "Mean daily turnover",
            _format_percent(candidate_metrics.get("mean_daily_turnover")),
        ),
    ]
    return "## Out-of-Sample Performance\n\n" + _two_column_table("Metric", "Value", rows)


def _benchmark_comparison_section(
    candidate_metrics: pd.Series,
    spy_metrics: pd.Series,
) -> str:
    rows = [
        (
            "Total return",
            _format_percent(candidate_metrics.get("total_return")),
            _format_percent(spy_metrics.get("total_return")),
            _format_delta_percent(
                candidate_metrics.get("total_return"),
                spy_metrics.get("total_return"),
            ),
        ),
        (
            "Sharpe-like",
            _format_number(candidate_metrics.get("sharpe_like")),
            _format_number(spy_metrics.get("sharpe_like")),
            _format_delta(candidate_metrics.get("sharpe_like"), spy_metrics.get("sharpe_like")),
        ),
        (
            "Max drawdown",
            _format_percent(candidate_metrics.get("max_drawdown")),
            _format_percent(spy_metrics.get("max_drawdown")),
            _format_delta_percent(
                candidate_metrics.get("max_drawdown"),
                spy_metrics.get("max_drawdown"),
            ),
        ),
        (
            "Final equity",
            _format_number(candidate_metrics.get("final_equity")),
            _format_number(spy_metrics.get("final_equity")),
            _format_delta(candidate_metrics.get("final_equity"), spy_metrics.get("final_equity")),
        ),
    ]
    return "## Benchmark Comparison\n\n" + _markdown_table(
        ["Metric", "Candidate", "SPY", "Candidate minus SPY"],
        rows,
    )


def _fold_by_fold_performance_section(fold_summary: pd.DataFrame) -> str:
    if fold_summary.empty:
        return "## Fold-by-Fold Performance\n\nNo fold summary rows were provided."

    display_columns = [
        column
        for column in [
            "fold_name",
            "test_start",
            "test_end",
            "strategy_total_return",
            "strategy_max_drawdown",
            "strategy_final_equity",
            "strategy_total_transaction_cost",
            "strategy_mean_daily_turnover",
        ]
        if column in fold_summary.columns
    ]
    rows = []
    for _, row in fold_summary.loc[:, display_columns].iterrows():
        rows.append(tuple(_format_table_cell(column, row[column]) for column in display_columns))

    return "## Fold-by-Fold Performance\n\n" + _markdown_table(display_columns, rows)


def _cost_sensitivity_section(
    candidate_name: str,
    candidate_metrics: pd.Series,
    cost_sensitivity: pd.DataFrame | None,
) -> str:
    lines = ["## Cost Sensitivity"]
    metric_rows = [
        (
            "Observed total transaction cost",
            _format_number(candidate_metrics.get("total_transaction_cost")),
        ),
        (
            "Observed mean daily turnover",
            _format_percent(candidate_metrics.get("mean_daily_turnover")),
        ),
    ]
    lines.append(_two_column_table("Metric", "Value", metric_rows))

    if cost_sensitivity is None or cost_sensitivity.empty:
        lines.append("No separate transaction-cost sensitivity table was provided.")
        return "\n\n".join(lines)

    sensitivity = cost_sensitivity.copy()
    if "candidate_name" in sensitivity.columns:
        sensitivity = sensitivity.loc[sensitivity["candidate_name"] == candidate_name]
    if sensitivity.empty:
        lines.append("No cost sensitivity rows matched this candidate.")
        return "\n\n".join(lines)

    if "cost_bps" in sensitivity.columns:
        sensitivity = sensitivity.assign(
            cost_bps=pd.to_numeric(sensitivity["cost_bps"], errors="coerce")
        )
    if "total_return" in sensitivity.columns:
        sensitivity = sensitivity.assign(
            total_return=pd.to_numeric(sensitivity["total_return"], errors="coerce")
        )

    display_columns = [
        column
        for column in [
            "cost_bps",
            "total_return",
            "sharpe_like",
            "max_drawdown",
            "final_equity",
            "total_transaction_cost",
            "mean_daily_turnover",
        ]
        if column in sensitivity.columns
    ]
    sensitivity = sensitivity.sort_values(
        [column for column in ["cost_bps", "candidate_name"] if column in sensitivity.columns],
        kind="mergesort",
    )
    rows = [
        tuple(_format_cost_sensitivity_cell(column, row[column]) for column in display_columns)
        for _, row in sensitivity.loc[:, display_columns].iterrows()
    ]
    lines.append(_markdown_table(display_columns, rows))
    lines.append(_cost_robustness_summary(sensitivity))
    return "\n\n".join(lines)


def _cost_robustness_summary(sensitivity: pd.DataFrame) -> str:
    if not {"cost_bps", "total_return"}.issubset(sensitivity.columns):
        return (
            "Cost robustness label: **fragile**\n\n"
            "The cost sensitivity table is missing cost_bps or total_return, so no "
            "break-even grid summary could be estimated."
        )

    tested = sensitivity.loc[:, ["cost_bps", "total_return"]].dropna()
    if tested.empty:
        return (
            "Cost robustness label: **fragile**\n\n"
            "No valid tested cost levels with total returns were available for a "
            "break-even grid summary."
        )

    tested = tested.sort_values("cost_bps", kind="mergesort")
    positive = tested.loc[tested["total_return"] > 0.0]
    non_positive = tested.loc[tested["total_return"] <= 0.0]

    highest_positive = (
        _format_bps(positive["cost_bps"].max()) if not positive.empty else "not observed"
    )
    first_negative = (
        _format_bps(non_positive["cost_bps"].min())
        if not non_positive.empty
        else "not observed"
    )

    summary_rows = [
        ("highest_positive_cost_bps", highest_positive),
        ("first_negative_cost_bps", first_negative),
    ]
    for target_bps in [10.0, 20.0, 50.0]:
        target_return = _return_at_exact_bps(tested, target_bps)
        if target_return is not None:
            summary_rows.append(
                (f"return_at_{_format_bps(target_bps)}bps", _format_percent(target_return))
            )

    label, label_basis, used_nearest = _cost_robustness_label(tested)
    break_even_note = _break_even_note(highest_positive, first_negative)
    grid_note = (
        " The label is based on the nearest available tested grid levels because "
        "one or more 5/10/20 bps thresholds were not tested exactly."
        if used_nearest
        else ""
    )
    notes = [
        f"Cost robustness label: **{label}** ({label_basis}).{grid_note}",
        break_even_note,
        _cost_plain_english_note(label),
    ]
    return (
        "Cost robustness summary:\n\n"
        + _two_column_table("Metric", "Value", summary_rows)
        + "\n\nRobustness notes:\n\n"
        + _bullet_list(notes)
    )


def _cost_robustness_label(tested: pd.DataFrame) -> tuple[str, str, bool]:
    five = _return_for_threshold(tested, 5.0)
    ten = _return_for_threshold(tested, 10.0)
    twenty = _return_for_threshold(tested, 20.0)
    used_nearest = five[2] or ten[2] or twenty[2]

    if twenty[1] > 0.0:
        return "high", f"positive return at {_format_bps(twenty[0])} bps", used_nearest
    if ten[1] > 0.0:
        return (
            "medium",
            f"positive return at {_format_bps(ten[0])} bps but not {_format_bps(twenty[0])} bps",
            used_nearest,
        )
    if five[1] > 0.0:
        return (
            "low",
            f"positive return at {_format_bps(five[0])} bps but not {_format_bps(ten[0])} bps",
            used_nearest,
        )
    return (
        "fragile",
        f"not positive at the {_format_bps(five[0])} bps tested threshold",
        used_nearest,
    )


def _return_for_threshold(tested: pd.DataFrame, target_bps: float) -> tuple[float, float, bool]:
    exact = tested.loc[tested["cost_bps"] == target_bps]
    if not exact.empty:
        return target_bps, float(exact.iloc[0]["total_return"]), False

    nearest_idx = (tested["cost_bps"] - target_bps).abs().idxmin()
    nearest = tested.loc[nearest_idx]
    return float(nearest["cost_bps"]), float(nearest["total_return"]), True


def _return_at_exact_bps(tested: pd.DataFrame, target_bps: float) -> float | None:
    exact = tested.loc[tested["cost_bps"] == target_bps]
    if exact.empty:
        return None
    return float(exact.iloc[0]["total_return"])


def _break_even_note(highest_positive: str, first_negative: str) -> str:
    if highest_positive == "not observed":
        return "Break-even status: no tested cost level had positive total return."
    if first_negative == "not observed":
        return (
            "Break-even status: not observed on the tested grid; total return stayed "
            f"positive through {highest_positive} bps."
        )
    return (
        "Break-even status: approximate break-even lies between "
        f"{highest_positive} bps and {first_negative} bps on the tested grid."
    )


def _cost_plain_english_note(label: str) -> str:
    notes = {
        "high": (
            "Plain-English robustness: the candidate has meaningful cushion against "
            "moderate transaction-cost assumptions in this diagnostic grid."
        ),
        "medium": (
            "Plain-English robustness: the candidate tolerates light costs, but the "
            "edge weakens before moderate transaction-cost assumptions."
        ),
        "low": (
            "Plain-English robustness: the candidate only clears a very light cost "
            "assumption and should be treated as cost-sensitive."
        ),
        "fragile": (
            "Plain-English robustness: the candidate does not show a positive edge "
            "under the tested cost threshold used for this label."
        ),
    }
    return notes[label]


def _holdings_and_contributors_section(
    holdings_summary: pd.DataFrame,
    contribution_summary: pd.DataFrame,
) -> str:
    selected_lines = _top_selected_symbols(holdings_summary)
    contributor_lines = _top_contribution_symbols(contribution_summary)
    return (
        "## Holdings and Contributors\n\n"
        "Top selected symbols:\n\n"
        + _bullet_list(selected_lines)
        + "\n\nTop contribution symbols:\n\n"
        + _bullet_list(contributor_lines)
    )


def _concentration_risk_section(concentration: pd.Series) -> str:
    rows = [
        (
            "Overall concentration risk",
            _format_object(concentration.get("overall_concentration_risk")),
        ),
        ("Risk score", _format_number(concentration.get("risk_score"), decimals=0)),
        (
            "Top selected symbol",
            _format_object(concentration.get("top_symbol_by_selection")),
        ),
        (
            "Top selected share",
            _format_percent(concentration.get("top_symbol_selected_share")),
        ),
        (
            "Top contribution symbol",
            _format_object(concentration.get("top_symbol_by_contribution")),
        ),
        (
            "Top positive contribution share",
            _format_percent(concentration.get("top_symbol_positive_contribution_share")),
        ),
        (
            "Effective selected symbols",
            _format_number(concentration.get("effective_selected_symbols")),
        ),
        (
            "Effective contributing symbols",
            _format_number(concentration.get("effective_contributing_symbols")),
        ),
    ]
    return "## Concentration Risk\n\n" + _two_column_table("Metric", "Value", rows)


def _holding_period_section(
    candidate_metrics: pd.Series,
    holding_period: pd.Series,
) -> str:
    rows = [
        ("Average holding days", _format_number(holding_period.get("avg_holding_days"))),
        ("Median holding days", _format_number(holding_period.get("median_holding_days"))),
        (
            "Single-day holding share",
            _format_percent(holding_period.get("single_day_holding_share")),
        ),
        (
            "Mean daily turnover",
            _format_percent(
                _first_available(
                    holding_period.get("mean_turnover"),
                    candidate_metrics.get("mean_daily_turnover"),
                )
            ),
        ),
        (
            "Maximum daily turnover",
            _format_percent(
                _first_available(
                    holding_period.get("max_turnover"),
                    candidate_metrics.get("max_daily_turnover"),
                )
            ),
        ),
    ]
    return "## Holding Period / Turnover\n\n" + _two_column_table("Metric", "Value", rows)


def _strengths_section(
    candidate_metrics: pd.Series,
    spy_metrics: pd.Series,
    concentration: pd.Series,
    holding_period: pd.Series,
) -> str:
    strengths: list[str] = []
    if _beats(candidate_metrics.get("total_return"), spy_metrics.get("total_return")):
        strengths.append("Out-of-sample total return is above SPY over the evaluated period.")
    if _beats(candidate_metrics.get("sharpe_like"), spy_metrics.get("sharpe_like")):
        strengths.append("Sharpe-like efficiency is above SPY in the same benchmark table.")
    if _metric_greater_than(candidate_metrics.get("max_drawdown"), -0.40):
        strengths.append("Maximum drawdown is inside the heuristic research threshold of -40%.")
    if _normalized_risk(concentration.get("overall_concentration_risk")) == "low":
        strengths.append("Concentration diagnostics are low-risk under the current heuristic.")
    if _metric_greater_than(holding_period.get("median_holding_days"), 1.0):
        strengths.append("Median holding period is longer than one strategy date.")

    if not strengths:
        strengths.append(
            "No major heuristic strengths were identified from the supplied diagnostics."
        )
    return "## Strengths\n\n" + _bullet_list(strengths)


def _weaknesses_section(
    candidate_metrics: pd.Series,
    spy_metrics: pd.Series,
    concentration: pd.Series,
    holding_period: pd.Series,
) -> str:
    weaknesses: list[str] = []
    if not _beats(candidate_metrics.get("total_return"), spy_metrics.get("total_return")):
        weaknesses.append("Total return does not clearly beat SPY in the supplied metrics.")
    if not _beats(candidate_metrics.get("sharpe_like"), spy_metrics.get("sharpe_like")):
        weaknesses.append("Sharpe-like efficiency does not clearly beat SPY.")
    if not _metric_greater_than(candidate_metrics.get("max_drawdown"), -0.40):
        weaknesses.append("Maximum drawdown breaches the heuristic -40% research threshold.")

    risk = _normalized_risk(concentration.get("overall_concentration_risk"))
    if risk in {"medium", "high"}:
        weaknesses.append(
            f"Concentration risk is {risk}, so performance may depend on few symbols."
        )
    if _metric_at_least(holding_period.get("single_day_holding_share"), 0.30):
        weaknesses.append("A large share of holding periods last only one strategy date.")
    if _metric_at_least(candidate_metrics.get("mean_daily_turnover"), 0.50):
        weaknesses.append("Mean daily turnover is high, making cost assumptions important.")

    if not weaknesses:
        weaknesses.append(
            "No major heuristic weaknesses were identified from the supplied diagnostics."
        )
    return "## Weaknesses / Risks\n\n" + _bullet_list(weaknesses)


def _research_verdict_section(verdict: str) -> str:
    descriptions = {
        VERDICT_PROMISING: (
            "The candidate clears this deterministic research-triage screen versus SPY, "
            "with drawdown and concentration diagnostics inside the preferred thresholds."
        ),
        VERDICT_NEEDS_VALIDATION: (
            "The candidate beats SPY on the headline return and Sharpe-like checks, but "
            "drawdown or concentration diagnostics require deeper validation before paper tracking."
        ),
        VERDICT_NOT_PROMISING: (
            "The candidate does not clear the current SPY-relative research-triage screen."
        ),
    }
    return (
        "## Research Verdict\n\n"
        f"**{verdict}**\n\n"
        "This is a heuristic research triage label, not an investment recommendation. "
        + descriptions[verdict]
    )


def _disclaimer_section() -> str:
    return (
        "## Disclaimer: research only, not financial advice\n\n"
        "This model card is for research documentation only. It is not financial, "
        "investment, tax, legal, or trading advice, and it should not be used as a "
        "standalone basis for live allocation decisions."
    )


def _research_verdict(
    candidate_metrics: pd.Series,
    spy_metrics: pd.Series,
    concentration: pd.Series,
) -> str:
    beats_spy = _beats(candidate_metrics.get("total_return"), spy_metrics.get("total_return")) and (
        _beats(candidate_metrics.get("sharpe_like"), spy_metrics.get("sharpe_like"))
    )
    if not beats_spy:
        return VERDICT_NOT_PROMISING

    drawdown_ok = _metric_greater_than(candidate_metrics.get("max_drawdown"), -0.40)
    risk = _normalized_risk(concentration.get("overall_concentration_risk"))
    if drawdown_ok and risk == "low":
        return VERDICT_PROMISING
    return VERDICT_NEEDS_VALIDATION


def _series_metrics(metrics: pd.DataFrame, series_name: str) -> pd.Series:
    if metrics.empty or "series_name" not in metrics.columns:
        return pd.Series(dtype="object")

    matches = metrics.loc[metrics["series_name"].astype(str) == series_name]
    if matches.empty:
        return pd.Series(dtype="object")
    return matches.iloc[0]


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    return frame.iloc[0]


def _top_selected_symbols(holdings_summary: pd.DataFrame, limit: int = 5) -> list[str]:
    if holdings_summary.empty or "symbol" not in holdings_summary.columns:
        return ["No holdings summary rows were provided."]

    sort_columns = [column for column in ["selected_days", "symbol"] if column in holdings_summary]
    ascending = [False if column == "selected_days" else True for column in sort_columns]
    holdings = holdings_summary.copy()
    if "selected_days" in holdings:
        holdings["selected_days"] = pd.to_numeric(holdings["selected_days"], errors="coerce")
    if sort_columns:
        holdings = holdings.sort_values(sort_columns, ascending=ascending, kind="mergesort")

    lines = []
    for _, row in holdings.head(limit).iterrows():
        symbol = _format_object(row.get("symbol"))
        selected_days = _format_number(row.get("selected_days"), decimals=0)
        avg_weight = _format_percent(row.get("avg_target_weight"))
        lines.append(f"{symbol}: selected days {selected_days}, average target weight {avg_weight}")
    return lines or ["No holdings summary rows were provided."]


def _top_contribution_symbols(
    contribution_summary: pd.DataFrame,
    limit: int = 5,
) -> list[str]:
    if contribution_summary.empty or "symbol" not in contribution_summary.columns:
        return ["No contribution summary rows were provided."]

    contributions = contribution_summary.copy()
    sort_col = (
        "total_position_return"
        if "total_position_return" in contributions.columns
        else "symbol"
    )
    if sort_col == "total_position_return":
        contributions[sort_col] = pd.to_numeric(contributions[sort_col], errors="coerce")
        contributions = contributions.sort_values(
            [sort_col, "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
    else:
        contributions = contributions.sort_values(sort_col, kind="mergesort")

    lines = []
    for _, row in contributions.head(limit).iterrows():
        symbol = _format_object(row.get("symbol"))
        total_position_return = _format_percent(row.get("total_position_return"))
        lines.append(f"{symbol}: total position return {total_position_return}")
    return lines or ["No contribution summary rows were provided."]


def _two_column_table(
    first_header: str,
    second_header: str,
    rows: Sequence[tuple[object, object]],
) -> str:
    return _markdown_table([first_header, second_header], rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    escaped_headers = [_escape_markdown_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _bullet_list(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _format_table_cell(column: str, value: object) -> str:
    if "return" in column or "drawdown" in column or "turnover" in column:
        return _format_percent(value)
    if column.endswith("cost") or column.endswith("equity") or column.endswith("bps"):
        return _format_number(value)
    return _format_object(value)


def _format_cost_sensitivity_cell(column: str, value: object) -> str:
    if column in {"total_return", "max_drawdown", "mean_daily_turnover"}:
        return _format_percent(value)
    if column == "sharpe_like":
        return _format_number(value, decimals=4)
    if column == "cost_bps":
        return _format_bps(value)
    if column in {"final_equity", "total_transaction_cost"}:
        return _format_number(value)
    return _format_table_cell(column, value)


def _format_delta(candidate_value: object, benchmark_value: object) -> str:
    candidate_number = _metric_float(candidate_value)
    benchmark_number = _metric_float(benchmark_value)
    if pd.isna(candidate_number) or pd.isna(benchmark_number):
        return "n/a"
    return _format_number(candidate_number - benchmark_number)


def _format_delta_percent(candidate_value: object, benchmark_value: object) -> str:
    candidate_number = _metric_float(candidate_value)
    benchmark_number = _metric_float(benchmark_value)
    if pd.isna(candidate_number) or pd.isna(benchmark_number):
        return "n/a"
    return _format_percent(candidate_number - benchmark_number)


def _format_percent(value: object) -> str:
    number = _metric_float(value)
    if pd.isna(number):
        return "n/a"
    return f"{number:.2%}"


def _format_number(value: object, decimals: int = 4) -> str:
    number = _metric_float(value)
    if pd.isna(number):
        return "n/a"
    return f"{number:.{decimals}f}"


def _format_bps(value: object) -> str:
    number = _metric_float(value)
    if pd.isna(number):
        return "n/a"
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _format_object(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format_object(val)}" for key, val in sorted(value.items()))
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_object(item) for item in value)
    if pd.isna(value):
        return "n/a"
    return str(value)


def _metric_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, Real):
        return float(value)
    try:
        converted = pd.to_numeric(value, errors="coerce")
    except (TypeError, ValueError):
        return math.nan
    if isinstance(converted, Real):
        return float(converted)
    return math.nan


def _beats(candidate_value: object, benchmark_value: object) -> bool:
    candidate_number = _metric_float(candidate_value)
    benchmark_number = _metric_float(benchmark_value)
    return (
        not pd.isna(candidate_number)
        and not pd.isna(benchmark_number)
        and candidate_number > benchmark_number
    )


def _metric_greater_than(value: object, threshold: float) -> bool:
    number = _metric_float(value)
    return not pd.isna(number) and number > threshold


def _metric_at_least(value: object, threshold: float) -> bool:
    number = _metric_float(value)
    return not pd.isna(number) and number >= threshold


def _normalized_risk(value: object) -> str:
    risk = str(value).strip().lower()
    if risk in {"low", "medium", "high"}:
        return risk
    return "unknown"


def _first_available(*values: object) -> object:
    for value in values:
        if not pd.isna(_metric_float(value)):
            return value
    return None


def _escape_markdown_cell(value: object) -> str:
    return _format_object(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "VERDICT_NEEDS_VALIDATION",
    "VERDICT_NOT_PROMISING",
    "VERDICT_PROMISING",
    "build_candidate_model_card",
]
