"""Create the Tiingo-only QQQ defensive timing validation report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_validation import (
    COST_SENSITIVITY_NAME,
    COVERAGE_NAME,
    DAILY_EQUITY_NAME,
    DEFAULT_OUTPUT_DIR,
    FULL_PERIOD_NAME,
    REPORT_NAME,
    SUBPERIODS_NAME,
    TRADES_NAME,
    WALK_FORWARD_SUMMARY_NAME,
    WALK_FORWARD_WINDOWS_NAME,
)


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingValidationReportData:
    """CSV-backed data for the validation Markdown report."""

    coverage: pd.DataFrame
    full_period: pd.DataFrame
    walk_forward_windows: pd.DataFrame
    walk_forward_summary: pd.DataFrame
    subperiods: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame


def run_tiingo_qqq_defensive_timing_validation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the validation Markdown report."""

    args = _parse_args(argv)
    report_data = read_tiingo_qqq_defensive_timing_validation_artifacts(args.output_dir)
    report = render_tiingo_qqq_defensive_timing_validation_markdown(report_data)
    report_path = (
        args.report_path if args.report_path is not None else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("US Market ML Tiingo QQQ defensive timing validation report")
    print(f"output dir: {args.output_dir}")
    print(f"report path: {report_path}")
    print(f"rolling windows: {len(report_data.walk_forward_windows)}")
    print(f"trade rows: {len(report_data.trades)}")
    return 0


def read_tiingo_qqq_defensive_timing_validation_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TiingoQQQDefensiveTimingValidationReportData:
    """Read validation CSV artifacts from ``output_dir``."""

    return TiingoQQQDefensiveTimingValidationReportData(
        coverage=_read_required_csv(output_dir / COVERAGE_NAME),
        full_period=_read_required_csv(output_dir / FULL_PERIOD_NAME),
        walk_forward_windows=_read_required_csv(output_dir / WALK_FORWARD_WINDOWS_NAME),
        walk_forward_summary=_read_required_csv(output_dir / WALK_FORWARD_SUMMARY_NAME),
        subperiods=_read_required_csv(output_dir / SUBPERIODS_NAME),
        cost_sensitivity=_read_required_csv(output_dir / COST_SENSITIVITY_NAME),
        daily_equity=_read_required_csv(output_dir / DAILY_EQUITY_NAME),
        trades=_read_required_csv(output_dir / TRADES_NAME),
    )


def render_tiingo_qqq_defensive_timing_validation_markdown(
    report_data: TiingoQQQDefensiveTimingValidationReportData,
) -> str:
    """Render the QQQ defensive timing validation Markdown report."""

    coverage = _first_row(report_data.coverage)
    full = _first_row(report_data.full_period)
    all_walk_forward = _summary_row(report_data.walk_forward_summary, "all")
    decision = _validation_decision(full, all_walk_forward)

    lines = [
        "# Tiingo QQQ Defensive Timing Validation Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(full, coverage, all_walk_forward, decision),
        "",
        "## 2. Candidate configuration",
        "",
        _markdown_table(
            _dict_to_frame(
                full,
                [
                    "strategy_name",
                    "risk_on_asset",
                    "risk_off_asset",
                    "signal_name",
                    "slow_sma_days",
                    "buffer_pct",
                    "min_hold_days",
                    "transaction_cost_bps",
                    "research_status",
                    "production_ready",
                ],
            )
        ),
        "",
        "## 3. Data coverage / whether longer-history validation was possible",
        "",
        _coverage_text(coverage),
        "",
        _markdown_table(_format_for_markdown(report_data.coverage)),
        "",
        "## 4. Full-period validation versus QQQ/SPY/cash",
        "",
        _markdown_table(_full_period_comparison(full)),
        "",
        "## 5. Walk-forward robustness summary",
        "",
        _markdown_table(_format_for_markdown(report_data.walk_forward_summary)),
        "",
        "## 6. Worst windows and failure modes",
        "",
        _worst_window_text(report_data.walk_forward_windows),
        "",
        "## 7. Named subperiod robustness",
        "",
        _markdown_table(_format_for_markdown(report_data.subperiods)),
        "",
        "## 8. Transaction cost sensitivity",
        "",
        _cost_sensitivity_text(report_data.cost_sensitivity),
        "",
        _markdown_table(_format_for_markdown(report_data.cost_sensitivity)),
        "",
        "## 9. What this validates",
        "",
        _what_this_validates(),
        "",
        "## 10. What remains unvalidated",
        "",
        _what_remains_unvalidated(),
        "",
        "## 11. Decision",
        "",
        _decision_text(decision),
        "",
        "## 12. Next action",
        "",
        _next_action_text(decision),
        "",
        "## 13. Caveats",
        "",
        _caveats_text(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(
    full: dict[str, object],
    coverage: dict[str, object],
    walk_forward: dict[str, object],
    decision: dict[str, object],
) -> str:
    if not full:
        return "No full-period validation row was available; keep this research-only."
    return (
        f"`{full.get('strategy_name')}` remains research-only with "
        "`production_ready = false`. Full-period total return is "
        f"`{_format_number(full.get('total_return'))}` versus QQQ "
        f"`{_format_number(full.get('qqq_buy_hold_total_return'))}` and SPY "
        f"`{_format_number(full.get('spy_buy_hold_total_return'))}`. "
        f"Return-to-drawdown is `{_format_number(full.get('return_to_drawdown'))}` "
        "versus QQQ "
        f"`{_format_number(full.get('qqq_buy_hold_return_to_drawdown'))}`. "
        f"Walk-forward underperformance rate versus QQQ is "
        f"`{_format_number(walk_forward.get('qqq_underperformance_rate'))}`. "
        f"Decision: `{decision['label']}`. "
        f"Validation range: `{coverage.get('selected_start_date')}` to "
        f"`{coverage.get('selected_end_date')}`."
    )


def _coverage_text(coverage: dict[str, object]) -> str:
    if not coverage:
        return "Coverage discovery did not produce a row."
    longer_history = _as_bool(coverage.get("longer_history_available"))
    if longer_history:
        availability = (
            "Longer QQQ/SPY history was available locally, so this validation starts "
            "at the earliest common Tiingo-only coverage date."
        )
    else:
        availability = (
            "Longer-history validation was limited by local Tiingo-only QQQ/SPY "
            "coverage; this report does not pretend to validate a longer period."
        )
    return "\n".join(
        [
            availability,
            "",
            f"- QQQ coverage: `{coverage.get('qqq_first_date')}` to "
            f"`{coverage.get('qqq_last_date')}`.",
            f"- SPY coverage: `{coverage.get('spy_first_date')}` to "
            f"`{coverage.get('spy_last_date')}`.",
            f"- Common coverage: `{coverage.get('common_start_date')}` to "
            f"`{coverage.get('common_end_date')}`.",
            f"- Selected validation window: `{coverage.get('selected_start_date')}` to "
            f"`{coverage.get('selected_end_date')}` with "
            f"`{coverage.get('selected_trading_days')}` trading days.",
            f"- Coverage note: {coverage.get('coverage_note')}",
        ]
    )


def _full_period_comparison(full: dict[str, object]) -> pd.DataFrame:
    if not full:
        return pd.DataFrame()
    rows = [
        {
            "comparison": "strategy",
            "total_return": full.get("total_return"),
            "max_drawdown": full.get("max_drawdown"),
            "return_to_drawdown": full.get("return_to_drawdown"),
            "vs_qqq": full.get("strategy_vs_qqq"),
        },
        {
            "comparison": "qqq_buy_hold",
            "total_return": full.get("qqq_buy_hold_total_return"),
            "max_drawdown": full.get("qqq_buy_hold_max_drawdown"),
            "return_to_drawdown": full.get("qqq_buy_hold_return_to_drawdown"),
            "vs_qqq": 0.0,
        },
        {
            "comparison": "spy_buy_hold",
            "total_return": full.get("spy_buy_hold_total_return"),
            "max_drawdown": full.get("spy_buy_hold_max_drawdown"),
            "return_to_drawdown": np.nan,
            "vs_qqq": _as_float(full.get("spy_buy_hold_total_return"))
            - _as_float(full.get("qqq_buy_hold_total_return")),
        },
        {
            "comparison": "cash",
            "total_return": full.get("cash_total_return"),
            "max_drawdown": 0.0,
            "return_to_drawdown": np.nan,
            "vs_qqq": _as_float(full.get("cash_total_return"))
            - _as_float(full.get("qqq_buy_hold_total_return")),
        },
    ]
    return _format_for_markdown(pd.DataFrame(rows))


def _worst_window_text(windows: pd.DataFrame) -> str:
    if windows.empty:
        return "_No rolling windows were available._"
    display_cols = [
        "window_label",
        "start_date",
        "end_date",
        "strategy_vs_qqq",
        "strategy_max_drawdown",
        "qqq_max_drawdown",
        "drawdown_improvement_vs_qqq",
        "transaction_count",
        "exposure_rate_to_qqq",
    ]
    weakest_return = windows.sort_values("strategy_vs_qqq", kind="mergesort").head(5)
    weakest_drawdown = windows.sort_values("strategy_max_drawdown", kind="mergesort").head(5)
    return "\n\n".join(
        [
            "Weakest QQQ-relative windows:",
            _markdown_table(_format_for_markdown(weakest_return.loc[:, display_cols])),
            "Worst strategy drawdown windows:",
            _markdown_table(_format_for_markdown(weakest_drawdown.loc[:, display_cols])),
            (
                "Main failure mode: the defensive switch can trail QQQ in strong QQQ "
                "bull markets, while SMA whipsaws can add cost without enough drawdown "
                "benefit."
            ),
        ]
    )


def _cost_sensitivity_text(cost_sensitivity: pd.DataFrame) -> str:
    if cost_sensitivity.empty:
        return "No transaction cost sensitivity rows were available."
    changed = cost_sensitivity.loc[
        cost_sensitivity["decision_changed_vs_anchor"].astype(bool)
    ]
    if changed.empty:
        result = "The research-only decision does not change across tested cost levels."
    else:
        result = (
            "The decision changes at one or more tested cost levels; review the table "
            "before any paper-tracking design."
        )
    return (
        f"{result} Optional explicit slippage sensitivity is deferred; this table varies "
        "transaction cost assumptions only."
    )


def _what_this_validates() -> str:
    return "\n".join(
        [
            "- The exact selected candidate over the longest local common QQQ/SPY window.",
            "- close T signal applied from the next trading day.",
            "- QQQ, SPY, and cash references in the same Tiingo-only frame.",
            "- Rolling evaluation windows without refitting or parameter changes.",
            "- Transaction cost sensitivity at 0, 5, 10, and 20 bps when defaults are used.",
        ]
    )


def _what_remains_unvalidated() -> str:
    return "\n".join(
        [
            "- External or longer history not present in the local Tiingo-only dataset.",
            "- Live paper-tracking mechanics and operational monitoring.",
            "- Broker-specific execution, taxes, borrow, and realized slippage.",
            "- Any production signal export path.",
            "- Any tree, MLP, or benchmark_gap_features_v4 follow-on model.",
        ]
    )


def _validation_decision(
    full: dict[str, object],
    walk_forward: dict[str, object],
) -> dict[str, object]:
    qqq_capture = _as_float(full.get("qqq_return_capture"))
    strategy_vs_qqq = _as_float(full.get("strategy_vs_qqq"))
    drawdown_improvement = _as_float(full.get("drawdown_improvement_vs_qqq"))
    return_to_drawdown = _as_float(full.get("return_to_drawdown"))
    qqq_return_to_drawdown = _as_float(full.get("qqq_buy_hold_return_to_drawdown"))
    underperformance_rate = _as_float(walk_forward.get("qqq_underperformance_rate"))
    drawdown_rate = _as_float(walk_forward.get("drawdown_improvement_window_rate"))
    average_drawdown = _as_float(walk_forward.get("average_drawdown_improvement_vs_qqq"))

    near_qqq = qqq_capture >= 0.95 or strategy_vs_qqq >= -0.10
    improves_full_rtd = return_to_drawdown > qqq_return_to_drawdown
    improves_full_drawdown = drawdown_improvement > 0.0
    persistent_underperformance = underperformance_rate >= 0.70
    minimal_drawdown_benefit = drawdown_rate < 0.40 and average_drawdown <= 0.01

    if not near_qqq and not improves_full_rtd:
        return {
            "label": "reject_deprioritize",
            "keep_best": False,
            "keep_research_only": True,
            "mixed": False,
            "reason": (
                "Longer-history validation materially trails QQQ and does not improve "
                "return-to-drawdown versus QQQ buy-and-hold."
            ),
        }
    if persistent_underperformance and minimal_drawdown_benefit:
        return {
            "label": "reject_deprioritize",
            "keep_best": False,
            "keep_research_only": True,
            "mixed": False,
            "reason": (
                "Walk-forward robustness shows persistent QQQ underperformance with "
                "minimal drawdown benefit."
            ),
        }
    if near_qqq and improves_full_rtd and improves_full_drawdown and persistent_underperformance:
        return {
            "label": "mixed_research_only",
            "keep_best": True,
            "keep_research_only": True,
            "mixed": True,
            "reason": (
                "Full-period validation is close to QQQ and improves return-to-drawdown, "
                "but rolling-window robustness is weak."
            ),
        }
    if near_qqq and improves_full_rtd and improves_full_drawdown:
        return {
            "label": "keep_as_best_qqq_defensive_timing_candidate",
            "keep_best": True,
            "keep_research_only": True,
            "mixed": False,
            "reason": (
                "Full-period validation remains near QQQ while improving drawdown and "
                "return-to-drawdown, with acceptable rolling evidence."
            ),
        }
    return {
        "label": "keep_research_only_review_required",
        "keep_best": False,
        "keep_research_only": True,
        "mixed": True,
        "reason": (
            "Evidence is not strong enough for promotion; keep research-only until "
            "validation review."
        ),
    }


def _decision_text(decision: dict[str, object]) -> str:
    return "\n".join(
        [
            f"- Decision: `{decision['label']}`.",
            (
                "- Keep as best QQQ defensive timing candidate: "
                f"{'yes' if decision['keep_best'] else 'no'}."
            ),
            "- Keep research-only: yes.",
            (
                "- Reject/deprioritize: "
                f"{'yes' if decision['label'] == 'reject_deprioritize' else 'no'}."
            ),
            "- Production-ready: no; `production_ready = false`.",
            f"- Reason: {decision['reason']}",
        ]
    )


def _next_action_text(decision: dict[str, object]) -> str:
    if decision["label"] == "reject_deprioritize":
        first = "- Review rejection/deprioritization before spending time on paper tracking."
    else:
        first = "- Live paper-tracking design after validation review."
    return "\n".join(
        [
            first,
            "- Longer external history if available.",
            "- Strategy model-card update with this validation bundle.",
            "- Production promotion: no.",
        ]
    )


def _caveats_text() -> str:
    return "\n".join(
        [
            "- Tiingo-only local research data.",
            "- No S3/cloud paths were used.",
            "- No production signal export was created or updated.",
            "- This report does not mark the candidate production-ready.",
            "- QQQ buy-and-hold remains the primary benchmark.",
        ]
    )


def _summary_row(summary: pd.DataFrame, window_group: str) -> dict[str, object]:
    if summary.empty or "window_group" not in summary.columns:
        return {}
    rows = summary.loc[summary["window_group"].astype(str).eq(window_group)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation artifact does not exist: {path}")
    return pd.read_csv(path)


def _first_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _dict_to_frame(row: dict[str, object], keys: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": row.get(key, np.nan)} for key in keys],
        columns=["metric", "value"],
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = _format_for_markdown(frame)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].map(_format_value)
    return out


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, float | int | np.floating | np.integer) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


def _format_number(value: object) -> str:
    number = _as_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.4f}"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo-only QQQ defensive timing validation report."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


create_tiingo_qqq_defensive_timing_validation_report_from_args = (
    run_tiingo_qqq_defensive_timing_validation_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_validation_report_from_args())
