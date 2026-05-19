"""Create the selected Tiingo-only QQQ defensive timing replay report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    REPORT_NAME,
    SUBPERIOD_SUMMARY_NAME,
    SUMMARY_NAME,
    TRADES_NAME,
)


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingSelectedReplayReportData:
    """CSV-backed data for the selected replay Markdown report."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    subperiod_summary: pd.DataFrame
    decision_matrix: pd.DataFrame


def run_tiingo_qqq_defensive_timing_selected_replay_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the selected replay Markdown report."""

    args = _parse_args(argv)
    report_data = read_tiingo_qqq_defensive_timing_selected_replay_artifacts(args.output_dir)
    report = render_tiingo_qqq_defensive_timing_selected_replay_markdown(report_data)
    report_path = (
        args.report_path if args.report_path is not None else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("US Market ML Tiingo QQQ defensive timing selected replay report")
    print(f"output dir: {args.output_dir}")
    print(f"report path: {report_path}")
    print(f"daily equity rows: {len(report_data.daily_equity)}")
    print(f"trade rows: {len(report_data.trades)}")
    return 0


def read_tiingo_qqq_defensive_timing_selected_replay_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TiingoQQQDefensiveTimingSelectedReplayReportData:
    """Read selected replay CSV artifacts from ``output_dir``."""

    return TiingoQQQDefensiveTimingSelectedReplayReportData(
        summary=_read_required_csv(output_dir / SUMMARY_NAME),
        daily_equity=_read_required_csv(output_dir / DAILY_EQUITY_NAME),
        trades=_read_required_csv(output_dir / TRADES_NAME),
        subperiod_summary=_read_required_csv(output_dir / SUBPERIOD_SUMMARY_NAME),
        decision_matrix=_read_required_csv(output_dir / DECISION_MATRIX_NAME),
    )


def render_tiingo_qqq_defensive_timing_selected_replay_markdown(
    report_data: TiingoQQQDefensiveTimingSelectedReplayReportData,
) -> str:
    """Render the selected QQQ defensive timing replay Markdown report."""

    summary = _first_row(report_data.summary)
    decision = _first_row(report_data.decision_matrix)
    comparison = _full_period_comparison(report_data.daily_equity, summary)
    drawdown_summary = _drawdown_summary(report_data.daily_equity)
    trade_preview = report_data.trades.copy().head(60)
    subperiods = report_data.subperiod_summary.copy()

    lines = [
        "# Tiingo QQQ SMA190 Defensive Replay Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(summary, decision),
        "",
        "## 2. Candidate configuration",
        "",
        _markdown_table(
            _dict_to_frame(
                summary,
                [
                    "strategy_name",
                    "risk_on_asset",
                    "risk_off_asset",
                    "signal_name",
                    "slow_sma_days",
                    "buffer_pct",
                    "min_hold_days",
                    "transaction_cost_bps",
                    "start_date",
                    "end_date",
                ],
            )
        ),
        "",
        "## 3. Why this candidate was selected",
        "",
        _selection_rationale(summary),
        "",
        "## 4. Full-period comparison vs QQQ/SPY/cash",
        "",
        _markdown_table(comparison),
        "",
        "## 5. Daily equity and drawdown summary",
        "",
        _markdown_table(drawdown_summary),
        "",
        "## 6. Trade timeline / turnover",
        "",
        _markdown_table(_format_for_markdown(trade_preview)),
        "",
        "## 7. Subperiod robustness",
        "",
        _markdown_table(_format_for_markdown(subperiods)),
        "",
        "## 8. Where it helps",
        "",
        _where_it_helps(summary, subperiods),
        "",
        "## 9. Where it still fails",
        "",
        _where_it_fails(summary, subperiods),
        "",
        "## 10. Decision",
        "",
        _decision_text(decision),
        "",
        _markdown_table(_format_for_markdown(report_data.decision_matrix)),
        "",
        "## 11. Next action",
        "",
        _next_action_text(),
        "",
        "## 12. Caveats",
        "",
        _caveats_text(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(summary: dict[str, object], decision: dict[str, object]) -> str:
    if not summary:
        return "No selected replay summary was available; keep this research-only."
    strategy_name = summary.get("strategy_name", "selected candidate")
    total_return = _format_number(summary.get("total_return"))
    qqq_return = _format_number(summary.get("qqq_buy_hold_total_return"))
    strategy_vs_qqq = _format_number(summary.get("strategy_vs_qqq"))
    drawdown_improvement = _format_number(summary.get("drawdown_improvement_vs_qqq"))
    decision_label = decision.get("decision", "research_only")
    return (
        f"`{strategy_name}` remains a research-only QQQ defensive timing candidate. "
        f"Replay total return is `{total_return}` versus QQQ buy-and-hold `{qqq_return}` "
        f"with strategy-vs-QQQ `{strategy_vs_qqq}` and drawdown improvement "
        f"`{drawdown_improvement}`. Decision: `{decision_label}`."
    )


def _selection_rationale(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            (
                "- The narrow sweep identified this candidate as the best QQQ defensive "
                "timing trade-off: it nearly matched QQQ while improving drawdown and "
                "return-to-drawdown."
            ),
            (
                f"- QQQ return capture in this replay is "
                f"`{_format_number(summary.get('qqq_return_capture'))}`."
            ),
            (
                f"- Return-to-drawdown is `{_format_number(summary.get('return_to_drawdown'))}` "
                "versus QQQ buy-and-hold "
                f"`{_format_number(summary.get('qqq_buy_hold_return_to_drawdown'))}`."
            ),
            (
                "- Timing convention remains leakage-safe: the close T signal applies "
                "starting the next trading day."
            ),
        ]
    )


def _where_it_helps(summary: dict[str, object], subperiods: pd.DataFrame) -> str:
    best_periods = _positive_period_names(subperiods)
    period_text = ", ".join(best_periods) if best_periods else "no subperiods versus QQQ"
    return "\n".join(
        [
            (
                "- It improves full-period drawdown versus QQQ by "
                f"`{_format_number(summary.get('drawdown_improvement_vs_qqq'))}`."
            ),
            (
                "- It improves return-to-drawdown when the defensive switch avoids "
                "enough QQQ downside without giving up much upside."
            ),
            f"- Positive or better QQQ-relative subperiods: {period_text}.",
        ]
    )


def _where_it_fails(summary: dict[str, object], subperiods: pd.DataFrame) -> str:
    weakest = _weakest_subperiod(subperiods)
    weak_text = (
        f"`{weakest['period_name']}` at `{_format_number(weakest['strategy_vs_qqq'])}`"
        if weakest
        else "not available"
    )
    return "\n".join(
        [
            (
                "- It still does not beat QQQ buy-and-hold on raw full-period return "
                f"(`{_format_number(summary.get('strategy_vs_qqq'))}` vs QQQ)."
            ),
            f"- Weakest QQQ-relative subperiod: {weak_text}.",
            (
                "- It is a defensive timing candidate, not evidence that QQQ "
                "buy-and-hold has been dethroned as the benchmark."
            ),
        ]
    )


def _decision_text(decision: dict[str, object]) -> str:
    decision_label = str(decision.get("decision", "unknown"))
    best = decision_label == "keep_as_best_qqq_defensive_timing_candidate"
    reject = decision_label == "reject_deprioritize"
    return "\n".join(
        [
            (
                "- Keep as best QQQ defensive timing candidate: "
                f"{'yes' if best else 'no'}."
            ),
            "- Keep as research-only defensive candidate: yes.",
            f"- Reject/deprioritize: {'yes' if reject else 'no'}.",
            "- Production-ready: no.",
            f"- Reason: {decision.get('decision_reason', 'research-only replay evidence')}",
        ]
    )


def _next_action_text() -> str:
    return "\n".join(
        [
            "- Model-card draft: allowed as research documentation.",
            "- Longer-history validation if data allows: next validation step.",
            "- Walk-forward robustness: next validation step.",
            "- Feature v4: do not add benchmark_gap_features_v4 from this replay.",
            "- tree_screen: do not run from this replay.",
            "- MLP: do not run from this replay.",
            "- Production promotion: do not promote or mark production-ready.",
        ]
    )


def _caveats_text() -> str:
    return "\n".join(
        [
            "- Tiingo-only input data.",
            "- Research-only output.",
            "- No S3/cloud paths were used.",
            "- No production signal export was created or updated.",
            "- QQQ buy-and-hold remains the benchmark.",
        ]
    )


def _full_period_comparison(
    daily_equity: pd.DataFrame,
    summary: dict[str, object],
) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    daily = daily_equity.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    for strategy_name, group in daily.groupby("strategy_name", sort=False):
        equity = pd.to_numeric(group["equity"], errors="coerce")
        returns = pd.to_numeric(group["daily_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "strategy_name": strategy_name,
                "active_assets": ", ".join(sorted(group["active_asset"].astype(str).unique())),
                "total_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": calculate_max_drawdown(equity),
                "daily_volatility": float(returns.std()),
                "exposure_rate_to_qqq": float(group["active_asset"].astype(str).eq("QQQ").mean()),
                "exposure_rate_to_spy": float(group["active_asset"].astype(str).eq("SPY").mean()),
            }
        )
    comparison = pd.DataFrame(rows)
    if summary:
        comparison["strategy_vs_selected"] = (
            comparison["total_return"] - _as_float(summary.get("total_return"))
        )
    return _format_for_markdown(comparison)


def _drawdown_summary(daily_equity: pd.DataFrame) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy_name, group in daily_equity.groupby("strategy_name", sort=False):
        drawdown = pd.to_numeric(group["drawdown"], errors="coerce")
        equity = pd.to_numeric(group["equity"], errors="coerce")
        rows.append(
            {
                "strategy_name": strategy_name,
                "start_equity": float(equity.iloc[0]),
                "end_equity": float(equity.iloc[-1]),
                "max_drawdown": float(drawdown.min()),
                "worst_drawdown_date": str(group.iloc[int(drawdown.argmin())]["date"]),
                "rows": int(len(group)),
            }
        )
    return _format_for_markdown(pd.DataFrame(rows))


def _positive_period_names(subperiods: pd.DataFrame) -> list[str]:
    if subperiods.empty or "strategy_vs_qqq" not in subperiods.columns:
        return []
    stress = subperiods.loc[subperiods["period_name"].ne("full")].copy()
    positive = stress.loc[pd.to_numeric(stress["strategy_vs_qqq"], errors="coerce") > 0.0]
    return positive["period_name"].astype(str).tolist()


def _weakest_subperiod(subperiods: pd.DataFrame) -> dict[str, object]:
    if subperiods.empty or "strategy_vs_qqq" not in subperiods.columns:
        return {}
    stress = subperiods.loc[subperiods["period_name"].ne("full")].copy()
    if stress.empty:
        stress = subperiods.copy()
    stress["strategy_vs_qqq"] = pd.to_numeric(stress["strategy_vs_qqq"], errors="coerce")
    return stress.sort_values("strategy_vs_qqq", kind="mergesort").iloc[0].to_dict()


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selected replay artifact does not exist: {path}")
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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the selected Tiingo-only QQQ defensive timing replay report."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


create_tiingo_qqq_defensive_timing_selected_replay_report_from_args = (
    run_tiingo_qqq_defensive_timing_selected_replay_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_selected_replay_report_from_args())
