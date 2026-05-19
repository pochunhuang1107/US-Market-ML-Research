"""Create research docs for the selected QQQ defensive timing candidate."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    DECISION_MATRIX_NAME,
    SUBPERIOD_SUMMARY_NAME,
    SUMMARY_NAME,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REPLAY_DIR,
)

DEFAULT_DOCS_DIR = Path("docs/research")
MODEL_CARD_NAME = "qqq_defensive_timing_model_card.md"
RESEARCH_DECISION_NAME = "qqq_defensive_timing_research_decision.md"


@dataclass(frozen=True)
class QQQDefensiveTimingModelCardData:
    """CSV-backed inputs for the selected QQQ defensive timing docs."""

    summary: pd.DataFrame
    subperiod_summary: pd.DataFrame
    decision_matrix: pd.DataFrame


def run_create_qqq_defensive_timing_model_card_from_args(
    argv: list[str] | None = None,
) -> int:
    """Read selected replay CSVs and write the research model card docs."""

    args = _parse_args(argv)
    outputs = write_qqq_defensive_timing_research_docs(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
    )

    print("US Market ML QQQ defensive timing research docs")
    print(f"replay dir: {args.replay_dir}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def write_qqq_defensive_timing_research_docs(
    *,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    output_dir: Path = DEFAULT_DOCS_DIR,
) -> dict[str, Path]:
    """Write the model card and decision docs from selected replay artifacts."""

    data = read_qqq_defensive_timing_model_card_data(replay_dir)
    model_card = render_qqq_defensive_timing_model_card(data)
    research_decision = render_qqq_defensive_timing_research_decision(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_card_path = output_dir / MODEL_CARD_NAME
    research_decision_path = output_dir / RESEARCH_DECISION_NAME
    model_card_path.write_text(model_card, encoding="utf-8")
    research_decision_path.write_text(research_decision, encoding="utf-8")

    return {
        "model card": model_card_path,
        "research decision": research_decision_path,
    }


def read_qqq_defensive_timing_model_card_data(
    replay_dir: Path = DEFAULT_REPLAY_DIR,
) -> QQQDefensiveTimingModelCardData:
    """Read selected replay CSV artifacts from ``replay_dir``."""

    return QQQDefensiveTimingModelCardData(
        summary=_read_required_csv(replay_dir / SUMMARY_NAME),
        subperiod_summary=_read_required_csv(replay_dir / SUBPERIOD_SUMMARY_NAME),
        decision_matrix=_read_required_csv(replay_dir / DECISION_MATRIX_NAME),
    )


def render_qqq_defensive_timing_model_card(
    data: QQQDefensiveTimingModelCardData,
) -> str:
    """Render the selected QQQ defensive timing model card."""

    summary = _first_row(data.summary)
    decision = _first_row(data.decision_matrix)
    strategy_name = str(summary.get("strategy_name", "qqq_sma190_buffer_p0p5_hold0_to_spy"))

    sections = [
        f"# QQQ Defensive Timing Model Card: {strategy_name}",
        _status_section(summary, decision),
        _strategy_rule_section(summary),
        _dataset_and_benchmarks_section(summary),
        _full_period_metrics_section(summary),
        _subperiod_metrics_section(data.subperiod_summary),
        _what_worked_section(summary, data.subperiod_summary),
        _what_failed_section(summary, data.subperiod_summary),
        _risks_and_limitations_section(),
        _why_not_production_ready_section(),
        _current_decision_section(decision),
        _next_validation_steps_section(),
    ]
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def render_qqq_defensive_timing_research_decision(
    data: QQQDefensiveTimingModelCardData,
) -> str:
    """Render the selected QQQ defensive timing research decision document."""

    summary = _first_row(data.summary)
    decision = _first_row(data.decision_matrix)
    strategy_name = str(summary.get("strategy_name", "qqq_sma190_buffer_p0p5_hold0_to_spy"))

    sections = [
        f"# QQQ Defensive Timing Research Decision: {strategy_name}",
        _research_path_section(),
        _candidate_decision_section(summary, decision),
        _why_qqq_benchmark_section(summary),
        _why_candidate_is_useful_section(summary, data.subperiod_summary),
        _explicit_no_go_section(),
        _recommended_next_task_section(),
    ]
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _status_section(summary: dict[str, object], decision: dict[str, object]) -> str:
    rows = [
        ("Strategy name", summary.get("strategy_name")),
        ("Strategy type", "QQQ defensive timing / risk-on-risk-off"),
        ("Family", summary.get("family", "total_return_spy_defensive")),
        ("Risk-on asset", summary.get("risk_on_asset")),
        ("Risk-off asset", summary.get("risk_off_asset")),
        ("Signal name", summary.get("signal_name")),
        ("Current decision", decision.get("decision")),
        ("Research status", decision.get("research_status")),
        ("Production-ready", "production_ready = false"),
    ]
    return "## 1. Status\n\n" + _two_column_table("Field", "Value", rows)


def _strategy_rule_section(summary: dict[str, object]) -> str:
    slow_sma_days = _format_integer(summary.get("slow_sma_days"), fallback="190")
    buffer = _as_float(summary.get("buffer_pct"))
    threshold = 1.0 + buffer if np.isfinite(buffer) else 1.005
    transaction_cost_bps = _format_number(summary.get("transaction_cost_bps"))
    min_hold_days = _format_integer(summary.get("min_hold_days"), fallback="0")
    return "\n".join(
        [
            "## 2. Strategy Rule",
            "",
            f"- Hold QQQ when QQQ close > SMA{slow_sma_days} * {_format_multiplier(threshold)}.",
            "- Otherwise hold SPY.",
            "- Signal from close T applies next trading day.",
            f"- Minimum hold days: `{min_hold_days}`.",
            f"- Transaction cost assumption: `{transaction_cost_bps}` bps per turnover unit.",
        ]
    )


def _dataset_and_benchmarks_section(summary: dict[str, object]) -> str:
    start_date = summary.get("start_date", "2019-01-02")
    end_date = summary.get("end_date", "2026-05-15")
    trading_days = _format_integer(summary.get("trading_days"), fallback="1853")
    return "\n".join(
        [
            "## 3. Dataset And Benchmarks",
            "",
            "- Dataset: Tiingo-only adjusted ETF research data.",
            f"- Range: `{start_date}` to `{end_date}`.",
            f"- Trading days in replay: `{trading_days}`.",
            "- Benchmarks: QQQ buy-and-hold, SPY buy-and-hold, and cash.",
        ]
    )


def _full_period_metrics_section(summary: dict[str, object]) -> str:
    rows = [
        ("Strategy total return", _format_number(summary.get("total_return"))),
        ("QQQ buy-and-hold total return", _format_number(summary.get("qqq_buy_hold_total_return"))),
        ("SPY buy-and-hold total return", _format_number(summary.get("spy_buy_hold_total_return"))),
        ("Strategy vs QQQ", _format_number(summary.get("strategy_vs_qqq"))),
        ("Strategy vs SPY", _format_number(summary.get("strategy_vs_spy"))),
        ("Annualized return", _format_number(summary.get("annualized_return"))),
        ("Daily volatility", _format_number(summary.get("daily_volatility"))),
        ("Sharpe-like", _format_number(summary.get("sharpe_like"))),
        ("Strategy max drawdown", _format_number(summary.get("max_drawdown"))),
        ("QQQ buy-and-hold max drawdown", _format_number(summary.get("qqq_buy_hold_max_drawdown"))),
        ("SPY buy-and-hold max drawdown", _format_number(summary.get("spy_buy_hold_max_drawdown"))),
        (
            "Drawdown improvement vs QQQ",
            _format_number(summary.get("drawdown_improvement_vs_qqq")),
        ),
        ("Return-to-drawdown", _format_number(summary.get("return_to_drawdown"))),
        (
            "QQQ buy-and-hold return-to-drawdown",
            _format_number(summary.get("qqq_buy_hold_return_to_drawdown")),
        ),
        ("QQQ return capture", _format_number(summary.get("qqq_return_capture"))),
        ("Transaction count", _format_integer(summary.get("transaction_count"))),
        ("Exposure rate to QQQ", _format_number(summary.get("exposure_rate_to_qqq"))),
        ("Exposure rate to SPY", _format_number(summary.get("exposure_rate_to_spy"))),
        ("Estimated transaction cost", _format_number(summary.get("estimated_transaction_cost"))),
    ]
    return (
        "## 4. Full-Period Metrics\n\n"
        "QQQ buy-and-hold comparison remains the primary readout: the strategy "
        "nearly matched QQQ return, but did not beat it on raw total return.\n\n"
        + _two_column_table("Metric", "Value", rows)
    )


def _subperiod_metrics_section(subperiod_summary: pd.DataFrame) -> str:
    display_columns = [
        "period_name",
        "period_start",
        "period_end",
        "strategy_total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "strategy_vs_qqq",
        "strategy_vs_spy",
        "strategy_max_drawdown",
        "qqq_max_drawdown",
        "spy_max_drawdown",
        "exposure_rate_to_qqq",
        "exposure_rate_to_spy",
        "transaction_count",
    ]
    available_columns = [
        column for column in display_columns if column in subperiod_summary.columns
    ]
    table = _markdown_table(_format_for_markdown(subperiod_summary.loc[:, available_columns]))
    return "\n".join(
        [
            "## 5. Subperiod Metrics",
            "",
            table,
            "",
            _subperiod_interpretation(subperiod_summary),
        ]
    )


def _what_worked_section(
    summary: dict[str, object],
    subperiod_summary: pd.DataFrame,
) -> str:
    positive_periods = _positive_strategy_vs_qqq_periods(subperiod_summary)
    positive_period_text = ", ".join(positive_periods) if positive_periods else "none"
    return "\n".join(
        [
            "## 6. What Worked",
            "",
            (
                "- The candidate preserved most QQQ upside with QQQ return capture of "
                f"`{_format_number(summary.get('qqq_return_capture'))}`."
            ),
            (
                "- It improved full-period max drawdown versus QQQ by "
                f"`{_format_number(summary.get('drawdown_improvement_vs_qqq'))}`."
            ),
            (
                "- It improved return-to-drawdown versus QQQ: "
                f"`{_format_number(summary.get('return_to_drawdown'))}` vs "
                f"`{_format_number(summary.get('qqq_buy_hold_return_to_drawdown'))}`."
            ),
            f"- It materially helped in QQQ-relative terms during: {positive_period_text}.",
        ]
    )


def _what_failed_section(
    summary: dict[str, object],
    subperiod_summary: pd.DataFrame,
) -> str:
    weakest = _weakest_strategy_vs_qqq_period(subperiod_summary)
    weakest_text = (
        f"{weakest.get('period_name')} (`{_format_number(weakest.get('strategy_vs_qqq'))}`)"
        if weakest
        else "not available"
    )
    return "\n".join(
        [
            "## 7. What Failed",
            "",
            (
                "- The strategy did not beat QQQ buy-and-hold on full-period raw return; "
                f"strategy vs QQQ was `{_format_number(summary.get('strategy_vs_qqq'))}`."
            ),
            "- The COVID crash/recovery window had worse drawdown than QQQ.",
            "- Strong QQQ bull markets remain hard for the defensive switch to match.",
            f"- Weakest QQQ-relative subperiod: {weakest_text}.",
        ]
    )


def _risks_and_limitations_section() -> str:
    return "\n".join(
        [
            "## 8. Risks And Limitations",
            "",
            "- History is short: the selected Tiingo-only replay starts in 2019.",
            (
                "- The rule was selected from research sweeps and needs out-of-sample "
                "robustness checks."
            ),
            "- It may whipsaw around the SMA threshold, especially with `min_hold_days = 0`.",
            "- SPY is still equity risk, not a cash or bond hedge.",
            "- Transaction costs and slippage are simplified.",
            (
                "- The result is sensitive to QQQ trend regimes and may lag during "
                "strong QQQ advances."
            ),
        ]
    )


def _why_not_production_ready_section() -> str:
    return "\n".join(
        [
            "## 9. Why It Is Not Production-Ready",
            "",
            "- `production_ready = false`.",
            "- This is research documentation, not a runtime signal specification.",
            "- No live paper-tracking process has been designed for this exact candidate.",
            "- No longer-history validation or walk-forward robustness result is attached yet.",
            "- No production signal export should be created or updated from this replay.",
            "- This is not evidence that QQQ buy-and-hold has been beaten.",
        ]
    )


def _current_decision_section(decision: dict[str, object]) -> str:
    decision_label = decision.get(
        "decision",
        "keep_as_best_qqq_defensive_timing_candidate",
    )
    research_status = decision.get(
        "research_status",
        "research_only_defensive_candidate",
    )
    return "\n".join(
        [
            "## 10. Current Decision",
            "",
            f"- Decision: `{decision_label}`.",
            f"- Research status: `{research_status}`.",
            "- Keep as best QQQ defensive timing candidate.",
            "- Keep research-only.",
            "- Do not mark production-ready.",
        ]
    )


def _next_validation_steps_section() -> str:
    return "\n".join(
        [
            "## 11. Next Validation Steps",
            "",
            "- Longer-history validation if data allows.",
            "- Walk-forward robustness.",
            "- Live paper-tracking design.",
            "- Transaction and slippage sensitivity.",
            "- No production promotion yet.",
        ]
    )


def _research_path_section() -> str:
    return "\n".join(
        [
            "## 1. Summary Of The Research Path",
            "",
            "- ML ETF rotation struggled versus QQQ.",
            (
                "- The QQQ-core overlay was diagnostic because it showed how much of "
                "the gap was QQQ exposure, but it is not the desired final direction."
            ),
            (
                "- Pure weighted allocation did not improve the QQQ gap enough to "
                "justify continuing that path now."
            ),
            "- The QQQ timing path produced a defensive candidate worth preserving for validation.",
        ]
    )


def _candidate_decision_section(
    summary: dict[str, object],
    decision: dict[str, object],
) -> str:
    return "\n".join(
        [
            "## 2. Current Candidate Decision",
            "",
            f"- Candidate: `{summary.get('strategy_name')}`.",
            f"- Decision: `{decision.get('decision')}`.",
            f"- Research status: `{decision.get('research_status')}`.",
            "- Production-ready: `false`.",
            (
                "- Full-period result: strategy total return "
                f"`{_format_number(summary.get('total_return'))}` vs QQQ buy-and-hold "
                f"`{_format_number(summary.get('qqq_buy_hold_total_return'))}`."
            ),
            (
                "- Risk-adjusted trade-off: return-to-drawdown "
                f"`{_format_number(summary.get('return_to_drawdown'))}` vs QQQ "
                f"`{_format_number(summary.get('qqq_buy_hold_return_to_drawdown'))}`."
            ),
        ]
    )


def _why_qqq_benchmark_section(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "## 3. Why QQQ Buy-And-Hold Remains The Primary Benchmark",
            "",
            (
                "- The research question is whether any ETF rotation or timing method "
                "improves the QQQ investor experience."
            ),
            "- This candidate still trails QQQ buy-and-hold on raw full-period total return.",
            (
                "- The observed strategy-vs-QQQ gap is "
                f"`{_format_number(summary.get('strategy_vs_qqq'))}`, so the bar has not moved."
            ),
            (
                "- SPY and cash are useful references, but they are not the hurdle "
                "rate for a QQQ defensive timing candidate."
            ),
        ]
    )


def _why_candidate_is_useful_section(
    summary: dict[str, object],
    subperiod_summary: pd.DataFrame,
) -> str:
    helpful_periods = _positive_strategy_vs_qqq_periods(subperiod_summary)
    period_text = (
        ", ".join(helpful_periods)
        if helpful_periods
        else "no QQQ-positive subperiods"
    )
    return "\n".join(
        [
            "## 4. Why This Candidate Is Useful Anyway",
            "",
            (
                "- It gives the research program a simple, interpretable defensive "
                "QQQ timing baseline."
            ),
            (
                "- It nearly captured QQQ return while allocating to SPY about "
                f"`{_format_number(summary.get('exposure_rate_to_spy'))}` of the time."
            ),
            (
                "- It improved the full-period drawdown profile and return-to-drawdown, "
                "which is the intended defensive use case."
            ),
            f"- It was most useful in: {period_text}.",
            "- It is a concrete candidate for longer-history and walk-forward validation.",
        ]
    )


def _explicit_no_go_section() -> str:
    return "\n".join(
        [
            "## 5. Explicit No-Go Decisions",
            "",
            "- No QQQ-core continuation for now.",
            "- No pure weighted allocation continuation for now.",
            "- No tree_screen from this result.",
            "- No MLP from this result.",
            "- No benchmark_gap_features_v4 from this result.",
            "- No production export.",
            "- No S3 workflow.",
            "- Do not modify the Market Data repository.",
        ]
    )


def _recommended_next_task_section() -> str:
    return "\n".join(
        [
            "## 6. Recommended Next Task",
            "",
            "- Run longer-history validation if data allows.",
            (
                "- Add walk-forward robustness validation for this exact QQQ "
                "defensive timing candidate."
            ),
            (
                "- Keep the result research-only until those validations and "
                "paper-tracking design exist."
            ),
        ]
    )


def _subperiod_interpretation(subperiod_summary: pd.DataFrame) -> str:
    period_lookup = {
        "covid_crash_recovery": (
            "COVID crash/recovery: strategy trails QQQ slightly and has worse "
            "drawdown than QQQ."
        ),
        "bear_2022": "Bear 2022: strategy beats QQQ materially by reducing QQQ exposure.",
        "ai_bull_2023_2024": (
            "AI bull 2023-2024: strategy trails QQQ because QQQ bull market "
            "strength dominates."
        ),
        "recent_2025_2026": (
            "Recent 2025-2026: strategy trails QQQ mildly while still beating SPY."
        ),
    }
    available = set(subperiod_summary.get("period_name", pd.Series(dtype=str)).astype(str))
    lines = [
        f"- {text}"
        for period_name, text in period_lookup.items()
        if period_name in available
    ]
    return "\n".join(lines)


def _positive_strategy_vs_qqq_periods(subperiod_summary: pd.DataFrame) -> list[str]:
    if subperiod_summary.empty or "strategy_vs_qqq" not in subperiod_summary.columns:
        return []
    stress = subperiod_summary.loc[
        subperiod_summary["period_name"].astype(str).ne("full")
    ].copy()
    stress["strategy_vs_qqq"] = pd.to_numeric(stress["strategy_vs_qqq"], errors="coerce")
    positive = stress.loc[stress["strategy_vs_qqq"] > 0.0]
    return positive["period_name"].astype(str).tolist()


def _weakest_strategy_vs_qqq_period(subperiod_summary: pd.DataFrame) -> dict[str, object]:
    if subperiod_summary.empty or "strategy_vs_qqq" not in subperiod_summary.columns:
        return {}
    stress = subperiod_summary.loc[
        subperiod_summary["period_name"].astype(str).ne("full")
    ].copy()
    if stress.empty:
        stress = subperiod_summary.copy()
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


def _two_column_table(
    first_header: str,
    second_header: str,
    rows: Iterable[tuple[str, object]],
) -> str:
    return _markdown_table(
        pd.DataFrame(
            [
                {first_header: label, second_header: _format_object(value)}
                for label, value in rows
            ]
        )
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_escape_markdown(str(value)) for value in row) + " |")
    return "\n".join(lines)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(_format_object)
    return formatted


def _format_object(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, float | int | np.floating | np.integer) and not isinstance(value, bool):
        return _format_number(value)
    return str(value)


def _format_number(value: object) -> str:
    number = _as_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.4f}"


def _format_integer(value: object, *, fallback: str = "n/a") -> str:
    number = _as_float(value)
    if not np.isfinite(number):
        return fallback
    return str(int(round(number)))


def _format_multiplier(value: float) -> str:
    if not np.isfinite(value):
        return "1.005"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create research-only model card and decision docs for the selected "
            "QQQ defensive timing replay."
        )
    )
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


create_qqq_defensive_timing_model_card_from_args = (
    run_create_qqq_defensive_timing_model_card_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_create_qqq_defensive_timing_model_card_from_args())
