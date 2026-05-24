"""Create the long-history QQQ SPY core-sleeve report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_market_ml.long_history_spy_core_sleeve import (
    DAILY_DRAWDOWN_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    RESULTS_NAME,
    ROLLING_WINDOWS_NAME,
    SUBPERIODS_NAME,
    build_spy_core_sleeve_decision_matrix,
    generate_core_sleeve_weight_grid,
)

REPORT_OUTPUT_PREFIX = "long_history_qqq_spy_core_sleeve"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_spy_core_sleeve")
SUMMARY_NAME = f"{REPORT_OUTPUT_PREFIX}_summary.csv"
REPORT_DECISION_MATRIX_NAME = f"{REPORT_OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{REPORT_OUTPUT_PREFIX}_report.md"


@dataclass(frozen=True)
class LongHistoryQQQSpyCoreSleeveReportData:
    """CSV-backed data for the SPY core-sleeve report."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    decision_matrix: pd.DataFrame


def run_long_history_qqq_spy_core_sleeve_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the SPY core-sleeve report artifacts."""

    args = _parse_args(argv)
    report_data = read_long_history_qqq_spy_core_sleeve_artifacts(args.sweep_output_dir)
    decision_matrix = build_spy_core_sleeve_decision_matrix(report_data.results)
    summary = build_long_history_qqq_spy_core_sleeve_summary(decision_matrix)
    rendered_data = LongHistoryQQQSpyCoreSleeveReportData(
        results=report_data.results,
        daily_equity=report_data.daily_equity,
        daily_drawdown=report_data.daily_drawdown,
        subperiods=report_data.subperiods,
        rolling_windows=report_data.rolling_windows,
        decision_matrix=decision_matrix,
    )
    report = render_long_history_qqq_spy_core_sleeve_markdown(rendered_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / SUMMARY_NAME, index=False)
    decision_matrix.to_csv(args.output_dir / REPORT_DECISION_MATRIX_NAME, index=False)
    report_path = (
        args.report_path
        if args.report_path is not None
        else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("US Market ML long-history QQQ SPY core-sleeve report")
    print(f"sweep output dir: {args.sweep_output_dir}")
    print(f"report output dir: {args.output_dir}")
    print(f"summary path: {args.output_dir / SUMMARY_NAME}")
    print(f"decision matrix path: {args.output_dir / REPORT_DECISION_MATRIX_NAME}")
    print(f"report path: {report_path}")
    print(f"configs: {len(decision_matrix)}")
    return 0


def read_long_history_qqq_spy_core_sleeve_artifacts(
    sweep_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> LongHistoryQQQSpyCoreSleeveReportData:
    """Read SPY core-sleeve CSV artifacts."""

    return LongHistoryQQQSpyCoreSleeveReportData(
        results=_read_required_csv(sweep_output_dir / RESULTS_NAME),
        daily_equity=_read_required_csv(sweep_output_dir / DAILY_EQUITY_NAME),
        daily_drawdown=_read_required_csv(sweep_output_dir / DAILY_DRAWDOWN_NAME),
        subperiods=_read_required_csv(sweep_output_dir / SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(sweep_output_dir / ROLLING_WINDOWS_NAME),
        decision_matrix=_read_required_csv(sweep_output_dir / DECISION_MATRIX_NAME),
    )


def build_long_history_qqq_spy_core_sleeve_summary(
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return a ranked summary table for the report output directory."""

    if decision_matrix.empty:
        return pd.DataFrame()
    decision_rank = {
        "best_core_sleeve_candidate": 0,
        "balanced_core_sleeve_candidate": 1,
        "return_candidate": 2,
        "defensive_candidate": 3,
        "reject_deprioritize": 4,
    }
    objective_score = decision_matrix.loc[:, [
        "objective_best_total_return_drawdown_controlled",
        "objective_best_return_to_drawdown",
        "objective_best_qqq_capture_drawdown_improvement",
        "objective_best_balanced_candidate",
    ]].apply(lambda row: int(_as_bool(row).sum()), axis=1)
    sort = decision_matrix.assign(
        _decision_rank=decision_matrix["decision"].map(decision_rank).fillna(9),
        _objective_score=objective_score,
    ).sort_values(
        [
            "_decision_rank",
            "_objective_score",
            "strategy_vs_qqq",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
        ],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    cols = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_name",
        "total_return",
        "strategy_vs_qqq",
        "qqq_return_capture",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "selected_objectives",
        "decision",
        "production_ready",
    ]
    return sort.loc[:, cols].reset_index(drop=True)


def render_long_history_qqq_spy_core_sleeve_markdown(
    report_data: LongHistoryQQQSpyCoreSleeveReportData,
) -> str:
    """Render the SPY core-sleeve Markdown report."""

    decision = build_spy_core_sleeve_decision_matrix(report_data.results)
    lines = [
        "# Long-History QQQ SPY Core-Sleeve Report",
        "",
        "This is a research-only SPY core plus candidate sleeve study. It does not "
        "mark anything production-ready and does not create production signal exports.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(decision),
        "",
        "## 2. Why SPY core + candidate sleeve",
        "",
        _why_core_sleeve(),
        "",
        "## 3. Candidate source recap",
        "",
        _candidate_source_recap(decision),
        "",
        "## 4. Core weight grid",
        "",
        _markdown_table(_format_for_markdown(generate_core_sleeve_weight_grid())),
        "",
        "## 5. Full-period comparison versus QQQ/SPY/standalone candidate",
        "",
        _markdown_table(_format_for_markdown(_full_period_comparison(decision))),
        "",
        "## 6. Best total-return portfolios",
        "",
        _markdown_table(_format_for_markdown(_best_total_return(decision))),
        "",
        "## 7. Best drawdown-controlled portfolios",
        "",
        _markdown_table(_format_for_markdown(_best_drawdown_controlled(decision))),
        "",
        "## 8. Best return-to-drawdown portfolios",
        "",
        _markdown_table(_format_for_markdown(_best_return_to_drawdown(decision))),
        "",
        "## 9. Subperiod robustness",
        "",
        _subperiod_robustness(report_data.subperiods),
        "",
        "## 10. Rolling-window robustness",
        "",
        _rolling_window_robustness(decision, report_data.rolling_windows),
        "",
        "## 11. Candidate_0001 sleeve results",
        "",
        _candidate_sleeve_result(decision, "candidate_0001"),
        "",
        "## 12. Candidate_0003 sleeve results",
        "",
        _candidate_sleeve_result(decision, "candidate_0003"),
        "",
        "## 13. Recommendation",
        "",
        _recommendation(decision),
        "",
        "## 14. Caveats",
        "",
        _caveats(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(decision: pd.DataFrame) -> str:
    if decision.empty:
        return "No SPY core-sleeve rows were available."
    decision_counts = decision["decision"].value_counts().to_dict()
    candidate_count = decision["candidate_id"].nunique()
    production_ready = int(_as_bool(decision["production_ready"]).sum())
    best_count = int(decision["decision"].eq("best_core_sleeve_candidate").sum())
    balanced_count = int(decision["decision"].eq("balanced_core_sleeve_candidate").sum())
    return (
        f"Evaluated `{len(decision)}` fixed-weight portfolios across `{candidate_count}` "
        "candidate sleeves, with QQQ buy-and-hold as the primary benchmark and SPY "
        "buy-and-hold as the core reference. "
        f"`{best_count}` rows met the strict best core-sleeve rule and `{balanced_count}` "
        f"met the balanced rule. Decision counts: `{decision_counts}`. "
        f"Production-ready candidates: `{production_ready}`."
    )


def _why_core_sleeve() -> str:
    return (
        "The candidate signals remain useful as active sleeves, but their standalone "
        "paths are concentrated. This study treats concentration as a sizing problem: "
        "SPY provides a stable core while the QQQ-relative candidate return stream is "
        "kept as the active sleeve. Portfolio weights are rebalanced daily in this "
        "first pass."
    )


def _candidate_source_recap(decision: pd.DataFrame) -> str:
    if decision.empty:
        return "_No source rows._"
    rows = []
    for candidate_id, group in decision.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        standalone = group.loc[pd.to_numeric(group["spy_core_weight"], errors="coerce").eq(0.0)]
        source = standalone.iloc[0] if not standalone.empty else first
        rows.append(
            {
                "candidate_id": candidate_id,
                "standalone_total_return": source.get("standalone_candidate_total_return"),
                "standalone_max_drawdown": source.get("standalone_candidate_max_drawdown"),
                "source_strategy_vs_qqq": first.get("source_strategy_vs_qqq"),
                "source_return_to_drawdown": first.get("source_return_to_drawdown"),
                "source_positive_vs_qqq_window_rate": first.get(
                    "source_positive_vs_qqq_window_rate"
                ),
                "source_top_symbol": first.get("source_top_symbol"),
                "source_top_symbol_exposure_pct": first.get("source_top_symbol_exposure_pct"),
                "source_top3_symbol_exposure_pct": first.get(
                    "source_top3_symbol_exposure_pct"
                ),
                "source_decision": first.get("source_decision"),
            }
        )
    return _markdown_table(_format_for_markdown(pd.DataFrame(rows)))


def _full_period_comparison(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    cols = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_name",
        "total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "standalone_candidate_total_return",
        "strategy_vs_qqq",
        "qqq_return_capture",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "decision",
    ]
    return decision.loc[:, cols].sort_values(
        ["candidate_id", "spy_core_weight"],
        kind="mergesort",
    )


def _best_total_return(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    return decision.sort_values("total_return", ascending=False, kind="mergesort").head(8).loc[
        :,
        [
            "candidate_id",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "total_return",
            "strategy_vs_qqq",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
            "selected_objectives",
            "decision",
        ],
    ]


def _best_drawdown_controlled(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    controlled = decision.loc[
        pd.to_numeric(decision["drawdown_improvement_vs_qqq"], errors="coerce").ge(0.0)
    ]
    if controlled.empty:
        return pd.DataFrame()
    return controlled.sort_values(
        ["total_return", "qqq_return_capture", "return_to_drawdown"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(8).loc[
        :,
        [
            "candidate_id",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "total_return",
            "qqq_return_capture",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
            "selected_objectives",
            "decision",
        ],
    ]


def _best_return_to_drawdown(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    return decision.sort_values(
        ["return_to_drawdown", "total_return"],
        ascending=[False, False],
        kind="mergesort",
    ).head(8).loc[
        :,
        [
            "candidate_id",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "total_return",
            "max_drawdown",
            "return_to_drawdown",
            "return_to_drawdown_vs_qqq",
            "qqq_return_capture",
            "selected_objectives",
            "decision",
        ],
    ]


def _subperiod_robustness(subperiods: pd.DataFrame) -> str:
    if subperiods.empty:
        return "_No subperiod rows._"
    grouped = (
        subperiods.groupby(["candidate_id", "period_name"], sort=False)
        .agg(
            best_strategy_vs_qqq=("strategy_vs_qqq", "max"),
            median_strategy_vs_qqq=("strategy_vs_qqq", "median"),
            best_drawdown_improvement=("drawdown_improvement_vs_qqq", "max"),
            rows=("strategy_vs_qqq", "size"),
        )
        .reset_index()
    )
    return _markdown_table(_format_for_markdown(grouped))


def _rolling_window_robustness(decision: pd.DataFrame, rolling_windows: pd.DataFrame) -> str:
    if decision.empty:
        return "_No rolling-window rows._"
    summary = decision.loc[
        :,
        [
            "candidate_id",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "positive_vs_qqq_window_rate",
            "qqq_underperformance_rate",
            "worst_strategy_vs_qqq",
            "selected_objectives",
            "decision",
        ],
    ].sort_values(
        ["positive_vs_qqq_window_rate", "worst_strategy_vs_qqq"],
        ascending=[False, False],
        kind="mergesort",
    ).head(10)
    detail = ""
    if not rolling_windows.empty:
        worst = rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(5)
        detail = "\n\nWorst rolling windows:\n\n" + _markdown_table(_format_for_markdown(worst))
    return _markdown_table(_format_for_markdown(summary)) + detail


def _candidate_sleeve_result(decision: pd.DataFrame, candidate_id: str) -> str:
    family = decision.loc[decision["candidate_id"].astype(str).eq(candidate_id)]
    if family.empty:
        return "_No rows._"
    display = family.sort_values(
        ["decision", "strategy_vs_qqq", "return_to_drawdown"],
        ascending=[True, False, False],
        kind="mergesort",
    ).loc[
        :,
        [
            "spy_core_weight",
            "candidate_sleeve_weight",
            "total_return",
            "strategy_vs_qqq",
            "qqq_return_capture",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
            "selected_objectives",
            "decision",
        ],
    ]
    return _markdown_table(_format_for_markdown(display))


def _recommendation(decision: pd.DataFrame) -> str:
    if decision.empty:
        return (
            "- Rich replay best core-sleeve candidate: `no rows`\n"
            "- Target sweep around core weight: `no`\n"
            "- Continue standalone candidate: `no rows`\n"
            "- MLP batch: `no`\n"
            "- Production promotion: `no`"
        )
    best = decision.loc[decision["decision"].eq("best_core_sleeve_candidate")]
    balanced = decision.loc[decision["decision"].eq("balanced_core_sleeve_candidate")]
    target = pd.concat([best, balanced], ignore_index=True)
    target_sweep = "yes" if not target.empty else "no"
    replay = "yes" if not best.empty else ("balanced only" if not balanced.empty else "no")
    standalone = decision.loc[pd.to_numeric(decision["spy_core_weight"], errors="coerce").eq(0.0)]
    continue_standalone = "yes" if standalone["strategy_vs_qqq"].gt(0.0).any() else "no"
    if not target.empty:
        target = target.sort_values(
            ["selected_objectives", "strategy_vs_qqq", "return_to_drawdown"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(1)
        row = target.iloc[0]
        target_text = (
            f"{row['candidate_id']} at SPY core `{row['spy_core_weight']}` / "
            f"candidate sleeve `{row['candidate_sleeve_weight']}`"
        )
    else:
        target_text = "none"
    return (
        f"- Rich replay best core-sleeve candidate: `{replay}` ({target_text})\n"
        f"- Target sweep around core weight: `{target_sweep}`\n"
        f"- Continue standalone candidate: `{continue_standalone}`\n"
        "- MLP batch: `no`\n"
        "- Production promotion: `no`"
    )


def _caveats() -> str:
    return (
        "- Research-only; no production-ready marking.\n"
        "- Tiingo-only source artifacts.\n"
        "- No production signal export and no model card.\n"
        "- QQQ buy-and-hold remains the primary benchmark.\n"
        "- Portfolio weights are daily rebalanced fixed weights, not drifted weights."
    )


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return out


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


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


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the long-history QQQ SPY core-sleeve report."
    )
    parser.add_argument("--sweep-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_spy_core_sleeve_report_from_args())
