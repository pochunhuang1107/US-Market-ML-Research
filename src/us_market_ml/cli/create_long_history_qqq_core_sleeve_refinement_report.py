"""Create the long-history QQQ core-sleeve refinement report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_refinement import (
    DAILY_DRAWDOWN_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    RESULTS_NAME,
    ROLLING_WINDOWS_NAME,
    SUBPERIODS_NAME,
    build_refined_core_sleeve_decision_matrix,
    generate_refinement_weight_grid,
)
from us_market_ml.long_history_spy_core_sleeve import (
    build_spy_core_sleeve_decision_matrix,
)

REPORT_OUTPUT_PREFIX = "long_history_qqq_core_sleeve_refinement"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_refinement")
DEFAULT_PREVIOUS_SPY_CORE_SLEEVE_RESULTS = Path(
    "artifacts/sweeps/long_history_qqq_spy_core_sleeve/"
    "long_history_qqq_spy_core_sleeve_results.csv"
)
SUMMARY_NAME = f"{REPORT_OUTPUT_PREFIX}_summary.csv"
REPORT_DECISION_MATRIX_NAME = f"{REPORT_OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{REPORT_OUTPUT_PREFIX}_report.md"

DECISION_RANK = {
    "best_refined_core_sleeve_candidate": 0,
    "balanced_refined_core_sleeve_candidate": 1,
    "return_candidate": 2,
    "defensive_candidate": 3,
    "reject_deprioritize": 4,
}
OBJECTIVE_COLUMNS = [
    "objective_best_total_return_drawdown_controlled",
    "objective_best_return_to_drawdown",
    "objective_best_total_return_drawdown_improvement",
    "objective_best_balanced_candidate",
]


@dataclass(frozen=True)
class LongHistoryQQQCoreSleeveRefinementReportData:
    """CSV-backed data for the refined core-sleeve report."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    decision_matrix: pd.DataFrame
    previous_spy_core_sleeve_results: pd.DataFrame


def run_long_history_qqq_core_sleeve_refinement_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the refinement report artifacts."""

    args = _parse_args(argv)
    report_data = read_long_history_qqq_core_sleeve_refinement_artifacts(
        sweep_output_dir=args.sweep_output_dir,
        previous_spy_core_sleeve_results=args.previous_spy_core_sleeve_results,
    )
    decision_matrix = build_refined_core_sleeve_decision_matrix(report_data.results)
    summary = build_long_history_qqq_core_sleeve_refinement_summary(decision_matrix)
    rendered_data = LongHistoryQQQCoreSleeveRefinementReportData(
        results=report_data.results,
        daily_equity=report_data.daily_equity,
        daily_drawdown=report_data.daily_drawdown,
        subperiods=report_data.subperiods,
        rolling_windows=report_data.rolling_windows,
        decision_matrix=decision_matrix,
        previous_spy_core_sleeve_results=report_data.previous_spy_core_sleeve_results,
    )
    report = render_long_history_qqq_core_sleeve_refinement_markdown(rendered_data)

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

    print("US Market ML long-history QQQ core-sleeve refinement report")
    print(f"sweep output dir: {args.sweep_output_dir}")
    print(f"report output dir: {args.output_dir}")
    print(f"summary path: {args.output_dir / SUMMARY_NAME}")
    print(f"decision matrix path: {args.output_dir / REPORT_DECISION_MATRIX_NAME}")
    print(f"report path: {report_path}")
    print(f"configs: {len(decision_matrix)}")
    return 0


def read_long_history_qqq_core_sleeve_refinement_artifacts(
    *,
    sweep_output_dir: Path = DEFAULT_OUTPUT_DIR,
    previous_spy_core_sleeve_results: Path = DEFAULT_PREVIOUS_SPY_CORE_SLEEVE_RESULTS,
) -> LongHistoryQQQCoreSleeveRefinementReportData:
    """Read refinement CSV artifacts and optional prior coarse results."""

    return LongHistoryQQQCoreSleeveRefinementReportData(
        results=_read_required_csv(sweep_output_dir / RESULTS_NAME),
        daily_equity=_read_required_csv(sweep_output_dir / DAILY_EQUITY_NAME),
        daily_drawdown=_read_required_csv(sweep_output_dir / DAILY_DRAWDOWN_NAME),
        subperiods=_read_required_csv(sweep_output_dir / SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(sweep_output_dir / ROLLING_WINDOWS_NAME),
        decision_matrix=_read_required_csv(sweep_output_dir / DECISION_MATRIX_NAME),
        previous_spy_core_sleeve_results=_read_optional_csv(previous_spy_core_sleeve_results),
    )


def build_long_history_qqq_core_sleeve_refinement_summary(
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return a ranked summary table for the report output directory."""

    if decision_matrix.empty:
        return pd.DataFrame()
    sort = _ranked_decision(decision_matrix)
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
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
        "positive_vs_qqq_window_rate",
        "worst_strategy_vs_qqq",
        "selected_objectives",
        "decision",
        "production_ready",
    ]
    return sort.loc[:, cols].reset_index(drop=True)


def render_long_history_qqq_core_sleeve_refinement_markdown(
    report_data: LongHistoryQQQCoreSleeveRefinementReportData,
) -> str:
    """Render the refined core-sleeve Markdown report."""

    decision = build_refined_core_sleeve_decision_matrix(report_data.results)
    lines = [
        "# Long-History QQQ Core-Sleeve Refinement Report",
        "",
        "This is a research-only portfolio-construction validation study. It does "
        "not mark anything production-ready and does not create production signal exports.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(decision),
        "",
        "## 2. Why refinement was needed",
        "",
        _why_refinement_was_needed(),
        "",
        "## 3. Prior coarse SPY core-sleeve recap",
        "",
        _prior_coarse_recap(report_data.previous_spy_core_sleeve_results),
        "",
        "## 4. Accounting modes tested",
        "",
        _accounting_modes_tested(decision),
        "",
        "## 5. Fine core-weight grid",
        "",
        _fine_core_weight_grid(decision),
        "",
        "## 6. Full-period results",
        "",
        _markdown_table(_format_for_markdown(_full_period_results(decision))),
        "",
        "## 7. Best drawdown-controlled results",
        "",
        _markdown_table(_format_for_markdown(_best_drawdown_controlled(decision))),
        "",
        "## 8. Best return-to-drawdown results",
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
        "## 11. Candidate_0001 refined result",
        "",
        _candidate_refined_result(decision, "candidate_0001"),
        "",
        "## 12. Candidate_0003 refined result",
        "",
        _candidate_refined_result(decision, "candidate_0003"),
        "",
        "## 13. Recommendation",
        "",
        _recommendation(decision),
        "",
        "## 14. Caveats",
        "",
        _caveats(decision),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(decision: pd.DataFrame) -> str:
    if decision.empty:
        return "No refined core-sleeve rows were available."
    decision_counts = decision["decision"].value_counts().to_dict()
    candidate_count = decision["candidate_id"].nunique()
    accounting_count = decision["portfolio_accounting_mode"].nunique()
    production_ready = int(_as_bool(decision["production_ready"]).sum())
    best_count = int(decision["decision"].eq("best_refined_core_sleeve_candidate").sum())
    balanced_count = int(
        decision["decision"].eq("balanced_refined_core_sleeve_candidate").sum()
    )
    leader = _top_candidate(decision)
    leader_text = (
        f" Current top refined row is `{leader['candidate_id']}` at SPY core "
        f"`{leader['spy_core_weight']}` using `{leader['portfolio_accounting_mode']}`."
        if leader is not None
        else ""
    )
    return (
        f"Evaluated `{len(decision)}` refined portfolios across `{candidate_count}` "
        f"candidate sleeves and `{accounting_count}` accounting modes, with QQQ "
        "buy-and-hold as the primary benchmark. "
        f"`{best_count}` rows met the strict refined core-sleeve rule and "
        f"`{balanced_count}` met the balanced rule. Decision counts: "
        f"`{decision_counts}`. Production-ready candidates: `{production_ready}`."
        f"{leader_text}"
    )


def _why_refinement_was_needed() -> str:
    return (
        "The first SPY core-sleeve pass used a coarse grid and daily rebalanced fixed "
        "weights. This refinement keeps the scope compact and tests whether the "
        "result survives finer SPY weights plus monthly, quarterly, and candidate "
        "rebalance-aligned drift accounting."
    )


def _prior_coarse_recap(previous: pd.DataFrame) -> str:
    if previous.empty:
        return "Prior SPY core-sleeve results were not provided to the report command."
    previous = _previous_with_decisions(previous)
    cols = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "rebalance_policy",
        "total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "return_to_drawdown",
        "decision",
    ]
    objective_cols = [
        "objective_best_total_return_drawdown_controlled",
        "objective_best_return_to_drawdown",
        "objective_best_qqq_capture_drawdown_improvement",
        "objective_best_balanced_candidate",
    ]
    for column in [*cols, *objective_cols]:
        if column not in previous.columns:
            previous[column] = pd.NA
    focused = previous.loc[
        previous["candidate_id"].astype(str).isin(["candidate_0001", "candidate_0003"])
    ].copy()
    selected_pieces: list[pd.DataFrame] = []
    for _, group in focused.groupby("candidate_id", sort=False):
        objective_mask = pd.Series(False, index=group.index)
        for column in objective_cols:
            objective_mask = objective_mask | _as_bool(group[column])
        objective_rows = group.loc[objective_mask]
        if not objective_rows.empty:
            selected_pieces.append(objective_rows)
        best_rows = group.loc[
            group["decision"].astype(str).isin(
                ["best_core_sleeve_candidate", "balanced_core_sleeve_candidate"]
            )
        ]
        if not best_rows.empty:
            selected_pieces.append(
                best_rows.sort_values(
                    ["strategy_vs_qqq", "return_to_drawdown"],
                    ascending=[False, False],
                    kind="mergesort",
                ).head(2)
            )
    if selected_pieces:
        focused = pd.concat(selected_pieces, ignore_index=True).drop_duplicates(
            subset=["candidate_id", "spy_core_weight", "candidate_sleeve_weight"],
            keep="first",
        )
    focused = focused.sort_values(
        ["candidate_id", "strategy_vs_qqq", "return_to_drawdown"],
        ascending=[True, False, False],
        kind="mergesort",
    ).head(8)
    return _markdown_table(_format_for_markdown(focused.loc[:, cols]))


def _accounting_modes_tested(decision: pd.DataFrame) -> str:
    if decision.empty:
        return "_No accounting modes._"
    rows = (
        decision.groupby("portfolio_accounting_mode", sort=False)
        .agg(
            configs=("portfolio_accounting_mode", "size"),
            best_total_return=("total_return", "max"),
            best_strategy_vs_qqq=("strategy_vs_qqq", "max"),
            best_drawdown_improvement=("drawdown_improvement_vs_qqq", "max"),
            best_return_to_drawdown=("return_to_drawdown", "max"),
        )
        .reset_index()
    )
    return _markdown_table(_format_for_markdown(rows))


def _fine_core_weight_grid(decision: pd.DataFrame) -> str:
    if decision.empty:
        return _markdown_table(generate_refinement_weight_grid())
    rows = (
        decision.loc[:, ["candidate_id", "spy_core_weight", "candidate_sleeve_weight"]]
        .drop_duplicates()
        .sort_values(["candidate_id", "spy_core_weight"], kind="mergesort")
    )
    return _markdown_table(_format_for_markdown(rows))


def _full_period_results(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "standalone_candidate_total_return",
        "strategy_vs_qqq",
        "qqq_return_capture",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "positive_vs_qqq_window_rate",
        "worst_strategy_vs_qqq",
        "decision",
    ]
    return decision.loc[:, cols].sort_values(
        ["candidate_id", "portfolio_accounting_mode", "spy_core_weight"],
        kind="mergesort",
    )


def _best_drawdown_controlled(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    controlled = decision.loc[
        pd.to_numeric(decision["drawdown_improvement_vs_qqq"], errors="coerce").ge(0.0)
    ]
    if controlled.empty:
        return pd.DataFrame()
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "qqq_return_capture",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "selected_objectives",
        "decision",
    ]
    return controlled.sort_values(
        ["total_return", "qqq_return_capture", "return_to_drawdown"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(10).loc[:, cols]


def _best_return_to_drawdown(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "max_drawdown",
        "return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "qqq_return_capture",
        "selected_objectives",
        "decision",
    ]
    return decision.sort_values(
        ["return_to_drawdown", "total_return"],
        ascending=[False, False],
        kind="mergesort",
    ).head(10).loc[:, cols]


def _subperiod_robustness(subperiods: pd.DataFrame) -> str:
    if subperiods.empty:
        return "_No subperiod rows._"
    grouped = (
        subperiods.groupby(["candidate_id", "portfolio_accounting_mode", "period_name"], sort=False)
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
    summary_cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "selected_objectives",
        "decision",
    ]
    summary = decision.loc[:, summary_cols].sort_values(
        ["positive_vs_qqq_window_rate", "worst_strategy_vs_qqq"],
        ascending=[False, False],
        kind="mergesort",
    ).head(12)
    detail = ""
    if not rolling_windows.empty:
        worst = rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(8)
        detail = "\n\nWorst rolling windows:\n\n" + _markdown_table(
            _format_for_markdown(worst)
        )
    return _markdown_table(_format_for_markdown(summary)) + detail


def _candidate_refined_result(decision: pd.DataFrame, candidate_id: str) -> str:
    family = decision.loc[decision["candidate_id"].astype(str).eq(candidate_id)]
    if family.empty:
        return "_No rows._"
    display = _ranked_decision(family).loc[
        :,
        [
            "portfolio_accounting_mode",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "total_return",
            "strategy_vs_qqq",
            "qqq_return_capture",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
            "positive_vs_qqq_window_rate",
            "worst_strategy_vs_qqq",
            "selected_objectives",
            "decision",
        ],
    ]
    return _markdown_table(_format_for_markdown(display))


def _recommendation(decision: pd.DataFrame) -> str:
    if decision.empty:
        return (
            "- Choose refined core-sleeve candidate for deep dive: `no rows`\n"
            "- Continue standalone candidate: `no rows`\n"
            "- Test QQQ core instead: `no rows`\n"
            "- MLP batch: `no`\n"
            "- Production promotion: `no`"
        )
    best = decision.loc[decision["decision"].eq("best_refined_core_sleeve_candidate")]
    balanced = decision.loc[
        decision["decision"].eq("balanced_refined_core_sleeve_candidate")
    ]
    target = pd.concat([best, balanced], ignore_index=True)
    if not target.empty:
        target = _ranked_decision(target).head(1)
        row = target.iloc[0]
        balanced_objective = decision.loc[_as_bool(decision["objective_best_balanced_candidate"])]
        balanced_text = ""
        if not balanced_objective.empty:
            balanced_row = _ranked_decision(balanced_objective).iloc[0]
            if (
                str(balanced_row["candidate_id"]) != str(row["candidate_id"])
                or float(balanced_row["spy_core_weight"]) != float(row["spy_core_weight"])
                or str(balanced_row["portfolio_accounting_mode"])
                != str(row["portfolio_accounting_mode"])
            ):
                balanced_text = (
                    f"; balanced check `{balanced_row['candidate_id']}` at SPY core "
                    f"`{balanced_row['spy_core_weight']}` using "
                    f"`{balanced_row['portfolio_accounting_mode']}`"
                )
        target_text = (
            f"`yes`: `{row['candidate_id']}` at SPY core `{row['spy_core_weight']}` / "
            f"candidate sleeve `{row['candidate_sleeve_weight']}` using "
            f"`{row['portfolio_accounting_mode']}`{balanced_text}"
        )
    else:
        target_text = "`no`: no refined SPY sleeve row cleared the balanced rule"
    standalone = decision.loc[pd.to_numeric(decision["spy_core_weight"], errors="coerce").eq(0.0)]
    continue_standalone = "yes" if standalone["strategy_vs_qqq"].gt(0.0).any() else "no"
    test_qqq_core = "yes" if target.empty else "optional after SPY refinement"
    return (
        f"- Choose refined core-sleeve candidate for deep dive: {target_text}\n"
        f"- Continue standalone candidate: `{continue_standalone}`\n"
        f"- Test QQQ core instead: `{test_qqq_core}`\n"
        "- MLP batch: `no`\n"
        "- Production promotion: `no`"
    )


def _caveats(decision: pd.DataFrame) -> str:
    modes = (
        ", ".join(sorted(decision["portfolio_accounting_mode"].astype(str).unique()))
        if not decision.empty
        else "none"
    )
    return (
        "- Research-only; `production_ready` is always false.\n"
        "- Tiingo-only source artifacts.\n"
        "- No production signal export, model card, S3/cloud write, or Market Data repo change.\n"
        "- QQQ buy-and-hold remains the primary benchmark.\n"
        f"- Accounting modes are explicitly stated per row: `{modes}`."
    )


def _top_candidate(decision: pd.DataFrame) -> pd.Series | None:
    if decision.empty:
        return None
    ranked = _ranked_decision(decision)
    return ranked.iloc[0] if not ranked.empty else None


def _ranked_decision(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return decision.copy()
    out = decision.copy()
    for column in OBJECTIVE_COLUMNS:
        if column not in out.columns:
            out[column] = False
    objective_score = out.loc[:, OBJECTIVE_COLUMNS].apply(
        lambda row: int(_as_bool(row).sum()),
        axis=1,
    )
    return out.assign(
        _decision_rank=out["decision"].map(DECISION_RANK).fillna(9),
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


def _previous_with_decisions(previous: pd.DataFrame) -> pd.DataFrame:
    if previous.empty:
        return previous
    if "decision" in previous.columns and previous["decision"].notna().any():
        return previous
    try:
        return build_spy_core_sleeve_decision_matrix(previous)
    except (KeyError, ValueError):
        return previous.copy()


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


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the long-history QQQ core-sleeve refinement report."
    )
    parser.add_argument("--sweep-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_OUTPUT_DIR)
    parser.add_argument(
        "--previous-spy-core-sleeve-results",
        type=Path,
        default=DEFAULT_PREVIOUS_SPY_CORE_SLEEVE_RESULTS,
    )
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_core_sleeve_refinement_report_from_args())
