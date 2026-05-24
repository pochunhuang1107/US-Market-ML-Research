"""Create the long-history QQQ concentration-control sweep report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.long_history_concentration_control import (
    CANDIDATES_NAME,
    COST_SENSITIVITY_NAME,
    DEFAULT_OUTPUT_DIR,
    RESULTS_NAME,
    ROLLING_WINDOWS_NAME,
    SELECTED_SYMBOLS_NAME,
    SUBPERIODS_NAME,
    SYMBOL_EXPOSURE_NAME,
    build_concentration_control_decision_matrix,
)

REPORT_OUTPUT_PREFIX = "long_history_qqq_concentration_control"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_concentration_control_sweep")
SUMMARY_NAME = f"{REPORT_OUTPUT_PREFIX}_summary.csv"
DECISION_MATRIX_NAME = f"{REPORT_OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{REPORT_OUTPUT_PREFIX}_report.md"


@dataclass(frozen=True)
class LongHistoryQQQConcentrationControlReportData:
    """CSV-backed data for the concentration-control report."""

    candidates: pd.DataFrame
    results: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame


def run_long_history_qqq_concentration_control_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the concentration-control report artifacts."""

    args = _parse_args(argv)
    report_data = read_long_history_qqq_concentration_control_artifacts(args.sweep_output_dir)
    summary = build_long_history_qqq_concentration_control_summary(report_data.results)
    decision_matrix = build_concentration_control_decision_matrix(report_data.results)
    report = render_long_history_qqq_concentration_control_markdown(
        LongHistoryQQQConcentrationControlReportData(
            candidates=report_data.candidates,
            results=decision_matrix,
            selected_symbols=report_data.selected_symbols,
            symbol_exposure=report_data.symbol_exposure,
            subperiods=report_data.subperiods,
            rolling_windows=report_data.rolling_windows,
            cost_sensitivity=report_data.cost_sensitivity,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / SUMMARY_NAME, index=False)
    decision_matrix.to_csv(args.output_dir / DECISION_MATRIX_NAME, index=False)
    report_path = (
        args.report_path
        if args.report_path is not None
        else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("US Market ML long-history QQQ concentration-control report")
    print(f"sweep output dir: {args.sweep_output_dir}")
    print(f"report output dir: {args.output_dir}")
    print(f"summary path: {args.output_dir / SUMMARY_NAME}")
    print(f"decision matrix path: {args.output_dir / DECISION_MATRIX_NAME}")
    print(f"report path: {report_path}")
    print(f"configs: {len(decision_matrix)}")
    return 0


def read_long_history_qqq_concentration_control_artifacts(
    sweep_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> LongHistoryQQQConcentrationControlReportData:
    """Read concentration-control sweep CSV artifacts."""

    return LongHistoryQQQConcentrationControlReportData(
        candidates=_read_required_csv(sweep_output_dir / CANDIDATES_NAME),
        results=_read_required_csv(sweep_output_dir / RESULTS_NAME),
        selected_symbols=_read_required_csv(sweep_output_dir / SELECTED_SYMBOLS_NAME),
        symbol_exposure=_read_required_csv(sweep_output_dir / SYMBOL_EXPOSURE_NAME),
        subperiods=_read_required_csv(sweep_output_dir / SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(sweep_output_dir / ROLLING_WINDOWS_NAME),
        cost_sensitivity=_read_required_csv(sweep_output_dir / COST_SENSITIVITY_NAME),
    )


def build_long_history_qqq_concentration_control_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return a ranked summary table for the report output directory."""

    if results.empty:
        return pd.DataFrame()
    decision_matrix = build_concentration_control_decision_matrix(results)
    decision_rank = {
        "keep_for_deep_dive": 0,
        "robust_candidate": 1,
        "concentration_improved_mixed": 2,
        "reject_deprioritize": 3,
    }
    sort = decision_matrix.assign(
        _decision_rank=decision_matrix["decision"].map(decision_rank).fillna(9),
        _diagnostic=decision_matrix["concentration_rule"].astype(str).str.contains("sanity"),
    )
    sort = sort.sort_values(
        [
            "_decision_rank",
            "_diagnostic",
            "strategy_vs_qqq",
            "positive_vs_qqq_window_rate",
            "top_symbol_exposure_pct",
        ],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    cols = [
        "candidate_id",
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "strategy_vs_qqq",
        "return_to_drawdown",
        "positive_vs_qqq_window_rate",
        "worst_strategy_vs_qqq",
        "top_symbol",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "cost_20bps_strategy_vs_qqq",
        "cost_30bps_strategy_vs_qqq",
        "decision",
        "production_ready",
    ]
    return sort.loc[:, cols].reset_index(drop=True)


def render_long_history_qqq_concentration_control_markdown(
    report_data: LongHistoryQQQConcentrationControlReportData,
) -> str:
    """Render the concentration-control sweep Markdown report."""

    results = build_concentration_control_decision_matrix(report_data.results)
    lines = [
        "# Long-History QQQ Concentration-Control Sweep Report",
        "",
        "This is a research-only concentration-control sweep. It does not mark any "
        "candidate production-ready and does not create production signal exports.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(results),
        "",
        "## 2. Source candidate recap",
        "",
        _source_candidate_recap(results),
        "",
        "## 3. Concentration-control rules tested",
        "",
        _rules_tested(report_data.candidates),
        "",
        "## 4. Full-period comparison versus QQQ",
        "",
        _markdown_table(_format_for_markdown(_full_period_table(results))),
        "",
        "## 5. Concentration impact",
        "",
        _concentration_impact(results),
        "",
        "## 6. Rolling-window robustness",
        "",
        _rolling_window_robustness(results, report_data.rolling_windows),
        "",
        "## 7. Subperiod robustness",
        "",
        _subperiod_robustness(report_data.subperiods),
        "",
        "## 8. Cost sensitivity",
        "",
        _cost_sensitivity(report_data.cost_sensitivity),
        "",
        "## 9. Candidate_0001 family result",
        "",
        _family_result(results, "candidate_0001"),
        "",
        "## 10. Candidate_0003 family result",
        "",
        _family_result(results, "candidate_0003"),
        "",
        "## 11. Best concentration-controlled candidates",
        "",
        _best_candidates(results),
        "",
        "## 12. Rejected/deprioritized controls",
        "",
        _rejected_controls(results),
        "",
        "## 13. Recommendation",
        "",
        _recommendation(results),
        "",
        "## 14. Caveats",
        "",
        _caveats(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return "No concentration-control rows were available."
    decision_counts = results["decision"].value_counts().to_dict()
    source_count = results["candidate_source_id"].nunique()
    production_ready = int(_as_bool(results["production_ready"]).sum())
    positive_edge = int(pd.to_numeric(results["strategy_vs_qqq"], errors="coerce").gt(0).sum())
    return (
        f"Evaluated `{len(results)}` concentration-control configs across `{source_count}` "
        "source candidates, with QQQ buy-and-hold as the primary benchmark. "
        f"`{positive_edge}` configs kept a positive full-period QQQ edge. "
        f"Decision counts: `{decision_counts}`. "
        f"Production-ready candidates: `{production_ready}`."
    )


def _source_candidate_recap(results: pd.DataFrame) -> str:
    if results.empty:
        return "_No source rows._"
    rows = []
    for source_id, group in results.groupby("candidate_source_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_source_id": source_id,
                "source_strategy_vs_qqq": first.get("source_strategy_vs_qqq"),
                "source_return_to_drawdown": first.get("source_return_to_drawdown"),
                "source_positive_vs_qqq_window_rate": first.get(
                    "source_positive_vs_qqq_window_rate"
                ),
                "source_top_symbol": first.get("source_top_symbol"),
                "source_top_symbol_exposure_pct": first.get("source_top_symbol_exposure_pct"),
                "source_top3_symbol_exposure_pct": first.get("source_top3_symbol_exposure_pct"),
                "source_decision": first.get("source_decision"),
            }
        )
    return _markdown_table(_format_for_markdown(pd.DataFrame(rows)))


def _rules_tested(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return "_No generated controls._"
    rules = (
        candidates.groupby(["concentration_rule", "control_value"], dropna=False)
        .size()
        .reset_index(name="config_count")
    )
    return _markdown_table(_format_for_markdown(rules))


def _full_period_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    cols = [
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "total_return",
        "qqq_buy_hold_total_return",
        "strategy_vs_qqq",
        "return_to_drawdown",
        "beats_qqq_return",
        "beats_qqq_return_to_drawdown",
        "decision",
    ]
    return results.loc[:, cols].sort_values("strategy_vs_qqq", ascending=False)


def _concentration_impact(results: pd.DataFrame) -> str:
    if results.empty:
        return "_No concentration rows._"
    cols = [
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "source_top_symbol_exposure_pct",
        "top_symbol",
        "top_symbol_exposure_pct",
        "source_top3_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "unique_symbol_count",
        "strategy_vs_qqq",
        "decision",
    ]
    table = results.loc[:, cols].sort_values(
        ["candidate_source_id", "top_symbol_exposure_pct", "strategy_vs_qqq"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    return _markdown_table(_format_for_markdown(table.head(30)))


def _rolling_window_robustness(results: pd.DataFrame, rolling_windows: pd.DataFrame) -> str:
    if results.empty:
        return "_No rolling-window rows._"
    summary_cols = [
        "candidate_source_id",
        "concentration_rule",
        "control_value",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
        "decision",
    ]
    weakest = (
        rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(12)
        if not rolling_windows.empty
        else pd.DataFrame()
    )
    return "\n\n".join(
        [
            "Rolling summary:",
            _markdown_table(_format_for_markdown(results.loc[:, summary_cols].head(30))),
            "Weakest QQQ-relative rolling windows:",
            _markdown_table(
                _format_for_markdown(weakest),
                [
                    "candidate_source_id",
                    "concentration_rule",
                    "window_label",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                ],
            ),
        ]
    )


def _subperiod_robustness(subperiods: pd.DataFrame) -> str:
    if subperiods.empty:
        return "_No subperiod rows._"
    table = subperiods.sort_values(
        ["candidate_source_id", "period_name", "strategy_vs_qqq"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    return _markdown_table(
        _format_for_markdown(table.head(60)),
        [
            "candidate_source_id",
            "concentration_rule",
            "period_name",
            "total_return",
            "qqq_total_return",
            "strategy_vs_qqq",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "top_symbol",
        ],
    )


def _cost_sensitivity(cost_sensitivity: pd.DataFrame) -> str:
    if cost_sensitivity.empty:
        return "_No cost sensitivity rows._"
    table = cost_sensitivity.sort_values(
        ["candidate_source_id", "concentration_rule", "transaction_cost_bps"],
        kind="mergesort",
    )
    return _markdown_table(
        _format_for_markdown(table.head(80)),
        [
            "candidate_source_id",
            "concentration_rule",
            "transaction_cost_bps",
            "total_return",
            "strategy_vs_qqq",
            "return_to_drawdown",
            "decision_at_cost",
        ],
    )


def _family_result(results: pd.DataFrame, source_id: str) -> str:
    family = results.loc[results["candidate_source_id"].astype(str).eq(source_id)].copy()
    if family.empty:
        return f"_No rows for `{source_id}`._"
    family = family.sort_values(
        ["decision", "strategy_vs_qqq"],
        ascending=[True, False],
        kind="mergesort",
    )
    return _markdown_table(
        _format_for_markdown(family.head(20)),
        [
            "concentration_rule",
            "control_value",
            "strategy_vs_qqq",
            "positive_vs_qqq_window_rate",
            "top_symbol",
            "top_symbol_exposure_pct",
            "top3_symbol_exposure_pct",
            "cost_30bps_strategy_vs_qqq",
            "decision",
        ],
    )


def _best_candidates(results: pd.DataFrame) -> str:
    if results.empty:
        return "_No best candidates._"
    controlled_mask = _controlled_candidate_mask(results)
    keep = results.loc[
        results["decision"].isin(["keep_for_deep_dive", "robust_candidate"])
        & controlled_mask
    ]
    if keep.empty:
        keep = results.loc[controlled_mask]
    decision_rank = {
        "keep_for_deep_dive": 0,
        "robust_candidate": 1,
        "concentration_improved_mixed": 2,
        "reject_deprioritize": 3,
    }
    keep = keep.sort_values(
        [
            "decision",
            "strategy_vs_qqq",
            "positive_vs_qqq_window_rate",
            "top_symbol_exposure_pct",
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
        key=lambda values: values.map(decision_rank).fillna(9)
        if values.name == "decision"
        else values,
    ).head(10)
    return _markdown_table(
        _format_for_markdown(keep),
        [
            "candidate_source_id",
            "concentration_rule",
            "control_value",
            "strategy_vs_qqq",
            "positive_vs_qqq_window_rate",
            "top_symbol_exposure_pct",
            "top3_symbol_exposure_pct",
            "decision",
            "rationale",
        ],
    )


def _rejected_controls(results: pd.DataFrame) -> str:
    rejected = results.loc[results["decision"].eq("reject_deprioritize")].copy()
    if rejected.empty:
        return "_No rejected controls._"
    rejected = rejected.sort_values(
        ["candidate_source_id", "strategy_vs_qqq"],
        ascending=[True, True],
        kind="mergesort",
    ).head(20)
    return _markdown_table(
        _format_for_markdown(rejected),
        [
            "candidate_source_id",
            "concentration_rule",
            "control_value",
            "strategy_vs_qqq",
            "top_symbol",
            "top_symbol_exposure_pct",
            "decision",
            "rationale",
        ],
    )


def _recommendation(results: pd.DataFrame) -> str:
    best = _best_non_diagnostic(results)
    best_label = (
        "none"
        if best is None
        else f"{best['candidate_source_id']} / {best['concentration_rule']} {best['control_value']}"
    )
    return "\n".join(
        [
            f"- Deep dive best concentration-controlled candidate: `{best_label}`.",
            (
                "- Targeted feature/model refinement: `yes`; use this sweep to guide "
                "the next narrow pass."
            ),
            "- MLP batch: `no`; keep deferred because this sweep is not a model zoo.",
            "- Production promotion: `no`; production_ready is always false.",
        ]
    )


def _best_non_diagnostic(results: pd.DataFrame) -> pd.Series | None:
    if results.empty:
        return None
    pool = results.loc[_controlled_candidate_mask(results)].copy()
    if pool.empty:
        return None
    decision_rank = {
        "keep_for_deep_dive": 0,
        "robust_candidate": 1,
        "concentration_improved_mixed": 2,
        "reject_deprioritize": 3,
    }
    pool = pool.assign(_rank=pool["decision"].map(decision_rank).fillna(9))
    pool = pool.sort_values(
        ["_rank", "strategy_vs_qqq", "top_symbol_exposure_pct"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return pool.iloc[0]


def _controlled_candidate_mask(results: pd.DataFrame) -> pd.Series:
    rule = results["concentration_rule"].astype(str)
    return ~rule.str.contains("sanity") & ~rule.eq("baseline_clone")


def _caveats() -> str:
    return "\n".join(
        [
            "- Research-only artifact bundle.",
            "- Tiingo-only local market data.",
            "- No production signal export, model card, or cloud write.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Exclusion sanity checks are diagnostics, not live trading rules.",
            "- Transaction costs are a simple turnover-times-bps approximation.",
        ]
    )


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required concentration-control artifact is missing: {path}")
    return pd.read_csv(path)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(_format_number)
    return out


def _format_number(value: object) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number):
        return ""
    return f"{number:.4f}"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


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
        "| "
        + " | ".join(_markdown_cell(value) for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the long-history QQQ concentration-control report."
    )
    parser.add_argument("--sweep-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_concentration_control_report_from_args())
