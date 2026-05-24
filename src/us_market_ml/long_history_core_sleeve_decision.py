"""Decision synthesis for the long-history QQQ core-sleeve refinement study.

This module only reads completed refinement artifacts. It does not run sweeps,
train models, write model cards, create production signal exports, or mark
anything production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_refinement import (
    CANDIDATE_REBALANCE_ALIGNED_WEIGHTS,
    DAILY_REBALANCED_FIXED_WEIGHTS,
    MONTHLY_REBALANCED_WEIGHTS,
    QUARTERLY_REBALANCED_WEIGHTS,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    DECISION_MATRIX_NAME as REFINEMENT_DECISION_MATRIX_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REFINEMENT_SWEEP_DIR,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    RESULTS_NAME as REFINEMENT_RESULTS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    ROLLING_WINDOWS_NAME as REFINEMENT_ROLLING_WINDOWS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    SUBPERIODS_NAME as REFINEMENT_SUBPERIODS_NAME,
)

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_decision"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_decision")
DEFAULT_REFINEMENT_REPORT_DIR = Path("artifacts/reports/long_history_qqq_core_sleeve_refinement")

REFINEMENT_REPORT_SUMMARY_NAME = "long_history_qqq_core_sleeve_refinement_summary.csv"
REFINEMENT_REPORT_DECISION_MATRIX_NAME = (
    "long_history_qqq_core_sleeve_refinement_decision_matrix.csv"
)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DECISION_MATRIX_NAME = "long_history_qqq_core_sleeve_decision_matrix.csv"
RECOMMENDATIONS_NAME = f"{OUTPUT_PREFIX}_recommendations.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

REALISTIC_ACCOUNTING_MODES = (
    MONTHLY_REBALANCED_WEIGHTS,
    QUARTERLY_REBALANCED_WEIGHTS,
    CANDIDATE_REBALANCE_ALIGNED_WEIGHTS,
)
MATERIAL_DRAWDOWN_IMPROVEMENT = 0.02

DECISION_RANK = {
    "best_core_sleeve_tracking_candidate": 0,
    "balanced_core_sleeve_tracking_candidate": 1,
    "raw_return_reference": 2,
    "defensive_reference": 3,
    "reject_deprioritize": 4,
}

OBJECTIVE_COLUMNS = [
    "objective_best_total_return_with_max_drawdown_lte_qqq",
    "objective_best_return_to_drawdown",
    "objective_best_drawdown_improvement_while_beating_qqq",
    "objective_best_practical_balanced_candidate",
]

DECISION_EXTRA_COLUMNS = [
    "realistic_accounting_mode_pass_count",
    "all_accounting_mode_pass_count",
    *OBJECTIVE_COLUMNS,
    "selected_objectives",
    "decision_category",
    "decision_rationale",
    "track_next",
    "rich_replay_next",
    "reference_only",
    "stop_spending_time",
    "production_ready",
]

SUMMARY_COLUMNS = [
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
    "realistic_accounting_mode_pass_count",
    "selected_objectives",
    "decision_category",
    "production_ready",
]

RECOMMENDATION_COLUMNS = [
    "recommendation_action",
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "weight_zone",
    "decision_category",
    "total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "rationale",
    "production_ready",
]


@dataclass(frozen=True)
class LongHistoryCoreSleeveDecisionInputs:
    """CSV-backed inputs for the decision synthesis report."""

    results: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    refinement_decision_matrix: pd.DataFrame
    refinement_summary: pd.DataFrame
    refinement_report_decision_matrix: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryCoreSleeveDecisionArtifacts:
    """CSV and Markdown outputs for the decision synthesis report."""

    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    recommendations: pd.DataFrame
    report_markdown: str


def read_long_history_core_sleeve_decision_inputs(
    *,
    refinement_sweep_dir: Path = DEFAULT_REFINEMENT_SWEEP_DIR,
    refinement_report_dir: Path = DEFAULT_REFINEMENT_REPORT_DIR,
) -> LongHistoryCoreSleeveDecisionInputs:
    """Read the completed refinement artifacts required for decision synthesis."""

    return LongHistoryCoreSleeveDecisionInputs(
        results=_read_required_csv(refinement_sweep_dir / REFINEMENT_RESULTS_NAME),
        subperiods=_read_required_csv(refinement_sweep_dir / REFINEMENT_SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(
            refinement_sweep_dir / REFINEMENT_ROLLING_WINDOWS_NAME
        ),
        refinement_decision_matrix=_read_required_csv(
            refinement_sweep_dir / REFINEMENT_DECISION_MATRIX_NAME
        ),
        refinement_summary=_read_required_csv(
            refinement_report_dir / REFINEMENT_REPORT_SUMMARY_NAME
        ),
        refinement_report_decision_matrix=_read_required_csv(
            refinement_report_dir / REFINEMENT_REPORT_DECISION_MATRIX_NAME
        ),
    )


def build_long_history_core_sleeve_decision_artifacts(
    inputs: LongHistoryCoreSleeveDecisionInputs,
) -> LongHistoryCoreSleeveDecisionArtifacts:
    """Build all decision report outputs from completed refinement inputs."""

    decision_matrix = build_long_history_core_sleeve_decision_matrix(inputs.results)
    recommendations = select_long_history_core_sleeve_recommendations(decision_matrix)
    summary = build_long_history_core_sleeve_decision_summary(decision_matrix)
    report = render_long_history_core_sleeve_decision_markdown(
        inputs=inputs,
        decision_matrix=decision_matrix,
        recommendations=recommendations,
    )
    return LongHistoryCoreSleeveDecisionArtifacts(
        summary=summary,
        decision_matrix=decision_matrix,
        recommendations=recommendations,
        report_markdown=report,
    )


def build_long_history_core_sleeve_decision_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Return the research-only decision matrix using the final decision vocabulary."""

    if results.empty:
        return pd.DataFrame(columns=[*results.columns, *DECISION_EXTRA_COLUMNS])
    matrix = results.copy()
    _ensure_decision_inputs(matrix)
    _add_accounting_survival_counts(matrix)
    for column in OBJECTIVE_COLUMNS:
        matrix[column] = False
    matrix["selected_objectives"] = ""
    _mark_objective_winners(matrix)

    decisions = matrix.apply(classify_core_sleeve_decision_category, axis=1)
    matrix["decision_category"] = [item[0] for item in decisions]
    matrix["decision_rationale"] = [item[1] for item in decisions]
    matrix["track_next"] = False
    matrix["rich_replay_next"] = False
    matrix["reference_only"] = matrix["decision_category"].eq("raw_return_reference")
    matrix["stop_spending_time"] = matrix["decision_category"].eq("reject_deprioritize")
    matrix["production_ready"] = False
    return matrix


def classify_core_sleeve_decision_category(row: pd.Series) -> tuple[str, str]:
    """Classify a single portfolio row for the final decision report."""

    beats_qqq = bool(row.get("beats_qqq_return", False))
    captures_90 = bool(row.get("captures_90pct_qqq_return", False))
    drawdown_improved = bool(row.get("improves_drawdown_vs_qqq", False))
    rtd_improved = bool(row.get("improves_return_to_drawdown_vs_qqq", False))
    is_core_sleeve = bool(row.get("is_core_sleeve", False))
    spy_weight = _to_float(row.get("spy_core_weight"))
    realistic_pass_count = int(_to_float(row.get("realistic_accounting_mode_pass_count")) or 0)
    drawdown_improvement = _to_float(row.get("drawdown_improvement_vs_qqq"))

    reasons: list[str] = []
    reasons.append("beats QQQ return" if beats_qqq else "does not beat QQQ return")
    reasons.append(
        "improves QQQ drawdown" if drawdown_improved else "does not improve QQQ drawdown"
    )
    reasons.append(
        "improves return-to-drawdown"
        if rtd_improved
        else "does not improve return-to-drawdown"
    )

    if spy_weight == 0.0 and beats_qqq and not drawdown_improved:
        return (
            "raw_return_reference",
            "; ".join([*reasons, "standalone sleeve remains reference-only"]),
        )
    if is_core_sleeve and beats_qqq and drawdown_improved and rtd_improved:
        if realistic_pass_count >= 2:
            return (
                "best_core_sleeve_tracking_candidate",
                "; ".join(
                    [
                        *reasons,
                        f"survives {realistic_pass_count} realistic accounting modes",
                    ]
                ),
            )
        return (
            "balanced_core_sleeve_tracking_candidate",
            "; ".join([*reasons, "needs more accounting-mode confirmation"]),
        )
    if is_core_sleeve and captures_90 and drawdown_improved and rtd_improved:
        return (
            "balanced_core_sleeve_tracking_candidate",
            "; ".join([*reasons, "captures at least 90% of QQQ return"]),
        )
    if (
        not beats_qqq
        and pd.notna(drawdown_improvement)
        and drawdown_improvement >= MATERIAL_DRAWDOWN_IMPROVEMENT
        and rtd_improved
    ):
        return (
            "defensive_reference",
            "; ".join([*reasons, "materially improves drawdown/risk profile"]),
        )
    return "reject_deprioritize", "; ".join(reasons)


def build_long_history_core_sleeve_decision_summary(
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact ranked summary table for the decision report."""

    if decision_matrix.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    ranked = _rank_decision_matrix(decision_matrix)
    return ranked.loc[:, SUMMARY_COLUMNS].reset_index(drop=True)


def accounting_mode_stability_summary(decision_matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize decision quality by accounting mode."""

    if decision_matrix.empty:
        return pd.DataFrame()
    grouped = (
        decision_matrix.groupby("portfolio_accounting_mode", sort=False)
        .agg(
            configs=("portfolio_accounting_mode", "size"),
            tracking_candidates=(
                "decision_category",
                lambda values: int(
                    values.isin(
                        [
                            "best_core_sleeve_tracking_candidate",
                            "balanced_core_sleeve_tracking_candidate",
                        ]
                    ).sum()
                ),
            ),
            best_candidates=(
                "decision_category",
                lambda values: int(values.eq("best_core_sleeve_tracking_candidate").sum()),
            ),
            best_total_return=("total_return", "max"),
            best_strategy_vs_qqq=("strategy_vs_qqq", "max"),
            best_drawdown_improvement=("drawdown_improvement_vs_qqq", "max"),
            best_return_to_drawdown=("return_to_drawdown", "max"),
        )
        .reset_index()
    )
    grouped["realistic_accounting_mode"] = grouped["portfolio_accounting_mode"].isin(
        REALISTIC_ACCOUNTING_MODES
    )
    return grouped


def select_long_history_core_sleeve_recommendations(
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return the compact tracking/reference recommendation table."""

    if decision_matrix.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    rows: list[dict[str, object]] = []
    tracking_rows = _tracking_recommendation_rows(decision_matrix)
    rich_replay_keys = set()
    for idx, row in enumerate(tracking_rows):
        action = "track_next"
        if idx in (0, 2):
            action = "rich_replay_next"
            rich_replay_keys.add(_row_key(row))
        rows.append(_recommendation_row(row, action))

    for candidate_id in ("candidate_0001", "candidate_0003"):
        standalone = _rank_decision_matrix(
            decision_matrix.loc[
                decision_matrix["candidate_id"].astype(str).eq(candidate_id)
                & pd.to_numeric(decision_matrix["spy_core_weight"], errors="coerce").eq(0.0)
            ]
        )
        if not standalone.empty:
            rows.append(
                _recommendation_row(
                    standalone.iloc[0],
                    "reference_only",
                    rationale=(
                        "Standalone sleeve is useful for signal review/raw return context, "
                        "not for production promotion."
                    ),
                )
            )

    rows.append(
        {
            "recommendation_action": "stop_spending_time",
            "candidate_id": "broad_sweep",
            "spy_core_weight": pd.NA,
            "candidate_sleeve_weight": pd.NA,
            "portfolio_accounting_mode": "not_applicable",
            "weight_zone": "not_applicable",
            "decision_category": "reject_deprioritize",
            "total_return": pd.NA,
            "strategy_vs_qqq": pd.NA,
            "max_drawdown": pd.NA,
            "drawdown_improvement_vs_qqq": pd.NA,
            "return_to_drawdown": pd.NA,
            "rationale": (
                "Do not run another broad sweep from this report; move to tracking/rich replay "
                "around the selected core-sleeve rows."
            ),
            "production_ready": False,
        }
    )
    out = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    out["production_ready"] = False
    return out.drop_duplicates(
        subset=[
            "recommendation_action",
            "candidate_id",
            "spy_core_weight",
            "portfolio_accounting_mode",
        ],
        keep="first",
    ).reset_index(drop=True)


def render_long_history_core_sleeve_decision_markdown(
    *,
    inputs: LongHistoryCoreSleeveDecisionInputs,
    decision_matrix: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    """Render the final long-history QQQ core-sleeve decision report."""

    lines = [
        "# Long-History QQQ Core-Sleeve Decision Report",
        "",
        "This is a concise research-only synthesis of the completed SPY core plus "
        "candidate sleeve refinement study. It reads finished local artifacts only.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(decision_matrix, recommendations),
        "",
        "## 2. What changed after realistic accounting",
        "",
        _realistic_accounting_summary(decision_matrix),
        "",
        "## 3. Standalone candidate comparison",
        "",
        _standalone_candidate_comparison(decision_matrix),
        "",
        "## 4. Candidate_0001 core-sleeve decision",
        "",
        _candidate_decision_text(decision_matrix, "candidate_0001"),
        "",
        "## 5. Candidate_0003 core-sleeve decision",
        "",
        _candidate_decision_text(decision_matrix, "candidate_0003"),
        "",
        "## 6. Accounting-mode stability",
        "",
        _accounting_mode_stability_text(decision_matrix),
        "",
        "## 7. Best portfolios by objective",
        "",
        _best_portfolios_by_objective(decision_matrix),
        "",
        "## 8. Subperiod caveats",
        "",
        _subperiod_caveats(inputs.subperiods, recommendations),
        "",
        "## 9. Final recommendation",
        "",
        _final_recommendation(recommendations),
        "",
        "## 10. Caveats",
        "",
        _caveats(decision_matrix),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_long_history_core_sleeve_decision_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveDecisionArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write decision synthesis artifacts to the report output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    artifacts.recommendations.to_csv(paths[RECOMMENDATIONS_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    return paths


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for the decision report."""

    return {
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        RECOMMENDATIONS_NAME: output_dir / RECOMMENDATIONS_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }


def _ensure_decision_inputs(matrix: pd.DataFrame) -> None:
    for column in [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_accounting_mode",
        "is_core_sleeve",
        "total_return",
        "strategy_vs_qqq",
        "qqq_return_capture",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "beats_qqq_return",
        "improves_drawdown_vs_qqq",
        "improves_return_to_drawdown_vs_qqq",
        "captures_90pct_qqq_return",
        "production_ready",
    ]:
        if column not in matrix.columns:
            matrix[column] = pd.NA


def _add_accounting_survival_counts(matrix: pd.DataFrame) -> None:
    pass_mask = (
        _as_bool(matrix["is_core_sleeve"])
        & _as_bool(matrix["beats_qqq_return"])
        & _as_bool(matrix["improves_drawdown_vs_qqq"])
        & _as_bool(matrix["improves_return_to_drawdown_vs_qqq"])
    )
    keys = ["candidate_id", "spy_core_weight", "candidate_sleeve_weight"]
    realistic_mask = matrix["portfolio_accounting_mode"].astype(str).isin(
        REALISTIC_ACCOUNTING_MODES
    )
    realistic_counts = (
        matrix.loc[pass_mask & realistic_mask]
        .groupby(keys, dropna=False)["portfolio_accounting_mode"]
        .nunique()
    )
    all_counts = (
        matrix.loc[pass_mask].groupby(keys, dropna=False)["portfolio_accounting_mode"].nunique()
    )
    row_keys = [
        (row.candidate_id, row.spy_core_weight, row.candidate_sleeve_weight)
        for row in matrix.loc[:, keys].itertuples(index=False)
    ]
    matrix["realistic_accounting_mode_pass_count"] = [
        int(realistic_counts.get(row_key, 0)) for row_key in row_keys
    ]
    matrix["all_accounting_mode_pass_count"] = [
        int(all_counts.get(row_key, 0)) for row_key in row_keys
    ]


def _mark_objective_winners(matrix: pd.DataFrame) -> None:
    objective_map = {
        "objective_best_total_return_with_max_drawdown_lte_qqq": (
            _best_total_return_drawdown_lte_qqq
        ),
        "objective_best_return_to_drawdown": _best_return_to_drawdown,
        "objective_best_drawdown_improvement_while_beating_qqq": (
            _best_drawdown_improvement_beating_qqq
        ),
        "objective_best_practical_balanced_candidate": _best_practical_balanced_candidate,
    }
    selected: dict[int, list[str]] = {}
    for column, selector in objective_map.items():
        idx = selector(matrix)
        if idx is None:
            continue
        matrix.loc[idx, column] = True
        label = column.replace("objective_", "")
        selected.setdefault(int(idx), []).append(label)
    for idx, labels in selected.items():
        matrix.loc[idx, "selected_objectives"] = ";".join(labels)


def _best_total_return_drawdown_lte_qqq(matrix: pd.DataFrame) -> int | None:
    rows = matrix.loc[pd.to_numeric(matrix["drawdown_improvement_vs_qqq"], errors="coerce").ge(0.0)]
    if rows.empty:
        return None
    return int(rows.sort_values("total_return", ascending=False, kind="mergesort").index[0])


def _best_return_to_drawdown(matrix: pd.DataFrame) -> int | None:
    if matrix.empty:
        return None
    return int(
        matrix.sort_values(
            ["return_to_drawdown", "total_return"],
            ascending=[False, False],
            kind="mergesort",
        ).index[0]
    )


def _best_drawdown_improvement_beating_qqq(matrix: pd.DataFrame) -> int | None:
    rows = matrix.loc[_as_bool(matrix["beats_qqq_return"])]
    if rows.empty:
        return None
    return int(
        rows.sort_values(
            ["drawdown_improvement_vs_qqq", "total_return"],
            ascending=[False, False],
            kind="mergesort",
        ).index[0]
    )


def _best_practical_balanced_candidate(matrix: pd.DataFrame) -> int | None:
    rows = matrix.loc[
        _as_bool(matrix["is_core_sleeve"])
        & _as_bool(matrix["captures_90pct_qqq_return"])
        & _as_bool(matrix["improves_drawdown_vs_qqq"])
        & _as_bool(matrix["improves_return_to_drawdown_vs_qqq"])
        & pd.to_numeric(matrix["realistic_accounting_mode_pass_count"], errors="coerce").ge(2)
    ].copy()
    if rows.empty:
        return None
    rows["_balance_score"] = (
        pd.to_numeric(rows["strategy_vs_qqq"], errors="coerce").fillna(-999.0)
        + pd.to_numeric(rows["drawdown_improvement_vs_qqq"], errors="coerce").fillna(0.0) * 10.0
    )
    return int(
        rows.sort_values(
            ["_balance_score", "positive_vs_qqq_window_rate", "total_return"],
            ascending=[False, False, False],
            kind="mergesort",
        ).index[0]
    )


def _tracking_recommendation_rows(decision_matrix: pd.DataFrame) -> list[pd.Series]:
    picks: list[pd.Series] = []
    pick_specs = [
        ("candidate_0001", (0.35, 0.40), "return"),
        ("candidate_0001", (0.40, 0.40), "balanced"),
        ("candidate_0003", (0.25, 0.30), "return_controlled"),
        ("candidate_0003", (0.35, 0.40), "defensive_balance"),
    ]
    for candidate_id, weight_range, objective in pick_specs:
        family = decision_matrix.loc[
            decision_matrix["candidate_id"].astype(str).eq(candidate_id)
            & pd.to_numeric(decision_matrix["spy_core_weight"], errors="coerce").between(
                weight_range[0],
                weight_range[1],
            )
            & decision_matrix["decision_category"].isin(
                [
                    "best_core_sleeve_tracking_candidate",
                    "balanced_core_sleeve_tracking_candidate",
                ]
            )
        ].copy()
        if family.empty:
            continue
        family["_mode_rank"] = family["portfolio_accounting_mode"].map(_accounting_mode_rank)
        if objective in {"return", "return_controlled"}:
            family = family.sort_values(
                ["_mode_rank", "strategy_vs_qqq", "drawdown_improvement_vs_qqq"],
                ascending=[True, False, False],
                kind="mergesort",
            )
        else:
            family = family.sort_values(
                ["_mode_rank", "drawdown_improvement_vs_qqq", "strategy_vs_qqq"],
                ascending=[True, False, False],
                kind="mergesort",
            )
        picks.append(family.iloc[0])

    deduped: list[pd.Series] = []
    seen: set[tuple[object, ...]] = set()
    for row in picks:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:4]


def _recommendation_row(
    row: pd.Series,
    action: str,
    *,
    rationale: str | None = None,
) -> dict[str, object]:
    candidate_id = str(row.get("candidate_id", ""))
    if candidate_id == "candidate_0001":
        zone = "35%-40% SPY core / 65%-60% candidate sleeve"
    elif candidate_id == "candidate_0003":
        zone = "25%-40% SPY core / 75%-60% candidate sleeve"
    else:
        zone = ""
    return {
        "recommendation_action": action,
        "candidate_id": candidate_id,
        "spy_core_weight": row.get("spy_core_weight"),
        "candidate_sleeve_weight": row.get("candidate_sleeve_weight"),
        "portfolio_accounting_mode": row.get("portfolio_accounting_mode"),
        "weight_zone": zone,
        "decision_category": row.get("decision_category"),
        "total_return": row.get("total_return"),
        "strategy_vs_qqq": row.get("strategy_vs_qqq"),
        "max_drawdown": row.get("max_drawdown"),
        "drawdown_improvement_vs_qqq": row.get("drawdown_improvement_vs_qqq"),
        "return_to_drawdown": row.get("return_to_drawdown"),
        "rationale": rationale or row.get("decision_rationale"),
        "production_ready": False,
    }


def _row_key(row: pd.Series) -> tuple[object, ...]:
    return (
        row.get("candidate_id"),
        round(float(row.get("spy_core_weight")), 10),
        row.get("portfolio_accounting_mode"),
    )


def _accounting_mode_rank(mode: object) -> int:
    order = {
        CANDIDATE_REBALANCE_ALIGNED_WEIGHTS: 0,
        MONTHLY_REBALANCED_WEIGHTS: 1,
        QUARTERLY_REBALANCED_WEIGHTS: 2,
        DAILY_REBALANCED_FIXED_WEIGHTS: 9,
    }
    return order.get(str(mode), 99)


def _executive_summary(
    decision_matrix: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    if decision_matrix.empty:
        return "No completed refinement rows were available."
    counts = decision_matrix["decision_category"].value_counts().to_dict()
    tracking = recommendations.loc[
        recommendations["recommendation_action"].isin(["track_next", "rich_replay_next"])
    ]
    tracking_labels = []
    for row in tracking.itertuples(index=False):
        if pd.notna(row.spy_core_weight):
            tracking_labels.append(
                f"`{row.candidate_id}` {float(row.spy_core_weight):.0%}/"
                f"{float(row.candidate_sleeve_weight):.0%}"
            )
    tracking_text = ", ".join(tracking_labels)
    production_ready = int(_as_bool(decision_matrix["production_ready"]).sum())
    return (
        f"The completed refinement produced `{len(decision_matrix)}` rows across "
        f"`{decision_matrix['candidate_id'].nunique()}` sleeves and "
        f"`{decision_matrix['portfolio_accounting_mode'].nunique()}` accounting modes. "
        "QQQ buy-and-hold remains the primary benchmark; SPY and standalone sleeves are "
        "secondary references. The compact next tracking set is "
        f"{tracking_text or '`none`'}. Decision counts: `{counts}`. "
        f"Production-ready rows: `{production_ready}`."
    )


def _realistic_accounting_summary(decision_matrix: pd.DataFrame) -> str:
    if decision_matrix.empty:
        return "_No rows._"
    return (
        "Daily fixed weights still describe the clean research baseline, but the decision "
        "survives only where monthly, quarterly, and candidate-aligned accounting do not "
        "erase the QQQ-relative edge. The important shift is that standalone raw return "
        "is separated from investable tracking candidates: candidate_0001 needs roughly "
        "35%-40% SPY core to neutralize drawdown, while candidate_0003 becomes most useful "
        "around 25%-40% SPY core because the blend materially improves drawdown behavior."
    )


def _standalone_candidate_comparison(decision_matrix: pd.DataFrame) -> str:
    standalone = decision_matrix.loc[
        pd.to_numeric(decision_matrix["spy_core_weight"], errors="coerce").eq(0.0)
    ]
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "decision_category",
    ]
    return _markdown_table(_format_for_markdown(_rank_decision_matrix(standalone).loc[:, cols]))


def _candidate_decision_text(decision_matrix: pd.DataFrame, candidate_id: str) -> str:
    family = decision_matrix.loc[
        decision_matrix["candidate_id"].astype(str).eq(candidate_id)
        & _as_bool(decision_matrix["is_core_sleeve"])
    ]
    if family.empty:
        return "_No rows._"
    zone = (
        "35%-50% SPY core is the useful return-oriented zone; 35% and 40% are the "
        "main tracking weights."
        if candidate_id == "candidate_0001"
        else "25%-45% SPY core is the useful risk-controlled zone; 25%-40% is the "
        "main tracking range."
    )
    display = _rank_decision_matrix(family).head(12)
    cols = [
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "realistic_accounting_mode_pass_count",
        "decision_category",
    ]
    return zone + "\n\n" + _markdown_table(_format_for_markdown(display.loc[:, cols]))


def _accounting_mode_stability_text(decision_matrix: pd.DataFrame) -> str:
    summary = accounting_mode_stability_summary(decision_matrix)
    if summary.empty:
        return "_No rows._"
    return _markdown_table(_format_for_markdown(summary))


def _best_portfolios_by_objective(decision_matrix: pd.DataFrame) -> str:
    if decision_matrix.empty:
        return "_No rows._"
    pieces = []
    labels = {
        "objective_best_total_return_with_max_drawdown_lte_qqq": (
            "best total return with max_drawdown <= QQQ"
        ),
        "objective_best_return_to_drawdown": "best return-to-drawdown",
        "objective_best_drawdown_improvement_while_beating_qqq": (
            "best drawdown improvement while still beating QQQ"
        ),
        "objective_best_practical_balanced_candidate": "best practical balanced candidate",
    }
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
        "return_to_drawdown",
        "decision_category",
    ]
    for column, label in labels.items():
        rows = decision_matrix.loc[_as_bool(decision_matrix[column])]
        pieces.append(f"**{label}**\n\n{_markdown_table(_format_for_markdown(rows.loc[:, cols]))}")
    return "\n\n".join(pieces)


def _subperiod_caveats(subperiods: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    if subperiods.empty:
        return "_No subperiod rows._"
    tracking = recommendations.loc[
        recommendations["recommendation_action"].isin(["track_next", "rich_replay_next"])
        & recommendations["candidate_id"].astype(str).str.startswith("candidate_")
    ]
    if tracking.empty:
        return "_No tracking recommendations._"
    keys = tracking.loc[
        :,
        ["candidate_id", "spy_core_weight", "candidate_sleeve_weight", "portfolio_accounting_mode"],
    ].drop_duplicates()
    merged = subperiods.merge(
        keys,
        on=[
            "candidate_id",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "portfolio_accounting_mode",
        ],
        how="inner",
    )
    periods = [
        "covid_crash_recovery",
        "bear_2022",
        "ai_bull_2023_2024",
        "recent_2025_2026",
    ]
    merged = merged.loc[merged["period_name"].isin(periods)]
    cols = [
        "candidate_id",
        "portfolio_accounting_mode",
        "spy_core_weight",
        "period_name",
        "total_return",
        "qqq_total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "drawdown_improvement_vs_qqq",
    ]
    caveat = (
        "Subperiod behavior is uneven: the selected blends are not a guarantee of QQQ "
        "outperformance in every regime. Covid crash/recovery and the AI bull window "
        "remain the main places to inspect lag; bear 2022 and recent 2025-2026 behavior "
        "are useful checks for drawdown control and current-regime drift."
    )
    return caveat + "\n\n" + _markdown_table(_format_for_markdown(merged.loc[:, cols]))


def _final_recommendation(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return "_No recommendations._"
    track = recommendations.loc[
        recommendations["recommendation_action"].isin(["track_next", "rich_replay_next"])
    ]
    rich = recommendations.loc[recommendations["recommendation_action"].eq("rich_replay_next")]
    refs = recommendations.loc[recommendations["recommendation_action"].eq("reference_only")]
    stops = recommendations.loc[recommendations["recommendation_action"].eq("stop_spending_time")]
    return (
        "- Track next: "
        + _recommendation_list(track)
        + "\n- Rich-replay next: "
        + _recommendation_list(rich)
        + "\n- Keep as references only: "
        + _recommendation_list(refs)
        + "\n- Stop spending time on: "
        + "; ".join(stops["rationale"].astype(str).tolist())
        + "\n- Run another broad sweep: `no`"
        + "\n- Promote to production: `no`"
    )


def _recommendation_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "`none`"
    labels = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.spy_core_weight):
            labels.append(f"`{row.candidate_id}`")
        else:
            labels.append(
                f"`{row.candidate_id}` {float(row.spy_core_weight):.0%} SPY / "
                f"{float(row.candidate_sleeve_weight):.0%} sleeve "
                f"(`{row.portfolio_accounting_mode}`)"
            )
    return "; ".join(labels)


def _caveats(decision_matrix: pd.DataFrame) -> str:
    modes = (
        ", ".join(sorted(decision_matrix["portfolio_accounting_mode"].astype(str).unique()))
        if not decision_matrix.empty
        else "none"
    )
    return (
        "- Research-only; `production_ready` is always false.\n"
        "- Tiingo-only input artifacts.\n"
        "- No production signal export, model card, S3/cloud write, or Market Data repo change.\n"
        "- QQQ buy-and-hold remains the primary benchmark.\n"
        f"- Core-sleeve conclusions depend on accounting mode: `{modes}`."
    )


def _rank_decision_matrix(decision_matrix: pd.DataFrame) -> pd.DataFrame:
    if decision_matrix.empty:
        return decision_matrix.copy()
    out = decision_matrix.copy()
    objective_score = out.loc[:, OBJECTIVE_COLUMNS].apply(
        lambda row: int(_as_bool(row).sum()),
        axis=1,
    )
    out["_decision_rank"] = out["decision_category"].map(DECISION_RANK).fillna(9)
    out["_objective_score"] = objective_score
    return out.sort_values(
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


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.fillna("")
    columns = table.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


__all__ = [
    "DECISION_MATRIX_NAME",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REFINEMENT_REPORT_DIR",
    "OUTPUT_PREFIX",
    "REALISTIC_ACCOUNTING_MODES",
    "RECOMMENDATIONS_NAME",
    "REPORT_NAME",
    "SUMMARY_NAME",
    "LongHistoryCoreSleeveDecisionArtifacts",
    "LongHistoryCoreSleeveDecisionInputs",
    "accounting_mode_stability_summary",
    "build_long_history_core_sleeve_decision_artifacts",
    "build_long_history_core_sleeve_decision_matrix",
    "build_long_history_core_sleeve_decision_summary",
    "classify_core_sleeve_decision_category",
    "output_paths",
    "read_long_history_core_sleeve_decision_inputs",
    "render_long_history_core_sleeve_decision_markdown",
    "select_long_history_core_sleeve_recommendations",
    "write_long_history_core_sleeve_decision_artifacts",
]
