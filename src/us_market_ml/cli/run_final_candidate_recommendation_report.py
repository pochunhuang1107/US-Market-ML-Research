"""Synthesize final research-stage candidate recommendations from report artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ALLOCATION_REPORT_DIR = Path("artifacts/reports/allocation_study_comparison")
DEFAULT_NARROW_ALLOCATION_REPORT_DIR = Path(
    "artifacts/reports/narrow_allocation_promotion_study"
)
DEFAULT_SLIDING_WINDOW_REPORT_DIR = Path("artifacts/reports/sliding_window_champion_study")
DEFAULT_SLIDING_WINDOW_BENCHMARK_DIR = Path(
    "artifacts/reports/sliding_window_benchmark_comparison"
)
DEFAULT_OVERNIGHT_CANDIDATE_REPORT_DIR = Path(
    "artifacts/reports/sliding_window_overnight_candidate_study"
)
DEFAULT_SHORTLIST_BENCHMARK_REPORT_DIR = Path(
    "artifacts/reports/sliding_window_shortlist_benchmark"
)
DEFAULT_LONGER_WINDOW_REPORT_DIR = Path(
    "artifacts/reports/sliding_window_longer_window_shortlist_study"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_candidate_recommendation")

SUMMARY_PATH_NAME = "final_candidate_summary.csv"
DECISION_MATRIX_PATH_NAME = "final_candidate_decision_matrix.csv"
REPORT_PATH_NAME = "final_candidate_recommendation.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

FIXED_FOLD_CHAMPION_ID = "full:linear_regression:top1:equal_weight:cost10"
BENCHMARK_RELATIVE_LEADER_ID = "broad_macro:linear_regression:top1:equal_weight:cost10"
LONGER_WINDOW_BALANCED_LEADER_ID = "full:ridge:top5:equal_weight:cost10"
FULL_UNIVERSE_REFERENCE_ID = "full:ridge:top3:equal_weight:cost10"
DEFENSIVE_BROAD_MACRO_REFERENCE_ID = "broad_macro:ridge:top5:equal_weight:cost10"
RANDOM_FOREST_3M_REFERENCE_ID = "broad_macro:random_forest:top3:equal_weight:cost10"


@dataclass(frozen=True)
class CandidateSpec:
    """Candidate identity used by the final synthesis report."""

    universe: str
    model_type: str
    top_k: int
    allocation_mode: str = "equal_weight"
    transaction_cost_bps: float = 10.0

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.universe}:{self.model_type}:top{self.top_k}:"
            f"{self.allocation_mode}:cost{_format_compact(self.transaction_cost_bps)}"
        )

    @property
    def candidate_label(self) -> str:
        return (
            f"{self.universe} / {self.model_type} / top{self.top_k} / "
            f"{self.allocation_mode} / cost{_format_compact(self.transaction_cost_bps)}"
        )


FINAL_CANDIDATE_SPECS = (
    CandidateSpec("full", "linear_regression", 1),
    CandidateSpec("broad_macro", "linear_regression", 1),
    CandidateSpec("broad_macro", "linear_regression", 2),
    CandidateSpec("full", "ridge", 3),
    CandidateSpec("full", "ridge", 5),
    CandidateSpec("broad_macro", "ridge", 5),
    CandidateSpec("broad_macro", "random_forest", 3),
)

FINAL_SUMMARY_COLUMNS = [
    "candidate_label",
    "fixed_fold_role",
    "allocation_result_role",
    "three_month_sliding_role",
    "three_month_benchmark_role",
    "longer_window_role",
    "mean_total_return_3m",
    "mean_total_return_longer",
    "positive_rate_3m",
    "positive_rate_longer",
    "beats_spy_rate_3m",
    "beats_spy_rate_longer",
    "beats_qqq_rate_3m",
    "beats_qqq_rate_longer",
    "worst_max_drawdown_3m",
    "worst_max_drawdown_longer",
    "recommendation_label",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_label",
    "return_strength",
    "benchmark_relative_strength",
    "drawdown_stability",
    "universe_stability",
    "model_complexity",
    "interpretability",
    "risk_of_overfitting",
    "product_explainability",
    "next_action",
]


@dataclass(frozen=True)
class FinalCandidateReportInputs:
    """Input report tables used by the final synthesis."""

    sliding_window_summary: pd.DataFrame
    sliding_window_benchmark_summary: pd.DataFrame
    overnight_candidate_metrics: pd.DataFrame
    shortlist_benchmark_summary: pd.DataFrame
    shortlist_candidate_summary: pd.DataFrame | None
    longer_window_candidate_summary: pd.DataFrame
    allocation_champion_comparison: pd.DataFrame | None
    narrow_allocation_champion_comparison: pd.DataFrame | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FinalCandidateRecommendationReportData:
    """Rendered table outputs and source warnings for the final report."""

    final_candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    warnings: tuple[str, ...]


def run_final_candidate_recommendation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the final candidate recommendation report."""

    args = _parse_args(argv)
    inputs = read_final_candidate_report_inputs(
        allocation_report_dir=args.allocation_report_dir,
        narrow_allocation_report_dir=args.narrow_allocation_report_dir,
        sliding_window_report_dir=args.sliding_window_report_dir,
        sliding_window_benchmark_dir=args.sliding_window_benchmark_dir,
        overnight_candidate_report_dir=args.overnight_candidate_report_dir,
        shortlist_benchmark_report_dir=args.shortlist_benchmark_report_dir,
        longer_window_report_dir=args.longer_window_report_dir,
    )
    report_data = build_final_candidate_recommendation_report(inputs)
    outputs = write_final_candidate_recommendation_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML final candidate recommendation report")
    print(f"allocation report dir: {args.allocation_report_dir}")
    print(f"narrow allocation report dir: {args.narrow_allocation_report_dir}")
    print(f"sliding-window report dir: {args.sliding_window_report_dir}")
    print(f"sliding-window benchmark dir: {args.sliding_window_benchmark_dir}")
    print(f"overnight candidate report dir: {args.overnight_candidate_report_dir}")
    print(f"shortlist benchmark report dir: {args.shortlist_benchmark_report_dir}")
    print(f"longer-window report dir: {args.longer_window_report_dir}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    if report_data.warnings:
        print()
        print("Warnings:")
        for warning in report_data.warnings:
            print(f"- {warning}")
    print()
    print(report_data.final_candidate_summary.to_string(index=False))
    return 0


def read_final_candidate_report_inputs(
    *,
    allocation_report_dir: Path = DEFAULT_ALLOCATION_REPORT_DIR,
    narrow_allocation_report_dir: Path = DEFAULT_NARROW_ALLOCATION_REPORT_DIR,
    sliding_window_report_dir: Path = DEFAULT_SLIDING_WINDOW_REPORT_DIR,
    sliding_window_benchmark_dir: Path = DEFAULT_SLIDING_WINDOW_BENCHMARK_DIR,
    overnight_candidate_report_dir: Path = DEFAULT_OVERNIGHT_CANDIDATE_REPORT_DIR,
    shortlist_benchmark_report_dir: Path = DEFAULT_SHORTLIST_BENCHMARK_REPORT_DIR,
    longer_window_report_dir: Path = DEFAULT_LONGER_WINDOW_REPORT_DIR,
) -> FinalCandidateReportInputs:
    """Read required and optional report CSV inputs from existing artifact directories."""

    warnings: list[str] = []
    sliding_window_summary = _read_required_csv(
        sliding_window_report_dir / "sliding_window_summary.csv",
        required_columns=[
            "mean_total_return",
            "positive_window_rate",
            "worst_max_drawdown",
        ],
    )
    sliding_window_benchmark_summary = _read_required_csv(
        sliding_window_benchmark_dir / "sliding_window_benchmark_summary.csv",
        required_columns=[
            "strategy_mean_total_return",
            "strategy_positive_window_rate",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
        ],
    )
    overnight_candidate_metrics = _read_required_csv(
        overnight_candidate_report_dir / "sliding_window_candidate_metrics.csv",
        required_columns=[
            "universe",
            "model_type",
            "top_k",
            "allocation_mode",
            "transaction_cost_bps",
            "avg_test_return",
            "positive_rate",
            "worst_max_drawdown",
        ],
    )
    shortlist_benchmark_summary = _read_required_csv(
        shortlist_benchmark_report_dir / "sliding_window_shortlist_benchmark_summary.csv",
        required_columns=[
            "candidate_id",
            "candidate_label",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "mean_total_return",
            "worst_max_drawdown",
        ],
    )
    longer_window_candidate_summary = _read_required_csv(
        longer_window_report_dir / "longer_window_shortlist_candidate_summary.csv",
        required_columns=[
            "candidate_id",
            "candidate_label",
            "mean_total_return",
            "positive_window_rate",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "worst_max_drawdown",
        ],
    )

    shortlist_candidate_summary = _read_optional_csv(
        shortlist_benchmark_report_dir / "sliding_window_shortlist_candidate_summary.csv",
        required_columns=[
            "candidate_id",
            "candidate_label",
            "mean_total_return",
            "positive_window_rate",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "worst_max_drawdown",
        ],
        warnings=warnings,
    )
    allocation_champion_comparison = _read_optional_csv(
        allocation_report_dir / "allocation_champion_comparison.csv",
        required_columns=["is_current_champion", "delta_mean_total_return_vs_champion"],
        warnings=warnings,
    )
    narrow_allocation_champion_comparison = _read_optional_csv(
        narrow_allocation_report_dir / "allocation_champion_comparison.csv",
        required_columns=["is_current_champion", "delta_mean_total_return_vs_champion"],
        warnings=warnings,
    )

    return FinalCandidateReportInputs(
        sliding_window_summary=sliding_window_summary,
        sliding_window_benchmark_summary=sliding_window_benchmark_summary,
        overnight_candidate_metrics=overnight_candidate_metrics,
        shortlist_benchmark_summary=shortlist_benchmark_summary,
        shortlist_candidate_summary=shortlist_candidate_summary,
        longer_window_candidate_summary=longer_window_candidate_summary,
        allocation_champion_comparison=allocation_champion_comparison,
        narrow_allocation_champion_comparison=narrow_allocation_champion_comparison,
        warnings=tuple(warnings),
    )


def build_final_candidate_recommendation_report(
    inputs: FinalCandidateReportInputs,
) -> FinalCandidateRecommendationReportData:
    """Create final candidate summary and decision matrix tables."""

    warnings = list(inputs.warnings)
    final_candidate_summary = create_final_candidate_summary(inputs, warnings=warnings)
    decision_matrix = create_final_candidate_decision_matrix(final_candidate_summary)
    return FinalCandidateRecommendationReportData(
        final_candidate_summary=final_candidate_summary,
        decision_matrix=decision_matrix,
        warnings=tuple(warnings),
    )


def create_final_candidate_summary(
    inputs: FinalCandidateReportInputs,
    *,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Create the final candidate summary table in the required row order."""

    warning_sink = [] if warnings is None else warnings
    rows = []
    for spec in FINAL_CANDIDATE_SPECS:
        candidate_id = spec.candidate_id
        overnight_row = _find_candidate_row(
            inputs.overnight_candidate_metrics,
            spec=spec,
            candidate_id_column="candidate_id",
        )
        shortlist_row = _find_by_candidate_id(
            inputs.shortlist_candidate_summary,
            candidate_id,
        )
        shortlist_benchmark_row = _find_by_candidate_id(
            inputs.shortlist_benchmark_summary,
            candidate_id,
        )
        longer_row = _find_by_candidate_id(
            inputs.longer_window_candidate_summary,
            candidate_id,
        )

        if candidate_id != FIXED_FOLD_CHAMPION_ID and overnight_row is None:
            warning_sink.append(
                f"Candidate `{spec.candidate_label}` was not found in overnight metrics."
            )
        if candidate_id != FIXED_FOLD_CHAMPION_ID and shortlist_benchmark_row is None:
            warning_sink.append(
                f"Candidate `{spec.candidate_label}` was not found in 3-month "
                "shortlist benchmark summary."
            )
        if candidate_id != FIXED_FOLD_CHAMPION_ID and longer_row is None:
            warning_sink.append(
                f"Candidate `{spec.candidate_label}` was not found in longer-window "
                "shortlist summary."
            )

        rows.append(
            {
                "candidate_label": spec.candidate_label,
                "fixed_fold_role": _fixed_fold_role(candidate_id),
                "allocation_result_role": _allocation_result_role(candidate_id, inputs),
                "three_month_sliding_role": _three_month_sliding_role(candidate_id),
                "three_month_benchmark_role": _three_month_benchmark_role(
                    candidate_id,
                    shortlist_benchmark_row,
                ),
                "longer_window_role": _longer_window_role(candidate_id),
                "mean_total_return_3m": _three_month_metric(
                    candidate_id,
                    inputs,
                    shortlist_row,
                    shortlist_benchmark_row,
                    overnight_row,
                    metric="mean_total_return",
                    overnight_metric="avg_test_return",
                    champion_metric="mean_total_return",
                ),
                "mean_total_return_longer": _source_value(
                    longer_row,
                    "mean_total_return",
                ),
                "positive_rate_3m": _three_month_metric(
                    candidate_id,
                    inputs,
                    shortlist_row,
                    shortlist_benchmark_row,
                    overnight_row,
                    metric="positive_window_rate",
                    overnight_metric="positive_rate",
                    champion_metric="positive_window_rate",
                ),
                "positive_rate_longer": _source_value(
                    longer_row,
                    "positive_window_rate",
                ),
                "beats_spy_rate_3m": _three_month_benchmark_metric(
                    candidate_id,
                    inputs,
                    shortlist_row,
                    shortlist_benchmark_row,
                    shortlist_metric="strategy_beats_spy_rate",
                    champion_metric="strategy_beats_spy_rate",
                ),
                "beats_spy_rate_longer": _source_value(
                    longer_row,
                    "strategy_beats_spy_rate",
                ),
                "beats_qqq_rate_3m": _three_month_benchmark_metric(
                    candidate_id,
                    inputs,
                    shortlist_row,
                    shortlist_benchmark_row,
                    shortlist_metric="strategy_beats_qqq_rate",
                    champion_metric="strategy_beats_qqq_rate",
                ),
                "beats_qqq_rate_longer": _source_value(
                    longer_row,
                    "strategy_beats_qqq_rate",
                ),
                "worst_max_drawdown_3m": _three_month_metric(
                    candidate_id,
                    inputs,
                    shortlist_row,
                    shortlist_benchmark_row,
                    overnight_row,
                    metric="worst_max_drawdown",
                    overnight_metric="worst_max_drawdown",
                    champion_metric="worst_max_drawdown",
                ),
                "worst_max_drawdown_longer": _source_value(
                    longer_row,
                    "worst_max_drawdown",
                ),
                "recommendation_label": _recommendation_label(candidate_id),
            }
        )
    return pd.DataFrame(rows, columns=FINAL_SUMMARY_COLUMNS)


def create_final_candidate_decision_matrix(
    final_candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create categorical candidate decision ratings from the final summary table."""

    rows = []
    for _, row in final_candidate_summary.iterrows():
        candidate_label = str(row["candidate_label"])
        candidate_id = _candidate_id_from_label(candidate_label)
        rows.append(
            {
                "candidate_label": candidate_label,
                "return_strength": _return_strength(row),
                "benchmark_relative_strength": _benchmark_relative_strength(row),
                "drawdown_stability": _drawdown_stability(row),
                "universe_stability": _universe_stability(candidate_id),
                "model_complexity": _model_complexity(candidate_id),
                "interpretability": _interpretability(candidate_id),
                "risk_of_overfitting": _risk_of_overfitting(candidate_id),
                "product_explainability": _product_explainability(candidate_id),
                "next_action": _next_action(candidate_id),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def write_final_candidate_recommendation_report(
    *,
    report_data: FinalCandidateRecommendationReportData,
    output_dir: Path,
) -> dict[str, Path]:
    """Write final synthesis CSVs and markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.final_candidate_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_final_candidate_recommendation_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_final_candidate_recommendation_markdown(
    report_data: FinalCandidateRecommendationReportData,
) -> str:
    """Render the final candidate recommendation report as Markdown."""

    warnings_section = []
    if report_data.warnings:
        warnings_section = [
            "## Input Warnings",
            "",
            *[f"- {warning}" for warning in report_data.warnings],
            "",
        ]

    lines = [
        "# Final Candidate Recommendation Report",
        "",
        "## Executive Summary",
        "",
        "This synthesis combines the recent allocation, sliding-window stability, "
        "benchmark comparison, overnight candidate, 3-month shortlist benchmark, "
        "and longer-window shortlist results. The evidence does not support a "
        "production-style promotion yet. The original fixed-fold return champion "
        "should stay in the research record, but it should not be treated as the "
        "final production candidate.",
        "",
        "The two strongest directions for the next confirmation work are "
        "`broad_macro / linear_regression / top1 / equal_weight / cost10` for "
        "benchmark-relative behavior and "
        "`full / ridge / top5 / equal_weight / cost10` for longer-window balanced "
        "behavior.",
        "",
        *warnings_section,
        "## Research Stages Completed",
        "",
        "- Allocation studies: allocation variants reduced drawdown in places but "
        "sacrificed too much return, so none replaced the fixed-fold return champion.",
        "- Sliding-window champion stability: the original champion was not confirmed "
        "under 3-month rolling windows.",
        "- Sliding-window benchmark comparison: the original champion underperformed "
        "SPY and QQQ over many same-period windows.",
        "- Overnight candidate study: broad_macro and diversified top-k candidates "
        "looked stronger than the original champion under 3-month rolling robustness.",
        "- Shortlist benchmark report: broad_macro linear top1 led benchmark-relative "
        "behavior; broad_macro random_forest top3 had strong 3-month return/risk.",
        "- Longer-window shortlist report: broad_macro linear top1 remained the "
        "benchmark-relative leader, while full ridge top5 became the balanced "
        "longer-window candidate.",
        "",
        "## What Changed From The Original Champion",
        "",
        "The original fixed-fold champion still explains the starting point of the "
        "research path, but its 3-month rolling return was close to flat and its "
        "same-window benchmark comparison was weak. The later candidate studies "
        "shifted attention from fixed-fold return leadership toward robustness "
        "across rolling windows, benchmark-relative behavior, and drawdown-aware "
        "candidate references.",
        "",
        "## Candidate Role Definitions",
        "",
        "- Fixed-fold return champion: the original fixed-fold return leader kept as "
        "a historical reference.",
        "- Benchmark-relative sliding-window candidate: the candidate with the best "
        "evidence versus SPY and QQQ across rolling windows.",
        "- Longer-window balanced candidate: the candidate with the strongest "
        "longer-window return/risk balance.",
        "- Full-universe balanced reference: a full-universe ridge reference kept for "
        "comparison against the balanced leader.",
        "- Defensive broad_macro reference: a broad_macro ridge reference kept for "
        "drawdown-oriented comparison.",
        "",
        "## Final Candidate Summary Table",
        "",
        _markdown_table(_format_final_summary_for_markdown(report_data.final_candidate_summary)),
        "",
        "## Decision Matrix",
        "",
        _markdown_table(report_data.decision_matrix),
        "",
        "## Recommended Research-Stage Labels",
        "",
        "- Fixed-fold return champion: "
        "`full / linear_regression / top1 / equal_weight / cost10`",
        "- Benchmark-relative sliding-window candidate: "
        "`broad_macro / linear_regression / top1 / equal_weight / cost10`",
        "- Longer-window balanced candidate: "
        "`full / ridge / top5 / equal_weight / cost10`",
        "- Full-universe balanced reference: "
        "`full / ridge / top3 / equal_weight / cost10`",
        "- Defensive broad_macro reference: "
        "`broad_macro / ridge / top5 / equal_weight / cost10`",
        "",
        "## Recommended Next Experiment",
        "",
        "First, update the project source docs after this final recommendation is "
        "reviewed. Then run a small confirmation study, or create a promotion-grade "
        "model card only after the candidate decision is stable. The confirmation "
        "study should focus on adjacent regimes, longer windows, and candidate "
        "robustness for broad_macro linear top1 and full ridge top5 before any daily "
        "signal export or model card is updated.",
        "",
        "## What Not To Do Next",
        "",
        "- Do not promote a production-style model from this evidence.",
        "- Do not treat the original fixed-fold champion as the final production "
        "candidate.",
        "- Do not rerun broad sweeps before converting this synthesis into reviewed "
        "project documentation.",
        "- Do not update daily signal export behavior or the model card until the "
        "candidate decision is stable.",
        "",
        "## Final Conclusion",
        "",
        "Do not promote a production-style model yet. Do not treat the original "
        "fixed-fold champion as the final production candidate. The next serious "
        "research should focus on confirming the two strongest directions: "
        "broad_macro linear top1 for benchmark-relative behavior and full ridge top5 "
        "for longer-window balanced behavior. A future follow-up could test adjacent "
        "regimes, longer windows, and candidate robustness before updating daily "
        "signal export or the model card.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a final synthesis and candidate recommendation report."
    )
    parser.add_argument(
        "--allocation-report-dir",
        type=Path,
        default=DEFAULT_ALLOCATION_REPORT_DIR,
    )
    parser.add_argument(
        "--narrow-allocation-report-dir",
        type=Path,
        default=DEFAULT_NARROW_ALLOCATION_REPORT_DIR,
    )
    parser.add_argument(
        "--sliding-window-report-dir",
        type=Path,
        default=DEFAULT_SLIDING_WINDOW_REPORT_DIR,
    )
    parser.add_argument(
        "--sliding-window-benchmark-dir",
        type=Path,
        default=DEFAULT_SLIDING_WINDOW_BENCHMARK_DIR,
    )
    parser.add_argument(
        "--overnight-candidate-report-dir",
        type=Path,
        default=DEFAULT_OVERNIGHT_CANDIDATE_REPORT_DIR,
    )
    parser.add_argument(
        "--shortlist-benchmark-report-dir",
        type=Path,
        default=DEFAULT_SHORTLIST_BENCHMARK_REPORT_DIR,
    )
    parser.add_argument(
        "--longer-window-report-dir",
        type=Path,
        default=DEFAULT_LONGER_WINDOW_REPORT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _read_required_csv(path: Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    frame = pd.read_csv(path)
    _validate_columns(frame, required_columns=required_columns, path=path)
    return frame


def _read_optional_csv(
    path: Path,
    *,
    required_columns: Sequence[str],
    warnings: list[str],
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Optional report input is missing: {path}")
        return None
    frame = pd.read_csv(path)
    try:
        _validate_columns(frame, required_columns=required_columns, path=path)
    except ValueError as exc:
        warnings.append(str(exc))
        return None
    return frame


def _validate_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    path: Path,
) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Report input {path} is missing required columns: {missing}")


def _find_candidate_row(
    frame: pd.DataFrame | None,
    *,
    spec: CandidateSpec,
    candidate_id_column: str,
) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    if "universe" in frame and "model_type" in frame and "top_k" in frame:
        numeric_cost = pd.to_numeric(frame.get("transaction_cost_bps"), errors="coerce")
        numeric_top_k = pd.to_numeric(frame["top_k"], errors="coerce")
        mask = (
            frame["universe"].eq(spec.universe)
            & frame["model_type"].eq(spec.model_type)
            & numeric_top_k.eq(spec.top_k)
            & frame.get("allocation_mode", "").eq(spec.allocation_mode)
            & np.isclose(
                numeric_cost.astype(float),
                spec.transaction_cost_bps,
                rtol=0.0,
                atol=1e-9,
            )
        )
        matches = frame.loc[mask]
        if not matches.empty:
            return matches.iloc[0]
    if candidate_id_column in frame.columns:
        matches = frame.loc[frame[candidate_id_column].astype(str).eq(spec.candidate_id)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _find_by_candidate_id(frame: pd.DataFrame | None, candidate_id: str) -> pd.Series | None:
    if frame is None or frame.empty or "candidate_id" not in frame.columns:
        return None
    matches = frame.loc[frame["candidate_id"].astype(str).eq(candidate_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _fixed_fold_role(candidate_id: str) -> str:
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return "fixed-fold return champion"
    return "not fixed-fold champion"


def _allocation_result_role(
    candidate_id: str,
    inputs: FinalCandidateReportInputs,
) -> str:
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return "retained after allocation studies"
    if (
        inputs.allocation_champion_comparison is None
        and inputs.narrow_allocation_champion_comparison is None
    ):
        return "allocation evidence unavailable"
    return "no allocation promotion"


def _three_month_sliding_role(candidate_id: str) -> str:
    roles = {
        FIXED_FOLD_CHAMPION_ID: "stability not confirmed",
        BENCHMARK_RELATIVE_LEADER_ID: "strong 3-month benchmark candidate",
        "broad_macro:linear_regression:top2:equal_weight:cost10": (
            "adjacent broad_macro linear candidate"
        ),
        FULL_UNIVERSE_REFERENCE_ID: "full-universe balanced reference",
        LONGER_WINDOW_BALANCED_LEADER_ID: "balanced shortlist candidate",
        DEFENSIVE_BROAD_MACRO_REFERENCE_ID: "defensive broad_macro reference",
        RANDOM_FOREST_3M_REFERENCE_ID: "strong 3-month return/risk candidate",
    }
    return roles[candidate_id]


def _three_month_benchmark_role(
    candidate_id: str,
    shortlist_benchmark_row: pd.Series | None,
) -> str:
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return "underperformed SPY/QQQ in many windows"
    if candidate_id == BENCHMARK_RELATIVE_LEADER_ID:
        return "best 3-month benchmark-relative candidate"
    if shortlist_benchmark_row is not None and "rank" in shortlist_benchmark_row:
        return f"shortlist benchmark rank {int(shortlist_benchmark_row['rank'])}"
    return "benchmark role unknown"


def _longer_window_role(candidate_id: str) -> str:
    roles = {
        FIXED_FOLD_CHAMPION_ID: "not in longer-window shortlist",
        BENCHMARK_RELATIVE_LEADER_ID: "longer-window benchmark-relative leader",
        "broad_macro:linear_regression:top2:equal_weight:cost10": (
            "adjacent longer-window reference"
        ),
        FULL_UNIVERSE_REFERENCE_ID: "full-universe balanced reference",
        LONGER_WINDOW_BALANCED_LEADER_ID: "longer-window balanced leader",
        DEFENSIVE_BROAD_MACRO_REFERENCE_ID: "defensive broad_macro reference",
        RANDOM_FOREST_3M_REFERENCE_ID: "3-month risk candidate not confirmed",
    }
    return roles[candidate_id]


def _recommendation_label(candidate_id: str) -> str:
    labels = {
        FIXED_FOLD_CHAMPION_ID: "keep as fixed-fold reference only",
        BENCHMARK_RELATIVE_LEADER_ID: "validate benchmark-relative direction",
        "broad_macro:linear_regression:top2:equal_weight:cost10": (
            "keep as adjacent broad_macro check"
        ),
        FULL_UNIVERSE_REFERENCE_ID: "keep as full-universe balanced reference",
        LONGER_WINDOW_BALANCED_LEADER_ID: "validate longer-window balanced direction",
        DEFENSIVE_BROAD_MACRO_REFERENCE_ID: "keep as defensive broad_macro reference",
        RANDOM_FOREST_3M_REFERENCE_ID: "deprioritize pending robustness",
    }
    return labels[candidate_id]


def _three_month_metric(
    candidate_id: str,
    inputs: FinalCandidateReportInputs,
    shortlist_row: pd.Series | None,
    shortlist_benchmark_row: pd.Series | None,
    overnight_row: pd.Series | None,
    *,
    metric: str,
    overnight_metric: str,
    champion_metric: str,
) -> float:
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return _source_value(inputs.sliding_window_summary.iloc[0], champion_metric)
    for row, column in [
        (shortlist_row, metric),
        (shortlist_benchmark_row, metric),
        (overnight_row, overnight_metric),
    ]:
        value = _source_value(row, column)
        if not pd.isna(value):
            return value
    return float("nan")


def _three_month_benchmark_metric(
    candidate_id: str,
    inputs: FinalCandidateReportInputs,
    shortlist_row: pd.Series | None,
    shortlist_benchmark_row: pd.Series | None,
    *,
    shortlist_metric: str,
    champion_metric: str,
) -> float:
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return _source_value(inputs.sliding_window_benchmark_summary.iloc[0], champion_metric)
    for row in [shortlist_row, shortlist_benchmark_row]:
        value = _source_value(row, shortlist_metric)
        if not pd.isna(value):
            return value
    return float("nan")


def _source_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return float("nan")
    return float(value)


def _return_strength(row: pd.Series) -> str:
    mean_return = _prefer_longer(row, "mean_total_return_longer", "mean_total_return_3m")
    positive_rate = _prefer_longer(row, "positive_rate_longer", "positive_rate_3m")
    if pd.isna(mean_return) or pd.isna(positive_rate):
        return "unknown"
    if mean_return >= 0.07 and positive_rate >= 0.70:
        return "strong"
    if mean_return >= 0.025 and positive_rate >= 0.55:
        return "moderate"
    return "weak"


def _benchmark_relative_strength(row: pd.Series) -> str:
    beats_spy = _prefer_longer(row, "beats_spy_rate_longer", "beats_spy_rate_3m")
    beats_qqq = _prefer_longer(row, "beats_qqq_rate_longer", "beats_qqq_rate_3m")
    if pd.isna(beats_spy) or pd.isna(beats_qqq):
        return "unknown"
    average = (beats_spy + beats_qqq) / 2.0
    if average >= 0.45:
        return "strong"
    if average >= 0.38:
        return "moderate"
    return "weak"


def _drawdown_stability(row: pd.Series) -> str:
    values = [
        _source_value(row, "worst_max_drawdown_3m"),
        _source_value(row, "worst_max_drawdown_longer"),
    ]
    known = [value for value in values if not pd.isna(value)]
    if not known:
        return "unknown"
    worst = min(known)
    if worst >= -0.25:
        return "strong"
    if worst >= -0.32:
        return "moderate"
    return "weak"


def _universe_stability(candidate_id: str) -> str:
    if candidate_id in {FULL_UNIVERSE_REFERENCE_ID, LONGER_WINDOW_BALANCED_LEADER_ID}:
        return "strong"
    if candidate_id == RANDOM_FOREST_3M_REFERENCE_ID:
        return "weak"
    if candidate_id == FIXED_FOLD_CHAMPION_ID:
        return "weak"
    return "moderate"


def _model_complexity(candidate_id: str) -> str:
    if "random_forest" in candidate_id:
        return "moderate"
    return "weak"


def _interpretability(candidate_id: str) -> str:
    if "random_forest" in candidate_id:
        return "moderate"
    return "strong"


def _risk_of_overfitting(candidate_id: str) -> str:
    if "random_forest" in candidate_id:
        return "moderate"
    if "top1" in candidate_id:
        return "moderate"
    return "weak"


def _product_explainability(candidate_id: str) -> str:
    if "random_forest" in candidate_id:
        return "moderate"
    return "strong"


def _next_action(candidate_id: str) -> str:
    actions = {
        FIXED_FOLD_CHAMPION_ID: "Keep as historical fixed-fold reference.",
        BENCHMARK_RELATIVE_LEADER_ID: "Prioritize confirmation for benchmark-relative behavior.",
        "broad_macro:linear_regression:top2:equal_weight:cost10": (
            "Use only as an adjacent broad_macro sensitivity check."
        ),
        FULL_UNIVERSE_REFERENCE_ID: "Keep as full-universe balanced comparison reference.",
        LONGER_WINDOW_BALANCED_LEADER_ID: (
            "Prioritize confirmation for longer-window balanced behavior."
        ),
        DEFENSIVE_BROAD_MACRO_REFERENCE_ID: (
            "Keep as defensive drawdown-oriented comparison reference."
        ),
        RANDOM_FOREST_3M_REFERENCE_ID: (
            "Deprioritize unless future robustness checks revive it."
        ),
    }
    return actions[candidate_id]


def _prefer_longer(row: pd.Series, longer_column: str, fallback_column: str) -> float:
    longer_value = _source_value(row, longer_column)
    if not pd.isna(longer_value):
        return longer_value
    return _source_value(row, fallback_column)


def _candidate_id_from_label(candidate_label: str) -> str:
    parts = [part.strip() for part in candidate_label.split("/")]
    if len(parts) != 5:
        raise ValueError(f"Invalid candidate label: {candidate_label}")
    universe, model_type, top_k, allocation_mode, cost = parts
    return f"{universe}:{model_type}:{top_k}:{allocation_mode}:{cost}"


def _format_final_summary_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_label",
        "fixed_fold_role",
        "three_month_sliding_role",
        "longer_window_role",
        "mean_total_return_3m",
        "mean_total_return_longer",
        "positive_rate_3m",
        "positive_rate_longer",
        "beats_spy_rate_3m",
        "beats_spy_rate_longer",
        "beats_qqq_rate_3m",
        "beats_qqq_rate_longer",
        "recommendation_label",
    ]
    output = summary[columns].copy()
    percent_columns = [
        "mean_total_return_3m",
        "mean_total_return_longer",
        "positive_rate_3m",
        "positive_rate_longer",
        "beats_spy_rate_3m",
        "beats_spy_rate_longer",
        "beats_qqq_rate_3m",
        "beats_qqq_rate_longer",
    ]
    for column in percent_columns:
        output[column] = output[column].map(_format_percent)
    return output


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    separator = ["---" for _ in headers]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_compact(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def main() -> int:
    return run_final_candidate_recommendation_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
