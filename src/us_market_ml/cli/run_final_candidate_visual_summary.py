"""Generate visual diagnostics for the final research-stage candidate shortlist."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FINAL_CANDIDATE_SUMMARY = Path(
    "artifacts/reports/final_candidate_recommendation/final_candidate_summary.csv"
)
DEFAULT_DECISION_MATRIX = Path(
    "artifacts/reports/final_candidate_recommendation/"
    "final_candidate_decision_matrix.csv"
)
DEFAULT_SHORTLIST_BENCHMARK_SUMMARY = Path(
    "artifacts/reports/sliding_window_shortlist_benchmark/"
    "sliding_window_shortlist_candidate_summary.csv"
)
DEFAULT_LONGER_WINDOW_SUMMARY = Path(
    "artifacts/reports/sliding_window_longer_window_shortlist_study/"
    "longer_window_shortlist_candidate_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_candidate_visual_summary")

REPORT_PATH_NAME = "final_candidate_visual_summary.md"
SUMMARY_PATH_NAME = "final_candidate_visual_summary.csv"
MEAN_RETURN_PNG = "candidate_mean_return.png"
POSITIVE_RATE_PNG = "candidate_positive_rate.png"
DRAWDOWN_PNG = "candidate_drawdown_comparison.png"
BENCHMARK_RELATIVE_PNG = "candidate_benchmark_relative.png"
RETURN_COMPARISON_PNG = "candidate_3m_vs_longer_return.png"
POSITIVE_RATE_COMPARISON_PNG = "candidate_3m_vs_longer_positive_rate.png"
RETURN_DRAWDOWN_SCATTER_PNG = "return_vs_worst_drawdown_scatter.png"
STABILITY_BALANCED_SCATTER_PNG = (
    "benchmark_stability_vs_balanced_score_scatter.png"
)

OUTPUT_PATH_NAMES = (
    REPORT_PATH_NAME,
    SUMMARY_PATH_NAME,
    MEAN_RETURN_PNG,
    POSITIVE_RATE_PNG,
    DRAWDOWN_PNG,
    BENCHMARK_RELATIVE_PNG,
    RETURN_COMPARISON_PNG,
    POSITIVE_RATE_COMPARISON_PNG,
    RETURN_DRAWDOWN_SCATTER_PNG,
    STABILITY_BALANCED_SCATTER_PNG,
)

FINAL_SUMMARY_REQUIRED_COLUMNS = [
    "candidate_label",
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
DECISION_MATRIX_REQUIRED_COLUMNS = [
    "candidate_label",
    "return_strength",
    "benchmark_relative_strength",
    "drawdown_stability",
    "next_action",
]
OPTIONAL_SUMMARY_REQUIRED_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "mean_total_return",
    "positive_window_rate",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "worst_max_drawdown",
]

FIXED_FOLD_CHAMPION_ID = "full:linear_regression:top1:equal_weight:cost10"
BENCHMARK_RELATIVE_LEADER_ID = (
    "broad_macro:linear_regression:top1:equal_weight:cost10"
)
LONGER_WINDOW_BALANCED_LEADER_ID = "full:ridge:top5:equal_weight:cost10"
FULL_UNIVERSE_REFERENCE_ID = "full:ridge:top3:equal_weight:cost10"
DEFENSIVE_BROAD_MACRO_REFERENCE_ID = "broad_macro:ridge:top5:equal_weight:cost10"
RANDOM_FOREST_3M_REFERENCE_ID = "broad_macro:random_forest:top3:equal_weight:cost10"

KEY_CANDIDATE_IDS = {
    FIXED_FOLD_CHAMPION_ID,
    BENCHMARK_RELATIVE_LEADER_ID,
    LONGER_WINDOW_BALANCED_LEADER_ID,
    FULL_UNIVERSE_REFERENCE_ID,
    DEFENSIVE_BROAD_MACRO_REFERENCE_ID,
    RANDOM_FOREST_3M_REFERENCE_ID,
}

PLOT_SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "short_label",
    "recommendation_label",
    "return_strength",
    "benchmark_relative_strength",
    "drawdown_stability",
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
    "benchmark_stability_score_3m",
    "return_risk_score_3m",
    "benchmark_stability_score_longer",
    "balanced_score_longer",
    "benchmark_stability_score",
    "balanced_score",
]


@dataclass(frozen=True)
class FinalCandidateVisualSummaryInputs:
    """Input tables used by the visual summary report."""

    final_candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    shortlist_benchmark_summary: pd.DataFrame | None
    longer_window_summary: pd.DataFrame | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FinalCandidateVisualSummaryReportData:
    """Prepared plotting table and source warnings."""

    plot_summary: pd.DataFrame
    warnings: tuple[str, ...]


def run_final_candidate_visual_summary_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write visual diagnostics for final candidate review."""

    args = _parse_args(argv)
    inputs = read_final_candidate_visual_summary_inputs(
        final_candidate_summary=args.final_candidate_summary,
        decision_matrix=args.decision_matrix,
        shortlist_benchmark_summary=args.shortlist_benchmark_summary,
        longer_window_summary=args.longer_window_summary,
    )
    report_data = build_final_candidate_visual_summary(inputs)
    outputs = write_final_candidate_visual_summary(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML final candidate visual summary")
    print(f"final candidate summary: {args.final_candidate_summary}")
    print(f"decision matrix: {args.decision_matrix}")
    print(f"3-month shortlist summary: {args.shortlist_benchmark_summary}")
    print(f"longer-window summary: {args.longer_window_summary}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    if report_data.warnings:
        print()
        print("Warnings:")
        for warning in report_data.warnings:
            print(f"- {warning}")
    print()
    print(report_data.plot_summary.to_string(index=False))
    return 0


def read_final_candidate_visual_summary_inputs(
    *,
    final_candidate_summary: Path = DEFAULT_FINAL_CANDIDATE_SUMMARY,
    decision_matrix: Path = DEFAULT_DECISION_MATRIX,
    shortlist_benchmark_summary: Path = DEFAULT_SHORTLIST_BENCHMARK_SUMMARY,
    longer_window_summary: Path = DEFAULT_LONGER_WINDOW_SUMMARY,
) -> FinalCandidateVisualSummaryInputs:
    """Read required final tables and optional shortlist score tables."""

    warnings: list[str] = []
    final_summary = _read_required_csv(
        final_candidate_summary,
        required_columns=FINAL_SUMMARY_REQUIRED_COLUMNS,
    )
    matrix = _read_required_csv(
        decision_matrix,
        required_columns=DECISION_MATRIX_REQUIRED_COLUMNS,
    )
    shortlist_summary = _read_optional_csv(
        shortlist_benchmark_summary,
        required_columns=OPTIONAL_SUMMARY_REQUIRED_COLUMNS,
        warnings=warnings,
    )
    longer_summary = _read_optional_csv(
        longer_window_summary,
        required_columns=OPTIONAL_SUMMARY_REQUIRED_COLUMNS,
        warnings=warnings,
    )
    return FinalCandidateVisualSummaryInputs(
        final_candidate_summary=final_summary,
        decision_matrix=matrix,
        shortlist_benchmark_summary=shortlist_summary,
        longer_window_summary=longer_summary,
        warnings=tuple(warnings),
    )


def build_final_candidate_visual_summary(
    inputs: FinalCandidateVisualSummaryInputs,
) -> FinalCandidateVisualSummaryReportData:
    """Prepare the plotting summary table from final and optional report inputs."""

    plot_summary = prepare_final_candidate_plot_data(
        final_candidate_summary=inputs.final_candidate_summary,
        decision_matrix=inputs.decision_matrix,
        shortlist_benchmark_summary=inputs.shortlist_benchmark_summary,
        longer_window_summary=inputs.longer_window_summary,
    )
    return FinalCandidateVisualSummaryReportData(
        plot_summary=plot_summary,
        warnings=inputs.warnings,
    )


def prepare_final_candidate_plot_data(
    *,
    final_candidate_summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    shortlist_benchmark_summary: pd.DataFrame | None = None,
    longer_window_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one enriched plotting row per final-stage candidate."""

    _validate_columns(
        final_candidate_summary,
        required_columns=FINAL_SUMMARY_REQUIRED_COLUMNS,
        path=Path("final_candidate_summary"),
    )
    _validate_columns(
        decision_matrix,
        required_columns=DECISION_MATRIX_REQUIRED_COLUMNS,
        path=Path("final_candidate_decision_matrix"),
    )

    summary = final_candidate_summary.copy()
    summary["candidate_id"] = summary["candidate_label"].map(_candidate_id_from_label)
    summary["short_label"] = summary["candidate_label"].map(_short_candidate_label)

    matrix_columns = [
        "candidate_label",
        "return_strength",
        "benchmark_relative_strength",
        "drawdown_stability",
        "next_action",
    ]
    output = summary.merge(
        decision_matrix[matrix_columns],
        on="candidate_label",
        how="left",
        validate="one_to_one",
    )

    if shortlist_benchmark_summary is not None:
        shortlist_scores = _prepare_optional_scores(
            shortlist_benchmark_summary,
            suffix="3m",
            score_columns=["benchmark_stability_score", "return_risk_score"],
        )
        output = output.merge(
            shortlist_scores,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )

    if longer_window_summary is not None:
        longer_scores = _prepare_optional_scores(
            longer_window_summary,
            suffix="longer",
            score_columns=["benchmark_stability_score", "balanced_score"],
        )
        output = output.merge(
            longer_scores,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )

    for column in PLOT_SUMMARY_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan

    output["benchmark_stability_score"] = output[
        "benchmark_stability_score_longer"
    ].combine_first(output["benchmark_stability_score_3m"])
    output["balanced_score"] = output["balanced_score_longer"]

    numeric_columns = [
        column
        for column in PLOT_SUMMARY_COLUMNS
        if column
        not in {
            "candidate_id",
            "candidate_label",
            "short_label",
            "recommendation_label",
            "return_strength",
            "benchmark_relative_strength",
            "drawdown_stability",
        }
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    return output[PLOT_SUMMARY_COLUMNS]


def write_final_candidate_visual_summary(
    *,
    report_data: FinalCandidateVisualSummaryReportData,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the plotting CSV, PNG charts, and markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / name for name in OUTPUT_PATH_NAMES}
    report_data.plot_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    _write_all_charts(report_data.plot_summary, output_dir=output_dir)
    outputs[REPORT_PATH_NAME].write_text(
        render_final_candidate_visual_summary_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_final_candidate_visual_summary_markdown(
    report_data: FinalCandidateVisualSummaryReportData,
) -> str:
    """Render the visual summary markdown report."""

    warning_lines = []
    if report_data.warnings:
        warning_lines = [
            "## Input Warnings",
            "",
            *[f"- {warning}" for warning in report_data.warnings],
            "",
        ]

    lines = [
        "# Final Candidate Visual Summary",
        "",
        "This visual summary is a downstream diagnostic report for the final "
        "research-stage candidate recommendation. It does not rerun sweeps, "
        "change model training, update daily signal export, or alter model card "
        "logic.",
        "",
        "Current conclusion: no production-ready candidate yet. The original "
        "fixed-fold champion is a historical fixed-fold reference only. The two "
        "strongest research directions are "
        "`broad_macro / linear_regression / top1 / equal_weight / cost10` for "
        "benchmark-relative behavior and "
        "`full / ridge / top5 / equal_weight / cost10` for longer-window balanced "
        "behavior.",
        "",
        *warning_lines,
        "## Mean Total Return By Candidate",
        "",
        "Compares 3-month and longer-window mean total return where values are "
        "available. The longer-window view helps separate short-horizon lift from "
        "candidate behavior that persists across wider validation windows.",
        "",
        f"![Mean total return]({MEAN_RETURN_PNG})",
        "",
        "## Positive Window Rate By Candidate",
        "",
        "Shows the share of validation windows with positive strategy return. This "
        "is a simple robustness check alongside average return.",
        "",
        f"![Positive window rate]({POSITIVE_RATE_PNG})",
        "",
        "## Worst Drawdown Comparison",
        "",
        "Shows each candidate's worst max drawdown from the 3-month and "
        "longer-window summaries. More negative values indicate deeper observed "
        "drawdown.",
        "",
        f"![Worst drawdown comparison]({DRAWDOWN_PNG})",
        "",
        "## Benchmark-Relative Performance",
        "",
        "Compares rates of beating SPY and QQQ, using longer-window values when "
        "available and falling back to 3-month values otherwise.",
        "",
        f"![Benchmark-relative performance]({BENCHMARK_RELATIVE_PNG})",
        "",
        "## 3-Month Vs Longer Mean Return",
        "",
        "A direct grouped comparison of candidate mean total return across the "
        "3-month and longer-window summaries.",
        "",
        f"![3-month vs longer return]({RETURN_COMPARISON_PNG})",
        "",
        "## 3-Month Vs Longer Positive Rate",
        "",
        "A direct grouped comparison of candidate positive-window rates across the "
        "3-month and longer-window summaries.",
        "",
        f"![3-month vs longer positive rate]({POSITIVE_RATE_COMPARISON_PNG})",
        "",
        "## Return Vs Worst Drawdown",
        "",
        "Plots mean return against worst max drawdown, preferring longer-window "
        "metrics where available. Key candidates are annotated so return/risk "
        "tradeoffs are easier to inspect.",
        "",
        f"![Return vs worst drawdown]({RETURN_DRAWDOWN_SCATTER_PNG})",
        "",
        "## Benchmark Stability Vs Balanced Score",
        "",
        "Plots benchmark stability against balanced score when the optional score "
        "columns are present. This is mainly a longer-window shortlist diagnostic.",
        "",
        f"![Benchmark stability vs balanced score]({STABILITY_BALANCED_SCATTER_PNG})",
        "",
        "## Interpretation",
        "",
        "- `broad_macro / linear_regression / top1 / equal_weight / cost10` has "
        "the best benchmark-relative direction.",
        "- `full / ridge / top5 / equal_weight / cost10` has stronger "
        "longer-window balanced behavior.",
        "- `broad_macro / random_forest / top3 / equal_weight / cost10` is "
        "deprioritized after longer-window validation.",
        "",
        "## Final Recommendation",
        "",
        "Review these visuals before running more sweeps. The next possible step "
        "after visual review is deep-dive bundles for the top 2-3 candidates; do "
        "not update the daily signal export or model card yet.",
        "",
    ]
    return "\n".join(lines)


def _write_all_charts(plot_summary: pd.DataFrame, *, output_dir: Path) -> None:
    _ensure_matplotlib_backend()
    _plot_grouped_metric(
        plot_summary,
        value_columns=["mean_total_return_3m", "mean_total_return_longer"],
        legend_labels=["3-month", "Longer-window"],
        output_path=output_dir / MEAN_RETURN_PNG,
        title="Mean Total Return By Candidate",
        ylabel="Mean total return",
        percent_axis=True,
    )
    _plot_grouped_metric(
        plot_summary,
        value_columns=["positive_rate_3m", "positive_rate_longer"],
        legend_labels=["3-month", "Longer-window"],
        output_path=output_dir / POSITIVE_RATE_PNG,
        title="Positive Window Rate By Candidate",
        ylabel="Positive window rate",
        percent_axis=True,
    )
    _plot_grouped_metric(
        plot_summary,
        value_columns=["worst_max_drawdown_3m", "worst_max_drawdown_longer"],
        legend_labels=["3-month", "Longer-window"],
        output_path=output_dir / DRAWDOWN_PNG,
        title="Worst Max Drawdown By Candidate",
        ylabel="Worst max drawdown",
        percent_axis=True,
    )
    _plot_benchmark_relative(
        plot_summary,
        output_path=output_dir / BENCHMARK_RELATIVE_PNG,
    )
    _plot_grouped_metric(
        plot_summary,
        value_columns=["mean_total_return_3m", "mean_total_return_longer"],
        legend_labels=["3-month", "Longer-window"],
        output_path=output_dir / RETURN_COMPARISON_PNG,
        title="3-Month Vs Longer-Window Mean Return",
        ylabel="Mean total return",
        percent_axis=True,
    )
    _plot_grouped_metric(
        plot_summary,
        value_columns=["positive_rate_3m", "positive_rate_longer"],
        legend_labels=["3-month", "Longer-window"],
        output_path=output_dir / POSITIVE_RATE_COMPARISON_PNG,
        title="3-Month Vs Longer-Window Positive Rate",
        ylabel="Positive window rate",
        percent_axis=True,
    )
    _plot_return_drawdown_scatter(
        plot_summary,
        output_path=output_dir / RETURN_DRAWDOWN_SCATTER_PNG,
    )
    _plot_stability_balanced_scatter(
        plot_summary,
        output_path=output_dir / STABILITY_BALANCED_SCATTER_PNG,
    )


def _plot_grouped_metric(
    plot_summary: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    legend_labels: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    percent_axis: bool,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    labels = plot_summary["short_label"].tolist()
    values = plot_summary[list(value_columns)].astype(float)
    x_positions = np.arange(len(labels), dtype=float)
    width = 0.36
    offsets = np.linspace(
        start=-(len(value_columns) - 1) * width / 2,
        stop=(len(value_columns) - 1) * width / 2,
        num=len(value_columns),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#3366AA", "#D55E00", "#009E73"]
    for index, (column, legend_label) in enumerate(zip(value_columns, legend_labels)):
        ax.bar(
            x_positions + offsets[index],
            values[column],
            width=width,
            label=legend_label,
            color=colors[index % len(colors)],
        )

    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if percent_axis:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=150)
    plt.close(fig)


def _plot_benchmark_relative(plot_summary: pd.DataFrame, *, output_path: Path) -> None:
    frame = plot_summary.copy()
    frame["beats_spy_rate"] = frame["beats_spy_rate_longer"].combine_first(
        frame["beats_spy_rate_3m"]
    )
    frame["beats_qqq_rate"] = frame["beats_qqq_rate_longer"].combine_first(
        frame["beats_qqq_rate_3m"]
    )
    _plot_grouped_metric(
        frame,
        value_columns=["beats_spy_rate", "beats_qqq_rate"],
        legend_labels=["Beats SPY", "Beats QQQ"],
        output_path=output_path,
        title="Benchmark-Relative Performance",
        ylabel="Window beat rate",
        percent_axis=True,
    )


def _plot_return_drawdown_scatter(
    plot_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    frame = plot_summary.copy()
    frame["mean_return"] = frame["mean_total_return_longer"].combine_first(
        frame["mean_total_return_3m"]
    )
    frame["worst_drawdown"] = frame["worst_max_drawdown_longer"].combine_first(
        frame["worst_max_drawdown_3m"]
    )
    frame = frame.dropna(subset=["mean_return", "worst_drawdown"])

    fig, ax = plt.subplots(figsize=(9, 6))
    if frame.empty:
        _draw_no_data_message(ax, "No return/drawdown data available")
    else:
        ax.scatter(
            frame["worst_drawdown"],
            frame["mean_return"],
            s=70,
            color="#3366AA",
            alpha=0.85,
        )
        _annotate_key_candidates(
            ax,
            frame,
            x_column="worst_drawdown",
            y_column="mean_return",
        )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.axvline(0.0, color="#444444", linewidth=0.8)
        ax.set_xlabel("Worst max drawdown")
        ax.set_ylabel("Mean total return")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.25)
    ax.set_title("Return Vs Worst Drawdown")
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=150)
    plt.close(fig)


def _plot_stability_balanced_scatter(
    plot_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    frame = plot_summary.dropna(
        subset=["benchmark_stability_score", "balanced_score"]
    ).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    if frame.empty:
        _draw_no_data_message(
            ax,
            "No benchmark stability/balanced score data available",
        )
    else:
        ax.scatter(
            frame["benchmark_stability_score"],
            frame["balanced_score"],
            s=70,
            color="#009E73",
            alpha=0.85,
        )
        _annotate_key_candidates(
            ax,
            frame,
            x_column="benchmark_stability_score",
            y_column="balanced_score",
        )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.axvline(0.0, color="#444444", linewidth=0.8)
        ax.set_xlabel("Benchmark stability score")
        ax.set_ylabel("Balanced score")
        ax.grid(alpha=0.25)
    ax.set_title("Benchmark Stability Vs Balanced Score")
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=150)
    plt.close(fig)


def _annotate_key_candidates(
    ax: object,
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
) -> None:
    for _, row in frame.iterrows():
        if row["candidate_id"] not in KEY_CANDIDATE_IDS:
            continue
        ax.annotate(
            str(row["short_label"]),
            xy=(row[x_column], row[y_column]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def _draw_no_data_message(ax: object, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _prepare_optional_scores(
    frame: pd.DataFrame,
    *,
    suffix: str,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    columns = ["candidate_id", *[column for column in score_columns if column in frame]]
    output = (
        frame[columns]
        .copy()
        .drop_duplicates(subset=["candidate_id"], keep="first")
    )
    rename_map = {
        column: f"{column}_{suffix}"
        for column in output.columns
        if column != "candidate_id"
    }
    return output.rename(columns=rename_map)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final candidate visual diagnostics from report CSVs."
    )
    parser.add_argument(
        "--final-candidate-summary",
        type=Path,
        default=DEFAULT_FINAL_CANDIDATE_SUMMARY,
    )
    parser.add_argument(
        "--decision-matrix",
        type=Path,
        default=DEFAULT_DECISION_MATRIX,
    )
    parser.add_argument(
        "--shortlist-benchmark-summary",
        type=Path,
        default=DEFAULT_SHORTLIST_BENCHMARK_SUMMARY,
    )
    parser.add_argument(
        "--longer-window-summary",
        type=Path,
        default=DEFAULT_LONGER_WINDOW_SUMMARY,
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


def _candidate_id_from_label(candidate_label: str) -> str:
    parts = [part.strip() for part in candidate_label.split("/")]
    if len(parts) != 5:
        raise ValueError(f"Invalid candidate label: {candidate_label}")
    universe, model_type, top_k, allocation_mode, cost = parts
    return f"{universe}:{model_type}:{top_k}:{allocation_mode}:{cost}"


def _short_candidate_label(candidate_label: str) -> str:
    parts = [part.strip() for part in candidate_label.split("/")]
    if len(parts) < 3:
        return candidate_label
    universe, model_type, top_k = parts[:3]
    universe_label = "broad" if universe == "broad_macro" else universe
    model_label = {
        "linear_regression": "linear",
        "random_forest": "rf",
    }.get(model_type, model_type)
    return f"{universe_label} {model_label} {top_k}"


def _ensure_matplotlib_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")


def main() -> int:
    return run_final_candidate_visual_summary_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
