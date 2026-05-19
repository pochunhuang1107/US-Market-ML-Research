"""Create a Markdown report for the Tiingo selected-candidate rich replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("artifacts/replays/tiingo_selected_candidate_rich_replay")
REPORT_PATH_NAME = "rich_replay_report.md"

DAILY_STRATEGY_EQUITY_NAME = "daily_strategy_equity.csv"
DAILY_HOLDINGS_NAME = "daily_holdings.csv"
REBALANCE_LOG_NAME = "rebalance_log.csv"
TRADE_TURNOVER_DIAGNOSTICS_NAME = "trade_turnover_diagnostics.csv"
FOLD_PERFORMANCE_NAME = "fold_performance.csv"
UNDERPERFORMANCE_WINDOWS_NAME = "underperformance_windows.csv"
SELECTED_SYMBOL_SUMMARY_NAME = "selected_symbol_summary.csv"
RICH_REPLAY_SUMMARY_NAME = "rich_replay_summary.csv"
REFERENCE_PERFORMANCE_NAME = "reference_performance.csv"
FEATURE_SET_COMPARISON_NAME = "feature_set_comparison.csv"
V2_VS_V1_UNDERPERFORMANCE_DELTA_NAME = "v2_vs_v1_underperformance_delta.csv"
V2_SELECTED_SYMBOL_SUMMARY_NAME = "v2_selected_symbol_summary.csv"


def run_tiingo_selected_candidate_rich_replay_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the rich replay Markdown report from CSV artifacts."""

    args = _parse_args(argv)
    artifact_data = read_rich_replay_artifacts(args.output_dir)
    report = render_tiingo_selected_candidate_rich_replay_markdown(artifact_data)
    report_path = (
        args.report_path if args.report_path is not None else args.output_dir / REPORT_PATH_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("US Market ML Tiingo selected-candidate rich replay report")
    print(f"output dir: {args.output_dir}")
    print(f"report path: {report_path}")
    print(f"SPY rows: {_benchmark_row_count(artifact_data.daily_strategy_equity, 'spy')}")
    print(f"QQQ rows: {_benchmark_row_count(artifact_data.daily_strategy_equity, 'qqq')}")
    return 0


class RichReplayReportData:
    """CSV-backed report data for the rich replay bundle."""

    def __init__(
        self,
        *,
        daily_strategy_equity: pd.DataFrame,
        daily_holdings: pd.DataFrame,
        rebalance_log: pd.DataFrame,
        trade_turnover_diagnostics: pd.DataFrame,
        fold_performance: pd.DataFrame,
        underperformance_windows: pd.DataFrame,
        selected_symbol_summary: pd.DataFrame,
        rich_replay_summary: pd.DataFrame,
        reference_performance: pd.DataFrame,
        feature_set_comparison: pd.DataFrame | None = None,
        v2_vs_v1_underperformance_delta: pd.DataFrame | None = None,
        v2_selected_symbol_summary: pd.DataFrame | None = None,
    ) -> None:
        self.daily_strategy_equity = daily_strategy_equity
        self.daily_holdings = daily_holdings
        self.rebalance_log = rebalance_log
        self.trade_turnover_diagnostics = trade_turnover_diagnostics
        self.fold_performance = fold_performance
        self.underperformance_windows = underperformance_windows
        self.selected_symbol_summary = selected_symbol_summary
        self.rich_replay_summary = rich_replay_summary
        self.reference_performance = reference_performance
        self.feature_set_comparison = (
            feature_set_comparison if feature_set_comparison is not None else pd.DataFrame()
        )
        self.v2_vs_v1_underperformance_delta = (
            v2_vs_v1_underperformance_delta
            if v2_vs_v1_underperformance_delta is not None
            else pd.DataFrame()
        )
        self.v2_selected_symbol_summary = (
            v2_selected_symbol_summary
            if v2_selected_symbol_summary is not None
            else pd.DataFrame()
        )


def read_rich_replay_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> RichReplayReportData:
    """Read rich replay CSV artifacts from ``output_dir``."""

    return RichReplayReportData(
        daily_strategy_equity=_read_csv_if_exists(output_dir / DAILY_STRATEGY_EQUITY_NAME),
        daily_holdings=_read_csv_if_exists(output_dir / DAILY_HOLDINGS_NAME),
        rebalance_log=_read_csv_if_exists(output_dir / REBALANCE_LOG_NAME),
        trade_turnover_diagnostics=_read_csv_if_exists(
            output_dir / TRADE_TURNOVER_DIAGNOSTICS_NAME
        ),
        fold_performance=_read_csv_if_exists(output_dir / FOLD_PERFORMANCE_NAME),
        underperformance_windows=_read_csv_if_exists(output_dir / UNDERPERFORMANCE_WINDOWS_NAME),
        selected_symbol_summary=_read_csv_if_exists(output_dir / SELECTED_SYMBOL_SUMMARY_NAME),
        rich_replay_summary=_read_csv_if_exists(output_dir / RICH_REPLAY_SUMMARY_NAME),
        reference_performance=_read_csv_if_exists(output_dir / REFERENCE_PERFORMANCE_NAME),
        feature_set_comparison=_read_csv_if_exists(output_dir / FEATURE_SET_COMPARISON_NAME),
        v2_vs_v1_underperformance_delta=_read_csv_if_exists(
            output_dir / V2_VS_V1_UNDERPERFORMANCE_DELTA_NAME
        ),
        v2_selected_symbol_summary=_read_csv_if_exists(
            output_dir / V2_SELECTED_SYMBOL_SUMMARY_NAME
        ),
    )


def render_tiingo_selected_candidate_rich_replay_markdown(
    artifact_data: RichReplayReportData,
) -> str:
    """Render the Tiingo selected-candidate rich replay Markdown report."""

    summary = _summary_row(artifact_data.rich_replay_summary)
    fold_performance = artifact_data.fold_performance.copy()
    underperformance = artifact_data.underperformance_windows.copy()
    holdings = artifact_data.selected_symbol_summary.copy()
    turnover = artifact_data.trade_turnover_diagnostics.copy()
    references = artifact_data.reference_performance.copy()
    feature_comparison = artifact_data.feature_set_comparison.copy()
    v2_v1_delta = artifact_data.v2_vs_v1_underperformance_delta.copy()
    v2_holdings = artifact_data.v2_selected_symbol_summary.copy()
    feature_set_version = str(summary.get("feature_set_version", "unknown"))
    overlay_name = str(summary.get("allocation_overlay_name", "equal_weight_top5_baseline"))
    overlay_description = str(
        summary.get("allocation_overlay_description", "equal-weight selected top5 baseline")
    )
    qqq_combination = str(summary.get("qqq_core_combination_handled", "not observed"))

    qqq_stronger = _as_float(summary.get("qqq_average_total_return")) > _as_float(
        summary.get("strategy_average_total_return")
    )
    qqq_sentence = (
        "QQQ buy-and-hold remains stronger on average raw return in this replay, "
        "so the selected candidate should not be promoted on internal ML comparisons alone."
        if qqq_stronger
        else "QQQ buy-and-hold is included as a promotion gate, not as an afterthought."
    )

    lines = [
        "# Tiingo Selected Candidate Rich Replay",
        "",
        "## Executive summary",
        "",
        "This is a local-only replay of the selected Tiingo-only ridge top5 candidate "
        "with daily strategy equity, drawdown, holdings, rebalance, turnover, and "
        "benchmark artifacts.",
        "",
        f"Feature set used: `{feature_set_version}`.",
        f"Overlay used: `{overlay_name}`.",
        f"Overlay construction: {overlay_description}.",
        f"QQQ core/satellite combination: {qqq_combination}.",
        f"Decision: `{summary.get('decision', 'unknown')}`.",
        f"Next action: `{summary.get('next_action', 'unknown')}`.",
        "",
        qqq_sentence,
        "",
        "The historical ML champion is retained as a continuity reference only. "
        "Promotion requires benchmark-aware evidence versus SPY and QQQ.",
        "",
        "## Strategy configuration",
        "",
        _markdown_table(
            _dict_to_frame(
                summary,
                [
                    "dataset_source",
                    "model_family",
                    "model_type",
                    "top_k",
                    "allocation_mode",
                    "allocation_overlay_name",
                    "allocation_overlay_kind",
                    "allocation_overlay_description",
                    "allocation_overlay_core_weights",
                    "allocation_overlay_satellite_weight",
                    "overlay_weight_valid",
                    "min_holding_days",
                    "switch_margin",
                    "transaction_cost_bps",
                    "train_window_years",
                    "validation_window_months",
                    "test_window_months",
                    "universe_mode",
                    "feature_selection_scope",
                    "min_history_rows",
                    "min_train_eligible_rows",
                    "feature_set_version",
                    "reference_feature_set_versions",
                    "target_col",
                    "label_horizon_days",
                ],
            )
        ),
        "",
        "## Comparison vs equal_weight_top5_baseline",
        "",
        _markdown_table(_equal_weight_comparison_frame(summary)),
        "",
        "## Comparison vs SPY and QQQ",
        "",
        _markdown_table(
            _dict_to_frame(
                summary,
                [
                    "strategy_average_total_return",
                    "spy_average_total_return",
                    "qqq_average_total_return",
                    "average_strategy_vs_spy",
                    "average_strategy_vs_qqq",
                    "delta_vs_equal_weight_strategy_vs_qqq",
                    "spy_underperformance_rate",
                    "qqq_underperformance_rate",
                ],
            )
        ),
        "",
        "## Comparison vs default_features_v1 selected candidate",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_for_feature_set(feature_comparison, "default_features_v1")
            )
        ),
        "",
        "## Comparison vs benchmark_gap_features_v1 selected candidate",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_for_feature_set(
                    feature_comparison,
                    "benchmark_gap_features_v1",
                )
            )
        ),
        "",
        "## Feature set comparison",
        "",
        _markdown_table(
            _format_for_markdown(
                feature_comparison[
                    [
                        column
                        for column in [
                            "comparison_role",
                            "feature_set_version",
                            "reference_label",
                            "fold_count",
                            "average_total_return",
                            "average_strategy_vs_spy",
                            "average_strategy_vs_qqq",
                            "average_max_drawdown",
                            "delta_average_total_return_vs_primary",
                            "delta_strategy_vs_spy_vs_primary",
                            "delta_strategy_vs_qqq_vs_primary",
                        ]
                        if column in feature_comparison.columns
                    ]
                ].head(20)
                if not feature_comparison.empty
                else feature_comparison
            )
        ),
        "",
        "## Drawdown and turnover impact",
        "",
        _markdown_table(
            _dict_to_frame(
                summary,
                [
                    "strategy_worst_max_drawdown",
                    "equal_weight_baseline_worst_max_drawdown",
                    "delta_vs_equal_weight_worst_max_drawdown",
                    "average_mean_daily_turnover",
                    "equal_weight_baseline_average_mean_daily_turnover",
                    "delta_vs_equal_weight_average_mean_daily_turnover",
                    "qqq_core_weight",
                    "satellite_weight",
                    "qqq_holding_days",
                    "qqq_core_only_days",
                    "qqq_combined_core_satellite_days",
                    "qqq_average_target_weight",
                    "qqq_max_target_weight",
                ],
            )
        ),
        "",
        "## Daily equity/drawdown summary",
        "",
        _markdown_table(
            _dict_to_frame(
                summary,
                [
                    "fold_count",
                    "daily_equity_rows",
                    "strategy_average_sharpe_like",
                    "strategy_average_max_drawdown",
                    "strategy_worst_max_drawdown",
                    "spy_average_max_drawdown",
                    "qqq_average_max_drawdown",
                ],
            )
        ),
        "",
        "## V2 vs V1 underperformance delta",
        "",
        _markdown_table(_format_for_markdown(v2_v1_delta.head(80))),
        "",
        "## Fold-by-fold performance",
        "",
        _markdown_table(
            _format_for_markdown(
                fold_performance[
                    [
                        column
                        for column in [
                            "fold",
                            "test_start",
                            "test_end",
                            "strategy_total_return",
                            "spy_total_return",
                            "qqq_total_return",
                            "strategy_vs_spy",
                            "strategy_vs_qqq",
                            "strategy_max_drawdown",
                            "total_transaction_cost",
                            "mean_daily_turnover",
                        ]
                        if column in fold_performance.columns
                    ]
                ].head(80)
            )
        ),
        "",
        "## Underperformance windows versus SPY and QQQ",
        "",
        _markdown_table(
            _format_for_markdown(
                underperformance.sort_values(
                    ["underperformed_benchmark", "excess_return_vs_benchmark"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(80)
                if not underperformance.empty
                else underperformance
            )
        ),
        "",
        "## Holdings / selected symbols",
        "",
        _markdown_table(_format_for_markdown(holdings.head(30))),
        "",
        "## V2 selected symbol summary",
        "",
        _markdown_table(_format_for_markdown(v2_holdings.head(30))),
        "",
        "## Turnover diagnostics",
        "",
        _markdown_table(_format_for_markdown(turnover.head(80))),
        "",
        "## Reference candidates",
        "",
        _markdown_table(_format_for_markdown(references.head(20))),
        "",
        "## Artifact limitations",
        "",
        "- This replay uses the existing research backtester and its simple "
        "turnover-bps transaction-cost approximation.",
        "- Target weights are close-generated and applied with the existing "
        "one-row-per-symbol lag.",
        "- SPY and QQQ are buy-and-hold references, not investable blended alternatives.",
        "- This is research evidence only; it is not production-ready.",
        "- This task did not run tree_screen, did not add MLP, did not add "
        "feature v4, did not use S3 readers, and did not export production signals.",
        "",
        "## Decision",
        "",
        f"Decision: `{summary.get('decision', 'unknown')}`.",
        "",
        "Decision rule applied: do not promote production-ready unless the selected "
        "candidate beats or is very close to QQQ with acceptable drawdown. If an "
        "allocation overlay improves the QQQ gap but still trails QQQ, classify it "
        "as best overlay candidate rather than production-ready.",
        "",
        "If v2 beats SPY on average but still trails QQQ, it is current best "
        "research feature-set evidence, not production-ready evidence.",
        "",
        "## Next action",
        "",
        f"Next action: `{summary.get('next_action', 'unknown')}`.",
        "",
        "Allowed next-action menu considered: `focused overnight sweep around QQQ core sizing`, "
        "`feature v4`, `tree_screen`, `MLP`, `model-card draft`.",
        "",
    ]
    return "\n".join(lines)


def _equal_weight_comparison_frame(summary: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "metric": "average_total_return",
            "overlay": summary.get("strategy_average_total_return", np.nan),
            "equal_weight_top5_baseline": summary.get(
                "equal_weight_baseline_average_total_return",
                np.nan,
            ),
            "delta_vs_equal_weight": summary.get(
                "delta_vs_equal_weight_average_total_return",
                np.nan,
            ),
        },
        {
            "metric": "average_strategy_vs_spy",
            "overlay": summary.get("average_strategy_vs_spy", np.nan),
            "equal_weight_top5_baseline": summary.get(
                "equal_weight_baseline_average_strategy_vs_spy",
                np.nan,
            ),
            "delta_vs_equal_weight": summary.get(
                "delta_vs_equal_weight_strategy_vs_spy",
                np.nan,
            ),
        },
        {
            "metric": "average_strategy_vs_qqq",
            "overlay": summary.get("average_strategy_vs_qqq", np.nan),
            "equal_weight_top5_baseline": summary.get(
                "equal_weight_baseline_average_strategy_vs_qqq",
                np.nan,
            ),
            "delta_vs_equal_weight": summary.get(
                "delta_vs_equal_weight_strategy_vs_qqq",
                np.nan,
            ),
        },
        {
            "metric": "worst_max_drawdown",
            "overlay": summary.get("strategy_worst_max_drawdown", np.nan),
            "equal_weight_top5_baseline": summary.get(
                "equal_weight_baseline_worst_max_drawdown",
                np.nan,
            ),
            "delta_vs_equal_weight": summary.get(
                "delta_vs_equal_weight_worst_max_drawdown",
                np.nan,
            ),
        },
        {
            "metric": "average_mean_daily_turnover",
            "overlay": summary.get("average_mean_daily_turnover", np.nan),
            "equal_weight_top5_baseline": summary.get(
                "equal_weight_baseline_average_mean_daily_turnover",
                np.nan,
            ),
            "delta_vs_equal_weight": summary.get(
                "delta_vs_equal_weight_average_mean_daily_turnover",
                np.nan,
            ),
        },
    ]
    return _format_for_markdown(pd.DataFrame(rows))


def _comparison_for_feature_set(
    feature_comparison: pd.DataFrame,
    feature_set_version: str,
) -> pd.DataFrame:
    if feature_comparison.empty or "feature_set_version" not in feature_comparison.columns:
        return pd.DataFrame()
    rows = feature_comparison.loc[
        feature_comparison["feature_set_version"].astype(str).eq(feature_set_version)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    columns = [
        column
        for column in [
            "comparison_role",
            "feature_set_version",
            "reference_label",
            "fold_count",
            "average_total_return",
            "average_strategy_vs_spy",
            "average_strategy_vs_qqq",
            "average_max_drawdown",
            "average_mean_daily_turnover",
            "delta_average_total_return_vs_primary",
            "delta_strategy_vs_spy_vs_primary",
            "delta_strategy_vs_qqq_vs_primary",
        ]
        if column in rows.columns
    ]
    return rows.loc[:, columns]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown report for the Tiingo selected-candidate rich replay."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _summary_row(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {}
    return summary.iloc[0].to_dict()


def _dict_to_frame(row: dict[str, object], keys: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": row.get(key, np.nan)} for key in keys],
        columns=["metric", "value"],
    )


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].map(_format_value)
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = _format_for_markdown(frame)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _as_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number


def _benchmark_row_count(frame: pd.DataFrame, prefix: str) -> int:
    equity_col = f"{prefix}_equity"
    if frame.empty or equity_col not in frame.columns:
        return 0
    return int(frame[equity_col].notna().sum())


if __name__ == "__main__":
    raise SystemExit(run_tiingo_selected_candidate_rich_replay_report_from_args())
