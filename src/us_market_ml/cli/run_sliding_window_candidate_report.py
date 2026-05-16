"""Generate a report from completed sliding-window candidate study results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/sliding_window_overnight_candidate_study_results.csv"
)
DEFAULT_FAILED_KEYS = Path(
    "artifacts/sweeps/sliding_window_overnight_candidate_study_failed_keys.txt"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/sliding_window_overnight_candidate_study"
)

METRICS_PATH_NAME = "sliding_window_candidate_metrics.csv"
TOP_RETURN_PATH_NAME = "sliding_window_candidate_top_return.csv"
TOP_SHARPE_PATH_NAME = "sliding_window_candidate_top_sharpe.csv"
TOP_DRAWDOWN_PATH_NAME = "sliding_window_candidate_top_drawdown.csv"
TOP_BALANCED_PATH_NAME = "sliding_window_candidate_top_balanced.csv"
BY_UNIVERSE_PATH_NAME = "sliding_window_candidate_by_universe.csv"
BY_MODEL_TYPE_PATH_NAME = "sliding_window_candidate_by_model_type.csv"
BY_TOPK_PATH_NAME = "sliding_window_candidate_by_topk.csv"
BY_ALLOCATION_MODE_PATH_NAME = "sliding_window_candidate_by_allocation_mode.csv"
REPORT_PATH_NAME = "sliding_window_candidate_report.md"

GROUP_COLUMNS = [
    "universe",
    "model_family",
    "model_type",
    "feature_set_version",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "allocation_mode",
    "score_temperature",
    "max_weight",
    "transaction_cost_bps",
    "target_col",
    "label_horizon_days",
    "random_state",
]

REQUIRED_COLUMNS = [
    "run_key",
    "fold",
    "split",
    *GROUP_COLUMNS,
    "total_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
]

REFERENCE_CHAMPION_CONFIG: dict[str, str | int | float] = {
    "universe": "full",
    "model_type": "linear_regression",
    "top_k": 1,
    "allocation_mode": "equal_weight",
    "transaction_cost_bps": 10.0,
    "feature_set_version": "default_features_v1",
    "random_state": 42,
}

REPORT_OUTPUTS = {
    METRICS_PATH_NAME,
    TOP_RETURN_PATH_NAME,
    TOP_SHARPE_PATH_NAME,
    TOP_DRAWDOWN_PATH_NAME,
    TOP_BALANCED_PATH_NAME,
    BY_UNIVERSE_PATH_NAME,
    BY_MODEL_TYPE_PATH_NAME,
    BY_TOPK_PATH_NAME,
    BY_ALLOCATION_MODE_PATH_NAME,
    REPORT_PATH_NAME,
}


@dataclass(frozen=True)
class SlidingWindowCandidateReportData:
    """All table outputs produced by the candidate report builder."""

    metrics: pd.DataFrame
    top_return: pd.DataFrame
    top_sharpe: pd.DataFrame
    top_drawdown: pd.DataFrame
    top_balanced: pd.DataFrame
    by_universe: pd.DataFrame
    by_model_type: pd.DataFrame
    by_topk: pd.DataFrame
    by_allocation_mode: pd.DataFrame
    reference_champion: pd.Series
    failed_key_count: int
    completed_key_count: int
    result_row_count: int
    test_row_count: int


def run_sliding_window_candidate_report_from_args(argv: list[str] | None = None) -> int:
    """Build a report pack from completed sliding-window candidate study CSV rows."""

    args = _parse_args(argv)
    report_data = build_sliding_window_candidate_report(
        results_csv=args.results_csv,
        failed_keys=args.failed_keys,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir)
    outputs = write_sliding_window_candidate_report(
        report_data=report_data,
        output_dir=output_dir,
        top_n=args.top_n,
    )

    print("US Market ML sliding-window candidate report")
    print(f"results CSV: {args.results_csv}")
    print(f"failed keys: {args.failed_keys}")
    for output_name in sorted(REPORT_OUTPUTS):
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.top_return.head(args.top_n).to_string(index=False))

    return 0


def build_sliding_window_candidate_report(
    *,
    results_csv: Path,
    failed_keys: Path | None = None,
    top_n: int = 20,
) -> SlidingWindowCandidateReportData:
    """Read completed overnight candidate results and return report tables."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    results = pd.read_csv(results_csv)
    _validate_results_columns(results)
    test_rows = _build_test_rows(results)
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    metrics = _build_candidate_metrics(test_rows)
    reference_champion = _identify_reference_champion(metrics)
    metrics = _add_reference_deltas(metrics, reference_champion)
    metrics = _add_balanced_score(metrics)
    metrics = _sort_candidates(
        metrics,
        by=["balanced_score", "avg_test_return", "avg_sharpe_like"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    reference_champion = _identify_reference_champion(metrics)

    top_return = _top_candidates(
        metrics,
        top_n=top_n,
        sort_columns=["avg_test_return", "avg_sharpe_like", "balanced_score"],
        ascending=[False, False, False],
    )
    top_sharpe = _top_candidates(
        metrics,
        top_n=top_n,
        sort_columns=["avg_sharpe_like", "avg_test_return", "balanced_score"],
        ascending=[False, False, False],
    )
    top_drawdown = _top_candidates(
        metrics,
        top_n=top_n,
        sort_columns=["avg_max_drawdown", "worst_max_drawdown", "avg_test_return"],
        ascending=[False, False, False],
    )
    top_balanced = _top_candidates(
        metrics,
        top_n=top_n,
        sort_columns=["balanced_score", "avg_test_return", "avg_sharpe_like"],
        ascending=[False, False, False],
    )

    by_universe = _build_group_comparison(metrics, ["universe"])
    by_model_type = _build_group_comparison(metrics, ["model_type"])
    by_topk = _build_group_comparison(metrics, ["top_k"])
    by_allocation_mode = _build_group_comparison(metrics, ["allocation_mode"])

    failed_key_count = _count_text_file_lines(failed_keys)
    completed_key_count = int(results["run_key"].nunique()) if "run_key" in results else 0

    return SlidingWindowCandidateReportData(
        metrics=metrics,
        top_return=top_return,
        top_sharpe=top_sharpe,
        top_drawdown=top_drawdown,
        top_balanced=top_balanced,
        by_universe=by_universe,
        by_model_type=by_model_type,
        by_topk=by_topk,
        by_allocation_mode=by_allocation_mode,
        reference_champion=reference_champion,
        failed_key_count=failed_key_count,
        completed_key_count=completed_key_count,
        result_row_count=len(results),
        test_row_count=len(test_rows),
    )


def write_sliding_window_candidate_report(
    *,
    report_data: SlidingWindowCandidateReportData,
    output_dir: Path,
    top_n: int = 20,
) -> dict[str, Path]:
    """Write report CSVs and markdown, returning the output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        METRICS_PATH_NAME: output_dir / METRICS_PATH_NAME,
        TOP_RETURN_PATH_NAME: output_dir / TOP_RETURN_PATH_NAME,
        TOP_SHARPE_PATH_NAME: output_dir / TOP_SHARPE_PATH_NAME,
        TOP_DRAWDOWN_PATH_NAME: output_dir / TOP_DRAWDOWN_PATH_NAME,
        TOP_BALANCED_PATH_NAME: output_dir / TOP_BALANCED_PATH_NAME,
        BY_UNIVERSE_PATH_NAME: output_dir / BY_UNIVERSE_PATH_NAME,
        BY_MODEL_TYPE_PATH_NAME: output_dir / BY_MODEL_TYPE_PATH_NAME,
        BY_TOPK_PATH_NAME: output_dir / BY_TOPK_PATH_NAME,
        BY_ALLOCATION_MODE_PATH_NAME: output_dir / BY_ALLOCATION_MODE_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }

    report_data.metrics.to_csv(outputs[METRICS_PATH_NAME], index=False)
    report_data.top_return.to_csv(outputs[TOP_RETURN_PATH_NAME], index=False)
    report_data.top_sharpe.to_csv(outputs[TOP_SHARPE_PATH_NAME], index=False)
    report_data.top_drawdown.to_csv(outputs[TOP_DRAWDOWN_PATH_NAME], index=False)
    report_data.top_balanced.to_csv(outputs[TOP_BALANCED_PATH_NAME], index=False)
    report_data.by_universe.to_csv(outputs[BY_UNIVERSE_PATH_NAME], index=False)
    report_data.by_model_type.to_csv(outputs[BY_MODEL_TYPE_PATH_NAME], index=False)
    report_data.by_topk.to_csv(outputs[BY_TOPK_PATH_NAME], index=False)
    report_data.by_allocation_mode.to_csv(
        outputs[BY_ALLOCATION_MODE_PATH_NAME],
        index=False,
    )
    outputs[REPORT_PATH_NAME].write_text(
        render_sliding_window_candidate_markdown(report_data, top_n=top_n),
        encoding="utf-8",
    )
    return outputs


def render_sliding_window_candidate_markdown(
    report_data: SlidingWindowCandidateReportData,
    *,
    top_n: int = 20,
) -> str:
    """Render the candidate report as Markdown."""

    champion = report_data.reference_champion
    best_return = report_data.top_return.iloc[0]
    best_sharpe = report_data.top_sharpe.iloc[0]
    best_drawdown = report_data.top_drawdown.iloc[0]
    best_balanced = report_data.top_balanced.iloc[0]

    return "\n".join(
        [
            "# Sliding-Window Overnight Candidate Study Report",
            "",
            "This report ranks the completed overnight sliding-window candidate study. "
            "Candidate metrics use only `split == \"test\"` rows and aggregate across "
            "rolling test runs.",
            "",
            "Balanced score is the average of five cross-candidate z-score components: "
            "`avg_test_return`, `avg_sharpe_like`, `positive_rate`, "
            "`avg_max_drawdown` (less negative is better), and `worst_max_drawdown` "
            "(less negative is better). It is a research ranking aid, not a "
            "production readiness score.",
            "",
            "## Completion Status",
            "",
            f"- Result rows: {report_data.result_row_count}",
            f"- Test rows used: {report_data.test_row_count}",
            f"- Completed configs: {report_data.completed_key_count}",
            f"- Failed configs: {report_data.failed_key_count}",
            f"- Candidate rows: {len(report_data.metrics)}",
            "",
            "## Current Reference Champion",
            "",
            "- Candidate name: `linear_smooth10_margin002_v1`",
            "- Candidate spec: "
            "`full / baseline / linear_regression / default_features_v1 / top1 / "
            "equal_weight / cost10`",
            "",
            _markdown_table(_format_candidate_rows(pd.DataFrame([champion]))),
            "",
            "The original fixed-fold champion is not the sliding-window robustness "
            "leader in this completed overnight study.",
            "",
            "## Top Return Candidates",
            "",
            _markdown_table(
                _format_candidate_rows(
                    report_data.top_return.head(top_n),
                    include_deltas=True,
                )
            ),
            "",
            "## Top Sharpe-Like Candidates",
            "",
            _markdown_table(
                _format_candidate_rows(
                    report_data.top_sharpe.head(top_n),
                    include_deltas=True,
                )
            ),
            "",
            "## Top Drawdown-Stability Candidates",
            "",
            _markdown_table(
                _format_candidate_rows(
                    report_data.top_drawdown.head(top_n),
                    include_deltas=True,
                )
            ),
            "",
            "## Balanced Ranking",
            "",
            _markdown_table(
                _format_candidate_rows(
                    report_data.top_balanced.head(top_n),
                    include_deltas=True,
                )
            ),
            "",
            "## Universe Comparison",
            "",
            _markdown_table(_format_group_rows(report_data.by_universe)),
            "",
            _universe_interpretation(report_data.by_universe),
            "",
            "## Model-Family Comparison",
            "",
            "All completed non-MLP overnight candidates in this study use the "
            "`baseline` model family, so the useful model-family comparison here is "
            "across baseline model types.",
            "",
            _markdown_table(_format_group_rows(report_data.by_model_type)),
            "",
            "## Top-K And Allocation-Mode Comparison",
            "",
            _markdown_table(_format_group_rows(report_data.by_topk)),
            "",
            _markdown_table(_format_group_rows(report_data.by_allocation_mode)),
            "",
            "## Preliminary Promotion Recommendation",
            "",
            _recommendation(
                champion=champion,
                best_return=best_return,
                best_sharpe=best_sharpe,
                best_drawdown=best_drawdown,
                best_balanced=best_balanced,
            ),
            "",
        ]
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a report-only ranking from completed sliding-window "
            "overnight candidate study results."
        )
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--failed-keys", type=Path, default=DEFAULT_FAILED_KEYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be positive.")
    return args


def _validate_results_columns(results: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(results.columns))
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")


def _build_test_rows(results: pd.DataFrame) -> pd.DataFrame:
    test_rows = results.loc[results["split"] == "test"].copy()
    metric_columns = ["total_return", "daily_volatility", "sharpe_like", "max_drawdown"]
    for column in metric_columns:
        test_rows[column] = pd.to_numeric(test_rows[column], errors="coerce")
    if test_rows[metric_columns].isna().any().any():
        raise ValueError("Test rows contain non-numeric candidate metric values.")
    return test_rows.reset_index(drop=True)


def _build_candidate_metrics(test_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = test_rows.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    metrics = grouped.agg(
        avg_test_return=("total_return", "mean"),
        median_test_return=("total_return", "median"),
        min_test_return=("total_return", "min"),
        max_test_return=("total_return", "max"),
        avg_sharpe_like=("sharpe_like", "mean"),
        median_sharpe_like=("sharpe_like", "median"),
        avg_max_drawdown=("max_drawdown", "mean"),
        worst_max_drawdown=("max_drawdown", "min"),
        avg_daily_volatility=("daily_volatility", "mean"),
        positive_rate=("total_return", lambda values: float((values > 0.0).mean())),
        runs=("fold", "nunique"),
    ).reset_index()

    metrics["candidate_id"] = metrics.apply(_candidate_id, axis=1)
    ordered_columns = [
        "candidate_id",
        *GROUP_COLUMNS,
        "avg_test_return",
        "median_test_return",
        "min_test_return",
        "max_test_return",
        "avg_sharpe_like",
        "median_sharpe_like",
        "avg_max_drawdown",
        "worst_max_drawdown",
        "avg_daily_volatility",
        "positive_rate",
        "runs",
    ]
    return metrics[ordered_columns]


def _identify_reference_champion(metrics: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=metrics.index)
    for column, expected in REFERENCE_CHAMPION_CONFIG.items():
        if isinstance(expected, float):
            mask &= np.isclose(metrics[column].astype(float), expected, rtol=0.0, atol=1e-9)
        else:
            mask &= metrics[column] == expected

    matches = metrics.loc[mask]
    if matches.empty:
        raise ValueError("Current reference champion config was not found in metrics.")
    if len(matches) > 1:
        raise ValueError("Current reference champion config matched multiple rows.")
    return matches.iloc[0]


def _add_reference_deltas(metrics: pd.DataFrame, champion: pd.Series) -> pd.DataFrame:
    output = metrics.copy()
    output["is_reference_champion"] = output["candidate_id"] == champion["candidate_id"]
    output["delta_avg_test_return"] = (
        output["avg_test_return"] - float(champion["avg_test_return"])
    )
    output["delta_avg_sharpe_like"] = (
        output["avg_sharpe_like"] - float(champion["avg_sharpe_like"])
    )
    output["avg_drawdown_improvement"] = (
        output["avg_max_drawdown"] - float(champion["avg_max_drawdown"])
    )
    output["worst_drawdown_improvement"] = (
        output["worst_max_drawdown"] - float(champion["worst_max_drawdown"])
    )
    output["delta_positive_rate"] = (
        output["positive_rate"] - float(champion["positive_rate"])
    )
    return output


def _add_balanced_score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["balanced_score"] = (
        _zscore(output["avg_test_return"])
        + _zscore(output["avg_sharpe_like"])
        + _zscore(output["positive_rate"])
        + _zscore(output["avg_max_drawdown"])
        + _zscore(output["worst_max_drawdown"])
    ) / 5.0
    return output


def _top_candidates(
    metrics: pd.DataFrame,
    *,
    top_n: int,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    return _sort_candidates(metrics, by=sort_columns, ascending=ascending).head(top_n)


def _sort_candidates(
    metrics: pd.DataFrame,
    *,
    by: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    sortable = metrics.assign(
        _reference_sort=metrics.get("is_reference_champion", False),
        _seed_sort=metrics["random_state"].eq(REFERENCE_CHAMPION_CONFIG["random_state"]),
    )
    sorted_frame = sortable.sort_values(
        by=[*by, "_reference_sort", "_seed_sort", "candidate_id"],
        ascending=[*ascending, False, False, True],
        kind="mergesort",
    )
    return sorted_frame.drop(columns=["_reference_sort", "_seed_sort"]).reset_index(
        drop=True
    )


def _build_group_comparison(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for name, group in metrics.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(group_columns, name, strict=True))
        row.update(
            {
                "candidate_count": int(len(group)),
                "mean_avg_test_return": group["avg_test_return"].mean(),
                "median_avg_test_return": group["avg_test_return"].median(),
                "best_avg_test_return": group["avg_test_return"].max(),
                "mean_avg_sharpe_like": group["avg_sharpe_like"].mean(),
                "best_avg_sharpe_like": group["avg_sharpe_like"].max(),
                "mean_avg_max_drawdown": group["avg_max_drawdown"].mean(),
                "best_avg_max_drawdown": group["avg_max_drawdown"].max(),
                "mean_worst_max_drawdown": group["worst_max_drawdown"].mean(),
                "best_worst_max_drawdown": group["worst_max_drawdown"].max(),
                "mean_positive_rate": group["positive_rate"].mean(),
                "best_balanced_score": group["balanced_score"].max(),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["best_balanced_score", "best_avg_test_return"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _candidate_id(row: pd.Series) -> str:
    return (
        f"{row['universe']}:{row['model_family']}:{row['model_type']}:"
        f"{row['feature_set_version']}:top{int(row['top_k'])}:"
        f"hold{int(row['min_holding_days'])}:margin{_format_compact(row['switch_margin'])}:"
        f"{row['allocation_mode']}:temp{_format_compact(row['score_temperature'])}:"
        f"maxw{_format_compact(row['max_weight'])}:"
        f"cost{_format_compact(row['transaction_cost_bps'])}:"
        f"{row['target_col']}:h{int(row['label_horizon_days'])}:"
        f"seed{int(row['random_state'])}"
    )


def _recommendation(
    *,
    champion: pd.Series,
    best_return: pd.Series,
    best_sharpe: pd.Series,
    best_drawdown: pd.Series,
    best_balanced: pd.Series,
) -> str:
    watchlist = [
        "broad_macro:baseline:linear_regression:default_features_v1:top1",
        "full:baseline:ridge:default_features_v1:top3",
        "broad_macro:baseline:random_forest:default_features_v1:top3",
        "broad_macro:baseline:ridge:default_features_v1:top5",
    ]
    return (
        "Do not call any candidate production-ready from this report. The original "
        "fixed-fold champion remains a useful reference, but it is not the "
        "sliding-window robustness leader: its average rolling test return is "
        f"{_format_percent(champion['avg_test_return'])}, Sharpe-like score is "
        f"{_format_percent(champion['avg_sharpe_like'])}, and positive-window rate is "
        f"{_format_percent(champion['positive_rate'])}. The research-stage candidates "
        "worth deeper follow-up are the high-return leader "
        f"`{best_return['candidate_id']}`, the Sharpe-like leader "
        f"`{best_sharpe['candidate_id']}`, the drawdown-stability leader "
        f"`{best_drawdown['candidate_id']}`, and the balanced-score leader "
        f"`{best_balanced['candidate_id']}`. The pre-specified candidates to discuss "
        f"remain `{watchlist[0]}`, `{watchlist[1]}`, `{watchlist[2]}`, and "
        f"`{watchlist[3]}`. Recommended next work is a narrower follow-up around the "
        "best 3-5 candidates, with SPY/QQQ benchmark comparison and possibly longer "
        "rolling test windows before any promotion decision."
    )


def _universe_interpretation(by_universe: pd.DataFrame) -> str:
    if {"full", "broad_macro"}.issubset(set(by_universe["universe"])):
        full = by_universe.loc[by_universe["universe"] == "full"].iloc[0]
        broad = by_universe.loc[by_universe["universe"] == "broad_macro"].iloc[0]
        return (
            "`broad_macro` versus `full`: broad_macro has best average test return "
            f"{_format_percent(broad['best_avg_test_return'])} versus "
            f"{_format_percent(full['best_avg_test_return'])} for full, and best "
            f"balanced score {_format_decimal(broad['best_balanced_score'])} versus "
            f"{_format_decimal(full['best_balanced_score'])} for full."
        )
    return "Universe comparison is limited because both `full` and `broad_macro` are not present."


def _format_candidate_rows(
    frame: pd.DataFrame,
    *,
    include_deltas: bool = False,
) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "avg_test_return",
        "avg_sharpe_like",
        "avg_max_drawdown",
        "worst_max_drawdown",
        "positive_rate",
        "runs",
        "balanced_score",
    ]
    if include_deltas:
        columns.extend(
            [
                "delta_avg_test_return",
                "delta_avg_sharpe_like",
                "worst_drawdown_improvement",
                "delta_positive_rate",
            ]
        )
    output = frame[columns].copy()
    for column in [
        "avg_test_return",
        "avg_sharpe_like",
        "avg_max_drawdown",
        "worst_max_drawdown",
        "positive_rate",
        "delta_avg_test_return",
        "delta_avg_sharpe_like",
        "worst_drawdown_improvement",
        "delta_positive_rate",
    ]:
        if column in output:
            output[column] = output[column].map(_format_percent)
    if "balanced_score" in output:
        output["balanced_score"] = output["balanced_score"].map(_format_decimal)
    return output


def _format_group_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column in {"candidate_count"} or column in GROUP_COLUMNS:
            continue
        if column == "best_balanced_score":
            output[column] = output[column].map(_format_decimal)
        else:
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


def _count_text_file_lines(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_decimal(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _format_compact(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def main() -> int:
    return run_sliding_window_candidate_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
