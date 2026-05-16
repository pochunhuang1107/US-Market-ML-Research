"""Generate allocation sweep comparison reports from completed sweep outputs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SWEEP_NAMES = [
    "allocation_core_modes_study",
    "allocation_capped_study",
    "allocation_softmax_study",
]

GROUP_COLUMNS = [
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

CHAMPION_CONFIG: dict[str, str | int | float] = {
    "model_family": "baseline",
    "model_type": "linear_regression",
    "feature_set_version": "default_features_v1",
    "top_k": 1,
    "min_holding_days": 10,
    "switch_margin": 0.002,
    "allocation_mode": "equal_weight",
    "score_temperature": 1.0,
    "max_weight": 1.0,
    "transaction_cost_bps": 10.0,
    "target_col": "excess_return_vs_spy_20d",
    "label_horizon_days": 20,
    "random_state": 42,
}

CHAMPION_FAMILY_FILTER: dict[str, str | int | float] = {
    "model_family": "baseline",
    "model_type": "linear_regression",
    "feature_set_version": "default_features_v1",
    "random_state": 42,
    "min_holding_days": 10,
    "switch_margin": 0.002,
    "target_col": "excess_return_vs_spy_20d",
    "label_horizon_days": 20,
    "transaction_cost_bps": 10.0,
}

CHAMPION_FAMILY_COLUMNS = [
    "candidate_id",
    "model_family",
    "model_type",
    "feature_set_version",
    "random_state",
    "min_holding_days",
    "switch_margin",
    "target_col",
    "label_horizon_days",
    "transaction_cost_bps",
    "top_k",
    "allocation_mode",
    "score_temperature",
    "max_weight",
    "mean_total_return",
    "mean_sharpe_like",
    "worst_max_drawdown",
    "mean_daily_volatility",
    "delta_mean_total_return_vs_champion",
    "worst_drawdown_improvement_vs_champion",
    "balanced_score",
]

METRICS_PATH_NAME = "allocation_candidate_metrics.csv"
TOP_CANDIDATES_PATH_NAME = "allocation_top_candidates.csv"
CHAMPION_COMPARISON_PATH_NAME = "allocation_champion_comparison.csv"
CHAMPION_FAMILY_COMPARISON_PATH_NAME = "allocation_champion_family_comparison.csv"
REPORT_PATH_NAME = "allocation_study_report.md"


@dataclass(frozen=True)
class SweepInput:
    """Files produced by one completed allocation sweep."""

    name: str
    results_csv: Path
    completed_keys: Path
    failed_keys: Path


def run_allocation_study_report_from_args(argv: list[str] | None = None) -> int:
    """Build allocation candidate rankings and a markdown recommendation report."""

    args = _parse_args(argv)
    sweep_inputs = _resolve_sweep_inputs(args.sweep, args.sweep_dir)

    outputs = build_allocation_study_report(
        sweep_inputs=sweep_inputs,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )

    print("US Market ML allocation study comparison report")
    print(f"candidate metrics CSV: {outputs[METRICS_PATH_NAME]}")
    print(f"top candidates CSV: {outputs[TOP_CANDIDATES_PATH_NAME]}")
    print(f"champion comparison CSV: {outputs[CHAMPION_COMPARISON_PATH_NAME]}")
    print(f"champion-family comparison CSV: {outputs[CHAMPION_FAMILY_COMPARISON_PATH_NAME]}")
    print(f"markdown report: {outputs[REPORT_PATH_NAME]}")

    return 0


def build_allocation_study_report(
    *,
    sweep_inputs: Sequence[SweepInput],
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, Path]:
    """Read completed allocation sweeps, write report artifacts, and return paths."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if not sweep_inputs:
        raise ValueError("At least one sweep input is required.")

    sweep_statuses = [_validate_sweep_status(sweep_input) for sweep_input in sweep_inputs]
    combined = _read_sweep_results(sweep_inputs)
    metrics = _build_candidate_metrics(combined)
    champion = _identify_champion(metrics)
    comparison = _build_champion_comparison(metrics, champion)
    champion_family_comparison = _build_champion_family_comparison(comparison)
    top_candidates = _build_top_candidates(comparison, top_n=top_n)
    recommendation = _build_recommendation(comparison, champion)
    report = _render_markdown_report(
        sweep_statuses=sweep_statuses,
        champion=champion,
        comparison=comparison,
        champion_family_comparison=champion_family_comparison,
        top_candidates=top_candidates,
        recommendation=recommendation,
        top_n=top_n,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        METRICS_PATH_NAME: output_dir / METRICS_PATH_NAME,
        TOP_CANDIDATES_PATH_NAME: output_dir / TOP_CANDIDATES_PATH_NAME,
        CHAMPION_COMPARISON_PATH_NAME: output_dir / CHAMPION_COMPARISON_PATH_NAME,
        CHAMPION_FAMILY_COMPARISON_PATH_NAME: output_dir / CHAMPION_FAMILY_COMPARISON_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    metrics.to_csv(outputs[METRICS_PATH_NAME], index=False)
    top_candidates.to_csv(outputs[TOP_CANDIDATES_PATH_NAME], index=False)
    comparison.to_csv(outputs[CHAMPION_COMPARISON_PATH_NAME], index=False)
    champion_family_comparison.to_csv(
        outputs[CHAMPION_FAMILY_COMPARISON_PATH_NAME],
        index=False,
    )
    outputs[REPORT_PATH_NAME].write_text(report, encoding="utf-8")

    return outputs


def parse_sweep_input(value: str) -> str | SweepInput:
    """Parse a sweep name or custom sweep input spec from CLI text."""

    parts = [part.strip() for part in value.split(":")]
    if len(parts) == 1 and parts[0]:
        return parts[0]
    if len(parts) != 4 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--sweep must be a sweep name or use format "
            "name:results_csv:completed_keys:failed_keys"
        )
    return SweepInput(
        name=parts[0],
        results_csv=Path(parts[1]),
        completed_keys=Path(parts[2]),
        failed_keys=Path(parts[3]),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed focused allocation sweep outputs."
    )
    parser.add_argument("--sweep-dir", type=Path, default=Path("artifacts/sweeps"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/allocation_study_comparison"),
    )
    parser.add_argument(
        "--sweep",
        action="append",
        type=parse_sweep_input,
        help=(
            "Optional repeatable custom sweep spec: "
            "either a sweep name resolved under --sweep-dir, or "
            "name:results_csv:completed_keys:failed_keys. When omitted, the three "
            "focused allocation sweeps are read from --sweep-dir."
        ),
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)

    if args.top_n <= 0:
        parser.error("--top-n must be positive.")

    return args


def _resolve_sweep_inputs(
    sweep_args: Sequence[str | SweepInput] | None,
    sweep_dir: Path,
) -> list[SweepInput]:
    if not sweep_args:
        return _default_sweep_inputs(sweep_dir)

    sweep_inputs: list[SweepInput] = []
    for sweep_arg in sweep_args:
        if isinstance(sweep_arg, SweepInput):
            sweep_inputs.append(sweep_arg)
        else:
            sweep_inputs.append(_sweep_input_from_name(sweep_arg, sweep_dir))
    return sweep_inputs


def _default_sweep_inputs(sweep_dir: Path) -> list[SweepInput]:
    return [
        _sweep_input_from_name(sweep_name, sweep_dir)
        for sweep_name in DEFAULT_SWEEP_NAMES
    ]


def _sweep_input_from_name(sweep_name: str, sweep_dir: Path) -> SweepInput:
    return SweepInput(
        name=sweep_name,
        results_csv=sweep_dir / f"{sweep_name}_results.csv",
        completed_keys=sweep_dir / f"{sweep_name}_completed_keys.txt",
        failed_keys=sweep_dir / f"{sweep_name}_failed_keys.txt",
    )


def _validate_sweep_status(sweep_input: SweepInput) -> dict[str, object]:
    for path in (
        sweep_input.results_csv,
        sweep_input.completed_keys,
        sweep_input.failed_keys,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing sweep artifact: {path}")

    failed_lines = _non_empty_lines(sweep_input.failed_keys)
    if failed_lines:
        raise ValueError(
            f"Sweep {sweep_input.name!r} has {len(failed_lines)} failed keys; "
            "allocation study report requires zero failures."
        )

    completed_lines = _non_empty_lines(sweep_input.completed_keys)
    rows = 0
    test_rows = 0
    for chunk in pd.read_csv(sweep_input.results_csv, usecols=["split"], chunksize=50_000):
        rows += len(chunk)
        test_rows += int((chunk["split"] == "test").sum())
    return {
        "sweep_name": sweep_input.name,
        "results_csv": str(sweep_input.results_csv),
        "completed_keys": len(completed_lines),
        "failed_keys": len(failed_lines),
        "result_rows": rows,
        "test_rows": test_rows,
        "status": "complete",
    }


def _read_sweep_results(sweep_inputs: Sequence[SweepInput]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_by_sweep: dict[str, list[str]] = {}

    for sweep_input in sweep_inputs:
        frame = pd.read_csv(sweep_input.results_csv)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            missing_by_sweep[sweep_input.name] = missing
            continue
        frame = frame.copy()
        frame["sweep_name"] = sweep_input.name
        frames.append(frame)

    if missing_by_sweep:
        formatted = "; ".join(
            f"{name}: {', '.join(columns)}" for name, columns in missing_by_sweep.items()
        )
        raise ValueError(f"Sweep result CSVs are missing required columns: {formatted}")
    if not frames:
        raise ValueError("No sweep result rows were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    for column in _numeric_group_columns():
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    for column in ["total_return", "daily_volatility", "sharpe_like", "max_drawdown"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    return combined


def _build_candidate_metrics(combined: pd.DataFrame) -> pd.DataFrame:
    test_rows = combined.loc[combined["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in allocation sweeps.")
    metric_columns = ["total_return", "daily_volatility", "sharpe_like", "max_drawdown"]
    if test_rows[metric_columns].isna().any().any():
        raise ValueError("Test rows contain non-numeric candidate metric values.")

    grouped = test_rows.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    metrics = grouped.agg(
        mean_total_return=("total_return", "mean"),
        median_total_return=("total_return", "median"),
        mean_sharpe_like=("sharpe_like", "mean"),
        median_sharpe_like=("sharpe_like", "median"),
        mean_max_drawdown=("max_drawdown", "mean"),
        worst_max_drawdown=("max_drawdown", "min"),
        mean_daily_volatility=("daily_volatility", "mean"),
        fold_count=("fold", "nunique"),
        test_row_count=("fold", "size"),
        positive_fold_rate=("total_return", lambda values: float((values > 0.0).mean())),
        source_sweeps=("sweep_name", _join_unique),
    ).reset_index()

    metrics["candidate_id"] = metrics.apply(_candidate_id, axis=1)
    metrics = _add_balanced_score(metrics)
    ordered_columns = [
        "candidate_id",
        *GROUP_COLUMNS,
        "mean_total_return",
        "median_total_return",
        "mean_sharpe_like",
        "median_sharpe_like",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_daily_volatility",
        "fold_count",
        "positive_fold_rate",
        "balanced_score",
        "test_row_count",
        "source_sweeps",
    ]
    return metrics[ordered_columns].sort_values(
        by=["balanced_score", "mean_total_return", "mean_sharpe_like", "candidate_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _identify_champion(metrics: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=metrics.index)
    for column, expected in CHAMPION_CONFIG.items():
        if isinstance(expected, float):
            mask &= np.isclose(metrics[column].astype(float), expected, rtol=0.0, atol=1e-9)
        else:
            mask &= metrics[column] == expected

    matches = metrics.loc[mask]
    if matches.empty:
        raise ValueError("Current top-1 champion config was not found in allocation metrics.")
    if len(matches) > 1:
        raise ValueError("Current top-1 champion config matched multiple candidate rows.")
    return matches.iloc[0]


def _build_champion_comparison(metrics: pd.DataFrame, champion: pd.Series) -> pd.DataFrame:
    comparison = metrics.copy()
    comparison["is_current_champion"] = comparison["candidate_id"] == champion["candidate_id"]
    comparison["delta_mean_total_return_vs_champion"] = (
        comparison["mean_total_return"] - float(champion["mean_total_return"])
    )
    comparison["delta_mean_sharpe_like_vs_champion"] = (
        comparison["mean_sharpe_like"] - float(champion["mean_sharpe_like"])
    )
    comparison["delta_mean_daily_volatility_vs_champion"] = (
        comparison["mean_daily_volatility"] - float(champion["mean_daily_volatility"])
    )
    comparison["mean_drawdown_improvement_vs_champion"] = (
        comparison["mean_max_drawdown"] - float(champion["mean_max_drawdown"])
    )
    comparison["worst_drawdown_improvement_vs_champion"] = (
        comparison["worst_max_drawdown"] - float(champion["worst_max_drawdown"])
    )
    comparison["rank_total_return"] = comparison["mean_total_return"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    comparison["rank_sharpe_like"] = comparison["mean_sharpe_like"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    comparison["rank_drawdown_improvement"] = comparison[
        "worst_drawdown_improvement_vs_champion"
    ].rank(method="min", ascending=False).astype(int)
    comparison["rank_balanced_score"] = comparison["balanced_score"].rank(
        method="min",
        ascending=False,
    ).astype(int)

    return _sort_with_research_tie_breakers(
        comparison,
        by=["rank_balanced_score", "rank_total_return"],
        ascending=[True, True],
    ).reset_index(drop=True)


def _build_top_candidates(comparison: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    ranking_specs = [
        ("best_total_return", "mean_total_return", False),
        ("best_sharpe_like", "mean_sharpe_like", False),
        ("best_drawdown_improvement", "worst_drawdown_improvement_vs_champion", False),
        ("best_balanced_score", "balanced_score", False),
    ]
    frames: list[pd.DataFrame] = []
    for ranking_name, sort_column, ascending in ranking_specs:
        ranked = _sort_with_research_tie_breakers(
            comparison,
            by=[sort_column, "mean_total_return"],
            ascending=[ascending, False],
        ).head(top_n).copy()
        ranked.insert(0, "ranking", ranking_name)
        ranked.insert(1, "ranking_position", range(1, len(ranked) + 1))
        ranked.insert(2, "ranking_metric_value", ranked[sort_column])
        frames.append(ranked)

    return pd.concat(frames, ignore_index=True)


def _build_champion_family_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=comparison.index)
    for column, expected in CHAMPION_FAMILY_FILTER.items():
        if isinstance(expected, float):
            mask &= np.isclose(comparison[column].astype(float), expected, rtol=0.0, atol=1e-9)
        else:
            mask &= comparison[column] == expected

    champion_family = comparison.loc[mask, CHAMPION_FAMILY_COLUMNS].copy()
    if champion_family.empty:
        raise ValueError("No champion-family allocation candidates were found.")

    return _sort_with_research_tie_breakers(
        champion_family,
        by=["balanced_score", "mean_total_return"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _add_balanced_score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["balanced_score"] = (
        _zscore(output["mean_total_return"])
        + _zscore(output["mean_sharpe_like"])
        + _zscore(output["mean_max_drawdown"])
        - _zscore(output["mean_daily_volatility"])
    ) / 4.0
    return output


def _build_recommendation(comparison: pd.DataFrame, champion: pd.Series) -> dict[str, str]:
    non_champion = comparison.loc[~comparison["is_current_champion"]].copy()
    best_balanced = _sort_with_research_tie_breakers(
        comparison,
        by=["balanced_score", "mean_total_return"],
        ascending=[False, False],
    ).iloc[0]

    if bool(best_balanced["is_current_champion"]):
        return {
            "decision": "Keep the current top-1 champion.",
            "candidate_id": str(champion["candidate_id"]),
            "reason": (
                "The current champion remains the best balanced candidate after "
                "penalizing drawdown and volatility."
            ),
        }

    cleaner_improvements = non_champion.loc[
        (non_champion["delta_mean_total_return_vs_champion"] > 0.0)
        & (non_champion["delta_mean_sharpe_like_vs_champion"] > 0.0)
        & (non_champion["worst_drawdown_improvement_vs_champion"] >= 0.0)
    ]
    cleaner_improvements = _sort_with_research_tie_breakers(
        cleaner_improvements,
        by=["balanced_score", "mean_total_return"],
        ascending=[False, False],
    )
    if not cleaner_improvements.empty:
        candidate = cleaner_improvements.iloc[0]
        return {
            "decision": "Promote an allocation candidate for research-stage follow-up.",
            "candidate_id": str(candidate["candidate_id"]),
            "reason": (
                "It improves mean total return and mean sharpe_like without a worse "
                "worst-fold drawdown in this completed sweep set."
            ),
        }

    positive_but_mixed = non_champion.loc[
        (non_champion["delta_mean_total_return_vs_champion"] > 0.0)
        | (non_champion["delta_mean_sharpe_like_vs_champion"] > 0.0)
        | (non_champion["worst_drawdown_improvement_vs_champion"] > 0.0)
    ]
    positive_but_mixed = _sort_with_research_tie_breakers(
        positive_but_mixed,
        by=["balanced_score", "mean_total_return"],
        ascending=[False, False],
    )
    if not positive_but_mixed.empty:
        candidate = positive_but_mixed.iloc[0]
        return {
            "decision": "Run a narrow follow-up allocation study.",
            "candidate_id": str(candidate["candidate_id"]),
            "reason": (
                "Some candidates improve one objective, but the return, sharpe_like, "
                "drawdown, and volatility tradeoffs are mixed versus the champion."
            ),
        }

    return {
        "decision": "Document allocation changes as not worth promoting from this study.",
        "candidate_id": str(champion["candidate_id"]),
        "reason": "No allocation candidate improves the champion on the core comparison metrics.",
    }


def _render_markdown_report(
    *,
    sweep_statuses: list[dict[str, object]],
    champion: pd.Series,
    comparison: pd.DataFrame,
    champion_family_comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
    recommendation: dict[str, str],
    top_n: int,
) -> str:
    champion_table = _records_to_markdown(
        [_compact_metric_record(champion)],
        [
            "candidate_id",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "mean_daily_volatility",
            "fold_count",
            "positive_fold_rate",
            "balanced_score",
        ],
    )
    status_table = _records_to_markdown(
        sweep_statuses,
        ["sweep_name", "status", "result_rows", "test_rows", "completed_keys", "failed_keys"],
    )
    top_return = _ranking_table(
        top_candidates,
        ranking="best_total_return",
        columns=[
            "ranking_position",
            "candidate_id",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "delta_mean_total_return_vs_champion",
        ],
    )
    top_balanced = _ranking_table(
        top_candidates,
        ranking="best_balanced_score",
        columns=[
            "ranking_position",
            "candidate_id",
            "balanced_score",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "mean_daily_volatility",
        ],
    )
    champion_family_table = _records_to_markdown(
        [
            _compact_champion_family_record(row)
            for _, row in champion_family_comparison.head(top_n).iterrows()
        ],
        [
            "top_k",
            "allocation_mode",
            "score_temperature",
            "max_weight",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "mean_daily_volatility",
            "delta_mean_total_return_vs_champion",
            "worst_drawdown_improvement_vs_champion",
            "balanced_score",
        ],
    )
    champion_family_interpretation = _build_champion_family_interpretation(
        champion_family_comparison,
        champion,
    )
    improved_drawdown = comparison.loc[
        comparison["worst_drawdown_improvement_vs_champion"] > 0.0
    ]
    improved_drawdown = _sort_with_research_tie_breakers(
        improved_drawdown,
        by=["worst_drawdown_improvement_vs_champion", "mean_total_return"],
        ascending=[False, False],
    ).head(top_n)
    drawdown_table = _records_to_markdown(
        [_compact_metric_record(row) for _, row in improved_drawdown.iterrows()],
        [
            "candidate_id",
            "worst_max_drawdown",
            "worst_drawdown_improvement_vs_champion",
            "mean_total_return",
            "mean_sharpe_like",
        ],
        empty_message="No candidate improved worst-fold drawdown versus the champion.",
    )
    tradeoff = _build_tradeoff_discussion(comparison, champion)

    return "\n".join(
        [
            "# Allocation Study Comparison Report",
            "",
            "This report compares the completed focused allocation sweeps at candidate level. "
            "Candidate metrics use only `split == \"test\"` rows and aggregate across folds.",
            "",
            "Balanced score is the average of four cross-candidate z-score components: "
            "`mean_total_return`, `mean_sharpe_like`, `mean_max_drawdown` (less negative is "
            "better), and negative `mean_daily_volatility`. It is a research ranking aid, "
            "not a production readiness score.",
            "",
            "## Sweep Completion Status",
            "",
            status_table,
            "",
            "## Current Champion",
            "",
            "Candidate name: `linear_smooth10_margin002_v1`",
            "",
            "Candidate spec: "
            "`baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002`",
            "",
            _config_bullets(champion),
            "",
            champion_table,
            "",
            "## Top Return Candidates",
            "",
            top_return,
            "",
            "## Top Balanced Candidates",
            "",
            top_balanced,
            "",
            "## Champion-Family Allocation Comparison",
            "",
            "This section holds the model family fixed to the current champion family: "
            "`baseline` / `linear_regression`, `default_features_v1`, seed `42`, "
            "10-day holding smoothing, `0.002` switch margin, 20-day excess-return "
            "target, and 10 bps transaction cost. It isolates allocation and `top_k` "
            "variants from broader model-family changes.",
            "",
            champion_family_table,
            "",
            champion_family_interpretation,
            "",
            "## Improved Drawdown Versus Champion",
            "",
            drawdown_table,
            "",
            "## Tradeoff Discussion",
            "",
            tradeoff,
            "",
            "## Final Recommendation",
            "",
            f"**{recommendation['decision']}**",
            "",
            f"Focus candidate: `{recommendation['candidate_id']}`.",
            "",
            recommendation["reason"],
            "",
            "This is a research-stage recommendation only. Do not treat any allocation "
            "candidate here as production-ready without additional validation.",
            "",
        ]
    )


def _build_champion_family_interpretation(
    champion_family_comparison: pd.DataFrame,
    champion: pd.Series,
) -> str:
    non_champion = champion_family_comparison.loc[
        champion_family_comparison["candidate_id"] != champion["candidate_id"]
    ]
    return_beaters = non_champion.loc[
        non_champion["delta_mean_total_return_vs_champion"] > 0.0
    ]
    drawdown_improved_return_sacrificed = non_champion.loc[
        (non_champion["worst_drawdown_improvement_vs_champion"] > 0.0)
        & (non_champion["delta_mean_total_return_vs_champion"] < 0.0)
    ]
    clearly_better = non_champion.loc[
        (non_champion["delta_mean_total_return_vs_champion"] > 0.0)
        & (non_champion["mean_sharpe_like"] > float(champion["mean_sharpe_like"]))
        & (non_champion["worst_drawdown_improvement_vs_champion"] >= 0.0)
    ]

    sentences: list[str] = []
    if return_beaters.empty:
        sentences.append(
            "Within the champion family, no allocation/top_k candidate beats the "
            "current champion on mean total return."
        )
    else:
        best_return = _sort_with_research_tie_breakers(
            return_beaters,
            by=["mean_total_return"],
            ascending=[False],
        ).iloc[0]
        sentences.append(
            f"Within the champion family, `{best_return['candidate_id']}` improves "
            "mean total return versus the champion."
        )

    if drawdown_improved_return_sacrificed.empty:
        sentences.append(
            "No champion-family candidate shows the specific pattern of better "
            "worst-fold drawdown with lower mean total return."
        )
    else:
        sentences.append(
            f"{len(drawdown_improved_return_sacrificed)} champion-family candidate(s) "
            "improve worst-fold drawdown but sacrifice mean total return, so the "
            "drawdown benefit is a tradeoff rather than a clean promotion signal."
        )

    if clearly_better.empty:
        sentences.append(
            "This section does not support promoting an allocation variant by itself; "
            "any follow-up should stay narrow and research-stage."
        )
    else:
        sentences.append(
            "At least one champion-family variant clears return, sharpe_like, and "
            "worst-drawdown checks, but it still needs follow-up validation before any "
            "promotion decision."
        )

    return " ".join(sentences)


def _build_tradeoff_discussion(comparison: pd.DataFrame, champion: pd.Series) -> str:
    best_return = _sort_with_research_tie_breakers(
        comparison,
        by=["mean_total_return"],
        ascending=[False],
    ).iloc[0]
    best_balanced = _sort_with_research_tie_breakers(
        comparison,
        by=["balanced_score", "mean_total_return"],
        ascending=[False, False],
    ).iloc[0]
    improved_drawdown_count = int(
        (comparison["worst_drawdown_improvement_vs_champion"] > 0.0).sum()
    )

    parts = [
        (
            f"The best-return candidate is `{best_return['candidate_id']}` with "
            f"mean total return {_format_value(best_return['mean_total_return'])}, "
            f"versus the champion at {_format_value(champion['mean_total_return'])}."
        ),
        (
            f"The best balanced candidate is `{best_balanced['candidate_id']}`. "
            "That ranking gives return and sharpe_like credit while charging candidates "
            "for deeper drawdowns and higher volatility."
        ),
        (
            f"{improved_drawdown_count} candidate(s) improved worst-fold drawdown versus "
            "the champion, but drawdown improvement should be weighed against return and "
            "sharpe_like changes rather than read as a standalone promotion signal."
        ),
    ]
    return " ".join(parts)


def _ranking_table(
    top_candidates: pd.DataFrame,
    *,
    ranking: str,
    columns: list[str],
) -> str:
    rows = top_candidates.loc[top_candidates["ranking"] == ranking]
    return _records_to_markdown(
        [_compact_metric_record(row) for _, row in rows.iterrows()],
        columns,
    )


def _records_to_markdown(
    records: Iterable[dict[str, object] | pd.Series],
    columns: list[str],
    *,
    empty_message: str = "No rows.",
) -> str:
    rows = [dict(record) for record in records]
    if not rows:
        return empty_message

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _compact_metric_record(row: pd.Series) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"],
        "mean_total_return": row["mean_total_return"],
        "median_total_return": row.get("median_total_return"),
        "mean_sharpe_like": row["mean_sharpe_like"],
        "median_sharpe_like": row.get("median_sharpe_like"),
        "mean_max_drawdown": row["mean_max_drawdown"],
        "worst_max_drawdown": row["worst_max_drawdown"],
        "mean_daily_volatility": row["mean_daily_volatility"],
        "fold_count": row["fold_count"],
        "positive_fold_rate": row["positive_fold_rate"],
        "balanced_score": row["balanced_score"],
        "delta_mean_total_return_vs_champion": row.get(
            "delta_mean_total_return_vs_champion"
        ),
        "delta_mean_sharpe_like_vs_champion": row.get(
            "delta_mean_sharpe_like_vs_champion"
        ),
        "worst_drawdown_improvement_vs_champion": row.get(
            "worst_drawdown_improvement_vs_champion"
        ),
        "ranking_position": row.get("ranking_position"),
    }


def _compact_champion_family_record(row: pd.Series) -> dict[str, object]:
    return {column: row[column] for column in CHAMPION_FAMILY_COLUMNS if column != "candidate_id"}


def _config_bullets(champion: pd.Series) -> str:
    lines = ["Champion config:"]
    for column in GROUP_COLUMNS:
        lines.append(f"- `{column}`: `{_format_value(champion[column])}`")
    return "\n".join(lines)


def _candidate_id(row: pd.Series) -> str:
    return (
        f"{row['model_family']}:{row['model_type']}:"
        f"{row['feature_set_version']}:top{_format_key_value(row['top_k'])}:"
        f"hold{_format_key_value(row['min_holding_days'])}:"
        f"margin{_format_key_value(row['switch_margin'])}:"
        f"{row['allocation_mode']}:temp{_format_key_value(row['score_temperature'])}:"
        f"maxw{_format_key_value(row['max_weight'])}:"
        f"cost{_format_key_value(row['transaction_cost_bps'])}:"
        f"{row['target_col']}:h{_format_key_value(row['label_horizon_days'])}:"
        f"seed{_format_key_value(row['random_state'])}"
    )


def _format_key_value(value: object) -> str:
    if pd.isna(value):
        return "none"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, np.floating):
        return f"{float(value):.6g}"
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


def _join_unique(values: pd.Series) -> str:
    return ",".join(sorted(str(value) for value in values.dropna().unique()))


def _sort_with_research_tie_breakers(
    frame: pd.DataFrame,
    *,
    by: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    if len(by) != len(ascending):
        raise ValueError("Sort columns and ascending flags must have the same length.")
    if frame.empty:
        return frame.copy()

    current_champion_sort = (
        frame["is_current_champion"] if "is_current_champion" in frame.columns else False
    )
    champion_seed_sort = (
        frame["random_state"].eq(CHAMPION_CONFIG["random_state"])
        if "random_state" in frame.columns
        else False
    )
    sortable = frame.assign(
        _is_current_champion_sort=current_champion_sort,
        _is_champion_seed_sort=champion_seed_sort,
    )
    sorted_frame = sortable.sort_values(
        by=[*by, "_is_current_champion_sort", "_is_champion_seed_sort", "candidate_id"],
        ascending=[*ascending, False, False, True],
        kind="mergesort",
    )
    return sorted_frame.drop(
        columns=["_is_current_champion_sort", "_is_champion_seed_sort"]
    )


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _non_empty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _numeric_group_columns() -> list[str]:
    return [
        "top_k",
        "min_holding_days",
        "switch_margin",
        "score_temperature",
        "max_weight",
        "transaction_cost_bps",
        "label_horizon_days",
        "random_state",
    ]


if __name__ == "__main__":
    raise SystemExit(run_allocation_study_report_from_args())
