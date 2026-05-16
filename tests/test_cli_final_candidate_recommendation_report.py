from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_final_candidate_recommendation_report import (
    DECISION_MATRIX_COLUMNS,
    FINAL_SUMMARY_COLUMNS,
    build_final_candidate_recommendation_report,
    create_final_candidate_decision_matrix,
    create_final_candidate_summary,
    read_final_candidate_report_inputs,
    render_final_candidate_recommendation_markdown,
    run_final_candidate_recommendation_report_from_args,
)


def test_reading_report_inputs_loads_required_and_optional_tables(tmp_path: Path) -> None:
    dirs = _write_synthetic_report_inputs(tmp_path)

    inputs = read_final_candidate_report_inputs(**dirs)

    assert len(inputs.sliding_window_summary) == 1
    assert len(inputs.sliding_window_benchmark_summary) == 1
    assert len(inputs.overnight_candidate_metrics) == 7
    assert len(inputs.shortlist_benchmark_summary) == 6
    assert inputs.shortlist_candidate_summary is not None
    assert len(inputs.longer_window_candidate_summary) == 6
    assert inputs.allocation_champion_comparison is not None
    assert inputs.narrow_allocation_champion_comparison is not None
    assert inputs.warnings == ()


def test_create_candidate_summary_combines_report_sources(tmp_path: Path) -> None:
    dirs = _write_synthetic_report_inputs(tmp_path)
    inputs = read_final_candidate_report_inputs(**dirs)

    summary = create_final_candidate_summary(inputs)

    assert list(summary.columns) == FINAL_SUMMARY_COLUMNS
    assert len(summary) == 7
    rows = summary.set_index("candidate_label")
    champion = rows.loc["full / linear_regression / top1 / equal_weight / cost10"]
    benchmark_leader = rows.loc[
        "broad_macro / linear_regression / top1 / equal_weight / cost10"
    ]
    longer_leader = rows.loc["full / ridge / top5 / equal_weight / cost10"]

    assert champion["fixed_fold_role"] == "fixed-fold return champion"
    assert champion["three_month_sliding_role"] == "stability not confirmed"
    assert champion["mean_total_return_3m"] == pytest.approx(0.01)
    assert champion["beats_spy_rate_3m"] == pytest.approx(0.40)
    assert pd.isna(champion["mean_total_return_longer"])

    assert benchmark_leader["three_month_benchmark_role"] == (
        "best 3-month benchmark-relative candidate"
    )
    assert benchmark_leader["mean_total_return_3m"] == pytest.approx(0.05)
    assert benchmark_leader["mean_total_return_longer"] == pytest.approx(0.09)
    assert benchmark_leader["beats_qqq_rate_longer"] == pytest.approx(0.48)

    assert longer_leader["longer_window_role"] == "longer-window balanced leader"
    assert longer_leader["recommendation_label"] == (
        "validate longer-window balanced direction"
    )


def test_create_decision_matrix_uses_simple_categorical_ratings(tmp_path: Path) -> None:
    dirs = _write_synthetic_report_inputs(tmp_path)
    inputs = read_final_candidate_report_inputs(**dirs)
    summary = create_final_candidate_summary(inputs)

    matrix = create_final_candidate_decision_matrix(summary)

    assert list(matrix.columns) == DECISION_MATRIX_COLUMNS
    assert len(matrix) == 7
    allowed = {"strong", "moderate", "weak", "unknown"}
    rating_columns = [
        "return_strength",
        "benchmark_relative_strength",
        "drawdown_stability",
        "universe_stability",
        "model_complexity",
        "interpretability",
        "risk_of_overfitting",
        "product_explainability",
    ]
    for column in rating_columns:
        assert set(matrix[column]).issubset(allowed)

    rows = matrix.set_index("candidate_label")
    assert rows.loc[
        "broad_macro / linear_regression / top1 / equal_weight / cost10",
        "benchmark_relative_strength",
    ] == "strong"
    assert rows.loc[
        "broad_macro / random_forest / top3 / equal_weight / cost10",
        "interpretability",
    ] == "moderate"
    assert "confirmation" in rows.loc[
        "full / ridge / top5 / equal_weight / cost10",
        "next_action",
    ]


def test_missing_optional_inputs_warn_without_crashing(tmp_path: Path) -> None:
    dirs = _write_synthetic_report_inputs(
        tmp_path,
        include_optional_allocation=False,
        include_shortlist_candidate_summary=False,
    )

    inputs = read_final_candidate_report_inputs(**dirs)
    report = build_final_candidate_recommendation_report(inputs)
    markdown = render_final_candidate_recommendation_markdown(report)

    assert len(report.warnings) == 3
    assert "Optional report input is missing" in markdown
    assert "## Input Warnings" in markdown
    assert len(report.final_candidate_summary) == 7
    benchmark_leader = report.final_candidate_summary.set_index("candidate_label").loc[
        "broad_macro / linear_regression / top1 / equal_weight / cost10"
    ]
    assert benchmark_leader["mean_total_return_3m"] == pytest.approx(0.05)
    assert benchmark_leader["positive_rate_3m"] == pytest.approx(0.62)


def test_run_final_candidate_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dirs = _write_synthetic_report_inputs(tmp_path)
    output_dir = tmp_path / "final"

    return_code = run_final_candidate_recommendation_report_from_args(
        [
            "--allocation-report-dir",
            str(dirs["allocation_report_dir"]),
            "--narrow-allocation-report-dir",
            str(dirs["narrow_allocation_report_dir"]),
            "--sliding-window-report-dir",
            str(dirs["sliding_window_report_dir"]),
            "--sliding-window-benchmark-dir",
            str(dirs["sliding_window_benchmark_dir"]),
            "--overnight-candidate-report-dir",
            str(dirs["overnight_candidate_report_dir"]),
            "--shortlist-benchmark-report-dir",
            str(dirs["shortlist_benchmark_report_dir"]),
            "--longer-window-report-dir",
            str(dirs["longer_window_report_dir"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "final candidate recommendation report" in captured.out

    expected_files = [
        "final_candidate_summary.csv",
        "final_candidate_decision_matrix.csv",
        "final_candidate_recommendation.md",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    summary = pd.read_csv(output_dir / "final_candidate_summary.csv")
    matrix = pd.read_csv(output_dir / "final_candidate_decision_matrix.csv")
    markdown = (output_dir / "final_candidate_recommendation.md").read_text(
        encoding="utf-8"
    )

    assert len(summary) == 7
    assert len(matrix) == 7
    assert "Executive Summary" in markdown
    assert "Recommended Research-Stage Labels" in markdown
    assert "Do not promote a production-style model yet" in markdown


def _write_synthetic_report_inputs(
    tmp_path: Path,
    *,
    include_optional_allocation: bool = True,
    include_shortlist_candidate_summary: bool = True,
) -> dict[str, Path]:
    allocation_report_dir = tmp_path / "allocation"
    narrow_allocation_report_dir = tmp_path / "narrow_allocation"
    sliding_window_report_dir = tmp_path / "sliding_window"
    sliding_window_benchmark_dir = tmp_path / "sliding_window_benchmark"
    overnight_candidate_report_dir = tmp_path / "overnight"
    shortlist_benchmark_report_dir = tmp_path / "shortlist"
    longer_window_report_dir = tmp_path / "longer"
    for directory in [
        allocation_report_dir,
        narrow_allocation_report_dir,
        sliding_window_report_dir,
        sliding_window_benchmark_dir,
        overnight_candidate_report_dir,
        shortlist_benchmark_report_dir,
        longer_window_report_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "test_window_count": 4,
                "mean_total_return": 0.01,
                "positive_window_rate": 0.50,
                "worst_max_drawdown": -0.42,
            }
        ]
    ).to_csv(sliding_window_report_dir / "sliding_window_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "window_count": 4,
                "strategy_mean_total_return": 0.01,
                "strategy_positive_window_rate": 0.50,
                "strategy_beats_spy_rate": 0.40,
                "strategy_beats_qqq_rate": 0.35,
            }
        ]
    ).to_csv(
        sliding_window_benchmark_dir / "sliding_window_benchmark_summary.csv",
        index=False,
    )

    overnight = _synthetic_candidate_rows(
        return_column="avg_test_return",
        positive_column="positive_rate",
        spy_column=None,
        qqq_column=None,
        include_champion=True,
    )
    overnight.to_csv(
        overnight_candidate_report_dir / "sliding_window_candidate_metrics.csv",
        index=False,
    )

    shortlist_benchmark = _synthetic_candidate_rows(
        return_column="mean_total_return",
        positive_column=None,
        spy_column="strategy_beats_spy_rate",
        qqq_column="strategy_beats_qqq_rate",
        include_champion=False,
    )
    shortlist_benchmark.insert(0, "rank", range(1, len(shortlist_benchmark) + 1))
    shortlist_benchmark.to_csv(
        shortlist_benchmark_report_dir / "sliding_window_shortlist_benchmark_summary.csv",
        index=False,
    )

    if include_shortlist_candidate_summary:
        shortlist_candidate = _synthetic_candidate_rows(
            return_column="mean_total_return",
            positive_column="positive_window_rate",
            spy_column="strategy_beats_spy_rate",
            qqq_column="strategy_beats_qqq_rate",
            include_champion=False,
        )
        shortlist_candidate.to_csv(
            shortlist_benchmark_report_dir
            / "sliding_window_shortlist_candidate_summary.csv",
            index=False,
        )

    longer = _synthetic_candidate_rows(
        return_column="mean_total_return",
        positive_column="positive_window_rate",
        spy_column="strategy_beats_spy_rate",
        qqq_column="strategy_beats_qqq_rate",
        include_champion=False,
        longer=True,
    )
    longer.to_csv(
        longer_window_report_dir / "longer_window_shortlist_candidate_summary.csv",
        index=False,
    )

    if include_optional_allocation:
        allocation_frame = pd.DataFrame(
            [
                {
                    "is_current_champion": True,
                    "delta_mean_total_return_vs_champion": 0.0,
                }
            ]
        )
        allocation_frame.to_csv(
            allocation_report_dir / "allocation_champion_comparison.csv",
            index=False,
        )
        allocation_frame.to_csv(
            narrow_allocation_report_dir / "allocation_champion_comparison.csv",
            index=False,
        )

    return {
        "allocation_report_dir": allocation_report_dir,
        "narrow_allocation_report_dir": narrow_allocation_report_dir,
        "sliding_window_report_dir": sliding_window_report_dir,
        "sliding_window_benchmark_dir": sliding_window_benchmark_dir,
        "overnight_candidate_report_dir": overnight_candidate_report_dir,
        "shortlist_benchmark_report_dir": shortlist_benchmark_report_dir,
        "longer_window_report_dir": longer_window_report_dir,
    }


def _synthetic_candidate_rows(
    *,
    return_column: str,
    positive_column: str | None,
    spy_column: str | None,
    qqq_column: str | None,
    include_champion: bool,
    longer: bool = False,
) -> pd.DataFrame:
    specs = [
        ("full", "linear_regression", 1, 0.01, 0.50, 0.40, 0.35, -0.42),
        ("broad_macro", "linear_regression", 1, 0.05, 0.62, 0.46, 0.47, -0.28),
        ("broad_macro", "linear_regression", 2, 0.03, 0.60, 0.42, 0.40, -0.22),
        ("full", "ridge", 3, 0.04, 0.58, 0.44, 0.41, -0.23),
        ("full", "ridge", 5, 0.04, 0.60, 0.39, 0.38, -0.24),
        ("broad_macro", "ridge", 5, 0.02, 0.68, 0.37, 0.34, -0.19),
        ("broad_macro", "random_forest", 3, 0.03, 0.68, 0.29, 0.30, -0.18),
    ]
    longer_overrides = {
        ("broad_macro", "linear_regression", 1): (0.09, 0.72, 0.46, 0.48, -0.38),
        ("broad_macro", "linear_regression", 2): (0.07, 0.70, 0.41, 0.40, -0.30),
        ("full", "ridge", 3): (0.08, 0.76, 0.45, 0.38, -0.28),
        ("full", "ridge", 5): (0.075, 0.80, 0.42, 0.35, -0.29),
        ("broad_macro", "ridge", 5): (0.045, 0.75, 0.31, 0.31, -0.27),
        ("broad_macro", "random_forest", 3): (0.055, 0.76, 0.30, 0.20, -0.31),
    }
    rows = []
    for universe, model_type, top_k, ret, pos, spy, qqq, drawdown in specs:
        if not include_champion and universe == "full" and model_type == "linear_regression":
            continue
        if longer:
            ret, pos, spy, qqq, drawdown = longer_overrides[(universe, model_type, top_k)]
        candidate_id = f"{universe}:{model_type}:top{top_k}:equal_weight:cost10"
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "candidate_label": (
                f"{universe} / {model_type} / top{top_k} / equal_weight / cost10"
            ),
            "universe": universe,
            "model_type": model_type,
            "top_k": top_k,
            "allocation_mode": "equal_weight",
            "transaction_cost_bps": 10.0,
            return_column: ret,
            "worst_max_drawdown": drawdown,
        }
        if positive_column is not None:
            row[positive_column] = pos
        if spy_column is not None:
            row[spy_column] = spy
        if qqq_column is not None:
            row[qqq_column] = qqq
        rows.append(row)
    return pd.DataFrame(rows)
