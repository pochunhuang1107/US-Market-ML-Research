from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_final_candidate_visual_summary import (
    OUTPUT_PATH_NAMES,
    PLOT_SUMMARY_COLUMNS,
    build_final_candidate_visual_summary,
    prepare_final_candidate_plot_data,
    read_final_candidate_visual_summary_inputs,
    render_final_candidate_visual_summary_markdown,
    run_final_candidate_visual_summary_from_args,
    write_final_candidate_visual_summary,
)


def test_reading_final_candidate_visual_summary_inputs_loads_tables(
    tmp_path: Path,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)

    inputs = read_final_candidate_visual_summary_inputs(
        final_candidate_summary=paths["final_candidate_summary"],
        decision_matrix=paths["decision_matrix"],
        shortlist_benchmark_summary=paths["shortlist_benchmark_summary"],
        longer_window_summary=paths["longer_window_summary"],
    )

    assert len(inputs.final_candidate_summary) == 7
    assert len(inputs.decision_matrix) == 7
    assert inputs.shortlist_benchmark_summary is not None
    assert inputs.longer_window_summary is not None
    assert inputs.warnings == ()


def test_prepare_plot_data_merges_scores_and_short_labels(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    inputs = read_final_candidate_visual_summary_inputs(
        final_candidate_summary=paths["final_candidate_summary"],
        decision_matrix=paths["decision_matrix"],
        shortlist_benchmark_summary=paths["shortlist_benchmark_summary"],
        longer_window_summary=paths["longer_window_summary"],
    )

    plot_data = prepare_final_candidate_plot_data(
        final_candidate_summary=inputs.final_candidate_summary,
        decision_matrix=inputs.decision_matrix,
        shortlist_benchmark_summary=inputs.shortlist_benchmark_summary,
        longer_window_summary=inputs.longer_window_summary,
    )

    assert list(plot_data.columns) == PLOT_SUMMARY_COLUMNS
    assert len(plot_data) == 7
    rows = plot_data.set_index("candidate_id")
    benchmark = rows.loc["broad_macro:linear_regression:top1:equal_weight:cost10"]
    longer = rows.loc["full:ridge:top5:equal_weight:cost10"]
    champion = rows.loc["full:linear_regression:top1:equal_weight:cost10"]

    assert benchmark["short_label"] == "broad linear top1"
    assert benchmark["benchmark_stability_score_3m"] == pytest.approx(0.45)
    assert benchmark["benchmark_stability_score_longer"] == pytest.approx(1.10)
    assert longer["balanced_score"] == pytest.approx(1.25)
    assert pd.isna(champion["balanced_score"])


def test_write_visual_summary_generates_all_expected_files(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    inputs = read_final_candidate_visual_summary_inputs(
        final_candidate_summary=paths["final_candidate_summary"],
        decision_matrix=paths["decision_matrix"],
        shortlist_benchmark_summary=paths["shortlist_benchmark_summary"],
        longer_window_summary=paths["longer_window_summary"],
    )
    report_data = build_final_candidate_visual_summary(inputs)
    output_dir = tmp_path / "visual"

    outputs = write_final_candidate_visual_summary(
        report_data=report_data,
        output_dir=output_dir,
    )

    for filename in OUTPUT_PATH_NAMES:
        assert outputs[filename].exists()
        assert outputs[filename].stat().st_size > 0

    summary = pd.read_csv(output_dir / "final_candidate_visual_summary.csv")
    markdown = (output_dir / "final_candidate_visual_summary.md").read_text(
        encoding="utf-8"
    )

    assert len(summary) == 7
    assert "Final Candidate Visual Summary" in markdown
    assert "candidate_mean_return.png" in markdown
    assert "do not update the daily signal export or model card yet" in markdown


def test_missing_optional_inputs_warn_and_still_generate_report(
    tmp_path: Path,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    missing_shortlist = tmp_path / "missing_shortlist.csv"
    missing_longer = tmp_path / "missing_longer.csv"

    inputs = read_final_candidate_visual_summary_inputs(
        final_candidate_summary=paths["final_candidate_summary"],
        decision_matrix=paths["decision_matrix"],
        shortlist_benchmark_summary=missing_shortlist,
        longer_window_summary=missing_longer,
    )
    report_data = build_final_candidate_visual_summary(inputs)
    markdown = render_final_candidate_visual_summary_markdown(report_data)
    output_dir = tmp_path / "visual_missing"

    outputs = write_final_candidate_visual_summary(
        report_data=report_data,
        output_dir=output_dir,
    )

    assert len(report_data.warnings) == 2
    assert "## Input Warnings" in markdown
    assert "Optional report input is missing" in markdown
    for filename in OUTPUT_PATH_NAMES:
        assert outputs[filename].exists()
        assert outputs[filename].stat().st_size > 0


def test_run_final_candidate_visual_summary_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "visual_cli"

    return_code = run_final_candidate_visual_summary_from_args(
        [
            "--final-candidate-summary",
            str(paths["final_candidate_summary"]),
            "--decision-matrix",
            str(paths["decision_matrix"]),
            "--shortlist-benchmark-summary",
            str(paths["shortlist_benchmark_summary"]),
            "--longer-window-summary",
            str(paths["longer_window_summary"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert "final candidate visual summary" in captured.out
    for filename in OUTPUT_PATH_NAMES:
        assert (output_dir / filename).exists()


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    final_candidate_summary = tmp_path / "final_candidate_summary.csv"
    decision_matrix = tmp_path / "final_candidate_decision_matrix.csv"
    shortlist_benchmark_summary = (
        tmp_path / "sliding_window_shortlist_candidate_summary.csv"
    )
    longer_window_summary = tmp_path / "longer_window_shortlist_candidate_summary.csv"

    _synthetic_final_summary().to_csv(final_candidate_summary, index=False)
    _synthetic_decision_matrix().to_csv(decision_matrix, index=False)
    _synthetic_optional_summary(longer=False).to_csv(
        shortlist_benchmark_summary,
        index=False,
    )
    _synthetic_optional_summary(longer=True).to_csv(longer_window_summary, index=False)

    return {
        "final_candidate_summary": final_candidate_summary,
        "decision_matrix": decision_matrix,
        "shortlist_benchmark_summary": shortlist_benchmark_summary,
        "longer_window_summary": longer_window_summary,
    }


def _synthetic_final_summary() -> pd.DataFrame:
    rows = []
    for spec in _candidate_specs():
        rows.append(
            {
                "candidate_label": _candidate_label(spec),
                "fixed_fold_role": "fixed-fold return champion"
                if spec[0] == "full" and spec[1] == "linear_regression"
                else "not fixed-fold champion",
                "allocation_result_role": "no allocation promotion",
                "three_month_sliding_role": "test role",
                "three_month_benchmark_role": "test benchmark role",
                "longer_window_role": "test longer role",
                "mean_total_return_3m": spec[3],
                "mean_total_return_longer": spec[4],
                "positive_rate_3m": spec[5],
                "positive_rate_longer": spec[6],
                "beats_spy_rate_3m": spec[7],
                "beats_spy_rate_longer": spec[8],
                "beats_qqq_rate_3m": spec[9],
                "beats_qqq_rate_longer": spec[10],
                "worst_max_drawdown_3m": spec[11],
                "worst_max_drawdown_longer": spec[12],
                "recommendation_label": spec[13],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_decision_matrix() -> pd.DataFrame:
    rows = []
    for spec in _candidate_specs():
        rows.append(
            {
                "candidate_label": _candidate_label(spec),
                "return_strength": "strong" if spec[4] >= 0.07 else "moderate",
                "benchmark_relative_strength": "strong"
                if spec[8] >= 0.45
                else "moderate",
                "drawdown_stability": "moderate",
                "universe_stability": "moderate",
                "model_complexity": "weak",
                "interpretability": "strong",
                "risk_of_overfitting": "weak",
                "product_explainability": "strong",
                "next_action": "Review visuals.",
            }
        )
    return pd.DataFrame(rows)


def _synthetic_optional_summary(*, longer: bool) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(_candidate_specs()):
        if spec[0] == "full" and spec[1] == "linear_regression":
            continue
        candidate_id = _candidate_id(spec)
        benchmark_score = 0.40 + index * 0.05
        balanced_score = 0.50 + index * 0.15
        if candidate_id == "broad_macro:linear_regression:top1:equal_weight:cost10":
            benchmark_score = 1.10 if longer else 0.45
        if candidate_id == "full:ridge:top5:equal_weight:cost10":
            balanced_score = 1.25
        row = {
            "candidate_id": candidate_id,
            "candidate_label": _candidate_label(spec),
            "universe": spec[0],
            "model_type": spec[1],
            "top_k": spec[2],
            "allocation_mode": "equal_weight",
            "transaction_cost_bps": 10.0,
            "window_count": 12,
            "mean_total_return": spec[4] if longer else spec[3],
            "median_total_return": spec[4] if longer else spec[3],
            "positive_window_rate": spec[6] if longer else spec[5],
            "mean_sharpe_like": 0.05,
            "mean_max_drawdown": -0.10,
            "worst_max_drawdown": spec[12] if longer else spec[11],
            "spy_mean_total_return": 0.04,
            "qqq_mean_total_return": 0.06,
            "strategy_beats_spy_rate": spec[8] if longer else spec[7],
            "strategy_beats_qqq_rate": spec[10] if longer else spec[9],
            "mean_strategy_minus_spy": 0.01,
            "mean_strategy_minus_qqq": -0.01,
            "benchmark_stability_score": benchmark_score,
        }
        if longer:
            row["balanced_score"] = balanced_score
        else:
            row["return_risk_score"] = 0.20 + index * 0.10
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_specs() -> list[tuple[object, ...]]:
    return [
        (
            "full",
            "linear_regression",
            1,
            0.01,
            float("nan"),
            0.50,
            float("nan"),
            0.40,
            float("nan"),
            0.35,
            float("nan"),
            -0.42,
            float("nan"),
            "keep as fixed-fold reference only",
        ),
        (
            "broad_macro",
            "linear_regression",
            1,
            0.05,
            0.09,
            0.62,
            0.72,
            0.43,
            0.46,
            0.46,
            0.48,
            -0.28,
            -0.38,
            "validate benchmark-relative direction",
        ),
        (
            "broad_macro",
            "linear_regression",
            2,
            0.03,
            0.07,
            0.61,
            0.70,
            0.42,
            0.41,
            0.41,
            0.40,
            -0.22,
            -0.30,
            "keep as adjacent broad_macro check",
        ),
        (
            "full",
            "ridge",
            3,
            0.04,
            0.08,
            0.58,
            0.76,
            0.44,
            0.45,
            0.42,
            0.38,
            -0.23,
            -0.28,
            "keep as full-universe balanced reference",
        ),
        (
            "full",
            "ridge",
            5,
            0.04,
            0.075,
            0.60,
            0.80,
            0.39,
            0.42,
            0.38,
            0.35,
            -0.24,
            -0.29,
            "validate longer-window balanced direction",
        ),
        (
            "broad_macro",
            "ridge",
            5,
            0.02,
            0.045,
            0.68,
            0.75,
            0.37,
            0.31,
            0.34,
            0.31,
            -0.19,
            -0.27,
            "keep as defensive broad_macro reference",
        ),
        (
            "broad_macro",
            "random_forest",
            3,
            0.03,
            0.055,
            0.68,
            0.76,
            0.29,
            0.30,
            0.30,
            0.20,
            -0.18,
            -0.31,
            "deprioritize pending robustness",
        ),
    ]


def _candidate_label(spec: tuple[object, ...]) -> str:
    return f"{spec[0]} / {spec[1]} / top{spec[2]} / equal_weight / cost10"


def _candidate_id(spec: tuple[object, ...]) -> str:
    return f"{spec[0]}:{spec[1]}:top{spec[2]}:equal_weight:cost10"
