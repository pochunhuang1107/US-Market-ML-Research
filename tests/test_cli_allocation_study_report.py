from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_allocation_study_report import (
    CHAMPION_CONFIG,
    SweepInput,
    build_allocation_study_report,
    run_allocation_study_report_from_args,
)


def test_allocation_study_report_writes_outputs_and_metrics(tmp_path: Path) -> None:
    sweep_inputs = _write_synthetic_sweeps(tmp_path / "sweeps")
    output_dir = tmp_path / "reports"

    outputs = build_allocation_study_report(
        sweep_inputs=sweep_inputs,
        output_dir=output_dir,
        top_n=2,
    )

    for path in outputs.values():
        assert path.exists()

    metrics = pd.read_csv(output_dir / "allocation_candidate_metrics.csv")
    comparison = pd.read_csv(output_dir / "allocation_champion_comparison.csv")
    champion_family = pd.read_csv(output_dir / "allocation_champion_family_comparison.csv")
    top_candidates = pd.read_csv(output_dir / "allocation_top_candidates.csv")
    report = (output_dir / "allocation_study_report.md").read_text(encoding="utf-8")

    assert len(metrics) == 4
    champion = metrics.loc[metrics["allocation_mode"] == "equal_weight"].iloc[0]
    assert champion["fold_count"] == 2
    assert champion["mean_total_return"] == pytest.approx(0.04)
    assert champion["positive_fold_rate"] == pytest.approx(0.5)
    assert pd.notna(metrics["balanced_score"]).all()

    assert comparison["is_current_champion"].sum() == 1
    champion_comparison = comparison.loc[comparison["is_current_champion"]].iloc[0]
    assert champion_comparison["random_state"] == CHAMPION_CONFIG["random_state"]
    assert champion_comparison["delta_mean_total_return_vs_champion"] == pytest.approx(0.0)
    assert champion_comparison["delta_mean_sharpe_like_vs_champion"] == pytest.approx(0.0)
    assert champion_comparison["worst_drawdown_improvement_vs_champion"] == pytest.approx(0.0)

    assert set(top_candidates["ranking"]) == {
        "best_total_return",
        "best_sharpe_like",
        "best_drawdown_improvement",
        "best_balanced_score",
    }
    assert len(champion_family) == 4
    assert set(champion_family["model_family"]) == {"baseline"}
    assert set(champion_family["model_type"]) == {"linear_regression"}
    assert set(champion_family["random_state"]) == {CHAMPION_CONFIG["random_state"]}
    assert "## Champion-Family Allocation Comparison" in report
    assert "Within the champion family" in report
    assert "Promote an allocation candidate for research-stage follow-up." in report
    assert "This is a research-stage recommendation only." in report


def test_allocation_study_report_cli_accepts_custom_sweeps(tmp_path: Path) -> None:
    sweep_inputs = _write_synthetic_sweeps(tmp_path / "sweeps")
    output_dir = tmp_path / "reports"

    return_code = run_allocation_study_report_from_args(
        [
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
            *[
                item
                for sweep_input in sweep_inputs
                for item in (
                    "--sweep",
                    (
                        f"{sweep_input.name}:{sweep_input.results_csv}:"
                        f"{sweep_input.completed_keys}:{sweep_input.failed_keys}"
                    ),
                )
            ],
        ]
    )

    assert return_code == 0
    assert (output_dir / "allocation_candidate_metrics.csv").exists()
    assert (output_dir / "allocation_top_candidates.csv").exists()
    assert (output_dir / "allocation_champion_comparison.csv").exists()
    assert (output_dir / "allocation_champion_family_comparison.csv").exists()
    assert (output_dir / "allocation_study_report.md").exists()


def test_allocation_study_report_cli_accepts_simple_sweep_names(tmp_path: Path) -> None:
    _write_synthetic_sweeps(tmp_path / "sweeps")
    output_dir = tmp_path / "reports"

    return_code = run_allocation_study_report_from_args(
        [
            "--sweep-dir",
            str(tmp_path / "sweeps"),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
            "--sweep",
            "allocation_core_modes_study",
            "--sweep",
            "allocation_capped_study",
            "--sweep",
            "allocation_softmax_study",
        ]
    )

    assert return_code == 0
    assert (output_dir / "allocation_champion_family_comparison.csv").exists()
    report = (output_dir / "allocation_study_report.md").read_text(encoding="utf-8")
    assert "## Champion-Family Allocation Comparison" in report


def test_allocation_study_report_rejects_failed_keys(tmp_path: Path) -> None:
    sweep_inputs = _write_synthetic_sweeps(tmp_path / "sweeps")
    sweep_inputs[0].failed_keys.write_text("failed-run-key\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires zero failures"):
        build_allocation_study_report(
            sweep_inputs=sweep_inputs,
            output_dir=tmp_path / "reports",
        )


def _write_synthetic_sweeps(sweep_dir: Path) -> list[SweepInput]:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = {
        "allocation_core_modes_study": [
            _row("champ", "fold_1", "train", total_return=9.0, sharpe_like=9.0),
            _row(
                "champ",
                "fold_1",
                "test",
                total_return=0.10,
                sharpe_like=0.02,
                max_drawdown=-0.10,
            ),
            _row(
                "champ",
                "fold_2",
                "test",
                total_return=-0.02,
                sharpe_like=-0.01,
                max_drawdown=-0.20,
            ),
            _row(
                "score",
                "fold_1",
                "test",
                top_k=2,
                allocation_mode="score_weighted",
                total_return=0.15,
                sharpe_like=0.03,
                max_drawdown=-0.12,
                daily_volatility=0.020,
            ),
            _row(
                "score",
                "fold_2",
                "test",
                top_k=2,
                allocation_mode="score_weighted",
                total_return=0.05,
                sharpe_like=0.02,
                max_drawdown=-0.18,
                daily_volatility=0.020,
            ),
        ],
        "allocation_capped_study": [
            _row(
                "capped",
                "fold_1",
                "test",
                top_k=2,
                allocation_mode="capped_score_weighted",
                max_weight=0.5,
                total_return=0.08,
                sharpe_like=0.025,
                max_drawdown=-0.08,
                daily_volatility=0.012,
            ),
            _row(
                "capped",
                "fold_2",
                "test",
                top_k=2,
                allocation_mode="capped_score_weighted",
                max_weight=0.5,
                total_return=0.07,
                sharpe_like=0.024,
                max_drawdown=-0.12,
                daily_volatility=0.012,
            ),
        ],
        "allocation_softmax_study": [
            _row(
                "softmax",
                "fold_1",
                "test",
                top_k=2,
                allocation_mode="softmax_score_weighted",
                score_temperature=0.5,
                total_return=0.06,
                sharpe_like=0.018,
                max_drawdown=-0.10,
                daily_volatility=0.018,
            ),
            _row(
                "softmax",
                "fold_2",
                "test",
                top_k=2,
                allocation_mode="softmax_score_weighted",
                score_temperature=0.5,
                total_return=0.03,
                sharpe_like=0.012,
                max_drawdown=-0.14,
                daily_volatility=0.018,
            ),
        ],
    }

    sweep_inputs: list[SweepInput] = []
    for sweep_name, rows in sweep_rows.items():
        results_csv = sweep_dir / f"{sweep_name}_results.csv"
        completed_keys = sweep_dir / f"{sweep_name}_completed_keys.txt"
        failed_keys = sweep_dir / f"{sweep_name}_failed_keys.txt"
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        completed_keys.write_text("fold_1\nfold_2\n", encoding="utf-8")
        failed_keys.write_text("", encoding="utf-8")
        sweep_inputs.append(
            SweepInput(
                name=sweep_name,
                results_csv=results_csv,
                completed_keys=completed_keys,
                failed_keys=failed_keys,
            )
        )

    return sweep_inputs


def _row(
    run_key: str,
    fold: str,
    split: str,
    *,
    total_return: float,
    sharpe_like: float,
    top_k: int = int(CHAMPION_CONFIG["top_k"]),
    allocation_mode: str = str(CHAMPION_CONFIG["allocation_mode"]),
    score_temperature: float = float(CHAMPION_CONFIG["score_temperature"]),
    max_weight: float = float(CHAMPION_CONFIG["max_weight"]),
    max_drawdown: float = -0.10,
    daily_volatility: float = 0.01,
) -> dict[str, object]:
    return {
        "run_key": run_key,
        "universe": "synthetic",
        "data_path": "synthetic.csv",
        "fold": fold,
        "split": split,
        "model_family": CHAMPION_CONFIG["model_family"],
        "model_type": CHAMPION_CONFIG["model_type"],
        "feature_set_version": CHAMPION_CONFIG["feature_set_version"],
        "feature_count": 3,
        "hidden_units": "none",
        "dropout": None,
        "scale_target": False,
        "learning_rate": None,
        "weight_decay": None,
        "batch_size": None,
        "max_epochs": None,
        "patience": None,
        "random_state": CHAMPION_CONFIG["random_state"],
        "top_k": top_k,
        "min_holding_days": CHAMPION_CONFIG["min_holding_days"],
        "switch_margin": CHAMPION_CONFIG["switch_margin"],
        "allocation_mode": allocation_mode,
        "score_temperature": score_temperature,
        "max_weight": max_weight,
        "transaction_cost_bps": CHAMPION_CONFIG["transaction_cost_bps"],
        "target_col": CHAMPION_CONFIG["target_col"],
        "label_horizon_days": CHAMPION_CONFIG["label_horizon_days"],
        "total_return": total_return,
        "mean_daily_return": 0.001,
        "daily_volatility": daily_volatility,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "final_equity": 1.0 + total_return,
        "epochs_trained": None,
    }
