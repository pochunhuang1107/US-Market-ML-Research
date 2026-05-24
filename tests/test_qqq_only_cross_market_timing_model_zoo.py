from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_market_ml.cli.run_qqq_only_cross_market_timing_model_zoo import (
    build_qqq_only_model_zoo_plan,
    preflight_qqq_only_model_zoo,
    run_qqq_only_cross_market_timing_model_zoo_from_args,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    EXPECTED_REFERENCE_SYMBOLS,
    build_qqq_only_feature_dataset,
    build_qqq_only_sweep_specs,
    discover_reference_series,
    load_reference_series,
    run_qqq_only_spec,
    write_qqq_only_report_artifacts,
)


def test_reference_series_loading_pivoting_excludes_dgs10_and_discovers_symbols(
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.csv"
    _synthetic_reference_long(include_dgs10=True).to_csv(ref_path, index=False)

    discovery = discover_reference_series(str(tmp_path / "*.csv"))
    reference = load_reference_series(tmp_path / "*.csv")

    assert set(discovery.expected_symbols) == set(EXPECTED_REFERENCE_SYMBOLS)
    assert set(discovery.found_symbols) == set(EXPECTED_REFERENCE_SYMBOLS)
    assert discovery.missing_symbols == ()
    assert set(reference.columns) == set(EXPECTED_REFERENCE_SYMBOLS)
    assert "DGS10" not in reference.columns
    assert reference.index.is_monotonic_increasing


def test_qqq_feature_generation_uses_etf_regime_features_and_past_only_reference_ffill() -> None:
    market = _synthetic_market_data()
    reference = _synthetic_reference_wide().iloc[::5].copy()

    dataset = build_qqq_only_feature_dataset(
        market,
        reference,
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    frame = dataset.frame.set_index("date")
    feature_names = set(dataset.feature_summary["feature_name"])

    assert "qqq_return_20d" in feature_names
    assert "qqq_close_to_sma_200" in feature_names
    assert "spy_return_20d" in feature_names
    assert "xlk_spy_return_ratio_20d" in feature_names
    assert "hyg_lqd_return_ratio_20d" in feature_names
    assert "vix_level" in feature_names
    assert not any(name.startswith("dgs10") for name in feature_names)
    assert "spy_return_20d" in dataset.feature_sets["qqq_self_cross_market"]
    assert "vix_level" in dataset.feature_sets["qqq_self_cross_market_reference"]

    first_ref_date = reference.index.min()
    before_first_ref = frame.index[frame.index < first_ref_date][0]
    after_first_ref = frame.index[
        (frame.index > first_ref_date) & (frame.index < reference.index[1])
    ][0]
    assert pd.isna(frame.loc[before_first_ref, "vix_level"])
    assert frame.loc[after_first_ref, "vix_level"] == reference.loc[first_ref_date, "VIX"]


def test_strategy_simulation_never_holds_anything_except_qqq_and_cash() -> None:
    dataset = build_qqq_only_feature_dataset(
        _synthetic_market_data(),
        _synthetic_reference_wide(),
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    spec = build_qqq_only_sweep_specs("smoke")[-1]

    output = run_qqq_only_spec(
        dataset,
        spec,
        start_date="2019-01-01",
        end_date="2020-06-30",
        prediction_cache={},
        min_train_rows=40,
    )

    assert set(output.exposure_log["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert "SPY" not in set(output.exposure_log["held_assets"])
    weights = output.exposure_log["qqq_weight"] + output.exposure_log["cash_weight"]
    assert np.allclose(weights, 1.0)
    assert set(output.exposure_log["effective_exposure"].round(6)).issubset({0.0, 1.0})
    assert output.result["production_ready"] is False
    assert output.result["feature_count"] > 0


def test_report_writes_required_sections_charts_and_production_ready_false(tmp_path: Path) -> None:
    dataset = build_qqq_only_feature_dataset(
        _synthetic_market_data(),
        _synthetic_reference_wide(),
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    specs = [build_qqq_only_sweep_specs("smoke")[0], build_qqq_only_sweep_specs("smoke")[-1]]
    cache: dict[tuple[object, ...], pd.DataFrame] = {}
    outputs = [
        run_qqq_only_spec(
            dataset,
            spec,
            start_date="2019-01-01",
            end_date="2020-06-30",
            prediction_cache=cache,
            min_train_rows=40,
        )
        for spec in specs
    ]

    paths = write_qqq_only_report_artifacts(
        results=pd.DataFrame([output.result for output in outputs]),
        daily_equity=pd.concat([output.daily_equity for output in outputs], ignore_index=True),
        daily_drawdown=pd.concat([output.daily_drawdown for output in outputs], ignore_index=True),
        exposure_log=pd.concat([output.exposure_log for output in outputs], ignore_index=True),
        feature_summary=dataset.feature_summary,
        output_dir=tmp_path,
    )

    summary = pd.read_csv(paths["summary"])
    decision = pd.read_csv(paths["decision_matrix"])
    markdown = paths["report"].read_text(encoding="utf-8")

    assert summary["production_ready"].eq(False).all()
    assert decision["production_ready"].eq(False).all()
    assert "## 1. Executive summary." in markdown
    assert "## 4. QQQ-only rule: holdings are QQQ/cash only." in markdown
    assert "## 13. Recommendation:" in markdown
    assert "## 14. Caveats:" in markdown
    for key, path in paths.items():
        if key.endswith("_chart"):
            assert path.exists()
            assert path.stat().st_size > 0


def test_cli_smoke_run_writes_outputs_and_resume_does_not_duplicate_rows(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv"
    reference_path = tmp_path / "reference.csv"
    _synthetic_market_data().to_csv(market_path, index=False)
    _synthetic_reference_long().to_csv(reference_path, index=False)
    output_dir = tmp_path / "sweep"
    report_dir = tmp_path / "report"

    args = [
        "--market-dataset-glob",
        str(market_path),
        "--reference-series-glob",
        str(reference_path),
        "--output-dir",
        str(output_dir),
        "--report-output-dir",
        str(report_dir),
        "--sweep-profile",
        "smoke",
        "--start-date",
        "2019-01-01",
        "--end-date",
        "2020-06-30",
        "--min-train-rows",
        "40",
        "--resume",
    ]

    assert run_qqq_only_cross_market_timing_model_zoo_from_args(args) == 0
    assert run_qqq_only_cross_market_timing_model_zoo_from_args(args) == 0

    results = pd.read_csv(output_dir / "qqq_only_cross_market_timing_model_zoo_results.csv")
    completed = (
        (output_dir / "qqq_only_cross_market_timing_model_zoo_completed_keys.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    exposure = pd.read_csv(output_dir / "qqq_only_cross_market_timing_model_zoo_exposure_log.csv")

    assert len(results) == len(completed) == 9
    assert results["production_ready"].eq(False).all()
    assert set(exposure["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert (report_dir / "qqq_only_cross_market_timing_model_zoo_report.md").exists()
    assert (report_dir / "qqq_only_cross_market_timing_equity_curve_top_candidates.png").exists()


def test_dry_run_prints_missing_reference_symbols_and_normal_run_fails_helpfully(
    tmp_path: Path,
    capsys,
) -> None:
    market_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(market_path, index=False)
    missing_ref_glob = str(tmp_path / "missing_reference" / "*.parquet")
    output_dir = tmp_path / "sweep"

    exit_code = run_qqq_only_cross_market_timing_model_zoo_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(market_path),
            "--reference-series-glob",
            missing_ref_glob,
            "--output-dir",
            str(output_dir),
            "--report-output-dir",
            str(tmp_path / "report"),
            "--sweep-profile",
            "smoke",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reference missing symbols:" in captured.out
    assert "normal run status: will fail" in captured.out
    assert (output_dir / "qqq_only_cross_market_timing_model_zoo_preflight.md").exists()

    plan = build_qqq_only_model_zoo_plan(
        market_dataset_glob=str(market_path),
        reference_series_glob=missing_ref_glob,
        output_dir=output_dir,
        report_output_dir=tmp_path / "report",
        sweep_profile="smoke",
    )
    preflight = preflight_qqq_only_model_zoo(plan)
    assert preflight.total_config_count == 9
    assert plan.reference_discovery.missing_symbols == EXPECTED_REFERENCE_SYMBOLS
    with pytest.raises(FileNotFoundError, match="Missing required local reference series"):
        run_qqq_only_cross_market_timing_model_zoo_from_args(
            [
                "--market-dataset-glob",
                str(market_path),
                "--reference-series-glob",
                missing_ref_glob,
                "--output-dir",
                str(output_dir),
                "--report-output-dir",
                str(tmp_path / "report"),
                "--sweep-profile",
                "smoke",
            ]
        )


def test_shell_runner_defaults_and_ctrl_c_helpers_are_present() -> None:
    script_path = Path("scripts/run_qqq_only_cross_market_timing_model_zoo.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert 'SWEEP_PROFILE="${SWEEP_PROFILE:-core}"' in script
    assert 'MAX_WORKERS="${MAX_WORKERS:-1}"' in script
    assert 'RESUME="${RESUME:-1}"' in script
    assert 'DRY_RUN="${DRY_RUN:-0}"' in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script
    assert "cleanup_on_interrupt" in script
    assert "sleep 60" in script
    assert "run_model_signal_export" not in script
    assert "run_model_card" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "QQQ_ONLY_CROSS_MARKET_TIMING_MODEL_ZOO_TEST_HELPERS=1; "
                "source scripts/run_qqq_only_cross_market_timing_model_zoo.sh; "
                "declare -f terminate_process_tree >/dev/null; "
                "declare -f wait_with_timeout >/dev/null"
            ),
        ],
        check=True,
    )


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2016-01-04", "2020-06-30")
    symbols = (
        "QQQ",
        "SPY",
        "IWM",
        "DIA",
        "VTI",
        "XLK",
        "XLY",
        "XLP",
        "XLF",
        "XLE",
        "XLU",
        "XLV",
        "XLRE",
        "TLT",
        "IEF",
        "HYG",
        "LQD",
        "GLD",
        "SLV",
        "MTUM",
        "QUAL",
        "VUG",
        "VTV",
        "ARKK",
    )
    for symbol_index, symbol in enumerate(symbols):
        drift = 0.04 + symbol_index * 0.001
        wave = np.sin(np.arange(len(dates)) / (28 + symbol_index)) * 0.15
        close_values = 80.0 + symbol_index * 3.0 + np.cumsum(drift + wave)
        for date, close in zip(dates, close_values, strict=False):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000 + symbol_index,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_reference_wide() -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-20", "2020-06-30")
    steps = np.arange(len(dates))
    return pd.DataFrame(
        {
            "DFF": 0.25 + steps * 0.001,
            "DGS2": 1.0 + np.sin(steps / 80.0),
            "T10Y2Y": 0.5 - steps * 0.0002,
            "VIX": 20.0 + np.sin(steps / 25.0) * 5.0,
            "VIX9D": 20.5 + np.sin(steps / 20.0) * 5.0,
            "VXEEM": 24.0 + np.cos(steps / 30.0) * 4.0,
        },
        index=dates,
    )


def _synthetic_reference_long(*, include_dgs10: bool = False) -> pd.DataFrame:
    wide = _synthetic_reference_wide()
    rows: list[dict[str, object]] = []
    for date, row in wide.iterrows():
        for symbol, value in row.items():
            rows.append({"date": date, "symbol": symbol, "value": value})
        if include_dgs10:
            rows.append({"date": date, "symbol": "DGS10", "value": 3.0})
    return pd.DataFrame(rows)
