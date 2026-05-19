from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_qqq_defensive_timing_validation_report import (
    read_tiingo_qqq_defensive_timing_validation_artifacts,
    render_tiingo_qqq_defensive_timing_validation_markdown,
    run_tiingo_qqq_defensive_timing_validation_report_from_args,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_validation import (
    REPORT_NAME,
    run_tiingo_qqq_defensive_timing_validation_from_args,
)


def test_validation_report_cli_writes_required_sections_and_research_language(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "validation"
    run_tiingo_qqq_defensive_timing_validation_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "auto",
            "--end-date",
            "2025-06-30",
        ]
    )

    exit_code = run_tiingo_qqq_defensive_timing_validation_report_from_args(
        ["--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")

    assert exit_code == 0
    assert "validation report" in captured.out
    for section in [
        "Executive summary",
        "Candidate configuration",
        "Data coverage / whether longer-history validation was possible",
        "Full-period validation versus QQQ/SPY/cash",
        "Walk-forward robustness summary",
        "Worst windows and failure modes",
        "Named subperiod robustness",
        "Transaction cost sensitivity",
        "What this validates",
        "What remains unvalidated",
        "Decision",
        "Next action",
        "Caveats",
    ]:
        assert section in markdown
    assert "qqq_sma190_buffer_p0p5_hold0_to_spy" in markdown
    assert "research-only" in markdown
    assert "production_ready = false" in markdown
    assert "Production-ready: no" in markdown
    assert "Longer-history validation was limited by local Tiingo-only QQQ/SPY coverage" in markdown
    assert "does not pretend to validate a longer period" in markdown
    assert "Tiingo-only local research data" in markdown
    assert "No S3/cloud paths were used" in markdown
    assert "No production signal export" in markdown
    assert "Production promotion: no" in markdown
    assert "Optional explicit slippage sensitivity is deferred" in markdown
    assert "Any tree, MLP, or benchmark_gap_features_v4" in markdown
    assert "QQQ buy-and-hold remains the primary benchmark" in markdown


def test_render_validation_report_from_loaded_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "validation"
    run_tiingo_qqq_defensive_timing_validation_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "auto",
            "--end-date",
            "2025-06-30",
        ]
    )

    report_data = read_tiingo_qqq_defensive_timing_validation_artifacts(output_dir)
    markdown = render_tiingo_qqq_defensive_timing_validation_markdown(report_data)

    assert "Tiingo QQQ Defensive Timing Validation Report" in markdown
    assert "Full-period validation versus QQQ/SPY/cash" in markdown
    assert "Walk-forward robustness summary" in markdown
    assert "Transaction cost sensitivity" in markdown
    assert "close T signal applied from the next trading day" in markdown
    assert "No production signal export was created or updated" in markdown


def _synthetic_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=1700)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 160.0, 300),
            np.linspace(158.0, 82.0, 300),
            np.linspace(84.0, 185.0, 300),
            np.linspace(183.0, 92.0, 300),
            np.linspace(94.0, 210.0, 500),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 165.0, len(dates)),
    }
    rows: list[dict[str, object]] = []
    for symbol, values in price_map.items():
        for date, close in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)
