from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_qqq_defensive_timing_selected_replay_report import (
    read_tiingo_qqq_defensive_timing_selected_replay_artifacts,
    render_tiingo_qqq_defensive_timing_selected_replay_markdown,
    run_tiingo_qqq_defensive_timing_selected_replay_report_from_args,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    REPORT_NAME,
    run_tiingo_qqq_defensive_timing_selected_replay_from_args,
)


def test_selected_replay_report_cli_writes_required_sections(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"
    run_tiingo_qqq_defensive_timing_selected_replay_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2021-06-30",
        ]
    )

    exit_code = run_tiingo_qqq_defensive_timing_selected_replay_report_from_args(
        ["--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")

    assert exit_code == 0
    assert "selected replay report" in captured.out
    for section in [
        "Executive summary",
        "Candidate configuration",
        "Why this candidate was selected",
        "Full-period comparison vs QQQ/SPY/cash",
        "Daily equity and drawdown summary",
        "Trade timeline / turnover",
        "Subperiod robustness",
        "Where it helps",
        "Where it still fails",
        "Decision",
        "Next action",
        "Caveats",
    ]:
        assert section in markdown
    assert "QQQ" in markdown
    assert "SPY" in markdown
    assert "cash" in markdown
    assert "research-only" in markdown
    assert "Keep as research-only defensive candidate: yes" in markdown
    assert "Production-ready: no" in markdown
    assert "No S3/cloud paths were used" in markdown
    assert "No production signal export" in markdown
    assert "QQQ buy-and-hold remains the benchmark" in markdown
    assert "do not add benchmark_gap_features_v4" in markdown
    assert "tree_screen: do not run" in markdown
    assert "MLP: do not run" in markdown
    assert "Production promotion: do not promote" in markdown


def test_render_selected_replay_report_from_loaded_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"
    run_tiingo_qqq_defensive_timing_selected_replay_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2021-06-30",
        ]
    )

    report_data = read_tiingo_qqq_defensive_timing_selected_replay_artifacts(output_dir)
    markdown = render_tiingo_qqq_defensive_timing_selected_replay_markdown(report_data)

    assert "qqq_sma190_buffer_p0p5_hold0_to_spy" in markdown
    assert "comparison vs QQQ/SPY/cash" in markdown
    assert "Trade timeline / turnover" in markdown
    assert "Subperiod robustness" in markdown
    assert "Timing convention remains leakage-safe" in markdown


def _synthetic_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=650)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 150.0, 260),
            np.linspace(149.0, 92.0, 170),
            np.linspace(94.0, 142.0, 220),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 132.0, len(dates)),
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
