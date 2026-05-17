from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_extended_history_start_year_viability_report import (
    CSV_NAME,
    MARKDOWN_NAME,
    SUMMARY_NAME,
    evaluate_start_years,
    run_extended_history_start_year_viability_report_from_args,
    write_start_year_viability_artifacts,
)


def test_candidate_start_years_are_evaluated() -> None:
    rows = evaluate_start_years(
        _synthetic_extended_market_data(),
        start_years=(1999, 2002),
        end_date="2005-12-31",
        test_start="2005-01-01",
        min_lookback_trading_days=30,
    )

    assert [row.start_year for row in rows] == [1999, 2002]
    assert all(row.usable_feature_label_rows_after_dropna > 0 for row in rows)


def test_insufficient_lookback_symbols_are_flagged() -> None:
    rows = evaluate_start_years(
        _synthetic_extended_market_data(short_symbol_days=20),
        start_years=(1999,),
        end_date="2005-12-31",
        test_start="2005-01-01",
        min_lookback_trading_days=30,
    )

    row = rows[0]

    assert "ARKK" not in row.symbols_with_enough_lookback.split(",")
    assert "symbols below 30-row lookback: ARKK" in row.warning_notes


def test_top_k_support_and_recommendations_distinguish_universe_modes() -> None:
    rows = evaluate_start_years(
        _synthetic_extended_market_data(),
        start_years=(1999, 2002),
        end_date="2005-12-31",
        test_start="2005-01-01",
        min_lookback_trading_days=30,
    )
    by_year = {row.start_year: row for row in rows}

    assert by_year[1999].supports_top1 is True
    assert by_year[1999].supports_top3 is True
    assert by_year[1999].supports_top5 is True
    assert by_year[1999].recommended_for_dynamic_universe is True
    assert by_year[1999].recommended_for_full_universe is False
    assert "symbols unavailable at start: ARKK,HYG" in by_year[1999].warning_notes

    assert by_year[2002].recommended_for_dynamic_universe is True
    assert by_year[2002].recommended_for_full_universe is True


def test_report_artifacts_can_be_written_to_temp_directory(tmp_path: Path) -> None:
    rows = evaluate_start_years(
        _synthetic_extended_market_data(),
        start_years=(1999, 2002),
        end_date="2005-12-31",
        test_start="2005-01-01",
        min_lookback_trading_days=30,
    )

    outputs = write_start_year_viability_artifacts(rows, output_dir=tmp_path)

    assert outputs["CSV report"] == tmp_path / CSV_NAME
    assert outputs["Markdown report"] == tmp_path / MARKDOWN_NAME
    assert outputs["JSON summary"] == tmp_path / SUMMARY_NAME
    assert pd.read_csv(tmp_path / CSV_NAME)["start_year"].tolist() == [1999, 2002]
    assert "Recommended start year for full-universe studies: 2002" in (
        tmp_path / MARKDOWN_NAME
    ).read_text(encoding="utf-8")
    assert '"recommended_dynamic_universe_start_year": 1999' in (
        tmp_path / SUMMARY_NAME
    ).read_text(encoding="utf-8")


def test_cli_reads_local_glob_and_writes_reports(
    tmp_path: Path,
    capsys,
) -> None:
    data_dir = tmp_path / "Market Data" / "year=synthetic"
    data_dir.mkdir(parents=True)
    _synthetic_extended_market_data().to_parquet(data_dir / "part.parquet", index=False)
    output_dir = tmp_path / "report"

    exit_code = run_extended_history_start_year_viability_report_from_args(
        [
            "--market-dataset-glob",
            str(tmp_path / "Market Data" / "**" / "*.parquet"),
            "--output-dir",
            str(output_dir),
            "--start-years",
            "1999",
            "2002",
            "--end-date",
            "2005-12-31",
            "--test-start",
            "2005-01-01",
            "--min-lookback-trading-days",
            "30",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Reading local market dataset" in captured.out
    assert str(output_dir / CSV_NAME) in captured.out
    assert (output_dir / CSV_NAME).exists()
    assert (output_dir / MARKDOWN_NAME).exists()
    assert (output_dir / SUMMARY_NAME).exists()


def _synthetic_extended_market_data(short_symbol_days: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbol_starts = {
        "SPY": "1999-01-01",
        "QQQ": "1999-01-01",
        "IWM": "1999-01-01",
        "TLT": "1999-01-01",
        "GLD": "1999-01-01",
        "LQD": "1999-01-01",
        "HYG": "2002-01-01",
        "ARKK": "2002-01-01",
    }
    end = pd.Timestamp("2005-12-31")
    for symbol_index, (symbol, start) in enumerate(symbol_starts.items()):
        dates = pd.date_range(start, end, freq="B")
        if symbol == "ARKK" and short_symbol_days is not None:
            dates = dates[:short_symbol_days]
        base_price = 50.0 + symbol_index * 10.0
        drift = 0.03 + symbol_index * 0.005
        for day_index, date in enumerate(dates):
            seasonal = (day_index % 13) * 0.02
            close = base_price + day_index * drift + seasonal
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.4,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 10_000 + (day_index % 37) * 1_000,
                    "vwap": close - 0.02,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
