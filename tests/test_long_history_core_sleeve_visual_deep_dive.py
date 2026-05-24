from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_visual_deep_dive import (
    CHART_NAMES,
    REPORT_NAME,
    RICH_REPLAY_PREFIX,
    SIGNAL_REVIEW_PREFIX,
    SUMMARY_NAME,
    TRACKING_PREFIX,
    LongHistoryCoreSleeveVisualDeepDiveInputs,
    build_long_history_core_sleeve_visual_deep_dive_artifacts,
    primary_visual_rows,
    read_long_history_core_sleeve_visual_deep_dive_inputs,
    secondary_visual_rows,
    select_visual_deep_dive_rows,
    write_long_history_core_sleeve_visual_deep_dive_artifacts,
)


def test_loading_visual_deep_dive_inputs(tmp_path: Path) -> None:
    inputs = _inputs()
    rich_dir = tmp_path / "rich"
    signal_dir = tmp_path / "signal"
    tracking_dir = tmp_path / "tracking"
    _write_input_dirs(inputs, rich_dir, signal_dir, tracking_dir)

    loaded = read_long_history_core_sleeve_visual_deep_dive_inputs(
        rich_replay_dir=rich_dir,
        signal_review_dir=signal_dir,
        tracking_dir=tracking_dir,
    )

    assert len(loaded.rich_replay_daily_equity) == len(inputs.rich_replay_daily_equity)
    assert len(loaded.signal_review_symbol_path) == len(inputs.signal_review_symbol_path)
    assert len(loaded.tracking_summary) == len(inputs.tracking_summary)


def test_filtering_primary_secondary_and_reference_rows() -> None:
    selected = select_visual_deep_dive_rows(_tracking_summary())

    assert selected["candidate_id"].tolist() == [
        "candidate_0001",
        "candidate_0003",
        "candidate_0001",
        "candidate_0003",
    ]
    assert len(primary_visual_rows(selected)) == 2
    assert len(secondary_visual_rows(selected)) == 2
    assert selected["production_ready"].eq(False).all()


def test_chart_output_file_names_and_summary_decisions(tmp_path: Path) -> None:
    inputs = _inputs()
    artifacts = build_long_history_core_sleeve_visual_deep_dive_artifacts(inputs)
    paths = write_long_history_core_sleeve_visual_deep_dive_artifacts(
        artifacts=artifacts,
        inputs=inputs,
        output_dir=tmp_path,
    )

    for name in CHART_NAMES:
        assert paths[name].exists()
        assert paths[name].stat().st_size > 0
    assert artifacts.chart_names == CHART_NAMES
    assert artifacts.summary["production_ready"].eq(False).all()
    assert "keep_tracking_with_concentration_caution" in artifacts.summary["review_status"].tolist()
    assert "keep_tracking_with_drawdown_caution" in artifacts.summary["review_status"].tolist()
    assert (
        artifacts.summary.loc[
            artifacts.summary["visual_role"].eq("secondary_40_60_reference"),
            "review_status",
        ]
        .eq("watchlist_sizing_reference")
        .all()
    )


def test_report_writes_required_sections_and_links(tmp_path: Path) -> None:
    inputs = _inputs()
    artifacts = build_long_history_core_sleeve_visual_deep_dive_artifacts(inputs)
    paths = write_long_history_core_sleeve_visual_deep_dive_artifacts(
        artifacts=artifacts,
        inputs=inputs,
        output_dir=tmp_path,
    )

    assert paths[SUMMARY_NAME].exists()
    assert paths[REPORT_NAME].exists()
    assert pd.read_csv(paths[SUMMARY_NAME])["production_ready"].eq(False).all()

    markdown = paths[REPORT_NAME].read_text(encoding="utf-8")
    for section in [
        "## 1. Executive summary",
        "## 2. Current strategies being visualized",
        "## 3. Equity curve comparison",
        "## 4. Drawdown comparison",
        "## 5. Rolling QQQ-relative behavior",
        "## 6. Subperiod performance",
        "## 7. candidate_0001 ETF sleeve timeline and exposure",
        "## 8. candidate_0003 ETF sleeve timeline and exposure",
        "## 9. What the charts reveal",
        "## 10. Current decision",
        "## 11. Caveats",
    ]:
        assert section in markdown
    for name in CHART_NAMES:
        assert f"]({name})" in markdown
    assert "No production signal export" in markdown
    assert "No model card" in markdown
    assert "QQQ remains the primary benchmark" in markdown
    assert "SPY is the core sleeve asset" in markdown
    assert "s3://" not in markdown.lower()
    assert "artifacts/production" not in markdown


def test_no_cloud_model_card_or_production_export_paths_in_script() -> None:
    script = Path("scripts/create_long_history_qqq_core_sleeve_visual_deep_dive.sh").read_text(
        encoding="utf-8"
    )

    assert "s3://" not in script.lower()
    assert "gs://" not in script.lower()
    assert "model_card" not in script
    assert "production signal export" in script
    assert "production-ready marking" in script


def _write_input_dirs(
    inputs: LongHistoryCoreSleeveVisualDeepDiveInputs,
    rich_dir: Path,
    signal_dir: Path,
    tracking_dir: Path,
) -> None:
    rich_dir.mkdir()
    signal_dir.mkdir()
    tracking_dir.mkdir()
    inputs.rich_replay_daily_equity.to_csv(
        rich_dir / f"{RICH_REPLAY_PREFIX}_daily_equity.csv",
        index=False,
    )
    inputs.rich_replay_daily_drawdown.to_csv(
        rich_dir / f"{RICH_REPLAY_PREFIX}_daily_drawdown.csv",
        index=False,
    )
    inputs.rich_replay_summary.to_csv(rich_dir / f"{RICH_REPLAY_PREFIX}_summary.csv", index=False)
    inputs.rich_replay_subperiods.to_csv(
        rich_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv",
        index=False,
    )
    inputs.rich_replay_rolling_windows.to_csv(
        rich_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv",
        index=False,
    )
    inputs.signal_review_symbol_regime_breakdown.to_csv(
        signal_dir / f"{SIGNAL_REVIEW_PREFIX}_symbol_regime_breakdown.csv",
        index=False,
    )
    inputs.signal_review_symbol_path.to_csv(
        signal_dir / f"{SIGNAL_REVIEW_PREFIX}_symbol_path.csv",
        index=False,
    )
    inputs.signal_review_summary.to_csv(
        signal_dir / f"{SIGNAL_REVIEW_PREFIX}_summary.csv",
        index=False,
    )
    inputs.tracking_summary.to_csv(tracking_dir / f"{TRACKING_PREFIX}_summary.csv", index=False)
    inputs.tracking_recent_windows.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_recent_windows.csv",
        index=False,
    )
    inputs.tracking_drawdown_watch.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_drawdown_watch.csv",
        index=False,
    )
    inputs.tracking_symbol_watch.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_symbol_watch.csv",
        index=False,
    )


def _inputs() -> LongHistoryCoreSleeveVisualDeepDiveInputs:
    return LongHistoryCoreSleeveVisualDeepDiveInputs(
        rich_replay_daily_equity=_daily_curves(),
        rich_replay_daily_drawdown=_daily_curves(),
        rich_replay_summary=_rich_summary(),
        rich_replay_subperiods=_subperiods(),
        rich_replay_rolling_windows=_rolling_windows(),
        signal_review_symbol_regime_breakdown=_symbol_regimes(),
        signal_review_symbol_path=_symbol_path(),
        signal_review_summary=_signal_summary(),
        tracking_summary=_tracking_summary(),
        tracking_recent_windows=_recent_windows(),
        tracking_drawdown_watch=_drawdown_watch(),
        tracking_symbol_watch=_symbol_watch(),
    )


def _tracking_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _summary_row(
                "candidate_0001", 0.35, 0.65, "primary return-oriented tracking candidate"
            ),
            _summary_row(
                "candidate_0003",
                0.25,
                0.75,
                "primary balanced/risk-controlled tracking candidate",
            ),
            _summary_row("candidate_0001", 0.40, 0.60, "secondary tracking reference"),
            _summary_row("candidate_0003", 0.40, 0.60, "secondary tracking reference"),
            _summary_row("candidate_0001", 0.00, 1.00, "raw return reference only"),
        ]
    )


def _rich_summary() -> pd.DataFrame:
    rows = []
    for row in _tracking_summary().itertuples(index=False):
        rows.append(
            {
                **row._asdict(),
                "recommendation_action": "rich_replay_next",
                "portfolio_name": row.candidate_id,
                "total_return": 1.4 if row.candidate_id == "candidate_0001" else 1.3,
                "qqq_buy_hold_total_return": 1.0,
                "spy_buy_hold_total_return": 0.7,
                "strategy_vs_qqq": 0.4,
                "strategy_vs_spy": 0.7,
                "max_drawdown": -0.24,
                "qqq_buy_hold_max_drawdown": -0.25,
                "spy_buy_hold_max_drawdown": -0.22,
                "drawdown_improvement_vs_qqq": 0.01,
                "positive_vs_qqq_window_rate": 0.6,
                "worst_strategy_vs_qqq": -0.1,
                "top_symbol": "XLE" if row.candidate_id == "candidate_0001" else "ARKK",
                "top_symbol_exposure_pct": 0.44,
                "top3_symbol_exposure_pct": 0.86,
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _summary_row(candidate_id: str, spy: float, sleeve: float, role: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "spy_core_weight": spy,
        "candidate_sleeve_weight": sleeve,
        "portfolio_accounting_mode": "candidate_rebalance_aligned_weights",
        "practical_role": role,
        "total_return": 1.2,
        "strategy_vs_qqq": 0.2,
        "max_drawdown": -0.2,
        "drawdown_improvement_vs_qqq": 0.02,
        "return_to_drawdown": 6.0,
        "positive_vs_qqq_window_rate": 0.6,
        "worst_strategy_vs_qqq": -0.1,
        "top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "production_ready": False,
    }


def _daily_curves() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    for selected in _tracking_summary().itertuples(index=False):
        for index, date in enumerate(dates):
            equity = 1.0 + (index * 0.03)
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "candidate_id": selected.candidate_id,
                    "spy_core_weight": selected.spy_core_weight,
                    "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                    "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                    "portfolio_name": selected.candidate_id,
                    "equity": equity,
                    "drawdown": -0.01 * index,
                    "candidate_equity": equity,
                    "candidate_drawdown": -0.01 * index,
                    "qqq_equity": 1.0 + (index * 0.02),
                    "qqq_drawdown": -0.015 * index,
                    "spy_equity": 1.0 + (index * 0.01),
                    "spy_drawdown": -0.012 * index,
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _rolling_windows() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        for years in (1, 2):
            rows.append(
                {
                    "candidate_id": selected.candidate_id,
                    "spy_core_weight": selected.spy_core_weight,
                    "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                    "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                    "window_label": f"{years}y_20200101_20201231",
                    "window_years": years,
                    "window_start": "2020-01-01",
                    "window_end": "2020-12-31",
                    "strategy_total_return": 0.2,
                    "qqq_total_return": 0.1,
                    "strategy_vs_qqq": 0.1 - (years * 0.01),
                    "strategy_max_drawdown": -0.2,
                    "qqq_max_drawdown": -0.25,
                    "drawdown_improvement_vs_qqq": 0.05 - (years * 0.01),
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _subperiods() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        for period in ("full", "bear_2022"):
            rows.append(
                {
                    "candidate_id": selected.candidate_id,
                    "spy_core_weight": selected.spy_core_weight,
                    "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                    "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                    "period_name": period,
                    "period_start": "2020-01-01",
                    "period_end": "2020-12-31",
                    "total_return": 0.2,
                    "qqq_total_return": 0.1,
                    "max_drawdown": -0.2,
                    "qqq_max_drawdown": -0.25,
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _signal_summary() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        path = "XLU -> XLE -> VUG" if selected.candidate_id == "candidate_0001" else "ARKK -> XLK"
        rows.append(
            {
                "candidate_id": selected.candidate_id,
                "spy_core_weight": selected.spy_core_weight,
                "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                "practical_role": selected.practical_role,
                "selected_symbols_in_order": path,
                "top_symbol": "XLE" if selected.candidate_id == "candidate_0001" else "ARKK",
                "top_symbol_exposure_pct": 0.44,
                "top3_symbol_exposure_pct": 0.86,
                "concentration_watch_level": "high"
                if selected.candidate_id == "candidate_0001"
                else "caution",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _symbol_regimes() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        if "primary" not in selected.practical_role:
            continue
        symbols = (
            ["XLU", "XLE", "VUG"] if selected.candidate_id == "candidate_0001" else ["ARKK", "XLK"]
        )
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "candidate_id": selected.candidate_id,
                    "spy_core_weight": selected.spy_core_weight,
                    "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                    "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                    "practical_role": selected.practical_role,
                    "symbol": symbol,
                    "start_date": f"2020-0{index + 1}-01",
                    "end_date": f"2020-0{index + 2}-01",
                    "trading_days": 21,
                    "exposure_pct": 1 / len(symbols),
                    "sleeve_weighted_exposure_pct": selected.candidate_sleeve_weight / len(symbols),
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _symbol_path() -> pd.DataFrame:
    rows = []
    for regime in _symbol_regimes().itertuples(index=False):
        rows.append(
            {
                "candidate_id": regime.candidate_id,
                "spy_core_weight": regime.spy_core_weight,
                "candidate_sleeve_weight": regime.candidate_sleeve_weight,
                "portfolio_accounting_mode": regime.portfolio_accounting_mode,
                "practical_role": regime.practical_role,
                "date": regime.start_date,
                "symbol": regime.symbol,
                "target_weight": 1.0,
                "sleeve_weighted_target_weight": regime.candidate_sleeve_weight,
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _recent_windows() -> pd.DataFrame:
    return _rolling_windows()


def _drawdown_watch() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        rows.append(
            {
                "candidate_id": selected.candidate_id,
                "spy_core_weight": selected.spy_core_weight,
                "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                "practical_role": selected.practical_role,
                "full_period_max_drawdown": -0.24,
                "recent_max_drawdown": -0.2,
                "worst_rolling_drawdown_gap_vs_qqq": -0.06,
                "recent_drawdown_worse_than_qqq": True,
                "drawdown_watch_level": "severe"
                if selected.candidate_id == "candidate_0003"
                else "caution",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _symbol_watch() -> pd.DataFrame:
    rows = []
    for selected in _tracking_summary().itertuples(index=False):
        rows.append(
            {
                "candidate_id": selected.candidate_id,
                "spy_core_weight": selected.spy_core_weight,
                "candidate_sleeve_weight": selected.candidate_sleeve_weight,
                "portfolio_accounting_mode": selected.portfolio_accounting_mode,
                "practical_role": selected.practical_role,
                "top_symbol": "XLE" if selected.candidate_id == "candidate_0001" else "ARKK",
                "top_symbol_exposure_pct": 0.44,
                "top3_symbol_exposure_pct": 0.86,
                "sleeve_weighted_top_symbol_exposure_pct": 0.28,
                "sleeve_weighted_top3_symbol_exposure_pct": 0.56,
                "concentration_watch_level": "high"
                if selected.candidate_id == "candidate_0001"
                else "caution",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)
