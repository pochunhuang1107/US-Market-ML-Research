from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.allocation_overlays import (
    AllocationOverlaySpec,
    apply_allocation_overlay,
    default_pure_weighted_allocation_specs,
    validate_overlay_weights,
)


def test_fixed_rank_allocation_sums_to_one() -> None:
    signals = _signals_for_date("2024-01-02")
    spec = AllocationOverlaySpec(
        name="rank_weighted_top5",
        kind="rank_weighted_topk",
        rank_weights=(0.30, 0.25, 0.20, 0.15, 0.10),
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].tolist() == pytest.approx([0.30, 0.25, 0.20, 0.15, 0.10])
    assert validate_overlay_weights(out)


def test_score_weighted_allocation_sums_to_one_and_respects_cap() -> None:
    signals = _signals_for_date(
        "2024-01-02",
        predictions={"AAA": 0.10, "BBB": 0.06, "CCC": 0.03, "DDD": 0.01, "EEE": -0.02},
    )
    spec = AllocationOverlaySpec(
        name="capped_score_weighted_top5_cap0.30",
        kind="capped_score_weighted_topk",
        max_weight=0.30,
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].max() <= 0.30 + 1.0e-12
    assert validate_overlay_weights(out)


def test_top1_allocation_sums_to_one() -> None:
    signals = _signals_for_date("2024-01-02")
    spec = _pure_spec("top1_only")

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["symbol"].tolist() == ["AAA"]
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].tolist() == pytest.approx([1.0])
    assert validate_overlay_weights(out)


@pytest.mark.parametrize(
    ("rule_name", "expected_count", "expected_weight"),
    [
        ("top2_equal_weight", 2, 0.50),
        ("top3_equal_weight", 3, 1.0 / 3.0),
        ("equal_weight_top5_baseline", 5, 0.20),
    ],
)
def test_topn_equal_allocations_sum_to_one(
    rule_name: str,
    expected_count: int,
    expected_weight: float,
) -> None:
    signals = _signals_for_date("2024-01-02")
    spec = _pure_spec(rule_name)

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert len(selected) == expected_count
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].tolist() == pytest.approx(
        [expected_weight] * expected_count
    )
    assert validate_overlay_weights(out)


@pytest.mark.parametrize(
    ("rule_name", "expected_weights"),
    [
        ("top5_rank_decay_mild", [0.30, 0.25, 0.20, 0.15, 0.10]),
        ("top3_rank_decay", [0.50, 0.30, 0.20]),
    ],
)
def test_pure_rank_decay_allocations_sum_to_one(
    rule_name: str,
    expected_weights: list[float],
) -> None:
    signals = _signals_for_date("2024-01-02")
    spec = _pure_spec(rule_name)

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].tolist() == pytest.approx(expected_weights)
    assert validate_overlay_weights(out)


@pytest.mark.parametrize(
    ("rule_name", "expected_count", "cap"),
    [
        ("score_weighted_top3_cap050", 3, 0.50),
        ("score_weighted_top5_cap035", 5, 0.35),
    ],
)
def test_pure_score_weighted_allocations_sum_to_one_and_respect_caps(
    rule_name: str,
    expected_count: int,
    cap: float,
) -> None:
    signals = _signals_for_date(
        "2024-01-02",
        predictions={"AAA": 0.10, "BBB": 0.06, "CCC": 0.03, "DDD": 0.01, "EEE": -0.02},
    )
    spec = _pure_spec(rule_name)

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert len(selected) == expected_count
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].max() <= cap + 1.0e-12
    assert validate_overlay_weights(out)


@pytest.mark.parametrize(
    ("rule_name", "expected_count", "cap"),
    [
        ("softmax_top3_temp_low", 3, 0.50),
        ("softmax_top5_temp_low", 5, 0.35),
    ],
)
def test_softmax_allocations_sum_to_one_and_respect_caps(
    rule_name: str,
    expected_count: int,
    cap: float,
) -> None:
    signals = _signals_for_date(
        "2024-01-02",
        predictions={"AAA": 0.12, "BBB": 0.06, "CCC": 0.03, "DDD": 0.01, "EEE": 0.00},
    )
    spec = _pure_spec(rule_name)

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert len(selected) == expected_count
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].max() <= cap + 1.0e-12
    assert validate_overlay_weights(out)


def test_score_weighted_allocation_falls_back_to_equal_when_scores_are_negative() -> None:
    signals = _signals_for_date(
        "2024-01-02",
        predictions={"AAA": -0.01, "BBB": -0.02, "CCC": -0.03, "DDD": -0.04, "EEE": -0.05},
    )
    spec = AllocationOverlaySpec(
        name="capped_score_weighted_top5_cap0.40",
        kind="capped_score_weighted_topk",
        max_weight=0.40,
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["target_weight"].tolist() == pytest.approx([0.20] * 5)


def test_qqq_ranked_rule_does_not_force_qqq_when_qqq_is_not_selected() -> None:
    signals = _signals_for_date("2024-01-02")
    spec = _pure_spec("qqq_if_ranked_top3_else_model")

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert "QQQ" not in selected["symbol"].tolist()
    assert len(selected) == 5
    assert selected["target_weight"].sum() == pytest.approx(1.0)


def test_qqq_ranked_rule_uses_fallback_when_qqq_rank_is_too_low() -> None:
    signals = _signals_for_date(
        "2024-01-02",
        selected_symbols=("AAA", "BBB", "CCC", "QQQ", "DDD"),
    )
    spec = _pure_spec("qqq_if_ranked_top3_else_model")

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert selected["symbol"].tolist() == ["AAA", "BBB", "CCC", "QQQ", "DDD"]
    assert selected.loc[selected["symbol"].eq("QQQ"), "target_weight"].iloc[0] == pytest.approx(
        0.20
    )
    assert len(selected) == 5
    assert validate_overlay_weights(out)


@pytest.mark.parametrize(
    ("rule_name", "selected_symbols"),
    [
        ("qqq_if_ranked_top3_else_model", ("AAA", "BBB", "QQQ", "CCC", "DDD")),
        ("qqq_if_ranked_top2_else_top5", ("AAA", "QQQ", "BBB", "CCC", "DDD")),
    ],
)
def test_qqq_ranked_rules_include_qqq_when_rank_condition_is_met(
    rule_name: str,
    selected_symbols: tuple[str, ...],
) -> None:
    signals = _signals_for_date("2024-01-02", selected_symbols=selected_symbols)
    spec = _pure_spec(rule_name)

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out)

    assert "QQQ" in selected["symbol"].tolist()
    assert len(selected) == 3
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert validate_overlay_weights(out)


def test_qqq_core_allocation_combines_when_qqq_is_selected() -> None:
    signals = _signals_for_date("2024-01-02", selected_symbols=("QQQ", "AAA", "BBB", "CCC", "DDD"))
    spec = AllocationOverlaySpec(
        name="qqq_core_20_satellite_top5",
        kind="static_core_satellite",
        core_weights=(("QQQ", 0.20),),
        satellite_weight=0.80,
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out).set_index("symbol")

    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected.loc["QQQ", "target_weight"] == pytest.approx(0.36)
    assert selected.loc["AAA", "target_weight"] == pytest.approx(0.16)


def test_spy_qqq_core_allocation_combines_when_selected() -> None:
    signals = _signals_for_date("2024-01-02", selected_symbols=("SPY", "QQQ", "AAA", "BBB", "CCC"))
    spec = AllocationOverlaySpec(
        name="spy_qqq_core_40_satellite_top5",
        kind="static_core_satellite",
        core_weights=(("SPY", 0.20), ("QQQ", 0.20)),
        satellite_weight=0.60,
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    selected = _selected(out).set_index("symbol")

    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected.loc["SPY", "target_weight"] == pytest.approx(0.32)
    assert selected.loc["QQQ", "target_weight"] == pytest.approx(0.32)
    assert selected.loc["AAA", "target_weight"] == pytest.approx(0.12)


def test_dynamic_benchmark_core_triggers_only_when_regime_flag_active() -> None:
    signals = pd.concat(
        [
            _signals_for_date("2024-01-02", regime_score=0.0),
            _signals_for_date("2024-01-03", regime_score=1.0),
        ],
        ignore_index=True,
    )
    spec = AllocationOverlaySpec(
        name="dynamic_spy_qqq_core_when_benchmark_strong",
        kind="dynamic_core_satellite",
        core_weights=(("SPY", 0.20), ("QQQ", 0.20)),
        satellite_weight=0.60,
        regime_columns=("max_spy_qqq_strong_regime_score",),
    )

    out = apply_allocation_overlay(signals, spec, top_k=5)
    inactive = out.loc[out["date"].eq(pd.Timestamp("2024-01-02"))]
    active = out.loc[out["date"].eq(pd.Timestamp("2024-01-03"))]

    assert inactive.loc[inactive["symbol"].eq("SPY"), "target_weight"].iloc[0] == pytest.approx(0.0)
    assert inactive.loc[inactive["symbol"].eq("QQQ"), "target_weight"].iloc[0] == pytest.approx(0.0)
    assert active.loc[active["symbol"].eq("SPY"), "target_weight"].iloc[0] == pytest.approx(0.20)
    assert active.loc[active["symbol"].eq("QQQ"), "target_weight"].iloc[0] == pytest.approx(0.20)
    assert validate_overlay_weights(out)


def _signals_for_date(
    date: str,
    *,
    selected_symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE"),
    predictions: dict[str, float] | None = None,
    regime_score: float = 0.0,
) -> pd.DataFrame:
    symbols = ("SPY", "QQQ", "AAA", "BBB", "CCC", "DDD", "EEE")
    prediction_by_symbol = {
        "SPY": 0.00,
        "QQQ": 0.01,
        "AAA": 0.09,
        "BBB": 0.08,
        "CCC": 0.07,
        "DDD": 0.06,
        "EEE": 0.05,
    }
    if predictions is not None:
        prediction_by_symbol.update(predictions)
    rows: list[dict[str, object]] = []
    rank_by_symbol = {symbol: rank for rank, symbol in enumerate(selected_symbols, start=1)}
    for index, symbol in enumerate(symbols):
        selected = symbol in selected_symbols
        rows.append(
            {
                "date": pd.Timestamp(date),
                "symbol": symbol,
                "close": 100.0 + index,
                "prediction": prediction_by_symbol[symbol],
                "signal_rank": rank_by_symbol.get(symbol, pd.NA),
                "selected": selected,
                "target_weight": 0.20 if selected else 0.0,
                "max_spy_qqq_strong_regime_score": regime_score,
                "qqq_strong_regime_score": regime_score,
            }
        )
    return pd.DataFrame(rows)


def _selected(signals: pd.DataFrame) -> pd.DataFrame:
    return (
        signals.loc[signals["selected"]]
        .sort_values(["signal_rank", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _pure_spec(name: str) -> AllocationOverlaySpec:
    specs = {spec.name: spec for spec in default_pure_weighted_allocation_specs()}
    return specs[name]
