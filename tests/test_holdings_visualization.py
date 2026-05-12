from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.visualization import (
    build_monthly_selection_matrix,
    plot_contribution_by_symbol,
    plot_monthly_selection_heatmap,
    plot_selection_frequency,
)


def test_plot_selection_frequency_creates_png_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    holdings_summary = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA", "CCC"],
            "selected_days": [2, 3, 1],
        }
    )
    output_path = tmp_path / "plots" / "selection_frequency.png"

    returned_path = plot_selection_frequency(holdings_summary, output_path, top_n=2)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_contribution_by_symbol_creates_png_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    contribution_summary = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "total_position_return": [0.05, -0.02, 0.10],
        }
    )
    output_path = tmp_path / "plots" / "contribution_by_symbol.png"

    returned_path = plot_contribution_by_symbol(contribution_summary, output_path, top_n=2)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_build_monthly_selection_matrix_counts_selected_days_by_symbol() -> None:
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-02-01",
                    "2021-02-03",
                ]
            ),
            "symbol": ["BBB", "AAA", "BBB", "AAA", "CCC"],
        }
    )

    matrix = build_monthly_selection_matrix(selected_holdings)

    expected = pd.DataFrame(
        {
            "AAA": [1, 1],
            "BBB": [2, 0],
            "CCC": [0, 1],
        },
        index=pd.Index(["2021-01", "2021-02"], name="month"),
    )
    expected.columns.name = "symbol"
    pd.testing.assert_frame_equal(matrix, expected)


def test_build_monthly_selection_matrix_empty_input_returns_empty_frame() -> None:
    matrix = build_monthly_selection_matrix(pd.DataFrame(columns=["date", "symbol"]))

    assert matrix.empty


def test_plot_monthly_selection_heatmap_creates_png_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01"]),
            "symbol": ["AAA", "BBB", "AAA"],
        }
    )
    output_path = tmp_path / "plots" / "monthly_selection_heatmap.png"

    returned_path = plot_monthly_selection_heatmap(selected_holdings, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_monthly_selection_heatmap_empty_input_creates_png_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    output_path = tmp_path / "plots" / "empty_heatmap.png"

    returned_path = plot_monthly_selection_heatmap(
        pd.DataFrame(columns=["date", "symbol"]),
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("plot_function", "frame"),
    [
        (plot_selection_frequency, pd.DataFrame({"symbol": ["AAA"]})),
        (plot_contribution_by_symbol, pd.DataFrame({"symbol": ["AAA"]})),
        (plot_monthly_selection_heatmap, pd.DataFrame({"date": ["2021-01-01"]})),
    ],
)
def test_holdings_visualizations_raise_for_missing_required_columns(
    plot_function,
    frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        plot_function(frame, tmp_path / "bad.png")


def test_build_monthly_selection_matrix_raises_for_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="symbol"):
        build_monthly_selection_matrix(pd.DataFrame({"date": ["2021-01-01"]}))
