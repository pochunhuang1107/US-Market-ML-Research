from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.visualization import plot_equity_curves


def test_plot_equity_curves_creates_png_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    combined = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"]
            ),
            "series_name": ["strategy", "strategy", "SPY", "SPY"],
            "daily_return": [0.0, 0.05, 0.0, 0.02],
            "equity": [1.0, 1.05, 1.0, 1.02],
        }
    )
    output_path = tmp_path / "plots" / "equity.png"

    returned_path = plot_equity_curves(combined, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_equity_curves_raises_for_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="series_name"):
        plot_equity_curves(pd.DataFrame({"date": [], "equity": []}), tmp_path / "bad.png")
