from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.visualization import plot_cost_sensitivity


def test_plot_cost_sensitivity_creates_png(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    sensitivity = pd.DataFrame(
        {
            "candidate_name": ["linear_top1", "linear_top1", "mlp_top1", "mlp_top1"],
            "cost_bps": [0.0, 10.0, 0.0, 10.0],
            "total_return": [0.12, 0.10, 0.15, 0.11],
        }
    )
    output_path = tmp_path / "charts" / "cost_sensitivity.png"

    returned_path = plot_cost_sensitivity(sensitivity, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_cost_sensitivity_validates_required_columns(tmp_path: Path) -> None:
    sensitivity = pd.DataFrame({"candidate_name": ["linear_top1"], "cost_bps": [0.0]})

    with pytest.raises(ValueError, match="total_return"):
        plot_cost_sensitivity(sensitivity, tmp_path / "cost_sensitivity.png")
