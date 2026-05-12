"""Static cost sensitivity plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_cost_sensitivity(
    sensitivity: pd.DataFrame,
    output_path: str | Path,
    metric_col: str = "total_return",
    title: str = "Cost Sensitivity",
) -> Path:
    """Save a PNG with one cost-sensitivity line per candidate and return its path."""

    _validate_required_columns(sensitivity, ["candidate_name", "cost_bps", metric_col])

    import matplotlib.pyplot as plt

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_sensitivity = sensitivity.sort_values(
        by=["candidate_name", "cost_bps"],
        kind="mergesort",
    )
    for candidate_name, candidate_df in sorted_sensitivity.groupby(
        "candidate_name",
        sort=True,
        dropna=False,
    ):
        ax.plot(
            candidate_df["cost_bps"],
            candidate_df[metric_col],
            marker="o",
            label=str(candidate_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel(metric_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(resolved_output_path, format="png")
    plt.close(fig)

    return resolved_output_path


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required cost sensitivity columns: {missing}")


__all__ = ["plot_cost_sensitivity"]
