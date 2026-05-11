"""Static equity curve plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_equity_curves(
    combined_equity: pd.DataFrame,
    output_path: str | Path,
    title: str = "Strategy vs Benchmark Equity",
) -> Path:
    """Save a PNG with one equity line per series and return its path."""

    _validate_required_columns(combined_equity, ["date", "series_name", "equity"])

    import matplotlib.pyplot as plt

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_equity = combined_equity.sort_values(
        by=["series_name", "date"],
        kind="mergesort",
    )
    for series_name, series_df in sorted_equity.groupby("series_name", sort=True, dropna=False):
        ax.plot(series_df["date"], series_df["equity"], label=str(series_name))

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(resolved_output_path, format="png")
    plt.close(fig)

    return resolved_output_path


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required equity curve columns: {missing}")


__all__ = ["plot_equity_curves"]
