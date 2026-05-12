"""Static holdings and contribution plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_selection_frequency(
    holdings_summary: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
    title: str = "Selected Days by Symbol",
) -> Path:
    """Save a bar chart of selected days by symbol and return its path."""

    _validate_required_columns(holdings_summary, ["symbol", "selected_days"])

    import matplotlib.pyplot as plt

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_data = (
        holdings_summary.sort_values("selected_days", ascending=False, kind="mergesort")
        .head(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_data["symbol"].astype(str), chart_data["selected_days"])
    ax.set_title(title)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Selected Days")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(resolved_output_path, format="png")
    plt.close(fig)

    return resolved_output_path


def plot_contribution_by_symbol(
    contribution_summary: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
    title: str = "Total Position Return by Symbol",
) -> Path:
    """Save a bar chart of total position return by symbol and return its path."""

    _validate_required_columns(contribution_summary, ["symbol", "total_position_return"])

    import matplotlib.pyplot as plt

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_data = (
        contribution_summary.sort_values(
            "total_position_return",
            ascending=False,
            kind="mergesort",
        )
        .head(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_data["symbol"].astype(str), chart_data["total_position_return"])
    ax.set_title(title)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Total Position Return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(resolved_output_path, format="png")
    plt.close(fig)

    return resolved_output_path


def build_monthly_selection_matrix(
    selected_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Return monthly selected-day counts with months as rows and symbols as columns."""

    _validate_required_columns(selected_holdings, ["date", "symbol"])
    if selected_holdings.empty:
        return pd.DataFrame()

    holdings = selected_holdings.copy()
    holdings["month"] = pd.to_datetime(holdings["date"]).dt.to_period("M").astype(str)

    matrix = pd.crosstab(holdings["month"], holdings["symbol"])
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def plot_monthly_selection_heatmap(
    selected_holdings: pd.DataFrame,
    output_path: str | Path,
    title: str = "Monthly Selected Days by Symbol",
) -> Path:
    """Save a heatmap of monthly selected-day counts by symbol and return its path."""

    matrix = build_monthly_selection_matrix(selected_holdings)

    import matplotlib.pyplot as plt

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No selected holdings", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(resolved_output_path, format="png")
        plt.close(fig)
        return resolved_output_path

    fig_width = max(8.0, min(16.0, 0.45 * len(matrix.columns) + 3.0))
    fig_height = max(5.0, min(14.0, 0.35 * len(matrix.index) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Month")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.astype(str))
    fig.colorbar(image, ax=ax, label="Selected Days")
    fig.tight_layout()
    fig.savefig(resolved_output_path, format="png")
    plt.close(fig)

    return resolved_output_path


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required holdings columns: {missing}")


__all__ = [
    "build_monthly_selection_matrix",
    "plot_contribution_by_symbol",
    "plot_monthly_selection_heatmap",
    "plot_selection_frequency",
]
