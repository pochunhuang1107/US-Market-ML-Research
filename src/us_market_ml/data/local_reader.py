"""Local file readers for curated market research input data.

These utilities load local CSV and parquet exports into the OHLCV DataFrame
shape expected by the research pipelines. They are intended for development
and offline research inputs only: S3 reading is a future task, feature/label
construction belongs in the dataset builder, and source files are never
mutated.
"""

from __future__ import annotations

import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LocalMarketDataConfig:
    """Column configuration for local market data loading.

    The defaults match curated ETF daily exports from the upstream
    ``US-Market-Data`` pipeline. Extra columns such as ``year``, ``month``, and
    ``transactions`` are preserved for downstream research use.
    """

    date_col: str = "date"
    symbol_col: str = "symbol"
    required_columns: tuple[str, ...] = (
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
    )


def read_local_market_data(
    path: str | Path,
    config: LocalMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Read local CSV/parquet market data into a normalized research frame.

    ``path`` may point to a single ``.parquet``/``.csv`` file or a directory.
    Directories are searched recursively, preferring parquet files and falling
    back to CSV only when no parquet files exist. This reader is intentionally
    local-only: it does not read from S3, build features or labels, write
    artifacts, or mutate source files.
    """

    path_text = str(path)
    glob_paths = _expand_supported_glob(path_text)
    if glob_paths is not None:
        return _read_supported_paths(glob_paths, config)

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Local market data path does not exist: {resolved_path}")

    if resolved_path.is_dir():
        parquet_paths = _find_supported_files(resolved_path, ".parquet")
        if parquet_paths:
            return read_parquet_files(parquet_paths, config)

        csv_paths = _find_supported_files(resolved_path, ".csv")
        if csv_paths:
            return read_csv_files(csv_paths, config)

        raise ValueError(f"Directory contains no supported market data files: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet_files([resolved_path], config)
    if suffix == ".csv":
        return read_csv_files([resolved_path], config)

    raise ValueError(
        "Unsupported local market data file extension "
        f"{resolved_path.suffix!r}; expected .parquet or .csv."
    )


def read_local_market_data_glob(
    pattern: str | Path,
    config: LocalMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Read local market data files matched by a glob pattern.

    This is a convenience wrapper around ``read_local_market_data`` for callers
    that want to be explicit about glob input. Paths are expanded locally only;
    S3 URIs and remote filesystems are intentionally outside this reader.
    """

    glob_paths = _expand_supported_glob(str(pattern), require_glob=True)
    if glob_paths is None:
        raise ValueError(f"Local market data glob pattern contains no glob magic: {pattern}")
    return _read_supported_paths(glob_paths, config)


def read_parquet_files(
    paths: Sequence[str | Path],
    config: LocalMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Read one or more local parquet files and normalize the result.

    This local research/dev reader does not perform S3 access, feature/label
    construction, artifact writing, or source-file mutation.
    """

    resolved_paths = _resolve_non_empty_paths(paths)
    frames = [pd.read_parquet(path) for path in resolved_paths]
    return normalize_market_data_frame(pd.concat(frames, ignore_index=True), config)


def read_csv_files(
    paths: Sequence[str | Path],
    config: LocalMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Read one or more local CSV files and normalize the result.

    This local research/dev reader does not perform S3 access, feature/label
    construction, artifact writing, or source-file mutation.
    """

    resolved_paths = _resolve_non_empty_paths(paths)
    frames = [pd.read_csv(path) for path in resolved_paths]
    return normalize_market_data_frame(pd.concat(frames, ignore_index=True), config)


def validate_market_data_columns(
    df: pd.DataFrame,
    config: LocalMarketDataConfig | None = None,
) -> None:
    """Validate that required OHLCV research input columns are present."""

    resolved_config = config if config is not None else LocalMarketDataConfig()
    missing_columns = [
        column
        for column in dict.fromkeys(
            (
                resolved_config.date_col,
                resolved_config.symbol_col,
                *resolved_config.required_columns,
            )
        )
        if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Market data is missing required columns: {missing}")


def normalize_market_data_frame(
    df: pd.DataFrame,
    config: LocalMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Return a copied, validated, date-parsed, symbol/date-sorted frame.

    Extra columns are preserved. This function prepares local input data for
    downstream research pipelines only; it does not build features or labels
    and does not mutate the caller's DataFrame.
    """

    resolved_config = config if config is not None else LocalMarketDataConfig()
    validate_market_data_columns(df, resolved_config)

    out = df.copy()
    out[resolved_config.date_col] = pd.to_datetime(out[resolved_config.date_col])
    return out.sort_values(
        by=[resolved_config.symbol_col, resolved_config.date_col],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_non_empty_paths(paths: Sequence[str | Path]) -> list[Path]:
    resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ValueError("At least one local market data file path is required.")
    return resolved_paths


def _expand_supported_glob(pattern: str, *, require_glob: bool = False) -> list[Path] | None:
    if not glob.has_magic(pattern):
        if require_glob:
            return None
        return None

    matched_paths = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    file_paths = [path for path in matched_paths if path.is_file()]
    if not file_paths:
        raise FileNotFoundError(f"Local market data glob matched no files: {pattern}")
    return file_paths


def _read_supported_paths(
    paths: Sequence[Path],
    config: LocalMarketDataConfig | None,
) -> pd.DataFrame:
    parquet_paths = [path for path in paths if path.suffix.lower() == ".parquet"]
    if parquet_paths:
        return read_parquet_files(parquet_paths, config)

    csv_paths = [path for path in paths if path.suffix.lower() == ".csv"]
    if csv_paths:
        return read_csv_files(csv_paths, config)

    raise ValueError("Local market data glob matched no supported .parquet or .csv files.")


def _find_supported_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == suffix
    )


__all__ = [
    "LocalMarketDataConfig",
    "normalize_market_data_frame",
    "read_csv_files",
    "read_local_market_data",
    "read_local_market_data_glob",
    "read_parquet_files",
    "validate_market_data_columns",
]
