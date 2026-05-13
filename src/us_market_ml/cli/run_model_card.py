"""Generate Markdown model cards from candidate deep-dive outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from us_market_ml.reports import build_candidate_model_card


def run_model_card_from_args(argv: list[str] | None = None) -> int:
    """Read local candidate diagnostic CSVs and write a Markdown model card."""

    args = _parse_args(argv)
    candidate_dir = Path(args.candidate_dir)
    candidate_name = args.candidate_name

    metrics = _read_candidate_csv(candidate_dir, candidate_name, "metrics")
    fold_summary = _read_candidate_csv(candidate_dir, candidate_name, "fold_summary")
    concentration_report = _read_candidate_csv(
        candidate_dir,
        candidate_name,
        "concentration_report",
    )
    holding_period_summary = _read_candidate_csv(
        candidate_dir,
        candidate_name,
        "holding_period_summary",
    )
    holdings_summary = _read_candidate_csv(candidate_dir, candidate_name, "holdings_summary")
    contribution_summary = _read_candidate_csv(
        candidate_dir,
        candidate_name,
        "contribution_summary",
    )
    cost_sensitivity = (
        pd.read_csv(args.cost_sensitivity_path)
        if args.cost_sensitivity_path is not None
        else None
    )

    markdown = build_candidate_model_card(
        candidate_name=candidate_name,
        candidate_family=args.candidate_family,
        model_type=args.model_type,
        config=args.config_json if args.config_json is not None else {},
        metrics=metrics,
        fold_summary=fold_summary,
        concentration_report=concentration_report,
        holding_period_summary=holding_period_summary,
        holdings_summary=holdings_summary,
        contribution_summary=contribution_summary,
        cost_sensitivity=cost_sensitivity,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"output model card Markdown: {output_path}")
    return 0


def parse_config_json(value: str) -> dict[str, object]:
    """Parse optional JSON object metadata for the model card."""

    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--config-json must be a JSON object.")
    return parsed


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a research-only Markdown model card from local candidate CSVs."
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-family", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--config-json", type=parse_config_json)
    parser.add_argument("--cost-sensitivity-path", type=Path)
    return parser.parse_args(argv)


def _read_candidate_csv(candidate_dir: Path, candidate_name: str, suffix: str) -> pd.DataFrame:
    path = candidate_dir / f"{candidate_name}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate diagnostic CSV: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    raise SystemExit(run_model_card_from_args())
