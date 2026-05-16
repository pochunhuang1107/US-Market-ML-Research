from __future__ import annotations

from pathlib import Path

import pytest

from us_market_ml.cli import run_final_candidate_deep_dive_bundles as bundles


def test_final_candidate_specs_are_current_research_stage_shortlist() -> None:
    groups = bundles.build_final_candidate_groups()

    assert [group.name for group in groups] == ["full", "broad_macro"]
    specs = {
        candidate.name: candidate
        for group in groups
        for candidate in group.candidates
    }

    assert list(specs) == [
        "fixed_fold_reference_full_linear_top1",
        "longer_window_balanced_full_ridge_top5",
        "full_balanced_reference_ridge_top3",
        "benchmark_relative_broadmacro_linear_top1",
        "defensive_broadmacro_ridge_top5",
    ]
    assert specs[
        "fixed_fold_reference_full_linear_top1"
    ].candidate_arg == (
        "baseline:fixed_fold_reference_full_linear_top1:linear_regression:1:42"
    )
    assert specs[
        "benchmark_relative_broadmacro_linear_top1"
    ].candidate_arg == (
        "baseline:benchmark_relative_broadmacro_linear_top1:linear_regression:1:42"
    )
    assert specs[
        "longer_window_balanced_full_ridge_top5"
    ].research_label == "full / ridge / top5 / equal_weight / cost10"
    assert specs["defensive_broadmacro_ridge_top5"].top_k == 5
    assert all(candidate.allocation_mode == "equal_weight" for candidate in specs.values())
    assert all(
        candidate.transaction_cost_bps == pytest.approx(10.0)
        for candidate in specs.values()
    )


def test_output_directory_naming_and_command_construction(tmp_path: Path) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    output_dir = tmp_path / "deep_dive"
    groups = bundles.build_final_candidate_groups(
        full_data_path=full_path,
        broad_macro_data_path=broad_path,
        output_dir=output_dir,
    )
    full_group, broad_group = groups

    assert full_group.output_dir == output_dir / "full"
    assert broad_group.output_dir == output_dir / "broad_macro"
    assert full_group.report_name == "final_candidate_deep_dive_full"
    assert broad_group.report_name == "final_candidate_deep_dive_broad_macro"

    command = bundles.build_deep_dive_command(full_group, python_bin="/tmp/python")
    assert command[:4] == [
        "/tmp/python",
        "-m",
        "us_market_ml.cli.run_candidate_deep_dive_report",
        "--data-path",
    ]
    assert str(full_path) in command
    assert str(output_dir / "full") in command
    assert command.count("--fold") == 4
    assert command.count("--candidate") == 3
    assert "--report-name" in command
    assert "final_candidate_deep_dive_full" in command
    assert "--transaction-cost-bps" in command
    assert "10" in command
    assert "--min-holding-days" in command
    assert "10" in command
    assert "--switch-margin" in command
    assert "0.002" in command
    assert "--allocation-mode" in command
    assert "equal_weight" in command


def test_dry_run_prints_commands_without_running_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    full_path.write_text("placeholder", encoding="utf-8")
    broad_path.write_text("placeholder", encoding="utf-8")
    calls: list[list[str]] = []

    def _fake_run(argv: list[str] | None = None) -> int:
        calls.append(list(argv or []))
        return 0

    monkeypatch.setattr(bundles, "run_candidate_deep_dive_report_from_args", _fake_run)

    return_code = bundles.run_final_candidate_deep_dive_bundles_from_args(
        [
            "--full-data-path",
            str(full_path),
            "--broad-macro-data-path",
            str(broad_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--python-bin",
            "/tmp/python",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert calls == []
    assert "Dry run: commands that would run" in captured.out
    assert captured.out.count("us_market_ml.cli.run_candidate_deep_dive_report") == 2
    assert "final_candidate_deep_dive_full" in captured.out
    assert "final_candidate_deep_dive_broad_macro" in captured.out


def test_broad_macro_missing_file_warns_and_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    full_path = tmp_path / "full.parquet"
    missing_broad_path = tmp_path / "missing_broad.parquet"
    full_path.write_text("placeholder", encoding="utf-8")
    calls: list[list[str]] = []

    def _fake_run(argv: list[str] | None = None) -> int:
        calls.append(list(argv or []))
        return 0

    monkeypatch.setattr(bundles, "run_candidate_deep_dive_report_from_args", _fake_run)

    return_code = bundles.run_final_candidate_deep_dive_bundles_from_args(
        [
            "--full-data-path",
            str(full_path),
            "--broad-macro-data-path",
            str(missing_broad_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert len(calls) == 1
    assert "WARNING: broad_macro data file is missing" in captured.out
    assert "final_candidate_deep_dive_full" in calls[0]
    assert "final_candidate_deep_dive_broad_macro" not in calls[0]


def test_missing_full_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required full universe data file"):
        bundles.run_final_candidate_deep_dive_bundles_from_args(
            [
                "--full-data-path",
                str(tmp_path / "missing_full.parquet"),
                "--broad-macro-data-path",
                str(tmp_path / "missing_broad.parquet"),
                "--output-dir",
                str(tmp_path / "reports"),
                "--dry-run",
            ]
        )


def test_summary_markdown_explains_candidate_labels_and_outputs(tmp_path: Path) -> None:
    groups = bundles.build_final_candidate_groups(output_dir=tmp_path / "deep_dive")
    summary_path = bundles.write_final_candidate_deep_dive_summary(
        output_dir=tmp_path / "deep_dive",
        groups=groups,
    )

    markdown = summary_path.read_text(encoding="utf-8")

    assert summary_path.name == "final_candidate_deep_dive_summary.md"
    assert "No candidate is production-ready yet." in markdown
    assert "full / linear_regression / top1 / equal_weight / cost10" in markdown
    assert "broad_macro / linear_regression / top1 / equal_weight / cost10" in markdown
    assert "final_candidate_deep_dive_full_candidate_equity.png" in markdown
    assert "monthly_selection_heatmap.png" in markdown
