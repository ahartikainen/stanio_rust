"""Thread scaling scheduler, throughput plots, and README publication tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import benchmarks.thread_sweep as sweep
from benchmarks.plot_threads import create_plots
from benchmarks.publish_threads import publish


def test_configurations_scale_only_controllable_readers() -> None:
    methods = ["staniorust_mmap_parallel", "pandas", "polars", "stanio"]
    configs = sweep.configurations(methods, [1, 4, 16])
    assert [config.key for config in configs] == [
        "staniorust_mmap_parallel@1",
        "staniorust_mmap_parallel@4",
        "staniorust_mmap_parallel@16",
        "pandas",
        "polars@1",
        "polars@4",
        "polars@16",
        "stanio",
    ]
    with pytest.raises(ValueError, match="unique"):
        sweep.configurations(["pandas", "pandas"], [1])
    with pytest.raises(ValueError, match="between 1 and 16"):
        sweep.configurations(["polars"], [17])


def test_case_specs_and_resource_guard() -> None:
    assert sweep.case_spec("100000x100") == (100_000, 100)
    with pytest.raises(argparse.ArgumentTypeError, match="ROWSxCOLUMNS"):
        sweep.case_spec("anything")
    allowed, skipped = sweep.select_cases(
        [(100, 10), (100, 10), (10_000_000, 100_000)],
        ram_bytes=4 * 2**30,
        disk_bytes=10 * 2**30,
        max_fixture_bytes=256 * 2**20,
        ram_fraction=0.2,
        disk_fraction=0.15,
    )
    assert len(allowed) == 1
    assert len(skipped) == 1


def example_records() -> list[dict]:
    """Several file sizes and shapes with the same total cell count."""
    records = []
    for rows, cols in [(100, 10), (200, 10), (10, 100), (20, 100)]:
        for method, threads in [
            ("staniorust_mmap_parallel", 1),
            ("staniorust_mmap_parallel", 4),
            ("polars", 1),
            ("polars", 4),
            ("pandas", None),
        ]:
            for iteration in (1, 2, 3):
                records.append(
                    {
                        "rows": rows,
                        "columns": cols,
                        "label": "synthetic",
                        "method": method,
                        "threads": threads,
                        "phase": "measure",
                        "status": "ok",
                        "input_bytes": rows * cols * 12,
                        "parse_time_s": (0.1 if method == "pandas" else 0.06)
                        * iteration
                        / (threads or 1),
                        "peak_rss_mib": 20.0,
                    }
                )
    return records


def test_summaries_keep_threads_separate_and_exclude_warmups() -> None:
    records = example_records()
    records.append(
        {
            **records[0],
            "phase": "warmup",
            "parse_time_s": 0.0001,
        }
    )
    records.append({**records[0], "status": "failed", "error": "timeout"})
    results = sweep.summarize(records)
    assert len(results) == 20
    rust_1 = next(
        row
        for row in results
        if row["rows"] == 100
        and row["columns"] == 10
        and row["method"] == "staniorust_mmap_parallel"
        and row["threads"] == 1
    )
    assert rust_1["n_ok"] == 3
    assert rust_1["n_failed"] == 1
    assert rust_1["median_s"] == pytest.approx(0.12)
    assert rust_1["median_mib_s"] == pytest.approx((100 * 10 * 12) / 2**20 / 0.12)
    assert rust_1["median_mvalues_s"] == pytest.approx(100 * 10 / 1e6 / 0.12)


def test_thread_scaling_plots_cover_all_families(tmp_path: Path) -> None:
    outputs = create_plots(sweep.summarize(example_records()), tmp_path, min_repeats=3)
    names = {path.name for path in outputs}
    assert "throughput_vs_threads_100x10.png" in names
    assert "throughput_vs_file_size_cols_10.png" in names
    assert "throughput_vs_columns_values_1000.png" in names
    assert all(path.stat().st_size > 0 for path in outputs)
    assert create_plots(sweep.summarize(example_records()), tmp_path, min_repeats=4) == []


def _fake_benchmark(_path: Path, method: str, threads: int, _timeout: int) -> dict:
    return {
        "status": "ok",
        "parse_time_s": (0.10 if method == "pandas" else 0.08) / threads,
        "time_s": 0.2,
        "peak_rss_mib": 12.0,
        "result_shape": None,
    }


def test_thread_sweep_runs_repeats_and_does_not_multiply_pandas(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sweep, "subprocess_benchmark", _fake_benchmark)
    output = tmp_path / "run"
    sweep.main(
        [
            "--case",
            "10x10",
            "--case",
            "20x10",
            "--methods",
            "pandas",
            "staniorust_mmap_parallel",
            "--threads",
            "1",
            "4",
            "--warmups",
            "1",
            "--repetitions",
            "3",
            "--output",
            str(output),
        ]
    )
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["resources"]["planned_process_runs"] == 2 * 3 * 4
    records = sweep.read_records(output / "results.jsonl")
    warmups = sweep.read_records(output / "warmups.jsonl")
    assert len(records) == 2 * 3 * 3
    assert len(warmups) == 2 * 3
    assert len([record for record in records if record["method"] == "pandas"]) == 6
    assert {record["threads"] for record in records} == {None, 1, 4}
    assert (output / "summary.csv").exists()
    assert list((output / "plots").glob("throughput_vs_threads_*.png"))
    assert list((output / "plots").glob("throughput_vs_file_size_cols_10.png"))


def test_dry_run_only_writes_preview(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    sweep.main(
        [
            "--case",
            "10x10",
            "--methods",
            "pandas",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    assert (output / "plan-preview.json").is_file()
    assert not (output / "results.jsonl").exists()


def test_publish_thread_gallery_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    readme = repo / "README.md"
    readme.write_text(
        "# demo\n\n<!-- BENCHMARKS:START -->old gallery<!-- BENCHMARKS:END -->\n\n"
        "## License\n\nApache-2.0\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "data"
    results_dir.mkdir()
    records = example_records()
    (results_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    (results_dir / "warmups.jsonl").write_text("", encoding="utf-8")
    sweep.save_summary(records, results_dir / "summary.csv")
    (results_dir / "plan.json").write_text(
        json.dumps(
            {
                "resources": {
                    "run_id": "20261104-fixture",
                    "threads_tested": [1, 4],
                    "repetitions": 3,
                    "warmups": 1,
                },
                "environment": {"os": "CI", "python": "3.12", "packages": {"pandas": "test"}},
            }
        ),
        encoding="utf-8",
    )
    folder = publish(results_dir, repo)
    assert (folder / "METHOD.md").is_file()
    assert (folder / "summary.csv").is_file()
    assert list((folder / "plots").glob("throughput_vs_threads_*.png"))
    after = readme.read_text(encoding="utf-8")
    assert after.count("<!-- THREAD-SWEEP:START -->") == 1
    assert "old gallery" in after
    assert "throughput_vs_file_size_cols_10.png" in after
    publish(results_dir, repo)
    assert readme.read_text(encoding="utf-8") == after


def test_publisher_rejects_single_run_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# test\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "plan.json").write_text(json.dumps({"resources": {"run_id": "test"}}))
    (results / "results.jsonl").write_text(json.dumps(example_records()[0]) + "\n")
    with pytest.raises(ValueError, match="sufficiently repeated"):
        publish(results, repo)
    assert not (repo / "docs").exists()
