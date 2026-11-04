"""Regression checks for randomized repeats, uncertainty and README snapshots."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from benchmarks._stats import aggregate, percentile
from benchmarks.publish import publish
from benchmarks.run import Case, balanced_method_rounds, main, verify_result_shape


def test_balanced_randomized_order_and_seed() -> None:
    methods = [f"reader_{index}" for index in range(7)]
    rounds = balanced_method_rounds(methods, 7, random.Random(41))
    assert rounds == balanced_method_rounds(methods, 7, random.Random(41))
    assert rounds != [methods] * 7
    for position in range(7):
        assert {order[position] for order in rounds} == set(methods)
    assert all(Counter(order) == Counter(methods) for order in rounds)
    assert balanced_method_rounds(methods, 2, random.Random(41), randomize=False) == [
        methods,
        methods,
    ]


def test_aggregate_repeat_stats_excludes_warmups() -> None:
    records = [
        {"rows": 10, "columns": 2, "method": "pandas", "status": "ok", "parse_time_s": t}
        for t in (1.0, 2.0, 3.0, 4.0, 10.0)
    ]
    records.extend(
        [
            {
                "rows": 10,
                "columns": 2,
                "method": "pandas",
                "status": "ok",
                "phase": "warmup",
                "parse_time_s": 1e-6,
            },
            {"rows": 10, "columns": 2, "method": "pandas", "status": "failed"},
        ]
    )
    row = aggregate(records)[0]
    assert row["n_ok"] == 5
    assert row["n_failed"] == 1
    assert row["median_s"] == 3.0
    assert row["p25_s"] == 2.0
    assert row["p75_s"] == 4.0
    assert percentile([5, 1, 3], 0.5) == 3


def synthetic_records() -> list[dict]:
    return [
        {
            "rows": rows,
            "columns": columns,
            "method": method,
            "phase": "measure",
            "status": "ok",
            "run_id": "20261104-test",
            "parse_time_s": (0.1 if method == "pandas" else 0.05) * iteration,
            "time_s": (0.15 if method == "pandas" else 0.08) * iteration,
            "peak_rss_mib": 40,
        }
        for rows, columns in ((10, 1), (20, 1), (10, 10), (20, 10))
        for method in ("pandas", "polars")
        for iteration in (1, 2, 3, 4, 5)
    ]


def test_readme_gallery_is_versioned_and_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "# My package\n\n## Benchmarks\n\nInstructions.\n\n## License\n\nApache.\n"
    )
    results = tmp_path / "results"
    results.mkdir()
    raw = synthetic_records()
    raw.append(
        {
            "rows": 10,
            "columns": 1,
            "method": "stanio",
            "status": "failed",
            "error": r"C:\Users\someone\private-file.csv: error",
        }
    )
    (results / "results.jsonl").write_text(
        "\n".join(json.dumps(record) for record in raw) + "\n", encoding="utf-8"
    )
    (results / "plan.json").write_text(
        json.dumps(
            {
                "resources": {
                    "run_id": "20261104-test",
                    "repetitions": 5,
                    "warmups": 1,
                    "seed": 17,
                    "threads": 8,
                    "randomize": True,
                },
                "environment": {
                    "os": "test OS",
                    "python": "3.12",
                    "packages": {"pandas": "test version"},
                },
            }
        ),
        encoding="utf-8",
    )
    first = publish(results, repo)
    assert first == repo / "docs" / "benchmarks" / "runs" / "20261104-test"
    assert (first / "summary.csv").exists()
    assert (first / "METHOD.md").exists()
    assert (first / "plan.json").exists()
    assert "C:\\Users" not in (first / "results.jsonl").read_text()
    assert "[redacted:" in (first / "results.jsonl").read_text()
    assert len(list((first / "plots").glob("*.png"))) == 4
    text = (repo / "README.md").read_text()
    assert text.count("<!-- BENCHMARKS:START -->") == 1
    assert "time_vs_samples_cols_1.png" in text
    assert "time_vs_columns_samples_20.png" in text
    assert "Speedup vs pandas" in text
    assert "## License" in text
    publish(results, repo)
    assert (repo / "README.md").read_text() == text


def test_rejects_unrepeated_results_before_publishing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# test\n")
    results = tmp_path / "results"
    results.mkdir()
    (results / "plan.json").write_text(json.dumps({"resources": {"run_id": "test"}}))
    (results / "results.jsonl").write_text(
        json.dumps(
            {"rows": 1, "columns": 1, "method": "pandas", "status": "ok", "parse_time_s": 0.1}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="No sufficiently repeated measurements"):
        publish(results, repo)
    assert not (repo / "docs").exists()


def test_runner_saves_repetitions_warmups_and_summary(tmp_path: Path, monkeypatch) -> None:
    import benchmarks.run as runner

    def fake_benchmark(path: Path, method: str, threads: int, timeout_s: int) -> dict:
        return {
            "status": "ok",
            "parse_time_s": 0.1 if method == "pandas" else 0.05,
            "time_s": 0.2,
            "peak_rss_mib": 10.0,
            "result_shape": None,
        }

    monkeypatch.setattr(runner, "subprocess_benchmark", fake_benchmark)
    output = tmp_path / "results"
    main(
        [
            "--rows",
            "1",
            "10",
            "--cols",
            "1",
            "--max-cases",
            "2",
            "--repetitions",
            "3",
            "--warmups",
            "1",
            "--seed",
            "41",
            "--methods",
            "pandas",
            "polars",
            "--output",
            str(output),
        ]
    )
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (output / "warmups.jsonl").read_text().splitlines()]
    assert len(results) == 2 * 2 * 3
    assert len(warmups) == 2 * 2
    assert {r["repetition"] for r in results} == {1, 2, 3}
    assert all(r["phase"] == "measure" for r in results)
    assert all(r["phase"] == "warmup" for r in warmups)
    assert (output / "summary.csv").exists()
    assert list((output / "plots_grouped").glob("*.png"))
    assert json.loads((output / "plan.json").read_text())["resources"]["seed"] == 41


def test_readers_with_wrong_cell_count_are_rejected() -> None:
    case = Case(4, 3, (3,), 1024, 96, "wide+long")
    assert (
        verify_result_shape({"status": "ok", "result_shape": [4, 2]}, case)["status"] == "invalid"
    )
    assert verify_result_shape({"status": "ok", "result_shape": [1, 4, 3]}, case)["status"] == "ok"


def test_dry_run_does_not_overwrite_measured_plan(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    plan_file = output / "plan.json"
    plan_file.write_text('{"resources": {"run_id": "preserve-me"}}', encoding="utf-8")
    main(
        [
            "--rows",
            "1",
            "--cols",
            "1",
            "--methods",
            "pandas",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    assert json.loads(plan_file.read_text())["resources"]["run_id"] == "preserve-me"
    assert (output / "plan-preview.json").exists()
