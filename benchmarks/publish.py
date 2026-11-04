"""Publish repeat-validated benchmark plots and a versioned README gallery.

From the repository root::

    python -m benchmarks.publish
    python -m benchmarks.publish --min-repeats 5 --baseline pandas

Only complete groups with enough successful measurements are published.
Existing exploratory single-run results remain in ``benchmark_results/``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from benchmarks._stats import aggregate, write_summary
from benchmarks.plot_grouped import create_plots, load_results

START = "<!-- BENCHMARKS:START -->"
END = "<!-- BENCHMARKS:END -->"


def read_records(path: Path) -> list[dict]:
    """Load an appendable JSON Lines result file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_validated(records: list[dict], min_repeats: int) -> tuple[list[dict], list[dict]]:
    """Keep only case/method combinations backed by enough successful runs."""
    if min_repeats < 2:
        raise ValueError("Publication requires at least two repetitions per method/case")
    summary = aggregate(records)
    accepted = {
        (r["rows"], r["columns"], r["method"])
        for r in summary
        if r["n_ok"] >= min_repeats and r["median_s"] is not None
    }
    filtered = [
        r
        for r in records
        if r.get("phase", "measure") == "measure"
        and r.get("status") == "ok"
        and (r["rows"], r["columns"], r["method"]) in accepted
    ]
    return filtered, [r for r in summary if (r["rows"], r["columns"], r["method"]) in accepted]


def relative_link(path: Path, repo: Path) -> str:
    """Return a Markdown-compatible repository-relative path."""
    return path.resolve().relative_to(repo.resolve()).as_posix()


def comparison_rows(summary: list[dict], baseline: str, max_rows: int = 10) -> list[str]:
    """Choose a spread of genuinely measurable cases for README comparisons."""
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in summary:
        grouped.setdefault((row["rows"], row["columns"]), []).append(row)
    comparisons: list[tuple[tuple[int, int], dict, dict]] = []
    for key, entries in grouped.items():
        baseline_row = next((r for r in entries if r["method"] == baseline), None)
        alternatives = [r for r in entries if r["method"] != baseline]
        # Very short timings are disproportionately sensitive to timer noise.
        if not baseline_row or not alternatives or baseline_row["median_s"] < 0.01:
            continue
        fastest = min(entries, key=lambda row: row["median_s"])
        comparisons.append((key, baseline_row, fastest))
    comparisons.sort(key=lambda row: (row[0][0] * row[0][1], *row[0]))
    if len(comparisons) > max_rows:
        indices = {
            round(index * (len(comparisons) - 1) / (max_rows - 1)) for index in range(max_rows)
        }
        comparisons = [comparisons[index] for index in sorted(indices)]
    return [
        f"| {rows:,} | {cols:,} | `{fastest['method']}` | "
        f"{fastest['median_s']:.4g} | {base['median_s'] / fastest['median_s']:.2f}× |"
        for (rows, cols), base, fastest in comparisons
    ]


def gallery_markdown(
    snapshot: Path,
    plots: list[Path],
    repo: Path,
    plan: dict,
    summary: list[dict],
    baseline: str,
    min_repeats: int,
) -> str:
    """Create an independently meaningful, clickable benchmark README section."""
    fixed_cols = sorted(
        (p for p in plots if p.stem.startswith("time_vs_samples_cols_")),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    fixed_rows = sorted(
        (p for p in plots if p.stem.startswith("time_vs_columns_samples_")),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )

    def thumb(path: Path) -> str:
        link = relative_link(path, repo)
        return f'<a href="{link}"><img src="{link}" width="420" alt="{path.stem}"></a>'

    cols_hero = next((p for p in fixed_cols if p.stem.endswith("_100")), None)
    cols_hero = cols_hero or (fixed_cols[len(fixed_cols) // 2] if fixed_cols else None)
    rows_hero = next((p for p in fixed_rows if p.stem.endswith("_1000")), None)
    rows_hero = rows_hero or (fixed_rows[len(fixed_rows) // 2] if fixed_rows else None)
    resources = plan.get("resources", {})
    environment = plan.get("environment", {})
    run_id = resources.get("run_id", snapshot.name)
    metadata_link = relative_link(snapshot / "METHOD.md", repo)
    summary_link = relative_link(snapshot / "summary.csv", repo)
    intro = [
        START,
        "### Measured performance comparisons",
        "",
        f"**Run:** `{run_id}` · **Minimum successful repeats:** {min_repeats} "
        f"· **Scheduled repeats:** {resources.get('repetitions', 'unknown')} "
        f"· **Warmups:** {resources.get('warmups', 'unknown')} "
        f"· **Random seed:** {resources.get('seed', 'unknown')}",
        "",
        f"**Environment:** {environment.get('os', 'not recorded')}, "
        f"Python {environment.get('python', 'unknown')}, "
        f"{resources.get('threads', 'unknown')} configured Rust threads.",
        "",
        "Times are the median of independent subprocess runs; shaded bands show "
        "the interquartile range (25th–75th percentile). Only measured "
        "configurations with enough successful runs are shown. These are "
        "warm-cache-oriented measurements, **not** cold-disk benchmarks.",
        "",
        f"[Methodology and environment]({metadata_link}) · "
        f"[All summary results (CSV)]({summary_link})",
        "",
    ]
    if cols_hero and rows_hero:
        intro.extend(
            [
                "| Fixed columns — time vs samples | Fixed samples — time vs columns |",
                "|:---:|:---:|",
                f"| {thumb(cols_hero)} | {thumb(rows_hero)} |",
                "",
            ]
        )
    intro.extend(
        [
            f"#### Representative results relative to {baseline}",
            "",
            f"| Samples | Columns | Fastest method | Median seconds | Speedup vs {baseline} |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    comparisons = comparison_rows(summary, baseline)
    intro.extend(comparisons or ["| — | — | No comparable cases ≥10 ms | — | — |"])
    intro.extend(
        [
            "",
            f"Speedup is {baseline} median time divided by the fastest method's median time "
            f"for the **same file shape**. Cases with {baseline} timing below 10 ms "
            "are omitted from this table to avoid misleading microbenchmarks.",
            "",
            "<details>",
            f"<summary>View all {len(plots)} benchmark comparison plots</summary>",
            "",
            "**Time vs sample count (fixed number of columns)**",
            "",
            "| Columns | Plot |",
            "|---:|:---|",
        ]
    )
    intro.extend(f"| {int(path.stem.rsplit('_', 1)[1]):,} | {thumb(path)} |" for path in fixed_cols)
    intro.extend(
        [
            "",
            "**Time vs column count (fixed number of samples)**",
            "",
            "| Samples | Plot |",
            "|---:|:---|",
        ]
    )
    intro.extend(f"| {int(path.stem.rsplit('_', 1)[1]):,} | {thumb(path)} |" for path in fixed_rows)
    intro.extend(["", "</details>", END, ""])
    return "\n".join(intro)


def update_readme(readme: Path, generated: str) -> None:
    """Replace an owned marked region without touching the rest of README."""
    content = readme.read_text(encoding="utf-8")
    if START in content and END in content:
        before, remainder = content.split(START, maxsplit=1)
        _previous, after = remainder.split(END, maxsplit=1)
        updated = before + generated.rstrip("\n") + "\n\n" + after.lstrip("\n")
    elif "## Benchmarks" in content:
        before, rest = content.split("## Benchmarks", maxsplit=1)
        offset = rest.find("\n## ")
        if offset == -1:
            updated = content.rstrip() + "\n\n" + generated
        else:
            position = len(before) + len("## Benchmarks") + offset
            updated = (
                content[:position].rstrip()
                + "\n\n"
                + generated.rstrip("\n")
                + "\n\n"
                + content[position:].lstrip("\n")
            )
    else:
        updated = content.rstrip() + "\n\n## Benchmarks\n\n" + generated
    readme.write_text(updated.rstrip() + "\n", encoding="utf-8")


def publish(
    results: Path,
    repo: Path,
    *,
    docs: Path | None = None,
    min_repeats: int = 3,
    baseline: str = "pandas",
    run_id: str | None = None,
) -> Path:
    """Publish validated data, grouped plots, environment and README gallery."""
    records_file = results / "results.jsonl"
    plan_file = results / "plan.json"
    if not records_file.is_file() or not plan_file.is_file():
        raise FileNotFoundError("Expected results.jsonl and plan.json in benchmark results folder")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    raw = read_records(records_file)
    selected, summary = select_validated(raw, min_repeats)
    if not selected:
        raise ValueError(
            "No sufficiently repeated measurements. Run with "
            f"--repetitions {max(7, min_repeats)} before publishing."
        )
    raw_ids = {r["run_id"] for r in selected if "run_id" in r}
    if len(raw_ids) > 1:
        raise ValueError("Results contain multiple run IDs; do not mix separate runs")
    source_id = plan.get("resources", {}).get("run_id")
    if source_id and any(r.get("run_id") != source_id for r in selected):
        raise ValueError("plan.json and results.jsonl do not describe the same run")
    identifier = run_id or source_id or "legacy-results"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", identifier):
        raise ValueError("Run ID must contain only letters, digits, dots, hyphens or underscores")
    docs = docs or (repo / "docs" / "benchmarks")
    snapshot = docs / "runs" / identifier
    plot_dir = snapshot / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Only plot the validated groups; don't quietly publish exploratory single runs.
    temporary_results = snapshot / "validated_results.jsonl"
    temporary_results.write_text(
        "\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8"
    )
    pictures = create_plots(load_results(temporary_results), plot_dir, formats=("png",))
    temporary_results.unlink()
    # Failed worker stderr may contain local usernames and absolute file paths.
    # Preserve all timing/status fields, but do not publish raw exception text.
    for source in ("results.jsonl", "warmups.jsonl"):
        original = results / source
        if original.is_file():
            with (snapshot / source).open("w", encoding="utf-8") as public_output:
                for record in read_records(original):
                    if "error" in record:
                        record["error"] = "[redacted: see local benchmark_results]"
                    public_output.write(json.dumps(record) + "\n")
    shutil.copy2(plan_file, snapshot / "plan.json")
    write_summary(raw, snapshot / "summary.csv")
    environment = plan.get("environment", {})
    resources = plan.get("resources", {})
    methodology = [
        f"# Benchmark run {identifier}",
        "",
        "## Protocol",
        "",
        f"- Measured repetitions per method/case (planned): {resources.get('repetitions', 'unknown')}",
        f"- Warmups per method/case: {resources.get('warmups', 'unknown')}",
        f"- Scheduling seed: {resources.get('seed', 'unknown')}",
        f"- Balanced randomized ordering: {resources.get('randomize', 'unknown')}",
        f"- Rust reader threads: {resources.get('threads', 'unknown')}",
        "- Each attempt runs in a new process; reader import is outside `parse_time_s`.",
        "- One CSV file per shape is shared among readers, with warmup reads before timing.",
        "- No OS cache clearing: these results represent mostly warm-cache behavior.",
        "- Plot lines show medians; shaded regions show the 25th–75th percentile.",
        f"- Only groups with at least {min_repeats} successful measured attempts are plotted.",
        "- Timeout/failed attempts are preserved in raw results, not converted to zero.",
        "- Microsecond-scale cases are dominated by fixed overhead and should not "
        "be used for general speedup claims.",
        "",
        "## Environment",
        "",
        f"- OS: {environment.get('os', 'unknown')}",
        f"- Python: {environment.get('python', 'unknown')}",
        f"- Processor: {environment.get('processor', 'unknown')}",
        f"- Logical / physical cores: {environment.get('logical_cores', '?')} / "
        f"{environment.get('physical_cores', '?')}",
        f"- Git commit: {environment.get('git_commit', 'unknown')}",
        "",
        "## Installed package versions",
        "",
    ]
    methodology.extend(
        f"- {name}: {version or 'not installed'}"
        for name, version in sorted(environment.get("packages", {}).items())
    )
    methodology.extend(
        [
            "",
            "## Reproducibility files",
            "",
            "- `results.jsonl`: all measured attempts, statuses and run sequence; "
            "raw stderr details are redacted from this public snapshot",
            "- `warmups.jsonl`: warmup attempts (when available)",
            "- `plan.json`: resource planner, skipped cases and system details",
            "- `summary.csv`: per-method, per-shape aggregate measurements",
            "",
            "**Synthetic fixtures are deterministic and deliberately repetitive; "
            "results may differ on real CmdStan outputs, cold storage or other hardware.**",
            "",
        ]
    )
    (snapshot / "METHOD.md").write_text("\n".join(methodology), encoding="utf-8")
    readme = repo / "README.md"
    text = gallery_markdown(snapshot, pictures, repo, plan, summary, baseline, min_repeats)
    update_readme(readme, text)
    print(f"Published {len(pictures)} grouped plots and benchmark snapshot to {snapshot}")
    return snapshot


def main(argv: list[str] | None = None) -> None:
    """CLI that publishes plots to versioned docs and refreshes README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--baseline", default="pandas")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    publish(
        args.results.resolve(),
        args.repo.resolve(),
        min_repeats=args.min_repeats,
        baseline=args.baseline,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
