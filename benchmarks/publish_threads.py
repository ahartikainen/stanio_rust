"""Publish a versioned thread-scaling gallery to GitHub's README.

Only reader/thread/file combinations with at least three successful measured
repetitions are published. The existing benchmark gallery is untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from benchmarks.plot_threads import create_plots
from benchmarks.thread_sweep import ROOT, read_records, summarize

START = "<!-- THREAD-SWEEP:START -->"
END = "<!-- THREAD-SWEEP:END -->"


def update_readme(readme: Path, content: str) -> None:
    """Replace the managed gallery, preserving hand-written README sections."""
    original = readme.read_text(encoding="utf-8")
    if START in original and END in original:
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
        updated, count = pattern.subn(lambda _: content, original, count=1)
        if count != 1:
            raise ValueError("Failed to locate thread-sweep gallery markers")
    else:
        found = re.search(r"(?m)^## License\s*$", original)
        if found:
            updated = original[: found.start()] + content + "\n\n" + original[found.start() :]
        else:
            updated = original.rstrip() + "\n\n" + content + "\n"
    readme.write_text(updated, encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def publish(
    results_dir: Path,
    repo: Path = ROOT,
    *,
    min_repeats: int = 3,
) -> Path:
    """Snapshot validated thread-scaling figures and link them from README."""
    if min_repeats < 3:
        raise ValueError("Publishing requires >=3 successful repeats")
    records = read_records(results_dir / "results.jsonl")
    plan = json.loads((results_dir / "plan.json").read_text(encoding="utf-8"))
    raw_summary = summarize(records)
    good = [
        row for row in raw_summary if row["n_ok"] >= min_repeats and row["median_s"] is not None
    ]
    if not good:
        raise ValueError("No sufficiently repeated measurements for publication")
    run_id = plan.get("resources", {}).get("run_id", "unknown-run")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Unsafe benchmark run identifier")
    destination = repo / "docs" / "benchmarks" / "thread_sweeps" / run_id
    plots_dir = destination / "plots"
    if destination.exists():
        shutil.rmtree(destination)
    plots_dir.mkdir(parents=True)
    # Plot only validated groups so failed or single-run configurations cannot
    # accidentally appear in the public README gallery.
    selected_summary = [row for row in good]
    figures = create_plots(selected_summary, plots_dir, min_repeats=min_repeats)
    if not figures:
        raise ValueError("Not enough successful combinations to create figures")

    shutil.copy2(results_dir / "plan.json", destination / "plan.json")
    shutil.copy2(results_dir / "summary.csv", destination / "summary.csv")
    # Preserve timing provenance without publishing traceback paths/usernames.
    for name in ("results.jsonl", "warmups.jsonl"):
        source = results_dir / name
        if not source.exists():
            continue
        with (destination / name).open("w", encoding="utf-8") as output:
            for record in read_records(source):
                if "error" in record:
                    record["error"] = "[redacted: see local benchmark_results]"
                output.write(json.dumps(record) + "\n")

    resources = plan.get("resources", {})
    environment = plan.get("environment", {})
    (destination / "METHOD.md").write_text(
        "\n".join(
            [
                "# Thread-scaling benchmark methodology",
                "",
                f"- Run: `{run_id}`",
                f"- OS: {environment.get('os', 'unknown')}",
                f"- CPU: {environment.get('processor', 'unknown')}",
                f"- Python: {environment.get('python', 'unknown')}",
                f"- Git commit: {environment.get('git_commit', 'unknown')}",
                f"- Configured thread counts: {resources.get('threads_tested', [])}",
                f"- Measured rounds: {resources.get('repetitions', 'unknown')}",
                f"- Warmups: {resources.get('warmups', 'unknown')}",
                f"- Seed: {resources.get('seed', 'unknown')}",
                f"- Minimum published successful repeats: {min_repeats}",
                "",
                "Throughput = actual CSV file size (MiB) / measured reader time (s).",
                "Import and subprocess startup are excluded from throughput timing.",
                "Every reader/thread configuration runs in an independent subprocess.",
                "Rust mmap and Polars use the configured worker count; baseline",
                "readers use their normal thread behavior and are run once per shape.",
                "Filesystem caches are not cleared; this is mostly warm-cache testing.",
                "The CSV fixtures repeat deterministic numeric patterns. These",
                "results are not substitutes for tests on actual CmdStan outputs.",
                "The median reflects repeated attempts, not a guarantee on future files.",
                "Failed or timed-out runs are not included in throughput statistics.",
                "",
                "## Installed versions",
                "",
                "```json",
                json.dumps(environment.get("packages", {}), indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    thread_hero = next((p for p in figures if p.name.startswith("throughput_vs_threads_")), None)
    size_hero = next((p for p in figures if p.name.startswith("throughput_vs_file_size_")), None)
    shape_hero = next((p for p in figures if p.name.startswith("throughput_vs_columns_")), None)

    def image_link(path: Path) -> str:
        link = _relative(path, repo)
        return f'<a href="{link}"><img src="{link}" width="410" alt="{path.stem}"></a>'

    section = [
        START,
        "### Thread scaling and file-shape throughput",
        "",
        f"**Run:** `{run_id}` · {resources.get('repetitions', '?')} measured repeats · "
        f"{resources.get('warmups', '?')} warmup(s) · "
        f"threads {resources.get('threads_tested', [])}.",
        "",
        "The plots compare **measured CSV throughput in MiB/s** (larger is "
        "better). Each method uses consistent colors; thread counts use different "
        "markers and linestyles. Methods that do not expose a worker-count "
        "setting are measured once per file, not duplicated at every count.",
        "",
        f"[Methodology and environment]({_relative(destination / 'METHOD.md', repo)}) · "
        f"[Summary CSV]({_relative(destination / 'summary.csv', repo)})",
        "",
    ]
    if thread_hero and size_hero:
        section.extend(
            [
                "| Throughput vs threads | Throughput vs file size |",
                "|:---:|:---:|",
                f"| {image_link(thread_hero)} | {image_link(size_hero)} |",
                "",
            ]
        )
    if shape_hero:
        section.extend(
            ["**Equal-value-count comparison across file shapes**", "", image_link(shape_hero), ""]
        )
    section.extend(
        [
            "<details>",
            f"<summary>Show all {len(figures)} throughput figures</summary>",
            "",
        ]
    )
    for path in figures:
        section.extend([f"**{path.stem.replace('_', ' ')}**", "", image_link(path), ""])
    section.extend(["</details>", END])
    update_readme(repo / "README.md", "\n".join(section))
    print(f"Published {len(figures)} figures in {destination}")
    return destination


def main(argv: list[str] | None = None) -> None:
    """Publish an existing finished thread-sweep run without rebenchmarking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark_results/thread_sweep"),
    )
    parser.add_argument("--min-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    publish(args.results.resolve(), ROOT, min_repeats=args.min_repeats)


if __name__ == "__main__":
    main()
