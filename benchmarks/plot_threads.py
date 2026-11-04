"""Generate throughput and thread-scaling figures from benchmark summaries.

All throughput figures use *measured CSV file size* divided by reader time.
They do not include Python import or process startup in the denominator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.thread_sweep import read_records, save_summary

METHOD_COLORS = {
    "staniorust_mmap_parallel": "#0072B2",  # blue
    "polars": "#CC79A7",  # magenta
    "pandas": "#D55E00",  # vermillion
    "stanio": "#009E73",  # green
    "numpy_loadtxt": "#4A4A4A",  # charcoal
    "staniorust_buffered": "#E69F00",  # gold
    "staniorust_mmap_1": "#56B4E9",  # sky blue
}
METHOD_NAMES = {
    "staniorust_mmap_parallel": "Rust mmap",
    "polars": "Polars",
    "pandas": "pandas",
    "stanio": "StanIO",
    "numpy_loadtxt": "NumPy loadtxt",
    "staniorust_buffered": "Rust buffered",
    "staniorust_mmap_1": "Rust mmap (fixed 1)",
}
THREAD_STYLES = {1: ("-", "o"), 2: ("--", "s"), 4: ("-.", "^"), 8: (":", "D"), 16: ("-", "P")}
THREAD_DASHES = ("-", "--", "-.", ":")
THREAD_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


def series_label(record: dict) -> str:
    """Human-readable algorithm plus thread configuration."""
    name = METHOD_NAMES.get(record["method"], record["method"])
    threads = record.get("threads")
    return f"{name} ({threads}t)" if threads is not None else name


def _style(record: dict) -> dict:
    """Keep one color per algorithm and vary line/marker for thread count."""
    threads = record.get("threads")
    if threads is None:
        return {
            "color": METHOD_COLORS.get(record["method"], "#777777"),
            "linestyle": "--",
            "marker": "x",
        }
    linestyle, marker = THREAD_STYLES.get(
        threads,
        (THREAD_DASHES[threads % len(THREAD_DASHES)], THREAD_MARKERS[threads % 10]),
    )
    return {
        "color": METHOD_COLORS.get(record["method"], "#777777"),
        "linestyle": linestyle,
        "marker": marker,
    }


def _clean(summary: list[dict], minimum_repeats: int) -> list[dict]:
    """Only plot successful, finite values backed by enough repetitions."""
    if minimum_repeats < 1:
        raise ValueError("minimum_repeats must be >=1")
    return [
        record
        for record in summary
        if record["n_ok"] >= minimum_repeats
        and isinstance(record.get("median_mib_s"), (int, float))
        and math.isfinite(record["median_mib_s"])
        and record["median_mib_s"] > 0
        and isinstance(record.get("input_bytes"), (int, float))
        and record["input_bytes"] > 0
    ]


def _key(record: dict) -> tuple[str, int | None]:
    return record["method"], record.get("threads")


def _decorate(ax, xlabel: str, title: str, *, log_x: bool = True) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Read throughput (MiB/s, higher is better)")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)


def create_plots(
    summary: list[dict],
    output_dir: Path,
    *,
    min_repeats: int = 3,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Plot thread scaling, file sizes, and shape effects at equal cell counts.

    Files:
      ``throughput_vs_threads_<rows>x<cols>.png`` for each CSV case;
      ``throughput_vs_file_size_cols_<cols>.png`` for each fixed width;
      ``throughput_vs_columns_values_<cells>.png`` for each equal-cell group.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not formats or any(fmt not in ("png", "svg") for fmt in formats):
        raise ValueError("Only PNG/SVG formats are supported")
    good = _clean(summary, min_repeats)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not good:
        return []

    written: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        for extension in formats:
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    # Scaling at fixed CSV dimensions. Baseline methods are horizontal lines.
    per_case: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in good:
        per_case[(record["rows"], record["columns"])].append(record)
    for (rows, columns), group in sorted(per_case.items()):
        threaded = [record for record in group if record.get("threads") is not None]
        distinct_threads = sorted({record["threads"] for record in threaded})
        if len(distinct_threads) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.8))
        by_method: dict[str, list[dict]] = defaultdict(list)
        for record in threaded:
            by_method[record["method"]].append(record)
        for method, values in sorted(by_method.items()):
            values.sort(key=lambda record: record["threads"])
            ax.plot(
                [record["threads"] for record in values],
                [record["median_mib_s"] for record in values],
                linewidth=2.1,
                markersize=6,
                label=METHOD_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, "#888888"),
                marker="o",
            )
        for record in group:
            if record.get("threads") is None:
                ax.axhline(
                    record["median_mib_s"],
                    label=series_label(record),
                    color=METHOD_COLORS.get(record["method"], "#777777"),
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.85,
                )
        ax.set_xticks(distinct_threads)
        _decorate(
            ax,
            "Configured worker threads",
            f"Thread scaling | {rows:,} samples x {columns:,} columns | "
            f"{group[0]['input_bytes'] / 2**20:.1f} MiB",
            log_x=False,
        )
        save(fig, f"throughput_vs_threads_{rows}x{columns}")

    # File-size scaling at fixed CSV width: algorithm/thread configurations are lines.
    for columns in sorted({record["columns"] for record in good}):
        group = [record for record in good if record["columns"] == columns]
        if len({record["rows"] for record in group}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        by_series: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
        for record in group:
            by_series[_key(record)].append(record)
        for values in by_series.values():
            values.sort(key=lambda record: record["input_bytes"])
            ax.plot(
                [record["input_bytes"] / 2**20 for record in values],
                [record["median_mib_s"] for record in values],
                linewidth=1.8,
                markersize=5,
                label=series_label(values[0]),
                **_style(values[0]),
            )
        _decorate(
            ax,
            "Actual CSV file size (MiB, log scale)",
            f"Reading speed vs file size | {columns:,} columns",
        )
        save(fig, f"throughput_vs_file_size_cols_{columns}")

    # Equal total number of values, variable width: isolates per-column overhead.
    totals = sorted({record["rows"] * record["columns"] for record in good})
    for values_count in totals:
        group = [record for record in good if record["rows"] * record["columns"] == values_count]
        if len({record["columns"] for record in group}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        by_series = defaultdict(list)
        for record in group:
            by_series[_key(record)].append(record)
        for values in by_series.values():
            values.sort(key=lambda record: record["columns"])
            ax.plot(
                [record["columns"] for record in values],
                [record["median_mib_s"] for record in values],
                linewidth=1.8,
                markersize=5,
                label=series_label(values[0]),
                **_style(values[0]),
            )
        _decorate(
            ax,
            "Number of columns (log scale)",
            f"File shape at {values_count:,} numerical values",
        )
        save(fig, f"throughput_vs_columns_values_{values_count}")

    return written


def main(argv: list[str] | None = None) -> None:
    """Regenerate thread-sweep figures from existing results without parsing CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark_results/thread_sweep/results.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/thread_sweep/plots"))
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--format", nargs="+", choices=("png", "svg"), default=["png"])
    args = parser.parse_args(argv)
    records = read_records(args.results)
    summary = save_summary(records, args.results.parent / "summary.csv")
    paths = create_plots(
        summary,
        args.output,
        min_repeats=args.min_repeats,
        formats=tuple(dict.fromkeys(args.format)),
    )
    print(json.dumps({"plots": len(paths), "directory": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
