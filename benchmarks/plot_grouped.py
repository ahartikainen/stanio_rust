"""Plot CmdStan CSV benchmark methods against rows and columns.

From the repository root::

    python -m benchmarks.plot_grouped
    python -m benchmarks.plot_grouped --metric time_s --format png svg

Reads ``benchmark_results/results.jsonl`` written by ``benchmarks.run``.
Every plot compares methods on the same CSV dimensions. Repeated measurements
of a given (rows, columns, method) use their median elapsed time.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from statistics import median

from benchmarks._stats import percentile

# Consistent, color-vision-friendly colors; redundant marker and dash encodings.
METHOD_STYLES = {
    "staniorust_mmap_parallel": ("#0072B2", "-", "o", "Rust mmap (parallel)"),
    "staniorust_mmap_1": ("#E69F00", "--", "s", "Rust mmap (1 thread)"),
    "staniorust_buffered": ("#009E73", "-.", "^", "Rust buffered"),
    "pandas": ("#D55E00", ":", "D", "pandas"),
    "polars": ("#CC79A7", "-", "P", "Polars"),
    "stanio": ("#56B4E9", "--", "X", "StanIO"),
    "numpy_loadtxt": ("#555555", "-.", "v", "NumPy loadtxt"),
}
FALLBACK_COLORS = ("#332288", "#AA4499", "#117733", "#882255", "#44AA99")
FALLBACK_LINES = ("-", "--", "-.", ":")
FALLBACK_MARKERS = ("h", "<", ">", "*", "8")


@dataclass(frozen=True)
class Measurement:
    """One successful benchmark result after combining any repeated runs."""

    rows: int
    columns: int
    method: str
    seconds: float
    p25: float | None = None
    p75: float | None = None
    repetitions: int = 1


def load_results(path: Path, metric: str = "parse_time_s") -> list[Measurement]:
    """Read JSON Lines, discard failed cases, and median-combine repetitions.

    ``parse_time_s`` excludes reader import/startup. ``time_s`` measures total
    subprocess time, including imports and process startup. Both are supported.
    """
    samples: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or record.get("phase", "measure") != "measure":
                continue
            if metric not in record:
                raise ValueError(f"Missing {metric!r} on line {line_number} of {path}")
            try:
                rows = int(record["rows"])
                cols = int(record["columns"])
                method = str(record["method"])
                seconds = float(record[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid benchmark result on line {line_number}") from exc
            if rows < 1 or cols < 1 or not method or not math.isfinite(seconds):
                raise ValueError(f"Invalid dimensions or timing on line {line_number}")
            if seconds <= 0:
                # Logarithmic y-axis cannot represent zero or negative timings.
                continue
            samples[(rows, cols, method)].append(seconds)
    if not samples:
        raise ValueError(f"No successful measurements with positive {metric} in {path}")
    return [
        Measurement(
            rows,
            cols,
            method,
            median(times),
            percentile(times, 0.25),
            percentile(times, 0.75),
            len(times),
        )
        for (rows, cols, method), times in sorted(samples.items())
    ]


def _styles(methods: set[str]) -> dict[str, tuple[str, str, str, str]]:
    """Use the same visual identity for a method in every generated figure."""
    styles = {method: METHOD_STYLES[method] for method in methods if method in METHOD_STYLES}
    others = sorted(methods - METHOD_STYLES.keys())
    for i, method in enumerate(others):
        styles[method] = (
            FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            FALLBACK_LINES[(i // len(FALLBACK_COLORS)) % len(FALLBACK_LINES)],
            FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)],
            method,
        )
    return styles


def _count_tick(value: float, _position: int) -> str:
    """Compact labels for logarithmic sample/column axes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def create_plots(
    measurements: list[Measurement],
    output_dir: Path,
    *,
    metric: str = "parse_time_s",
    x_axis: str = "both",
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Produce one figure per fixed column count and/or fixed sample count.

    ``x_axis='samples'``: for each column count, x=number of samples, lines=methods.
    ``x_axis='columns'``: for each sample count, x=number of columns, lines=methods.
    ``x_axis='both'`` produces both families, keeping method colors consistent.
    """
    import matplotlib

    matplotlib.use("Agg")  # Save PNG/SVG even without an interactive display.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if x_axis not in {"both", "samples", "columns"}:
        raise ValueError(f"Unrecognized x-axis option: {x_axis}")
    if not formats or any(fmt not in {"png", "svg"} for fmt in formats):
        raise ValueError("formats must be png and/or svg")
    if not measurements:
        raise ValueError("No benchmark measurements to plot")

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = _styles({m.method for m in measurements})
    method_order = [*METHOD_STYLES, *sorted(set(styles) - METHOD_STYLES.keys())]
    written: list[Path] = []
    groups = []
    if x_axis in ("both", "samples"):
        groups.extend(("samples", c) for c in sorted({m.columns for m in measurements}))
    if x_axis in ("both", "columns"):
        groups.extend(("columns", n) for n in sorted({m.rows for m in measurements}))

    for direction, fixed_count in groups:
        if direction == "samples":
            points = [m for m in measurements if m.columns == fixed_count]
            x = attrgetter("rows")
            xlabel = "Samples per chain (rows)"
            title = f"Read time vs samples | {fixed_count:,} columns"
            stem = f"time_vs_samples_cols_{fixed_count}"
        else:
            points = [m for m in measurements if m.rows == fixed_count]
            x = attrgetter("columns")
            xlabel = "Parameter columns"
            title = f"Read time vs columns | {fixed_count:,} samples"
            stem = f"time_vs_columns_samples_{fixed_count}"

        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_methods = 0
        for method in method_order:
            series = sorted((m for m in points if m.method == method), key=x)
            if not series:
                continue
            color, linestyle, marker, label = styles[method]
            xs = [x(m) for m in series]
            ax.plot(
                xs,
                [m.seconds for m in series],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.3,
                markersize=6,
                linewidth=2.0,
                label=label,
            )
            if any(m.repetitions >= 3 for m in series):
                ax.fill_between(
                    xs,
                    [m.p25 if m.p25 is not None else m.seconds for m in series],
                    [m.p75 if m.p75 is not None else m.seconds for m in series],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            plotted_methods += 1

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_count_tick))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            "Read time (seconds, including subprocess startup)"
            if metric == "time_s"
            else "Parse + conversion time (seconds)"
        )
        ax.set_title(title + " | median (IQR shading for >=3 repeats)")
        ax.grid(visible=True, which="major", alpha=0.32)
        ax.grid(visible=True, which="minor", alpha=0.10)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        fig.tight_layout()
        for fmt in formats:
            path = output_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        print(f"{stem}: {plotted_methods} methods, {len(points)} observations")
    return written


def main(argv: list[str] | None = None) -> None:
    """CLI for regenerating grouped benchmark plots without rerunning benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmark_results/results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/plots_grouped"))
    parser.add_argument("--metric", choices=("parse_time_s", "time_s"), default="parse_time_s")
    parser.add_argument("--x-axis", choices=("both", "samples", "columns"), default="both")
    parser.add_argument("--format", choices=("png", "svg"), nargs="+", default=["png"])
    args = parser.parse_args(argv)
    measurements = load_results(args.results, metric=args.metric)
    output_paths = create_plots(
        measurements,
        args.output,
        metric=args.metric,
        x_axis=args.x_axis,
        formats=tuple(dict.fromkeys(args.format)),
    )
    print(f"Saved {len(output_paths)} plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
