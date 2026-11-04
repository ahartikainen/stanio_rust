"""Thread scaling and shape-sensitive throughput benchmarks for Stan CSV.

Run from the repository root::

    python -m benchmarks.thread_sweep --dry-run
    python -m benchmarks.thread_sweep --profile quick --threads 1 2 4 8 16
    python -m benchmarks.thread_sweep --profile full --all-threads
    python -m benchmarks.thread_sweep --plot-only

Only Rust parallel mmap and Polars vary their thread count. Baseline readers
run once per file shape. Every configuration runs in a fresh subprocess.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import tempfile
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmarks._stats import percentile
from benchmarks.run import (
    METHODS,
    ROOT,
    Case,
    balanced_method_rounds,
    budgeted_free_ram,
    environment_details,
    plan,
    subprocess_benchmark,
    verify_result_shape,
    write_fixture,
)

# Equal numeric payloads at different widths help separate effects of
# numerical work from per-column overhead. All defaults are resource-gated.
QUICK_SHAPES = (
    (100_000, 10),
    (1_000_000, 10),
    (1_000, 1_000),
    (10_000, 1_000),
    (10, 100_000),
    (100, 100_000),
)
FULL_SHAPES = tuple(
    (values // columns, columns)
    for values in (1_000_000, 10_000_000, 100_000_000)
    for columns in (1, 10, 100, 1_000, 10_000, 100_000)
    if values % columns == 0 and 1 <= values // columns <= 10_000_000
)
DEFAULT_THREADS = (1, 2, 4, 8, 16)
SCALABLE_METHODS = frozenset({"staniorust_mmap_parallel", "polars"})
DEFAULT_METHODS = (
    "staniorust_mmap_parallel",
    "polars",
    "staniorust_buffered",
    "pandas",
    "stanio",
    "numpy_loadtxt",
)
SUMMARY_FIELDS = (
    "rows",
    "columns",
    "label",
    "method",
    "threads",
    "series",
    "input_bytes",
    "n_ok",
    "n_failed",
    "median_s",
    "p25_s",
    "p75_s",
    "median_mib_s",
    "p25_mib_s",
    "p75_mib_s",
    "median_mvalues_s",
    "median_peak_rss_mib",
)


@dataclass(frozen=True, slots=True)
class Configuration:
    """One reader with an optional explicitly configured worker count."""

    method: str
    threads: int | None

    @property
    def key(self) -> str:
        """Unique configuration label used for randomization and output."""
        return f"{self.method}@{self.threads}" if self.threads is not None else self.method


def configurations(
    methods: list[str] | tuple[str, ...], thread_counts: list[int] | tuple[int, ...]
) -> list[Configuration]:
    """Only multiply configurations for readers with controllable threading."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("At least one unique benchmark method is required")
    if any(method not in METHODS for method in methods):
        raise ValueError("Unknown benchmark method")
    if not thread_counts or any(not 1 <= value <= 16 for value in thread_counts):
        raise ValueError("Thread counts must be integers between 1 and 16")
    if len(set(thread_counts)) != len(thread_counts):
        raise ValueError("Thread counts must be unique")
    result = []
    for method in methods:
        if method in SCALABLE_METHODS:
            result.extend(Configuration(method, n) for n in sorted(thread_counts))
        else:
            result.append(Configuration(method, 1 if method == "staniorust_mmap_1" else None))
    return result


def case_spec(value: str) -> tuple[int, int]:
    """Parse a rectangular case, e.g. ``100000x100``."""
    try:
        fields = value.lower().split("x")
        if len(fields) != 2:
            raise ValueError
        rows, cols = map(int, fields)
        if not 1 <= rows <= 10_000_000 or not 1 <= cols <= 100_000:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use ROWSxCOLUMNS (up to 10M x 100k)") from exc
    return rows, cols


def select_cases(
    shapes: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    ram_bytes: int,
    disk_bytes: int,
    ram_fraction: float,
    disk_fraction: float,
    max_fixture_bytes: int,
) -> tuple[list[Case], list[dict]]:
    """Apply the existing safety planner to each chosen shape, not a cross product."""
    allowed: list[Case] = []
    skipped: list[dict] = []
    for rows, cols in dict.fromkeys(shapes):
        feasible, excluded = plan(
            [rows],
            [cols],
            ram_bytes=ram_bytes,
            disk_bytes=disk_bytes,
            max_fixture_bytes=max_fixture_bytes,
            ram_fraction=ram_fraction,
            disk_fraction=disk_fraction,
        )
        allowed.extend(feasible)
        skipped.extend(excluded)
    return allowed, skipped


def read_records(path: Path) -> list[dict]:
    """Read JSONL records while ignoring empty lines."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def summarize(records: list[dict]) -> list[dict]:
    """Group measured runs by shape, reader and thread count.

    Warmups are excluded. Failed attempts are counted, not fabricated as zero.
    Throughput is based on actual input file bytes / measured parse seconds.
    """
    groups: dict[tuple[int, int, str, int | None], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("phase", "measure") != "measure":
            continue
        if not {"rows", "columns", "method"}.issubset(record):
            continue
        key = (
            int(record["rows"]),
            int(record["columns"]),
            str(record["method"]),
            record.get("threads"),
        )
        groups[key].append(record)

    summary: list[dict] = []
    for (rows, columns, method, threads), attempts in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or 0)
    ):
        valid = [
            record
            for record in attempts
            if record.get("status") == "ok"
            and isinstance(record.get("parse_time_s"), (int, float))
            and math.isfinite(record["parse_time_s"])
            and record["parse_time_s"] > 0
            and isinstance(record.get("input_bytes"), (int, float))
            and record["input_bytes"] > 0
        ]
        seconds = [float(record["parse_time_s"]) for record in valid]
        throughput = [record["input_bytes"] / 2**20 / record["parse_time_s"] for record in valid]
        values_s = [rows * columns / 1_000_000 / t for t in seconds]
        rss = [
            float(record["peak_rss_mib"])
            for record in valid
            if isinstance(record.get("peak_rss_mib"), (int, float))
            and math.isfinite(record["peak_rss_mib"])
        ]
        sample = attempts[0]
        summary.append(
            {
                "rows": rows,
                "columns": columns,
                "label": sample.get("label", ""),
                "method": method,
                "threads": threads,
                "series": Configuration(method, threads).key,
                "input_bytes": sample.get("input_bytes"),
                "n_ok": len(valid),
                "n_failed": len(attempts) - len(valid),
                "median_s": statistics.median(seconds) if seconds else None,
                "p25_s": percentile(seconds, 0.25) if seconds else None,
                "p75_s": percentile(seconds, 0.75) if seconds else None,
                "median_mib_s": statistics.median(throughput) if throughput else None,
                "p25_mib_s": percentile(throughput, 0.25) if throughput else None,
                "p75_mib_s": percentile(throughput, 0.75) if throughput else None,
                "median_mvalues_s": statistics.median(values_s) if values_s else None,
                "median_peak_rss_mib": statistics.median(rss) if rss else None,
            }
        )
    return summary


def save_summary(records: list[dict], destination: Path) -> list[dict]:
    """Write a CSV summary retaining the method/thread dimension."""
    summary = summarize(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def regenerate(records: list[dict], destination: Path) -> list[Path]:
    """Refresh summary and figures from raw data without rerunning readers."""
    from benchmarks.plot_threads import create_plots

    summary = save_summary(records, destination / "summary.csv")
    return create_plots(summary, destination / "plots")


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=case_spec,
        help="Specific CSV shape ROWSxCOLS; repeat for multiple files",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=list(DEFAULT_THREADS),
        help="Thread counts for both parallel Rust mmap and Polars (1..16)",
    )
    parser.add_argument(
        "--all-threads",
        action="store_true",
        help="Test every thread count from 1 to 16",
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--ram-fraction", type=float, default=0.20)
    parser.add_argument("--disk-fraction", type=float, default=0.15)
    parser.add_argument("--max-file-mib", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/thread_sweep"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--keep-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Perform randomized rounds across reader/thread configurations."""
    args = _args(argv)
    if args.repetitions < 1 or args.warmups < 0 or args.timeout < 1:
        raise ValueError("Repetitions and timeout must be positive; warmups >= 0")
    if not 0 < args.ram_fraction <= 1 or not 0 < args.disk_fraction <= 1:
        raise ValueError("Resource fractions must be in (0, 1]")
    if args.max_file_mib is not None and args.max_file_mib <= 0:
        raise ValueError("Maximum file size must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("Maximum cases must be positive")
    if args.dry_run and args.publish:
        raise ValueError("Cannot publish a dry-run")
    if args.plot_only and (args.dry_run or args.publish):
        raise ValueError("Plot-only mode cannot run or publish")

    counts = list(range(1, 17)) if args.all_threads else args.threads
    configs = configurations(args.methods, counts)
    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / "results.jsonl"
    if args.plot_only:
        paths = regenerate(read_records(results_file), args.output)
        print(f"Generated {len(paths)} figures in {args.output / 'plots'}")
        return

    shapes = args.cases or (QUICK_SHAPES if args.profile == "quick" else FULL_SHAPES)
    free_ram = budgeted_free_ram()
    free_disk = shutil.disk_usage(args.output).free
    max_file_mib = args.max_file_mib or (256 if args.profile == "quick" else 4096)
    allowed, skipped = select_cases(
        shapes,
        ram_bytes=free_ram,
        disk_bytes=free_disk,
        ram_fraction=args.ram_fraction,
        disk_fraction=args.disk_fraction,
        max_fixture_bytes=max_file_mib * 2**20,
    )
    allowed = allowed[: args.max_cases]
    rng = random.Random(args.seed)
    rng.shuffle(allowed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    planned_runs = len(allowed) * len(configs) * (args.repetitions + args.warmups)
    report = {
        "run_id": run_id,
        "profile": args.profile,
        "free_ram_gib": round(free_ram / 2**30, 2),
        "free_disk_gib": round(free_disk / 2**30, 2),
        "ram_fraction": args.ram_fraction,
        "disk_fraction": args.disk_fraction,
        "max_file_mib": max_file_mib,
        "cases": len(allowed),
        "skipped_cases": len(skipped),
        "configurations": len(configs),
        "planned_process_runs": planned_runs,
        "threads_tested": counts,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "seed": args.seed,
        "cache_protocol": "warm-oriented (no OS page-cache flush)",
    }
    print(json.dumps(report, indent=2))
    plan_file = args.output / ("plan-preview.json" if args.dry_run else "plan.json")
    plan_file.write_text(
        json.dumps(
            {
                "resources": report,
                "environment": environment_details(),
                "configurations": [asdict(config) for config in configs],
                "cases": [asdict(case) for case in allowed],
                "skips": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.dry_run:
        for case in allowed:
            print(
                f"{case.rows:>10,} draws x {case.columns:>7,} cols "
                f"| estimated {case.estimated_csv_bytes / 2**20:>8.1f} MiB "
                f"| {case.label}"
            )
        return

    results_file.write_text("", encoding="utf-8")
    warmups_file = args.output / "warmups.jsonl"
    warmups_file.write_text("", encoding="utf-8")
    config_lookup = {config.key: config for config in configs}
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="staniorust-thread-bench-") as temp:
        for case_index, case in enumerate(allowed, 1):
            path = Path(temp) / "chain.csv"
            size = write_fixture(path, case)
            base = {**asdict(case), "input_bytes": size, "run_id": run_id}
            failed: set[str] = set()
            print(
                f"File {case_index}/{len(allowed)}: {case.rows:,} draws x "
                f"{case.columns:,} columns ({size / 2**20:.1f} MiB)",
                flush=True,
            )
            with (
                results_file.open("a", encoding="utf-8") as results_output,
                warmups_file.open("a", encoding="utf-8") as warmups_output,
            ):
                for warmup in range(1, args.warmups + 1):
                    order = list(config_lookup)
                    rng.shuffle(order)
                    for key in order:
                        if key in failed:
                            continue
                        config = config_lookup[key]
                        sequence += 1
                        outcome = verify_result_shape(
                            subprocess_benchmark(
                                path, config.method, config.threads or 1, args.timeout
                            ),
                            case,
                        )
                        record = {
                            **base,
                            "method": config.method,
                            "threads": config.threads,
                            "series": key,
                            "phase": "warmup",
                            "warmup": warmup,
                            "sequence": sequence,
                            **outcome,
                        }
                        warmups_output.write(json.dumps(record) + "\n")
                        warmups_output.flush()
                        if outcome["status"] != "ok":
                            failed.add(key)
                            measured = {
                                **record,
                                "phase": "measure",
                                "repetition": 0,
                                "error": (
                                    f"warmup failed: {outcome.get('error', outcome['status'])}"
                                ),
                            }
                            results_output.write(json.dumps(measured) + "\n")
                            results_output.flush()
                            print(f"  {key}: warmup {outcome['status']}", flush=True)

                rounds = balanced_method_rounds(list(config_lookup), args.repetitions, rng)
                for repetition, order in enumerate(rounds, 1):
                    for key in order:
                        if key in failed:
                            continue
                        config = config_lookup[key]
                        sequence += 1
                        outcome = verify_result_shape(
                            subprocess_benchmark(
                                path, config.method, config.threads or 1, args.timeout
                            ),
                            case,
                        )
                        record = {
                            **base,
                            "method": config.method,
                            "threads": config.threads,
                            "series": key,
                            "phase": "measure",
                            "repetition": repetition,
                            "sequence": sequence,
                            **outcome,
                        }
                        results_output.write(json.dumps(record) + "\n")
                        results_output.flush()
                        print(
                            f"  {key:>28} {repetition}/{args.repetitions}: "
                            f"{record['status']:<7} "
                            f"{record.get('parse_time_s', float('nan')):.4f}s",
                            flush=True,
                        )
                        if outcome["status"] != "ok":
                            failed.add(key)
            if args.keep_files:
                shutil.copy2(path, args.output / f"case_{case.rows}_{case.columns}.csv")
            path.unlink(missing_ok=True)

    records = read_records(results_file)
    paths = regenerate(records, args.output)
    failures = [row for row in records if row.get("status") != "ok"]
    print(
        f"Completed {run_id}: {len(paths)} plots, {len(failures)} failed configurations "
        f"or measured attempts. See {results_file.resolve()}."
    )
    if args.publish:
        from benchmarks.publish_threads import publish

        publish(args.output.resolve(), ROOT)


if __name__ == "__main__":
    main()
