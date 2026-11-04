"""RAM- and disk-budgeted benchmarks for CmdStan-like numeric CSV readers.

Run from the project directory::

    python -m benchmarks.run --dry-run
    python -m benchmarks.run --profile quick
    python -m benchmarks.run --profile full --ram-fraction 0.4 --max-file-mib 8192
    python -m benchmarks.run --methods pandas polars --max-cases 8

The same files are used for every reader. Each method runs in a fresh subprocess
so process RSS and initialization overhead can be measured independently.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

ROWS = (1, 10, 100, 1_000, 100_000, 1_000_000, 10_000_000)
COLS = (1, 10, 100, 1_000, 10_000, 100_000)
METHODS = (
    "staniorust_mmap_1",
    "staniorust_mmap_parallel",
    "staniorust_buffered",
    "pandas",
    "polars",
    "stanio",
    "numpy_loadtxt",
)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Case:
    """A reproducible workload with a Stan parameter container shape."""

    rows: int
    columns: int
    shape: tuple[int, ...]
    estimated_csv_bytes: int
    output_bytes: int
    label: str


def shape_for(columns: int) -> tuple[int, ...]:
    """Use a 1-D, 2-D, 3-D or 4-D Stan variable with <=100k columns."""
    if columns == 1:
        return ()
    presets = {
        10: (10,),
        100: (10, 10),
        1_000: (10, 10, 10),
        10_000: (10, 10, 10, 10),
        100_000: (10, 10, 10, 100),
    }
    return presets.get(columns, (columns,))


def container_labels(shape: tuple[int, ...]) -> list[str]:
    """Emit Stan 1-based n-D names in column-major (Fortran) order."""
    if not shape:
        return ["theta"]
    width = math.prod(shape)
    labels = []
    for offset in range(width):
        pos = offset
        coords = []
        for dim in shape:
            pos, index = divmod(pos, dim)
            coords.append(str(index + 1))
        labels.append("theta." + ".".join(coords))
    return labels


def budgeted_free_ram() -> int:
    """Respect process-available RAM and Linux container cgroup limits."""
    available = int(psutil.virtual_memory().available)
    for prefix in (Path("/sys/fs/cgroup"),):
        maximum = prefix / "memory.max"
        current = prefix / "memory.current"
        try:
            capacity = maximum.read_text().strip()
            if capacity != "max":
                remaining = int(capacity) - int(current.read_text())
                available = min(available, max(0, remaining))
        except (OSError, ValueError):
            pass
    return available


def plan(
    rows: list[int], columns: list[int], *, ram_bytes: int, disk_bytes: int,
    max_fixture_bytes: int, ram_fraction: float, disk_fraction: float,
) -> tuple[list[Case], list[dict]]:
    """Choose feasible combinations before allocating or creating files.

    Parameters
    ----------
    rows, columns : list[int]
        Target sample counts and flattened Stan parameter sizes.
    ram_bytes, disk_bytes : int
        Currently free memory and destination filesystem capacity.
    max_fixture_bytes : int
        Per-fixture cap, independent of larger resources.
    ram_fraction, disk_fraction : float
        Conservative fractions of *free* resources available for this run.

    Returns
    -------
    tuple[list[Case], list[dict]]
        Feasible workloads and planned skips with their reasons.
    """
    feasible, skipped = [], []
    for nrows in rows:
        for ncols in columns:
            if nrows < 1 or not 1 <= ncols <= 100_000:
                raise ValueError("rows must be >= 1 and columns must lie between 1 and 100,000")
            shape = shape_for(ncols)
            if (math.prod(shape) if shape else 1) != ncols:
                raise AssertionError("Stan container shape does not match requested columns")
            # Fixture values have about 9-11 bytes plus commas per cell.
            output = nrows * ncols * 8
            estimated = nrows * ncols * 16 + (ncols * 25) + 1024
            case = Case(nrows, ncols, shape, estimated, output, shape_label(nrows, ncols))
            # Pandas and Polars may keep both input buffers and data columns.
            # Allow 4x the numerical payload as a conservative eager budget.
            ram_needed = max(estimated + output * 4, output * 5)
            limits = (
                (estimated > max_fixture_bytes, "per-file fixture limit"),
                (estimated > disk_bytes * disk_fraction, "free disk-space budget"),
                (ram_needed > ram_bytes * ram_fraction, "available RAM budget"),
            )
            reason = next((label for condition, label in limits if condition), None)
            if reason:
                skipped.append({**asdict(case), "status": "skipped", "reason": reason})
            else:
                feasible.append(case)
    return feasible, skipped


def shape_label(rows: int, columns: int) -> str:
    """Classify workloads by their aspect ratio, without changing the grid."""
    if columns > 10 * rows:
        return "very wide"
    if rows > 10 * columns:
        return "very long"
    return "wide+long"


def write_fixture(path: Path, case: Case) -> int:
    """Write one test file with a fixed reusable block and bounded memory."""
    header = ",".join(container_labels(case.shape)).encode("ascii")
    row_variants = []
    for variant in range(8):
        # A few numerical formats, signs and exponents, repeated deterministically.
        words = (
            f"{((col % 19) - 9) * (1 + variant / 5):.6g}"
            if col % 17 else f"{(variant + 1) * 1.03125e-6:.8e}"
            for col in range(case.columns)
        )
        row_variants.append((",".join(words) + "\n").encode("ascii"))
    avg = sum(map(len, row_variants)) // len(row_variants)
    chunk_rows = max(1, (4 * 1024 * 1024) // max(avg, 1))
    pattern = b"".join(row_variants)
    with path.open("wb", buffering=4 * 1024 * 1024) as target:
        target.write(b"# method = sample\n# num_warmup = 0\n# save_warmup = 0\n# thin = 1\n")
        target.write(header)
        target.write(b"\n")
        remaining = case.rows
        while remaining:
            take = min(remaining, chunk_rows)
            complete, tail = divmod(take, len(row_variants))
            target.write(pattern * complete)
            if tail:
                target.write(b"".join(row_variants[:tail]))
            remaining -= take
    return path.stat().st_size


def run_worker(path: Path, method: str, threads: int) -> dict:
    """Load the entire CSV once, inside a fresh process."""
    if method == "staniorust_mmap_1":
        from staniorust import read_csv

        started = time.perf_counter()
        _, data = read_csv(path, threads=1, engine="mmap", memory_limit_bytes=2**63 - 1)
        shape = data.shape
    elif method == "staniorust_mmap_parallel":
        from staniorust import read_csv

        started = time.perf_counter()
        _, data = read_csv(path, threads=threads, engine="mmap", memory_limit_bytes=2**63 - 1)
        shape = data.shape
    elif method == "staniorust_buffered":
        from staniorust import read_csv

        started = time.perf_counter()
        _, data = read_csv(path, engine="buffered", memory_limit_bytes=2**63 - 1)
        shape = data.shape
    elif method == "pandas":
        import pandas as pd

        started = time.perf_counter()
        data = pd.read_csv(path, comment="#", dtype="float64", low_memory=False)
        shape = data.shape
    elif method == "polars":
        import polars as pl

        started = time.perf_counter()
        data = pl.read_csv(path, comment_prefix="#", infer_schema_length=1_000)
        shape = data.shape
    elif method == "stanio":
        from stanio.csv import read_csv as stanio_read_csv

        started = time.perf_counter()
        _, data = stanio_read_csv(str(path))
        shape = data.shape
    elif method == "numpy_loadtxt":
        import numpy as np

        started = time.perf_counter()
        # Four configuration rows and one header row in the generated file.
        data = np.loadtxt(path, delimiter=",", skiprows=5, ndmin=2)
        shape = data.shape
    else:
        raise ValueError(f"unsupported benchmark method: {method}")
    elapsed = time.perf_counter() - started
    return {"elapsed_s": elapsed, "shape": list(shape)}


def subprocess_benchmark(
    path: Path, method: str, threads: int, timeout_s: int,
) -> dict:
    """Measure wall-clock time and child RSS without mixing reader processes."""
    cmd = [sys.executable, "-m", "benchmarks.run", "--worker", "--path", str(path),
           "--method", method, "--threads", str(threads)]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    monitor = psutil.Process(proc.pid)
    peak = 0
    timed_out = False
    while proc.poll() is None:
        try:
            peak = max(peak, monitor.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if time.perf_counter() - started > timeout_s:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - started
    if timed_out:
        return {"status": "timeout", "time_s": elapsed, "peak_rss_mib": peak / 2**20}
    if proc.returncode:
        return {"status": "failed", "time_s": elapsed, "peak_rss_mib": peak / 2**20,
                "error": stderr.strip()[-1500:]}
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "failed", "error": "worker did not return JSON", "time_s": elapsed}
    return {"status": "ok", "time_s": elapsed, "peak_rss_mib": peak / 2**20,
            "parse_time_s": result["elapsed_s"], "result_shape": result["shape"]}


def render_plots(records: list[dict], destination: Path) -> list[Path]:
    """Draw time-vs-samples log-log curves, one line per column count.

    Figures are separated by implementation so the legend encodes only one
    variable (column count), as requested. Time is log-scaled because sample
    counts span seven orders of magnitude; this shows scaling slopes clearly.
    """
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    methods = sorted({record["method"] for record in records if record.get("status") == "ok"})
    for method in methods:
        subset = [r for r in records if r.get("status") == "ok" and r["method"] == method]
        for metric, ylabel, suffix in (
            ("parse_time_s", "Parse + conversion time (s)", "time"),
            ("peak_rss_mib", "Peak process RSS (MiB)", "rss"),
        ):
            fig, ax = plt.subplots(figsize=(10, 6))
            for cols in sorted({r["columns"] for r in subset}):
                series = sorted((r for r in subset if r["columns"] == cols),
                                key=lambda r: r["rows"])
                xs = [r["rows"] for r in series]
                ys = [r[metric] for r in series]
                if xs and all(x > 0 for x in ys):
                    ax.loglog(xs, ys, marker="o", label=f"{cols:,} columns")
            ax.set_xlabel("CmdStan samples per chain (draws)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{method}: {ylabel} vs draws (log-log)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(title="Flat parameter columns", loc="best", fontsize=8, ncol=2)
            fig.tight_layout()
            output = destination / f"{method}_{suffix}_loglog.png"
            fig.savefig(output, dpi=160)
            plt.close(fig)
            paths.append(output)
    return paths


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    parser.add_argument("--cols", type=int, nargs="+", default=list(COLS))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ram-fraction", type=float, default=0.20)
    parser.add_argument("--disk-fraction", type=float, default=0.15)
    parser.add_argument("--max-file-mib", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--keep-files", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the CLI, including subprocess worker mode and plot-only regeneration."""
    args = _args(argv)
    if args.worker:
        print(json.dumps(run_worker(args.path, args.method, args.threads)))
        return
    if not 0 < args.ram_fraction <= 1 or not 0 < args.disk_fraction <= 1:
        raise ValueError("RAM and disk fractions must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / "results.jsonl"
    if args.plot_only:
        existing = [json.loads(row) for row in results_file.read_text().splitlines()]
        print("Plots:", *render_plots(existing, args.output / "plots"), sep="\n  ")
        return
    free_ram = budgeted_free_ram()
    free_disk = shutil.disk_usage(args.output).free
    max_file = args.max_file_mib or (256 if args.profile == "quick" else 4096)
    allowed, skipped = plan(args.rows, args.cols, ram_bytes=free_ram, disk_bytes=free_disk,
                            max_fixture_bytes=max_file * 2**20,
                            ram_fraction=args.ram_fraction, disk_fraction=args.disk_fraction)
    max_cases = args.max_cases or (28 if args.profile == "quick" else len(allowed))
    # Round-robin across sample counts: quick runs must include long files,
    # not only the first few short-row combinations.
    buckets = {}
    for n in args.rows:
        bucket = [case for case in allowed if case.rows == n]
        # Alternate narrow and wide cases so quick runs cover 100k-column
        # tensors as well as long, single-column series.
        reordered = []
        while bucket:
            reordered.append(bucket.pop(0))
            if bucket:
                reordered.append(bucket.pop(-1))
        buckets[n] = reordered
    spread = []
    while len(spread) < max_cases and any(buckets.values()):
        for n in args.rows:
            if buckets[n] and len(spread) < max_cases:
                spread.append(buckets[n].pop(0))
    allowed = spread
    report = {
        "free_ram_gib": round(free_ram / 2**30, 2),
        "free_disk_gib": round(free_disk / 2**30, 2),
        "planned_cases": len(allowed),
        "skipped_cases": len(skipped),
        "methods": args.methods,
    }
    print(json.dumps(report, indent=2))
    (args.output / "plan.json").write_text(json.dumps({
        "resources": report, "cases": [asdict(c) for c in allowed], "skips": skipped,
    }, indent=2))
    if args.dry_run:
        for case in allowed:
            print(f"{case.rows:>10,} draws x {case.columns:>7,} cols | "
                  f"{case.estimated_csv_bytes / 2**20:>8.1f} MiB | {case.shape}")
        return
    results_file.write_text("", encoding="utf-8")
    all_results = []
    with tempfile.TemporaryDirectory(prefix="staniorust-bench-") as work:
        for case_number, case in enumerate(allowed, 1):
            path = Path(work) / "chain.csv"
            print(f"Case {case_number}/{len(allowed)}: {case.rows:,} draws x "
                  f"{case.columns:,} columns", flush=True)
            actual_size = write_fixture(path, case)
            for method in args.methods:
                record = {**asdict(case), "method": method, "input_bytes": actual_size,
                          **subprocess_benchmark(path, method, args.threads, args.timeout)}
                all_results.append(record)
                with results_file.open("a", encoding="utf-8") as target:
                    target.write(json.dumps(record) + "\n")
                print(f"  {method:>25}: {record['status']:<8} "
                      f"{record.get('parse_time_s', float('nan')):.4f} s", flush=True)
            if not args.keep_files:
                path.unlink(missing_ok=True)
            else:
                target = args.output / f"case_{case.rows}_{case.columns}.csv"
                shutil.move(path, target)
    plots = render_plots(all_results, args.output / "plots")
    print("Generated plots:", *plots, sep="\n  ")


if __name__ == "__main__":
    main()
