# Benchmark design: sample count × Stan container dimensions

The benchmark suite separates *model shape* (flattened output columns) from *sample count* (rows). The largest example variable is a **four-dimensional `array[10, 10, 10, 100] real theta`**, containing exactly **100,000 numeric CSV columns**.

## Experimental grid

| Samples per chain (rows) | Parameter columns | Stan variable dimensions |
| ---: | ---: | --- |
| 1 | 1 | scalar |
| 10 | 10 | `[10]` |
| 100 | 100 | `[10, 10]` |
| 1,000 | 1,000 | `[10, 10, 10]` |
| 100,000 | 10,000 | `[10, 10, 10, 10]` |
| 1,000,000 | 100,000 | `[10, 10, 10, 100]` |
| 10,000,000 | All widths from 1 to 100,000 | Feasible subsets only |

The runner evaluates the **Cartesian product** of the seven requested row counts and six column counts. A combination is skipped if its resource requirements exceed the available budget. The table above illustrates shape mapping, not an assertion that every row/column pair is feasible.

One file with 10,000,000 draws × 100,000 columns would require **8,000 GB (8 TB) of float64 output alone**, plus CSV text storage and parser overhead. It cannot be tested eagerly on 128 GB of RAM. That case is reported as *skipped*, not silently downsampled or labeled measured.

## Reference implementations

1. `staniorust_mmap_1`: native mmap parser, one Rayon thread.
2. `staniorust_mmap_parallel`: native mmap parser with configured Rayon thread count.
3. `staniorust_buffered`: native Rust buffered stream.
4. `pandas`: `pandas.read_csv(..., comment='#', dtype='float64')`.
5. `polars`: `polars.read_csv(..., comment_prefix='#')`.
6. `stanio`: official StanIO CSV reader.
7. `numpy_loadtxt`: `numpy.loadtxt` after the known synthetic header.

Compare **eager-to-eager** methods on the *same* generated file. Streaming-only methods and `summarize_csv` are different workloads, so they should not be mixed into eager time plots as if they return the same result.

## Memory and disk safety

- Free RAM is measured with `psutil.virtual_memory().available` (and Linux cgroup capacity when available); installed RAM is not used as a substitute.
- Default memory budget: **20% of free RAM**, with a conservative multiplier for pandas/Polars temporary copies.
- Default disk budget: **15% of free disk space**.
- Default quick maximum synthetic file estimate: **256 MiB**; full profile: **4,096 MiB** (both configurable).
- The quick profile selects a spread of feasible cases across sample-count levels and alternates narrow and wide shapes. The full profile includes every feasible case.
- One synthetic file is generated at a time into a temporary directory and normally deleted immediately after all readers have processed it.
- Use `--dry-run` to view the entire resource decision *before any CSV fixture files are created*.
- Use `--keep-files` only when you want to retain the synthetic input files; they are never written to CmdStan's folders.
- Default per-method timeout: **240 seconds**, configurable with `--timeout`.

## Repeated, randomized methodology (recommended)

The original exploratory implementation measured every reader **once and in a
fixed order**. Those old plots are suitable for a first look but **not** robust
speedup claims. This updated runner defaults to:

- **7 measured runs per method and case**, with **1 warmup run** before timing.
- Each measured run is a **fresh Python subprocess**. The timer starts after
  library imports but includes parsing plus conversion to the requested output.
- **Randomized case order** and a **seeded, balanced round-robin method order**:
  for seven readers and seven rounds, each reader runs at each position once.
- The same generated CSV file is reused for every method, while setup and file
  generation are excluded from reader timing. `POLARS_MAX_THREADS` is set to
  the benchmark `--threads` value, matching the configured Rust thread count.
- **Median times** in plots, with translucent **25th-to-75th-percentile bands**
  from the repeated observations. The bands show run-to-run spread, *not*
  confidence intervals on method speedups. Use `summary.csv` to inspect `cv`
  (coefficient of variation), number of successful runs and failures.
- **Warm-cache-oriented**: the benchmark does **not** flush the OS filesystem
  cache between reads. It does not establish cold-disk throughput. CPU load,
  antivirus, thermals and other processes can still affect measurements.
- For a close comparison or noisy runs, use `--repetitions 15`. Tiny files
  dominated by microsecond-level fixed overhead should not be used to claim
  broad performance superiority. Benchmark representative real CmdStan files
  as well as synthetic data.

A benchmark run writes `plan.json` (including the UTC run ID, random seed,
resource limits, versions, CPU and Git commit), `results.jsonl` with **every
measured run**, `warmups.jsonl` separately, `summary.csv`, `plots/`, and
`plots_grouped/`. Failed attempts remain in the raw records.

The `--dry-run` command writes **`plan-preview.json`** instead of replacing the
provenance of an earlier measured run.

### Publish a versioned benchmark gallery

Once at least three successful repetitions are available for a case/method,
publish all validated grouped plots and a compact comparison table to the
README. You can use `--publish` when running a benchmark or publish a
completed run separately:

```cmd
python -m benchmarks.publish
```

This creates `docs/benchmarks/runs/<run-id>/` with the plots, raw timings,
summary, machine metadata and methodology. Raw worker stderr is redacted from
published JSONL to avoid exposing local usernames and file paths; the original
local `benchmark_results/` retains the diagnostics. The README's generated benchmark
section is replaced **in place** on subsequent publications; older snapshot
folders remain intact. The publisher refuses old single-run-only data by
default rather than presenting it as statistically reliable.

You can require at least five successful runs before publication:

```cmd
python -m benchmarks.publish --min-repeats 5
```

Commit the README and the generated docs folder to Git. `benchmark_results/`
remains ignored. **Publish before starting another benchmark**, because a
new run replaces the unversioned working `results.jsonl`.

## Commands

```cmd
conda activate lumo
cd path\to\stanio_rust
python -m pip install ".[benchmark]"

python -m benchmarks.run --dry-run
python -m benchmarks.run --profile quick --repetitions 7 --warmups 1 --seed 20261104
python -m benchmarks.run --profile quick --repetitions 7 --publish
python -m benchmarks.run --profile full --ram-fraction 0.35 --max-file-mib 8192
python -m benchmarks.run --rows 1 10 100 1000 100000 1000000 10000000 --cols 1 10 100 1000 10000 100000
python -m benchmarks.run --methods staniorust_mmap_1 staniorust_mmap_parallel pandas polars
python -m benchmarks.run --profile full --repetitions 15 --warmups 1 --seed 20261104
python -m benchmarks.run --plot-only
python -m benchmarks.plot_grouped --format png svg
python -m benchmarks.publish
```

Outputs:

```text
benchmark_results/
    plan.json                     # case selection, free resources, skip reasons
    results.jsonl                 # each measured run, run order, timing and peak RSS
    warmups.jsonl                 # unmeasured warmup runs
    summary.csv                   # median, p25, p75, CV, failures per case/method
    plots_grouped/               # one method-comparison chart per fixed dimension
    plots/
        pandas_time_loglog.png
        pandas_rss_loglog.png
        polars_time_loglog.png
        ...
```

## Why log–log rather than only semilog-x?

The requested x-values span **seven powers of ten** (1 → 10,000,000). Parsing time should roughly scale with the total number of numeric values, but setup dominates tiny files, and caches/storage impose nonlinear behavior. A **logarithmic x-axis** lets all seven row counts remain readable; a **logarithmic y-axis** makes multi-order-of-magnitude time differences visible without flattening the fast readers.

On log–log plots, linear-time growth with sample count at a *fixed number of columns* has approximate slope **1** once fixed overhead is negligible. Curves for different column widths should be vertically separated approximately in proportion to their total decoded cells, unless throughput or allocation behavior changes.

A semilog-x plot can be useful for a narrower time range, but the full row-count sweep is better viewed with **log–log** scaling. Always state that missing curve points were skipped due to RAM, disk or timeout limitations; they are not zero timings.

## Fairness caveats

- Running each implementation in a separate process isolates Python caches and tracks peak process RSS. OS filesystem cache remains shared; the randomized, balanced order limits systematic positioning effects but is **not** a cold-cache test.
- Parser time excludes module import but includes object conversion, consistent with the eager output being used.
- Polars and pandas may differ in allocation behavior, especially at 100,000 columns; record versions and CPU thread counts when comparing results.
- The generated file repeats several deterministic numeric row patterns; real Stan models may have different string length distributions, numeric extremes, and value entropy. Also benchmark real CmdStan output files.
- Benchmark results should never be used as claimed speedups until the native extension has passed correctness tests on the target platform.
