# stanio_rust

**Rust-native CmdStan CSV and Stan JSON I/O, with mmap, parallel parsing, row indexing, and bounded-memory statistics.**

- **PyPI/distribution name:** `stanio_rust`
- **Python import:** `staniorust`
- **License:** Apache-2.0
- **Python:** 3.10+
- **Build:** [maturin](https://www.maturin.rs/) + Rust + a compatible native linker

> **Validation status (2026-11-04):** CI has successfully built the native extension and the latest reported test suite passed. The workflow now also defines wheel-build jobs, which require a successful CI run before their artifacts can be considered verified. Performance improvements remain unverified until benchmarked on representative CmdStan files.

## Features

| Task | Function | Behavior |
| --- | --- | --- |
| Read CmdStan samples | `read_csv()` | float64 NumPy `(chains, draws, columns)` or pandas/Polars |
| Reconstruct Stan arrays | `read_csv(..., reshape=True)` | Rust for real-valued arrays; StanIO-compatible NumPy extraction for complex values and tuples |
| Select specific draws | `read_csv(..., rows=slice(...))` | Only requested numeric values are decoded after row indexing |
| Repeated random access | `open_csv()` | Cache row-aligned byte partitions in memory; no sidecar files |
| Chunked CSV reads | `iter_csv()` | One chain/phase per bounded block |
| Lazy Dask | `read_csv_dask()` | Independently fetched local/cloud byte partitions |
| Streaming statistics | `summarize_csv()` | Mean, variance, min, max, finite count; approximate quantiles |
| Stan input JSON | `read_json()` | serde_json with optional selected keys and numeric NumPy arrays |
| Stan-compatible JSON output | `dump_stan_json()`, `write_stan_json()` | Bool, NumPy, pandas, complex and nested tuple serialization |
| StanIO parameter metadata | `parse_header()`, `stan_variables()` | Public `Variable`/`VariableType`, typed nested tuple reconstruction |
| Cloud | `engine="auto"` | mmap local completed files; fsspec buffered cloud reads |
| Benchmarks | `python -m benchmarks.run` | pandas, Polars, StanIO, NumPy and three Rust strategies |

## Installation

The extension must be compiled before using this source checkout. With Rust and a working C linker installed:

```cmd
conda activate lumo
python -m pip install maturin
cd path\to\stanio_rust
maturin develop --release
python -m pip install ".[pandas,polars,dask,cloud,benchmark,test,dev]"
python -m pytest -q
```

On Windows, the default Rust target expects the Microsoft C/C++ toolchain; alternatively configure a **matching GNU toolchain and linker**. Using `conda` to install maturin does not itself provide the native linker. You can also build an isolated wheel with `maturin build --release`.

### Prebuilt wheels from GitHub Actions

GitHub Actions builds **one `cp310-abi3` wheel per OS/architecture** (Linux x86-64,
Windows x64, macOS Intel and macOS Apple Silicon). Thanks to PyO3's
`abi3-py310` feature, each wheel supports standard GIL-enabled CPython
**3.10 and newer** on its matching platform; separate builds for 3.11, 3.12,
and so on are unnecessary. Free-threaded CPython needs separate wheel builds
and is not supported by this wheel matrix.

After a successful push or manual workflow run, open **Actions** →
**Quality, Native Tests and Wheels** → the completed run → **Artifacts**.
Download the artifact for your platform (`wheels-windows-amd64`,
`wheels-linux-x86_64`, `wheels-macos-x86_64`, or `wheels-macos-arm64`),
extract the `.whl` file, then install it with:

```cmd
python -m pip install stanio_rust-0.2.1-cp310-abi3-win_amd64.whl
```

This workflow stores downloadable artifacts for 14 days; it does **not**
automatically publish to PyPI or create GitHub Releases. Native wheels are
verified with an import smoke test. The separate CI test matrix runs against
CPython 3.10–3.14 on Windows and Linux.

**Format and check Python with Ruff:**

```cmd
ruff format python tests benchmarks examples scripts
ruff check python tests benchmarks examples scripts
```

Rust source formatting: `cargo fmt --all`. The CI workflow runs Ruff and the native integration tests on Linux and Windows.

## Quick usage with CmdStanPy

```python
import staniorust as sio

csv_files = fit.runset.csv_files  # list of CSVs from a completed CmdStanPy fit

# Native mmap + Rayon, chains x draws x flattened columns
header, samples = sio.read_csv(csv_files, threads=8)

# Reshape a vector or matrix using Stan's indexed field names
variables = sio.read_csv(csv_files, columns=["theta", "beta"], reshape=True)
# variables["beta"].shape == (chains, draws, *stan_dimensions)

# A draw range from every chain: byte index skips preceding draw conversions
header, selected = sio.read_csv(
    csv_files, rows=slice(100_000, 110_000), columns=["theta"], threads=8
)

# DataFrames retain flattened column labels plus chain/draw/warmup identifiers
pd_df = sio.read_csv(csv_files, output="pandas", columns=["theta"])
pl_df = sio.read_csv(csv_files, output="polars", columns=["theta"])

# Iterate without loading all draws at once
for chunk in sio.iter_csv(csv_files, chunksize=10_000, columns=["theta"]):
    print(chunk.chain, chunk.draw_start, chunk.warmup, chunk.data.shape)

# No per-draw output array: Rust maintains online statistics only
stats = sio.summarize_csv(csv_files, columns=["theta"], threads=8)

# P² approximate streaming quantiles, sequential rather than parallel
qstats = sio.summarize_csv(csv_files, columns=["theta"], quantiles=[0.05, 0.5, 0.95])

# Reuse a local mmap and row index for many queries
reader = sio.open_csv(csv_files[0])
print(reader.metadata, reader.nrows())
print(reader.read(rows=slice(1_000, 2_000), columns=["beta"])[1].shape)
```

### Dask and remote files

```python
# Local: indexed by Rust, then independently mapped per Dask partition
lazy = sio.read_csv_dask(csv_files, chunksize=50_000, columns=["beta"])

# S3/GCS/ABFS/etc: requires a compatible fsspec backend, plus credentials
# and the same URI/backend availability on every worker.
for chunk in sio.iter_csv("s3://my-bucket/model/chain-1.csv", columns=["beta"]):
    process(chunk.data)

# Local and remote files use the same auto-selection policy.
# engine="buffered" is safer for files that may change during reading.
```

**No files are ever written alongside the CSV inputs.** Index checkpoints are held in RAM. Dask index construction does make an initial byte scan, which is O(file size); the numerical parser decodes only the partitions requested by workers.

### JSON

```python
# Read selected Stan input variables without constructing Python objects
# for unrelated top-level keys.
values = sio.read_json("data.json", keys=["N", "X"], arrays="numpy")

# Complex values and tuple-like objects that Stan represents as JSON
# mappings remain dictionaries; numeric rectangular arrays become float64.
```


## Compatibility with stan-dev/stanio

The public `staniorust` package provides `Variable`, `VariableType`,
`parse_header`, `stan_variables`, `dump_stan_json`, and `write_stan_json`
for StanIO-style use. It also provides the `staniorust.csv`,
`staniorust.reshape`, and `staniorust.json` submodules.

**Run the *unaltered* upstream test suite** against this package (Rust
extension must be installed):

```cmd
git clone --depth 1 https://github.com/stan-dev/stanio.git upstream-stanio
python scripts/run_upstream_stanio_tests.py upstream-stanio
```

The runner aliases `stanio` imports to `staniorust` in memory; no upstream
tests, fixtures, or installed StanIO package need to be patched. Its CI job
clones StanIO and runs these tests. See `.github/workflows/ci.yml`.

**Validation:** the compatibility-focused pure-Python tests can run without
Rust. Full upstream compatibility **has not yet been established** until
the native extension is built and the upstream fixture suite succeeds.

### Writing Stan JSON

```python
import staniorust as sio

sio.write_stan_json("data.json", {"N": 3, "z": 1 + 2j, "pair": (1, [2, 3])})
serialized = sio.dump_stan_json({"flag": True})  # flag becomes 1
```

### Reconstructing complex or tuple parameters

```python
header, draws = sio.read_csv("chain.csv")
params = sio.parse_header(header)
values = sio.stan_variables(params, draws, object=False)
# Complex parameters become complex128; tuples become structured NumPy dtypes.
# Set object=True for Python tuple objects instead.
```

## mmap + Rayon architecture

The diagram below is **Mermaid**, so GitHub renders it automatically in
this README; no image export or GitHub Action is required.

```mermaid
flowchart TD
    A["CmdStan CSV / completed output"] --> B{"Local immutable file?"}
    B -->|Yes| C["Read-only mmap (memmap2)"]
    B -->|Cloud or active file| D["Buffered / range-based reading"]
    C --> E["Scan metadata and newline-aligned row boundaries"]
    E --> F{"Requested operation"}
    F -->|Read all or selected draws| G["Sparse row index + row-aligned partitions"]
    G --> H["Rayon worker threads: parse numeric columns"]
    H --> I["Write disjoint numeric output slices"]
    D --> J["Sequential chunk parser"]
    J --> I
    F -->|Streaming summary| K["Per-worker Welford statistics"]
    K --> L["Merge statistics, no full draws matrix"]
    I --> M["NumPy / pandas / Polars / Stan reshape"]
    F -->|Lazy Dask| N["Task graph of independent indexed ranges"]
    N --> H
```

The memory map avoids an intermediate copy of CSV **input bytes**; numeric
output still requires decoding text into float64 values. The algorithms are
correctness-checked against CmdStan and StanIO fixtures before benchmarking.

1. Read initial CmdStan comments and header to determine column names and saved warmup count.
2. Open a **read-only mmap** of a complete, immutable local CSV file.
3. Scan newline boundaries, omitting comments, and build *sparse* row-aligned partitions (default up to 16,384 rows or 32 MiB each).
4. For selected rows, skip irrelevant records without parsing their numerical values.
5. Use Rayon workers to parse disjoint partitions directly into disjoint output array slices. Python's GIL is released during numeric work.
6. For streaming statistics, maintain a per-worker Welford accumulator and merge moments with Chan's formulas; approximate P² quantiles are sequential.

The mmap provides efficient **input access, not zero-copy float64 arrays**. ASCII numbers still require parsing. Byte-range indexing means a random row selection has one O(file size) boundary scan on the first read, after which only relevant byte partitions need to be parsed. Reusing `StanReader` avoids rebuilding its default index for repeated reads.

**Do not map files concurrently being written or truncated.** mmap2 documents undefined behavior if another process modifies the mapped file. Use `engine="buffered"` for active files and cloud files.

## Stan correctness

- Leading and embedded `#` comments are skipped.
- Saved warmup rows are excluded by default, and chunks never mix warmup with sampling rows.
- The first index of an n-D container changes fastest in Stan's CSV flattening order; Rust reshaping uses **explicit 1-based column names**, rather than assuming C order.
- Floating-point values support `NaN`, `inf`, `-inf`, and scientific notation.
- Each numeric record is checked for the expected field count, even when projecting columns.
- Rows are zero-based *retained* row numbers per chain; `rows=slice(a, b)` is `[a, b)`.
- NumPy multi-chain eager output requires equal retained row counts; chunked DataFrame output does not.
- Tuple slots (`bar:1`, `bar:2.1`) and complex components (`z.real`, `z.imag`) are available as flat columns. When `reshape=True`, a NumPy compatibility path reconstructs them as typed Python tuple or complex arrays. Ordinary numeric array reshaping remains native Rust.
- Profiling CSVs with string columns are **not supported** by this numeric CSV reader.

## Memory behavior and limits

| Operation | Approximate application memory |
| --- | --- |
| mmap data source | Mapped virtual address range + OS page cache, not a copy of the entire CSV |
| Indexed metadata | O(number of partitions), not O(number of rows) |
| Streaming CSV | O(decoded bytes in one chunk) |
| Eager NumPy | O(8 × chains × draws × columns) bytes |
| Rust exact reductions | O(columns × active Rayon workers) |
| P² quantiles | O(columns × requested quantiles) |
| Dask | O(number of partitions) graph + memory per active worker |

For eager NumPy `read_csv` output, the API defaults to a limit of **65% of currently available RAM**, not total installed RAM (via psutil). Set `memory_limit_bytes` explicitly to customize. Additional allocations can arise in pandas, Polars, reshape, Dask, and the OS page cache; for very large workloads prefer `iter_csv` or `summarize_csv`.

### Known limitations and validation gaps

1. **Native build unverified in this authoring environment.** Treat this as the first fully specified source release candidate, not a verified wheel.
2. Cloud readers are *bounded for CSV*, but `read_json()` on cloud input currently fetches the full JSON byte document before handing it to Rust. Local JSON uses a streaming serde deserializer and can skip unselected top-level keys, but retains each selected JSON value in memory.
3. Dask indexing must scan the file once. Cloud indexing scans line boundaries using Python; local mmap indexing is in Rust.
4. Complex and tuple types are reconstructed in NumPy (not yet in Rust); Pathfinder/variational-specific interpretation and profiling CSV text fields remain unsupported.
5. `P²` streaming quantiles are **approximate** and sequential. Moments use finite values only, with nonfinite values counted separately.
6. Remote backends require their own `fsspec` dependencies and credentials. Concurrent mutation of a cloud object can invalidate range offsets.
7. Benchmark timings are hardware-, filesystem-, and version-dependent. No Rust speedup is claimed in this source release.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the experiment matrix, realistic resource limits and plotting rationale.

```cmd
python -m benchmarks.run --dry-run
python -m benchmarks.run --profile quick --repetitions 7 --warmups 1
python -m benchmarks.run --profile quick --publish  # also update README gallery
python -m benchmarks.run --profile full --ram-fraction 0.4 --max-file-mib 8192
python -m benchmarks.run --plot-only
python -m benchmarks.publish  # versioned gallery + performance summary in README
python -m benchmarks.thread_sweep --dry-run
python -m benchmarks.thread_sweep --profile quick --threads 1 2 4 8 16
python -m benchmarks.thread_sweep --profile full --all-threads --dry-run
python -m benchmarks.thread_sweep --profile quick --publish
```

Results are saved under `benchmark_results/` as JSON Lines, a `summary.csv` of repeated measurements, and standalone plots. Benchmarking now defaults to seven measured runs in a randomized balanced order plus one warmup per method/case; figures show the median and interquartile spread. Run `python -m benchmarks.publish` to generate a versioned, clickable README gallery of only sufficiently repeated measurements. Each plot uses **draw count on the x-axis, parse time on the y-axis, and a separate curve for each parameter-column count**. Both axes are logarithmic. A separate plot per implementation avoids overloading one legend with both algorithms and widths.

### Thread scaling and throughput across CSV file shapes

For fair comparisons, configure **the same worker count** for Rust parallel
mmap and Polars. Run the pandas, StanIO, NumPy and Rust buffered baselines once
per file shape; they do not have a comparable thread-count knob here. The
thread sweep defaults to **1, 2, 4, 8 and 16 threads**, one warmup and seven
measured repetitions, using randomized/balanced execution order.

```cmd
python -m benchmarks.thread_sweep --dry-run
python -m benchmarks.thread_sweep --profile quick --threads 1 2 4 8 16 --repetitions 7
python -m benchmarks.plot_threads --format png svg
python -m benchmarks.publish_threads
```

Figures show **throughput versus configured threads**, **throughput versus
actual CSV size** at each fixed column count, and **throughput versus column
count** at fixed total numeric values. File size is measured from generated
CSV bytes, not inferred from float64 payload size. Use
`--all-threads` to test all 16 thread counts once the coarse sweep identifies
where performance stops improving. The full sweep is long; use `--dry-run` to
inspect its resource and subprocess budgets before starting it.

`--publish` adds a separate versioned thread-scaling gallery under
`docs/benchmarks/thread_sweeps/<run-id>/` without replacing the existing
main benchmark gallery. See [BENCHMARKS.md](BENCHMARKS.md) for test shape
selection and interpretation caveats.

<!-- BENCHMARKS:START -->
### Measured performance comparisons

**Run:** `20261104T173933Z-bfdc6d` · **Minimum successful repeats:** 3 · **Scheduled repeats:** 15 · **Warmups:** 1 · **Random seed:** 20261104

**Environment:** Windows-11-10.0.26300-SP0, Python 3.12.9, 8 configured Rust threads.

Times are the median of independent subprocess runs; shaded bands show the interquartile range (25th–75th percentile). Only measured configurations with enough successful runs are shown. These are warm-cache-oriented measurements, **not** cold-disk benchmarks.

[Methodology and environment](docs/benchmarks/runs/20261104T173933Z-bfdc6d/METHOD.md) · [All summary results (CSV)](docs/benchmarks/runs/20261104T173933Z-bfdc6d/summary.csv)

| Fixed columns — time vs samples | Fixed samples — time vs columns |
|:---:|:---:|
| <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100.png" width="420" alt="time_vs_samples_cols_100"></a> | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000.png" width="420" alt="time_vs_columns_samples_1000"></a> |

#### Representative results relative to pandas

| Samples | Columns | Fastest method | Median seconds | Speedup vs pandas |
|---:|---:|---|---:|---:|
| 1 | 100,000 | `numpy_loadtxt` | 0.008684 | 80.21× |
| 10 | 100,000 | `staniorust_mmap_1` | 0.03904 | 18.20× |
| 100,000 | 10 | `staniorust_mmap_parallel` | 0.00778 | 2.81× |
| 1,000,000 | 1 | `staniorust_mmap_parallel` | 0.01971 | 2.21× |
| 100 | 100,000 | `staniorust_mmap_parallel` | 0.1381 | 6.50× |
| 10,000,000 | 1 | `polars` | 0.1253 | 3.69× |
| 1,000 | 100,000 | `staniorust_mmap_parallel` | 0.494 | 7.60× |
| 100,000 | 1,000 | `staniorust_mmap_parallel` | 0.4648 | 8.56× |
| 1,000,000 | 100 | `staniorust_mmap_parallel` | 0.4189 | 9.57× |
| 10,000,000 | 10 | `staniorust_mmap_parallel` | 0.5076 | 5.25× |

Speedup is pandas median time divided by the fastest method's median time for the **same file shape**. Cases with pandas timing below 10 ms are omitted from this table to avoid misleading microbenchmarks.

<details>
<summary>View all 12 benchmark comparison plots</summary>

**Time vs sample count (fixed number of columns)**

| Columns | Plot |
|---:|:---|
| 1 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_1.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_1.png" width="420" alt="time_vs_samples_cols_1"></a> |
| 10 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_10.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_10.png" width="420" alt="time_vs_samples_cols_10"></a> |
| 100 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100.png" width="420" alt="time_vs_samples_cols_100"></a> |
| 1,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_1000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_1000.png" width="420" alt="time_vs_samples_cols_1000"></a> |
| 100,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_samples_cols_100000.png" width="420" alt="time_vs_samples_cols_100000"></a> |

**Time vs column count (fixed number of samples)**

| Samples | Plot |
|---:|:---|
| 1 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1.png" width="420" alt="time_vs_columns_samples_1"></a> |
| 10 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_10.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_10.png" width="420" alt="time_vs_columns_samples_10"></a> |
| 100 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_100.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_100.png" width="420" alt="time_vs_columns_samples_100"></a> |
| 1,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000.png" width="420" alt="time_vs_columns_samples_1000"></a> |
| 100,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_100000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_100000.png" width="420" alt="time_vs_columns_samples_100000"></a> |
| 1,000,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_1000000.png" width="420" alt="time_vs_columns_samples_1000000"></a> |
| 10,000,000 | <a href="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_10000000.png"><img src="docs/benchmarks/runs/20261104T173933Z-bfdc6d/plots/time_vs_columns_samples_10000000.png" width="420" alt="time_vs_columns_samples_10000000"></a> |

</details>
<!-- BENCHMARKS:END -->

<!-- THREAD-SWEEP:START -->
### Thread scaling and file-shape throughput

**Run:** `20261104T200436Z-f6eec3` · 7 measured repeats · 1 warmup(s) · threads [1, 2, 4, 8, 16].

The plots compare **measured CSV throughput in MiB/s** (larger is better). Each method uses consistent colors; thread counts use different markers and linestyles. Methods that do not expose a worker-count setting are measured once per file, not duplicated at every count.

[Methodology and environment](docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/METHOD.md) · [Summary CSV](docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/summary.csv)

| Throughput vs threads | Throughput vs file size |
|:---:|:---:|
| <a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10x100000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10x100000.png" width="410" alt="throughput_vs_threads_10x100000"></a> | <a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1.png" width="410" alt="throughput_vs_file_size_cols_1"></a> |

**Equal-value-count comparison across file shapes**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_1000000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_1000000.png" width="410" alt="throughput_vs_columns_values_1000000"></a>

<details>
<summary>Show all 26 throughput figures</summary>

**throughput vs threads 10x100000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10x100000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10x100000.png" width="410" alt="throughput_vs_threads_10x100000"></a>

**throughput vs threads 100x10000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100x10000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100x10000.png" width="410" alt="throughput_vs_threads_100x10000"></a>

**throughput vs threads 100x100000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100x100000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100x100000.png" width="410" alt="throughput_vs_threads_100x100000"></a>

**throughput vs threads 1000x1000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x1000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x1000.png" width="410" alt="throughput_vs_threads_1000x1000"></a>

**throughput vs threads 1000x10000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x10000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x10000.png" width="410" alt="throughput_vs_threads_1000x10000"></a>

**throughput vs threads 1000x100000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x100000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000x100000.png" width="410" alt="throughput_vs_threads_1000x100000"></a>

**throughput vs threads 10000x100**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x100.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x100.png" width="410" alt="throughput_vs_threads_10000x100"></a>

**throughput vs threads 10000x1000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x1000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x1000.png" width="410" alt="throughput_vs_threads_10000x1000"></a>

**throughput vs threads 10000x10000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x10000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000x10000.png" width="410" alt="throughput_vs_threads_10000x10000"></a>

**throughput vs threads 100000x10**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x10.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x10.png" width="410" alt="throughput_vs_threads_100000x10"></a>

**throughput vs threads 100000x100**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x100.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x100.png" width="410" alt="throughput_vs_threads_100000x100"></a>

**throughput vs threads 100000x1000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x1000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_100000x1000.png" width="410" alt="throughput_vs_threads_100000x1000"></a>

**throughput vs threads 1000000x1**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x1.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x1.png" width="410" alt="throughput_vs_threads_1000000x1"></a>

**throughput vs threads 1000000x10**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x10.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x10.png" width="410" alt="throughput_vs_threads_1000000x10"></a>

**throughput vs threads 1000000x100**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x100.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_1000000x100.png" width="410" alt="throughput_vs_threads_1000000x100"></a>

**throughput vs threads 10000000x1**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000000x1.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000000x1.png" width="410" alt="throughput_vs_threads_10000000x1"></a>

**throughput vs threads 10000000x10**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000000x10.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_threads_10000000x10.png" width="410" alt="throughput_vs_threads_10000000x10"></a>

**throughput vs file size cols 1**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1.png" width="410" alt="throughput_vs_file_size_cols_1"></a>

**throughput vs file size cols 10**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_10.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_10.png" width="410" alt="throughput_vs_file_size_cols_10"></a>

**throughput vs file size cols 100**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_100.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_100.png" width="410" alt="throughput_vs_file_size_cols_100"></a>

**throughput vs file size cols 1000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_1000.png" width="410" alt="throughput_vs_file_size_cols_1000"></a>

**throughput vs file size cols 10000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_10000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_10000.png" width="410" alt="throughput_vs_file_size_cols_10000"></a>

**throughput vs file size cols 100000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_100000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_file_size_cols_100000.png" width="410" alt="throughput_vs_file_size_cols_100000"></a>

**throughput vs columns values 1000000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_1000000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_1000000.png" width="410" alt="throughput_vs_columns_values_1000000"></a>

**throughput vs columns values 10000000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_10000000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_10000000.png" width="410" alt="throughput_vs_columns_values_10000000"></a>

**throughput vs columns values 100000000**

<a href="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_100000000.png"><img src="docs/benchmarks/thread_sweeps/20261104T200436Z-f6eec3/plots/throughput_vs_columns_values_100000000.png" width="410" alt="throughput_vs_columns_values_100000000"></a>

</details>
<!-- THREAD-SWEEP:END -->

## License

This project is licensed under **Apache License, Version 2.0**. See [LICENSE](LICENSE).
