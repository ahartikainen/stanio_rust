# stanio_rust

**Rust-native CmdStan CSV and Stan JSON I/O, with mmap, parallel parsing, row indexing, and bounded-memory statistics.**

- **PyPI/distribution name:** `stanio_rust`
- **Python import:** `staniorust`
- **License:** Apache-2.0
- **Python:** 3.10+
- **Build:** [maturin](https://www.maturin.rs/) + Rust + a compatible native linker

> **Validation status (2026-11-04):** This is a full *source implementation candidate*, not a published or compiled release. Python wrapper tests and the benchmark planning tests pass. The current execution environment has no `rustc`, `cargo`, or `maturin` and cannot reach Rust's distribution server, so the native extension, native integration tests, and Rust performance have **not** been built or verified here. CI contains a Rust build/test matrix to perform that validation in a Rust-enabled environment. Treat native speedups as hypotheses until measured.

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
python -m benchmarks.run --profile quick
python -m benchmarks.run --profile full --ram-fraction 0.4 --max-file-mib 8192
python -m benchmarks.run --plot-only
```

Results are saved under `benchmark_results/` as JSON Lines and standalone plots. Each plot uses **draw count on the x-axis, parse time on the y-axis, and a separate curve for each parameter-column count**. Both axes are logarithmic. A separate plot per implementation avoids overloading one legend with both algorithms and widths.

## License

This project is licensed under **Apache License, Version 2.0**. See [LICENSE](LICENSE).
