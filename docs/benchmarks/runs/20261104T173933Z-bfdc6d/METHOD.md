# Benchmark run 20261104T173933Z-bfdc6d

## Protocol

- Measured repetitions per method/case (planned): 15
- Warmups per method/case: 1
- Scheduling seed: 20261104
- Balanced randomized ordering: True
- Rust reader threads: 8
- Each attempt runs in a new process; reader import is outside `parse_time_s`.
- One CSV file per shape is shared among readers, with warmup reads before timing.
- No OS cache clearing: these results represent mostly warm-cache behavior.
- Plot lines show medians; shaded regions show the 25th–75th percentile.
- Only groups with at least 3 successful measured attempts are plotted.
- Timeout/failed attempts are preserved in raw results, not converted to zero.
- Microsecond-scale cases are dominated by fixed overhead and should not be used for general speedup claims.

## Environment

- OS: Windows-11-10.0.26300-SP0
- Python: 3.12.9
- Processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- Logical / physical cores: 32 / 24
- Git commit: 874b41dc83247a219c4294e64c565d176ea1c928

## Installed package versions

- numpy: 2.5.4
- pandas: 3.1.0
- polars: 2.1.0
- psutil: 7.2.2
- stanio: 0.5.1
- stanio_rust: 0.2.1

## Reproducibility files

- `results.jsonl`: all measured attempts, statuses and run sequence; raw stderr details are redacted from this public snapshot
- `warmups.jsonl`: warmup attempts (when available)
- `plan.json`: resource planner, skipped cases and system details
- `summary.csv`: per-method, per-shape aggregate measurements

**Synthetic fixtures are deterministic and deliberately repetitive; results may differ on real CmdStan outputs, cold storage or other hardware.**
