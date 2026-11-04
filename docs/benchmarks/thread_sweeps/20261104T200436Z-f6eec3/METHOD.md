# Thread-scaling benchmark methodology

- Run: `20261104T200436Z-f6eec3`
- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- Python: 3.12.9
- Git commit: 874b41dc83247a219c4294e64c565d176ea1c928
- Configured thread counts: [1, 2, 4, 8, 16]
- Measured rounds: 7
- Warmups: 1
- Seed: 20261104
- Minimum published successful repeats: 3

Throughput = actual CSV file size (MiB) / measured reader time (s).
Import and subprocess startup are excluded from throughput timing.
Every reader/thread configuration runs in an independent subprocess.
Rust mmap and Polars use the configured worker count; baseline
readers use their normal thread behavior and are run once per shape.
Filesystem caches are not cleared; this is mostly warm-cache testing.
The CSV fixtures repeat deterministic numeric patterns. These
results are not substitutes for tests on actual CmdStan outputs.
The median reflects repeated attempts, not a guarantee on future files.
Failed or timed-out runs are not included in throughput statistics.

## Installed versions

```json
{
  "stanio_rust": "0.2.1",
  "numpy": "2.5.4",
  "pandas": "3.1.0",
  "polars": "2.1.0",
  "stanio": "0.5.1",
  "psutil": "7.2.2"
}
```
