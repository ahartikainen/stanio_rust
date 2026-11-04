"""Small, dependency-free aggregation helpers for repeatable CSV benchmarks."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated percentile on a finite, nonempty sample."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Expected nonempty values and percentile between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(records: list[dict]) -> list[dict]:
    """Aggregate successful measured attempts, never counting warmups.

    Older result files without an explicit ``phase`` are considered measured.
    One record is returned for each (rows, columns, method) with any attempts.
    """
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("phase", "measure") != "measure":
            continue
        if not {"rows", "columns", "method"}.issubset(record):
            continue
        key = (int(record["rows"]), int(record["columns"]), str(record["method"]))
        groups[key].append(record)

    summary: list[dict] = []
    for (rows, columns, method), attempts in sorted(groups.items()):
        successful = [
            r
            for r in attempts
            if r.get("status") == "ok"
            and isinstance(r.get("parse_time_s"), (int, float))
            and math.isfinite(r["parse_time_s"])
            and r["parse_time_s"] > 0
        ]
        times = [float(r["parse_time_s"]) for r in successful]
        walls = [
            float(r["time_s"])
            for r in successful
            if isinstance(r.get("time_s"), (int, float)) and math.isfinite(r["time_s"])
        ]
        rss = [
            float(r["peak_rss_mib"])
            for r in successful
            if isinstance(r.get("peak_rss_mib"), (int, float)) and math.isfinite(r["peak_rss_mib"])
        ]
        mean_time = statistics.mean(times) if times else None
        summary.append(
            {
                "rows": rows,
                "columns": columns,
                "method": method,
                "n_ok": len(successful),
                "n_failed": len(attempts) - len(successful),
                "median_s": statistics.median(times) if times else None,
                "p25_s": percentile(times, 0.25) if times else None,
                "p75_s": percentile(times, 0.75) if times else None,
                "min_s": min(times) if times else None,
                "max_s": max(times) if times else None,
                "cv": (
                    statistics.stdev(times) / mean_time if len(times) > 1 and mean_time else None
                ),
                "median_wall_s": statistics.median(walls) if walls else None,
                "median_peak_rss_mib": statistics.median(rss) if rss else None,
            }
        )
    return summary


def write_summary(records: list[dict], path: Path) -> list[dict]:
    """Write a row-oriented CSV summary and return its contents."""
    summary = aggregate(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "rows",
        "columns",
        "method",
        "n_ok",
        "n_failed",
        "median_s",
        "p25_s",
        "p75_s",
        "min_s",
        "max_s",
        "cv",
        "median_wall_s",
        "median_peak_rss_mib",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary
