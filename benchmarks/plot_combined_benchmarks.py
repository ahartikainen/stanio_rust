from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("benchmark_results/results.jsonl")
OUT_DIR = Path("benchmark_results/plots_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> pd.DataFrame:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    df = pd.DataFrame(records)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if df.empty:
        raise ValueError("No successful benchmark rows found")

    return df


def find_metric_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find any of these columns: {candidates}\nAvailable columns: {list(df.columns)}"
    )


def ensure_memory_mb(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Return (df, memory_column_name_in_mb).
    Supports common alternative memory field names.
    """
    if "peak_rss_mb" in df.columns:
        return df, "peak_rss_mb"
    if "rss_mb" in df.columns:
        return df, "rss_mb"
    if "peak_rss_mib" in df.columns:
        df = df.copy()
        df["peak_rss_mb"] = df["peak_rss_mib"] * 1.048576
        return df, "peak_rss_mb"
    if "peak_rss_bytes" in df.columns:
        df = df.copy()
        df["peak_rss_mb"] = df["peak_rss_bytes"] / 1_000_000
        return df, "peak_rss_mb"
    if "rss_bytes" in df.columns:
        df = df.copy()
        df["peak_rss_mb"] = df["rss_bytes"] / 1_000_000
        return df, "peak_rss_mb"

    raise KeyError(
        "Could not find a memory column. "
        "Expected one of: peak_rss_mb, rss_mb, peak_rss_mib, peak_rss_bytes, rss_bytes"
    )


def format_int(x: int) -> str:
    return f"{int(x):,}"


def build_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"rows", "columns", "method"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["rows"] = df["rows"].astype(int)
    df["columns"] = df["columns"].astype(int)
    df["case"] = df.apply(
        lambda r: f"{format_int(r['rows'])} rows\n{format_int(r['columns'])} cols",
        axis=1,
    )
    return df


def preferred_method_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "staniorust_mmap_parallel",
        "staniorust_mmap_1",
        "staniorust_buffered",
        "pandas",
        "polars",
        "stanio",
        "numpy_loadtxt",
    ]
    existing = list(df["method"].dropna().unique())
    ordered = [m for m in preferred if m in existing]
    ordered.extend(sorted([m for m in existing if m not in ordered]))
    return ordered


def case_order(df: pd.DataFrame) -> list[str]:
    tmp = (
        df[["rows", "columns", "case"]]
        .drop_duplicates()
        .sort_values(["rows", "columns"], kind="stable")
    )
    return tmp["case"].tolist()


def style_map(methods: list[str]) -> dict[str, dict[str, object]]:
    colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
    ]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    styles = {}
    for i, method in enumerate(methods):
        styles[method] = {
            "color": colors[i % len(colors)],
            "linestyle": linestyles[(i // len(colors)) % len(linestyles)],
            "marker": markers[i % len(markers)],
        }
    return styles


def plot_metric_by_case(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
    logy: bool = True,
) -> None:
    methods = preferred_method_order(df)
    cases = case_order(df)
    styles = style_map(methods)

    pivot = df.pivot_table(
        index="case",
        columns="method",
        values=metric_col,
        aggfunc="median",
    )
    pivot = pivot.reindex(index=cases)

    fig, ax = plt.subplots(figsize=(max(12, len(cases) * 0.6), 7))

    for method in methods:
        if method not in pivot.columns:
            continue
        y = pivot[method]
        ax.plot(
            range(len(pivot.index)),
            y.values,
            label=method,
            linewidth=2.2,
            markersize=7,
            **styles[method],
        )

    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Benchmark case")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_vs_baseline(
    df: pd.DataFrame,
    time_col: str,
    baseline: str,
    filename: str,
) -> None:
    methods = preferred_method_order(df)
    cases = case_order(df)
    styles = style_map(methods)

    pivot = df.pivot_table(
        index="case",
        columns="method",
        values=time_col,
        aggfunc="median",
    )
    pivot = pivot.reindex(index=cases)

    if baseline not in pivot.columns:
        raise KeyError(f"Baseline method '{baseline}' not found. Available: {list(pivot.columns)}")

    baseline_values = pivot[baseline]

    fig, ax = plt.subplots(figsize=(max(12, len(cases) * 0.6), 7))

    for method in methods:
        if method not in pivot.columns or method == baseline:
            continue
        speedup = baseline_values / pivot[method]
        ax.plot(
            range(len(pivot.index)),
            speedup.values,
            label=f"{method} vs {baseline}",
            linewidth=2.2,
            markersize=7,
            **styles[method],
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Speedup (baseline time / method time)")
    ax.set_xlabel("Benchmark case")
    ax.set_title(f"Speedup relative to {baseline}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_vs_rows_per_column_count(
    df: pd.DataFrame,
    time_col: str,
    filename: str,
) -> None:
    """
    Optional extra plot:
    x = rows, y = time, one line = method+column-count.
    Useful when you want the original style but combined.
    """
    methods = preferred_method_order(df)
    styles = style_map(methods)
    col_counts = sorted(df["columns"].unique())

    fig, ax = plt.subplots(figsize=(12, 7))

    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for _m_idx, method in enumerate(methods):
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue

        for c_idx, col_count in enumerate(col_counts):
            subset = method_df[method_df["columns"] == col_count].sort_values("rows")
            if subset.empty:
                continue

            ax.plot(
                subset["rows"],
                subset[time_col],
                label=f"{method} | {format_int(col_count)} cols",
                color=styles[method]["color"],
                linestyle=linestyle_cycle[c_idx % len(linestyle_cycle)],
                marker=marker_cycle[c_idx % len(marker_cycle)],
                linewidth=2.0,
                markersize=6,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Reading time (seconds)")
    ax.set_title("Combined benchmark: all methods and column counts")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_results(RESULTS_PATH)
    df = build_case_columns(df)

    time_col = find_metric_column(
        df,
        ["parse_time_s", "elapsed_s", "time_s", "duration_s"],
    )
    df, mem_col = ensure_memory_mb(df)

    plot_metric_by_case(
        df=df,
        metric_col=time_col,
        ylabel="Reading time (seconds)",
        title="Benchmark comparison by case",
        filename="combined_time_by_case.png",
        logy=True,
    )

    plot_metric_by_case(
        df=df,
        metric_col=mem_col,
        ylabel="Peak RSS (MB)",
        title="Peak memory comparison by case",
        filename="combined_memory_by_case.png",
        logy=True,
    )

    plot_speedup_vs_baseline(
        df=df,
        time_col=time_col,
        baseline="pandas",
        filename="combined_speedup_vs_pandas.png",
    )

    plot_time_vs_rows_per_column_count(
        df=df,
        time_col=time_col,
        filename="combined_time_loglog_all_methods.png",
    )

    print(f"Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
