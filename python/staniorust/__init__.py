"""Fast Rust-backed CmdStan CSV / Stan JSON readers and streaming statistics."""

from ._api import (
    StanChunk,
    StanReader,
    iter_csv,
    open_csv,
    read_csv,
    read_csv_dask,
    read_json,
    summarize_csv,
)

__all__ = [
    "StanChunk",
    "StanReader",
    "iter_csv",
    "open_csv",
    "read_csv",
    "read_csv_dask",
    "read_json",
    "summarize_csv",
]
__version__ = "0.2.0"
