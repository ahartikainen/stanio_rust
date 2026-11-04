"""Fast Rust-backed CmdStan I/O and StanIO-compatible parameter utilities.

Pure-Python compatibility modules are importable without the compiled
extension. The optimized read functions require ``maturin develop --release``.
"""

from __future__ import annotations

from .json import dump_stan_json, write_stan_json
from .reshape import Variable, VariableType, parse_header, stan_variables

_NATIVE_EXPORTS = frozenset(
    {
        "StanChunk",
        "StanReader",
        "iter_csv",
        "open_csv",
        "read_csv",
        "read_csv_dask",
        "read_json",
        "summarize_csv",
    }
)


def __getattr__(name: str):
    """Load native-backed APIs only when used, keeping JSON writers standalone."""
    if name in _NATIVE_EXPORTS:
        from . import _api

        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StanChunk",
    "StanReader",
    "Variable",
    "VariableType",
    "dump_stan_json",
    "iter_csv",
    "open_csv",
    "parse_header",
    "read_csv",
    "read_csv_dask",
    "read_json",
    "stan_variables",
    "summarize_csv",
    "write_stan_json",
]

__version__ = "0.2.1"
