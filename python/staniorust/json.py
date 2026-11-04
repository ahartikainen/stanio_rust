"""StanIO-compatible JSON serialization for CmdStan model inputs.

Writing accepts Python numeric types, NumPy, pandas, xarray, complex values,
and nested Stan tuples. Reading large JSON inputs uses the Rust API elsewhere.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np


def process_dictionary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a mapping without mutating its original contents."""
    return {key: process_value(value) for key, value in data.items()}


def process_value(value: Any) -> Any:
    """Convert Python/NumPy containers to Stan's JSON conventions."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, Mapping):
        return process_dictionary(value)
    if isinstance(value, tuple):
        return {str(i): process_value(part) for i, part in enumerate(value, 1)}
    if isinstance(value, list):
        return [process_value(part) for part in value]

    module = getattr(type(value), "__module__", "")
    if module.startswith(("numpy", "pandas", "xarray")):
        array = np.asanyarray(value)
        if array.dtype.kind in "iuf":
            return array.tolist()
        if array.dtype.kind == "c":
            # np.asarray is important: np.matrix otherwise changes the
            # broadcasting and output dimensions of the two-component stack.
            dense = np.asarray(array)
            return np.stack([dense.real, dense.imag], axis=-1).tolist()
        if array.dtype.kind == "b":
            return array.astype(np.int64).tolist()
        return process_value(array.tolist())
    return value


def dump_stan_json(data: Mapping[str, Any]) -> str:
    """Serialize a mapping as CmdStan-compatible JSON text."""
    return json.dumps(process_dictionary(data), allow_nan=True)


def write_stan_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Write CmdStan-compatible JSON to a UTF-8 text file."""
    with open(path, "w", encoding="utf-8") as stream:
        # Single source of truth so write and dump always produce identical text.
        stream.write(dump_stan_json(data))


__all__ = ["process_dictionary", "process_value", "dump_stan_json", "write_stan_json"]
