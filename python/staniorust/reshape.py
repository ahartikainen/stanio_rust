"""StanIO-compatible parameter metadata and structured-array extraction.

CSV parsing and ordinary numeric reshaping still run in Rust. This module
provides the public StanIO 0.5.1 parameter API, including complex values and
recursive tuple types, for arrays that are already resident in NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any

import numpy as np


class VariableType(Enum):
    """Type of a flattened Stan CSV parameter."""

    SCALAR = 1
    COMPLEX = 2
    TUPLE = 3


@dataclass
class Variable:
    """Parameter description with offsets relative to its parent container."""

    name: str
    start_idx: int
    end_idx: int
    dimensions: tuple[int, ...]
    type: VariableType
    contents: list[Variable]

    def dtype(self, top: bool = True) -> np.dtype[Any]:
        """The NumPy dtype for a parameter, optionally including array shape."""
        if self.type is VariableType.SCALAR:
            dtype = np.dtype(np.float64)
        elif self.type is VariableType.COMPLEX:
            dtype = np.dtype(np.complex128)
        else:
            dtype = np.dtype(
                [(str(i), child.dtype(top=False)) for i, child in enumerate(self.contents, 1)]
            )
        return dtype if top else np.dtype((dtype, self.dimensions))

    def columns(self) -> range:
        """Flat column positions in the containing output array."""
        return range(self.start_idx, self.end_idx)

    def num_elts(self) -> int:
        """Number of top-level array elements."""
        return prod(self.dimensions)

    def elt_size(self) -> int:
        """Width, in CSV columns, of this parameter."""
        return self.end_idx - self.start_idx

    def _extract_helper(self, source: np.ndarray, offset: int = 0) -> np.ndarray:
        begin = offset + self.start_idx
        end = offset + self.end_idx
        if self.type is VariableType.SCALAR:
            return source[..., begin:end].reshape(-1, *self.dimensions, order="F")
        if self.type is VariableType.COMPLEX:
            interleaved = source[..., begin:end].reshape(-1, 2, *self.dimensions, order="F")
            return interleaved[:, 0, ...] + 1j * interleaved[:, 1, ...]

        elements = self.num_elts()
        width = self.elt_size()
        if not elements or width % elements:
            raise ValueError(f"invalid tuple dimensions for {self.name!r}")
        block = width // elements
        rows = prod(source.shape[:-1])
        values = np.empty((rows, elements), dtype=object)
        for cell in range(elements):
            children = [
                child._extract_helper(source, begin + cell * block) for child in self.contents
            ]
            for row in range(rows):
                values[row, cell] = tuple(child[row] for child in children)
        return values.reshape(-1, *self.dimensions, order="F")

    def extract_reshape(self, src: np.ndarray, object: bool = True) -> np.ndarray:
        """Reconstruct the Stan parameter while preserving leading draw axes.

        When ``object=False``, tuples use nested NumPy structured dtypes.
        All Stan array indices follow column-major (Fortran) ordering.
        """
        source = np.asarray(src)
        if source.ndim == 0:
            raise ValueError("source must have a trailing flattened-column axis")
        if source.shape[-1] < self.end_idx:
            raise ValueError("parameter exceeds the flattened source array width")
        result = self._extract_helper(source)
        if not object:
            result = result.astype(self.dtype())
        if source.ndim > 1:
            return result.reshape(*source.shape[:-1], *self.dimensions, order="F")
        return result.squeeze(axis=0)


def _base_name(column: str) -> str:
    """Return the Stan parameter name before array indices or tuple slots."""
    return column.split(":", 1)[0].split(".", 1)[0]


def _parameters(columns: list[str]) -> list[Variable]:
    """Recursively parse contiguous StanIO CSV name groups."""
    parsed: list[Variable] = []
    start = 0
    while start < len(columns):
        name = _base_name(columns[start])
        stop = start + 1
        while stop < len(columns) and _base_name(columns[stop]) == name:
            stop += 1
        group = columns[start:stop]
        ending = group[-1]
        prefix = ending.split(":", 1)[0].split(".")[1:]
        if any(":" in column for column in group):
            dims = tuple(int(index) for index in prefix if index)
            kind = VariableType.TUPLE
            if not all(":" in column for column in group):
                raise ValueError(f"mixed tuple and scalar columns for {name!r}")
            nested = list(dict.fromkeys(f"field_{col.split(':', 1)[1]}" for col in group))
            children = _parameters(nested)
        elif any(column.endswith((".real", ".imag")) for column in group):
            kind = VariableType.COMPLEX
            dims = tuple(int(index) for index in prefix[:-1])
            children = []
        else:
            kind = VariableType.SCALAR
            dims = tuple(int(index) for index in prefix if index)
            children = []
        parsed.append(Variable(name, start, stop, dims, kind, children))
        start = stop
    return parsed


def parse_header(header: str) -> dict[str, Variable]:
    """Parse a comma-separated flattened Stan CSV header into variable specs."""
    if not header.strip():
        return {}
    columns = [name.strip() for name in header.strip().split(",")]
    if any(not name for name in columns):
        raise ValueError("Stan CSV header contains an empty column name")
    variables = _parameters(columns)
    if len({var.name for var in variables}) != len(variables):
        raise ValueError("Stan CSV variable columns must be grouped contiguously")
    return {var.name: var for var in variables}


def stan_variables(
    parameters: dict[str, Variable],
    source: np.ndarray,
    *,
    object: bool = True,
) -> dict[str, np.ndarray]:
    """Return Stan parameters reconstructed from an existing NumPy array."""
    return {
        key: parameter.extract_reshape(source, object=object)
        for key, parameter in parameters.items()
    }


__all__ = ["Variable", "VariableType", "parse_header", "stan_variables"]
