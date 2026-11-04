"""Minimal StanIO-compatible CSV reading interface.

The full :func:`staniorust.read_csv` includes projection, streaming, and other
features; this module preserves StanIO's basic headers and error messages.
"""

from __future__ import annotations

from os import PathLike

import numpy as np


def read_csv(filenames: str | PathLike[str] | list[str | PathLike[str]]) -> tuple[str, np.ndarray]:
    """Read CSVs with the original StanIO / NumPy loadtxt shape semantics.

    The enhanced ``staniorust.read_csv`` always returns a 3D NumPy array.
    This compatibility function removes singleton draw/column axes for
    each chain, as ``np.loadtxt`` does before ``np.stack`` in StanIO.
    """
    from ._api import read_csv as read_native

    try:
        header, values = read_native(filenames)
        if values.shape[1] == 0:
            # np.loadtxt returns shape (0,) when there are no data rows.
            return header, values.reshape(values.shape[0], 0)
        singleton_axes = tuple(axis for axis in (1, 2) if values.shape[axis] == 1)
        return header, values.squeeze(axis=singleton_axes)
    except ValueError as exc:
        text = str(exc)
        if "headers differ" in text or "header mismatch" in text:
            raise ValueError("Headers do not match") from exc
        if (
            ("unequal" in text and "draw counts" in text)
            or ("CSV record has" in text and "expected" in text)
            or ("expected" in text and "numeric values" in text)
            or "too many values on line" in text
        ):
            raise ValueError("all input arrays must have the same shape") from exc
        raise


__all__ = ["read_csv"]
