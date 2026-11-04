"""Minimal StanIO-compatible CSV reading interface.

The full :func:`staniorust.read_csv` includes projection, streaming, and other
features; this module preserves StanIO's basic headers and error messages.
"""

from __future__ import annotations

from os import PathLike

import numpy as np


def read_csv(filenames: str | PathLike[str] | list[str | PathLike[str]]) -> tuple[str, np.ndarray]:
    """Read Stan CSV files and stack equal-sized chains along axis zero."""
    from ._api import read_csv as read_native

    try:
        return read_native(filenames)
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
