"""Integration tests for Rust serde_json conversion."""

import numpy as np
import pytest

pytest.importorskip("staniorust._native", exc_type=ImportError)

from staniorust import read_json  # noqa: E402


def test_read_json_preserves_values(json_file):
    values = read_json(json_file)
    assert values["N"] == 3
    assert values["is_ready"] is True
    assert values["x"] == [[1, 2], [3, 4]]


def test_read_json_numeric_arrays(json_file):
    values = read_json(json_file, arrays="numpy", keys=["N", "x"])
    assert set(values) == {"N", "x"}
    assert values["N"] == 3
    np.testing.assert_array_equal(values["x"], [[1, 2], [3, 4]])
    assert values["x"].dtype == np.float64


def test_missing_key(json_file):
    with pytest.raises(ValueError, match="unknown JSON key"):
        read_json(json_file, keys=["missing"])
