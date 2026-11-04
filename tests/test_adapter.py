"""Test the public Python wrappers without requiring a compiled Rust extension."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    """Provide a tiny native stand-in and import the actual package API."""
    root = Path(__file__).resolve().parents[1] / "python" / "staniorust"
    package = ModuleType("staniorust")
    package.__path__ = [str(root)]
    native = ModuleType("staniorust._native")

    class FakeReader:
        def __init__(self, path):
            self.header = ["theta", "beta.1", "beta.2"]
            self.metadata = [("method", "sample")]

        def nrows(self, include_warmup):
            return 2

        def partitions(self, warmup, rows, size):
            return [(0, 100, 0, 0, 2, False)]

        def read(self, columns, warmup, start, stop, threads):
            names = self.header if columns is None else ["theta"]
            return names, np.array([[1.0], [2.0]])[start:stop]

        def summarize(self, columns, warmup, threads, quantiles):
            return {
                "column": ["theta"],
                "count": [2],
                "nonfinite": [0],
                "mean": np.array([1.5]),
                "variance": np.array([0.5]),
                "min": np.array([1.0]),
                "max": np.array([2.0]),
            }

    class FakeStream:
        def __init__(self, path, columns, warmup):
            self.header = ["theta", "beta.1", "beta.2"]
            self.columns = ["theta"]
            self.done = False

        def next_chunk(self, chunksize, columnar):
            if self.done:
                return None
            self.done = True
            values = {"theta": np.array([1.0, 2.0])} if columnar else np.array([[1.0], [2.0]])
            return values, 0, False

    native.MappedCsv = FakeReader
    native.CsvStream = FakeStream
    native.mmap_read_all = lambda *args: (["theta"], np.array([[[1.0], [2.0]]]))
    native.mmap_read_partition = lambda *args: {"theta": np.array([1.0, 2.0])}
    native.mmap_summarize = lambda *args: FakeReader("x").summarize(None, False, 1, None)
    native.reshape = lambda names, values: {"theta": values[..., 0]}
    native.read_json = lambda path, arrays, keys: {"N": 2}

    monkeypatch.setitem(sys.modules, "staniorust", package)
    monkeypatch.setitem(sys.modules, "staniorust._native", native)
    monkeypatch.delitem(sys.modules, "staniorust._api", raising=False)
    return importlib.import_module("staniorust._api")


def test_python_api(adapter):
    header, data = adapter.read_csv("run.csv", columns=["theta"])
    assert header == "theta"
    assert data.shape == (1, 2, 1)
    assert adapter.read_csv("run.csv", reshape=True)["theta"].shape == (1, 2)
    with pytest.raises(ValueError, match="requires output='numpy'"):
        adapter.read_csv("run.csv", output="pandas", reshape=True)
    assert adapter.read_json("in.json") == {"N": 2}


def test_random_access_and_summary(adapter):
    reader = adapter.open_csv("run.csv")
    assert reader.nrows() == 2
    assert reader.metadata["method"] == "sample"
    assert reader.read(rows=slice(0, 1), columns=["theta"])[1].shape == (1, 1)
    stats = adapter.summarize_csv("run.csv", columns=["theta"], output="dict")
    np.testing.assert_allclose(stats["mean"], [1.5])
    with pytest.raises(ValueError, match="step 1"):
        adapter.read_csv("run.csv", rows=slice(0, 3, 2))


def test_streaming_and_frame_adapter(adapter):
    chunk = next(adapter.iter_csv("run.csv", output="numpy", columns=["theta"]))
    assert chunk.chain == 1
    assert chunk.data.shape == (2, 1)
    assert chunk.draw_start == 0
    assert not chunk.warmup

    pytest.importorskip("pandas")
    frame = adapter.read_csv("run.csv", output="pandas", columns=["theta"])
    assert frame.columns.to_list() == ["chain__", "draw__", "warmup__", "theta"]
    assert frame["theta"].to_list() == [1.0, 2.0]
    assert frame["chain__"].to_list() == [1, 1]


def test_buffered(adapter):
    header, value = adapter.read_csv("run.csv", output="numpy", engine="buffered")
    assert header == "theta"
    assert value.shape == (1, 2, 1)
