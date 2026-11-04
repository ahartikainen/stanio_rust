"""Cloud-style fsspec memory:// integration tests."""

import numpy as np
import pytest

pytest.importorskip("staniorust._native", exc_type=ImportError)
fsspec = pytest.importorskip("fsspec")

from staniorust import iter_csv, read_csv, read_csv_dask, summarize_csv  # noqa: E402


def test_memory_filesystem_buffered():
    with fsspec.open("memory://staniorust/test.csv", "wb") as file:
        file.write(b"# method = sample\n# num_warmup = 1\n# save_warmup = 1\n"
                   b"x,vec.1,vec.2\n0,1,2\n# Adaptation terminated\n3,4,5\n6,7,8\n")
    header, values = read_csv("memory://staniorust/test.csv")
    assert header == "x,vec.1,vec.2"
    np.testing.assert_array_equal(values[0], [[3, 4, 5], [6, 7, 8]])
    blocks = list(iter_csv("memory://staniorust/test.csv", chunksize=1,
                           include_warmup=True))
    assert [block.warmup for block in blocks] == [True, False, False]
    result = summarize_csv("memory://staniorust/test.csv", columns=["x"], output="dict")
    np.testing.assert_allclose(result["mean"], [4.5])


def test_dask_remote():
    pytest.importorskip("dask.dataframe")
    with fsspec.open("memory://staniorust/dask.csv", "wb") as file:
        file.write(b"# method = sample\na\n1\n2\n3\n")
    frame = read_csv_dask("memory://staniorust/dask.csv", chunksize=1).compute()
    assert frame["a"].to_list() == [1, 2, 3]
