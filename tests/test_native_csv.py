"""Integration tests for Rust -> Python CmdStan CSV decoding."""

import numpy as np
import pytest

pytest.importorskip("staniorust._native", exc_type=ImportError)

from staniorust import iter_csv, read_csv, read_csv_dask  # noqa: E402


def test_numpy_read_and_chain_shape(csv_files):
    header, arr = read_csv(csv_files)
    assert arr.shape == (2, 2, 8)
    assert header.startswith("lp__,alpha,beta.1")
    np.testing.assert_array_equal(arr[:, 0, 1], [1, 2])
    assert arr.dtype == np.float64


def test_stan_reshape_uses_index_coordinates(csv_files):
    result = read_csv(csv_files, columns=["alpha", "beta", "mat"], reshape=True)
    assert result["alpha"].shape == (2, 2)
    assert result["beta"].shape == (2, 2, 2)
    assert result["mat"].shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result["beta"][0, 0], [1, 2])
    np.testing.assert_array_equal(result["mat"][0, 0], [[1, 2], [3, 4]])


def test_column_projection(csv_files):
    header, data = read_csv(csv_files, columns=["alpha"])
    assert header == "alpha"
    assert data.shape == (2, 2, 1)


def test_incomplete_variable_rejected(csv_files):
    with pytest.raises(ValueError, match="incomplete"):
        read_csv(csv_files, columns=["beta.2"], reshape=True)


def test_warmup_and_chunk_transitions(csv_files):
    chunks = list(iter_csv(csv_files[:1], output="numpy", chunksize=5, include_warmup=True))
    assert [(chunk.warmup, len(chunk.data), chunk.draw_start) for chunk in chunks] == [
        (True, 1, 0),
        (False, 2, 0),
    ]
    header, arr = read_csv(csv_files[:1], include_warmup=True)
    assert arr.shape == (1, 3, 8)
    np.testing.assert_array_equal(arr[0, :, 0], [-5, -1, -2])


def test_pandas(csv_files):
    pytest.importorskip("pandas")
    frame = read_csv(csv_files, output="pandas", columns=["alpha", "beta"])
    assert frame.shape == (4, 6)  # metadata + alpha + beta.1 + beta.2
    assert frame["chain__"].to_list() == [1, 1, 2, 2]
    assert frame["draw__"].to_list() == [0, 1, 0, 1]
    assert not frame["warmup__"].any()


def test_polars(csv_files):
    pytest.importorskip("polars")
    frame = read_csv(csv_files, output="polars", columns=["alpha"])
    assert frame.shape == (4, 4)
    assert frame["alpha"].to_list() == [1.0, 2.0, 2.0, 3.0]


def test_dask(csv_files):
    pytest.importorskip("dask.dataframe")
    ddf = read_csv_dask(csv_files, chunksize=1, columns=["alpha"])
    assert ddf.npartitions == 4
    frame = ddf.compute()
    assert frame["alpha"].to_list() == [1.0, 2.0, 2.0, 3.0]
    assert frame["draw__"].to_list() == [0, 1, 0, 1]


def test_missing_column(csv_files):
    with pytest.raises(ValueError, match="unknown Stan"):
        read_csv(csv_files[:1], columns=["not_a_parameter"])


def test_unequal_chains(csv_files):
    csv_files[1].write_text(csv_files[1].read_text() + "-3,2,1,2,1,2,3,4\n")
    with pytest.raises(ValueError, match="unequal draw counts"):
        read_csv(csv_files)


def test_thinning_and_nonfinite_values(tmp_path):
    path = tmp_path / "thin.csv"
    path.write_text(
        "# method = sample\n"
        "# num_warmup = 5\n"
        "# thin = 2\n"
        "# save_warmup = 1\n"
        "lp__,theta\n"
        "-9,1\n"
        "-8,2\n"
        "-7,3\n"
        "# Adaptation terminated\n"
        "inf,NaN\n"
        "-inf,+inf\n",
        encoding="utf-8",
    )
    header, post = read_csv(path)
    assert header == "lp__,theta"
    assert post.shape == (1, 2, 2)
    assert np.isposinf(post[0, 0, 0])
    assert np.isnan(post[0, 0, 1])
    assert np.isneginf(post[0, 1, 0])
    all_chunks = list(iter_csv(path, include_warmup=True, chunksize=10, output="numpy"))
    assert [(c.warmup, len(c.data)) for c in all_chunks] == [(True, 3), (False, 2)]
