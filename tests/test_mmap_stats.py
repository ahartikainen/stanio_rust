"""Native integration tests for mmap parallelism, indexing and online statistics."""

import numpy as np
import pytest

pytest.importorskip("staniorust._native", exc_type=ImportError)

from staniorust import open_csv, read_csv, summarize_csv  # noqa: E402


def test_parallel_matches_buffered(csv_files):
    _, one = read_csv(csv_files, engine="mmap", threads=1, include_warmup=True)
    _, many = read_csv(csv_files, engine="mmap", threads=4, include_warmup=True)
    _, buffered = read_csv(csv_files, engine="buffered", include_warmup=True)
    np.testing.assert_array_equal(one, many)
    np.testing.assert_array_equal(one, buffered)


def test_indexed_random_slice(csv_files):
    reader = open_csv(csv_files[0])
    assert reader.nrows() == 2
    assert reader.nrows(include_warmup=True) == 3
    assert reader.metadata["num_warmup"] == "1"
    header, part = reader.read(rows=slice(1, 2), columns=["alpha"])
    assert header == "alpha"
    assert part.shape == (1, 1)
    np.testing.assert_allclose(part[:, 0], [2])
    assert len(reader.partitions(chunksize=1)) == 2
    assert len(reader.partitions(chunksize=1, include_warmup=True)) == 3


def test_slice_and_chain_row_order(csv_files):
    _, all_rows = read_csv(csv_files, rows=slice(0, 2), columns=["alpha"])
    _, last = read_csv(csv_files, rows=slice(1, 2), columns=["alpha"])
    np.testing.assert_array_equal(last[:, 0, 0], all_rows[:, 1, 0])


def test_online_statistics_against_numpy(csv_files):
    _, array = read_csv(csv_files, columns=["alpha", "beta"])
    from_native = summarize_csv(csv_files, columns=["alpha", "beta"], threads=2)
    assert from_native["column"].tolist() == ["alpha", "beta.1", "beta.2"]
    np.testing.assert_allclose(from_native["mean"], array.mean(axis=(0, 1)))
    np.testing.assert_allclose(from_native["variance"], array.reshape(-1, 3).var(axis=0, ddof=1))
    assert from_native["count"].tolist() == [4, 4, 4]


def test_quantiles_and_nonfinite(tmp_path):
    path = tmp_path / "quant.csv"
    path.write_text(
        "# method = sample\n# num_warmup = 0\ntheta\n"
        + "".join(f"{i}\n" for i in range(1, 1001))
        + "NaN\ninf\n",
        encoding="ascii",
    )
    result = summarize_csv(path, quantiles=[0.0, 0.5, 1.0])
    row = result.iloc[0]
    assert row["count"] == 1000
    assert row["nonfinite"] == 2
    assert row["min"] == 1.0
    assert row["max"] == 1000.0
    assert abs(row["q0.5"] - 500.5) < 15.0  # P² is approximate.


def test_quantiles_multiple_chains(csv_files):
    result = summarize_csv(csv_files, columns=["alpha"], quantiles=[0.5])
    assert result.loc[0, "count"] == 4
    assert abs(result.loc[0, "q0.5"] - 2.0) < 1.0
