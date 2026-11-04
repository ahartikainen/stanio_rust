"""Benchmark planning and synthetic fixtures require no compiled Rust module."""

from benchmarks.run import Case, container_labels, plan, shape_for, write_fixture


def test_nd_flat_shapes_and_stan_fortran_index_order():
    assert shape_for(100_000) == (10, 10, 10, 100)
    assert len(container_labels(shape_for(100_000))) == 100_000
    assert container_labels((2, 3))[:4] == [
        "theta.1.1",
        "theta.2.1",
        "theta.1.2",
        "theta.2.2",
    ]


def test_resource_planner_does_not_allocate_impossible_matrix():
    good, skipped = plan(
        [1, 10_000_000],
        [1, 100_000],
        ram_bytes=16 * 2**30,
        disk_bytes=100 * 2**30,
        max_fixture_bytes=256 * 2**20,
        ram_fraction=0.2,
        disk_fraction=0.1,
    )
    assert any(item.rows == 1 and item.columns == 100_000 for item in good)
    assert any(item["rows"] == 10_000_000 and item["columns"] == 100_000 for item in skipped)


def test_fixture_counts_and_contents(tmp_path):
    case = Case(
        rows=3,
        columns=4,
        shape=(2, 2),
        estimated_csv_bytes=1024,
        output_bytes=96,
        label="wide+long",
    )
    filename = tmp_path / "case.csv"
    size = write_fixture(filename, case)
    content = filename.read_text().splitlines()
    assert len(content) == 4 + 1 + 3
    assert content[4].split(",") == ["theta.1.1", "theta.2.1", "theta.1.2", "theta.2.2"]
    assert size == filename.stat().st_size
