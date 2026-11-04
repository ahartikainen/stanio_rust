"""Small standalone regression cases modeled on StanIO's public contract.

The full official tests can be run unchanged via scripts/run_upstream_stanio_tests.py.
These tests do not require a compiled native extension.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from staniorust.json import dump_stan_json, write_stan_json
from staniorust.reshape import VariableType, parse_header, stan_variables


def test_stanio_csv_header_shapes_and_values():
    columns = "lp__,mu,v.1,v.2,m.1.1,m.2.1,m.1.2,m.2.2"
    values = np.array([[0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0]])
    out = stan_variables(parse_header(columns), values[None, :, :])
    assert out["lp__"].shape == (1, 1)
    assert out["v"].shape == (1, 1, 2)
    assert out["m"].shape == (1, 1, 2, 2)
    np.testing.assert_equal(out["m"][0, 0], [[10.0, 14.0], [12.0, 16.0]])


def test_stanio_complex_scalar_vector_and_matrix():
    names = (
        "z.real,z.imag,zv.1.real,zv.1.imag,zv.2.real,zv.2.imag,"
        "zm.1.1.real,zm.1.1.imag,zm.2.1.real,zm.2.1.imag,"
        "zm.1.2.real,zm.1.2.imag,zm.2.2.real,zm.2.2.imag"
    )
    values = np.arange(1, 15, dtype=np.float64).reshape(1, -1)
    params = parse_header(names)
    assert params["z"].type is VariableType.COMPLEX
    out = stan_variables(params, values)
    assert out["z"].shape == (1,)
    assert out["z"][0] == 1 + 2j
    np.testing.assert_equal(out["zv"][0], [3 + 4j, 5 + 6j])
    np.testing.assert_equal(out["zm"][0], [[7 + 8j, 11 + 12j], [9 + 10j, 13 + 14j]])
    assert out["zv"].dtype == np.dtype("complex128")


@pytest.mark.parametrize("object_mode", [True, False])
def test_stanio_recursive_tuple_and_array_of_tuples(object_mode):
    names = (
        "pair:1,pair:2,nested:1,nested:2:1,nested:2:2.real,nested:2:2.imag,"
        "arr.1:1,arr.1:2,arr.2:1,arr.2:2"
    )
    src = np.arange(1, 11, dtype=np.float64).reshape(1, -1)
    parsed = parse_header(names)
    data = stan_variables(parsed, src, object=object_mode)
    assert data["pair"].shape == (1,)
    assert data["arr"].shape == (1, 2)
    if object_mode:
        assert data["pair"][0] == (1.0, 2.0)
        assert data["nested"][0] == (3.0, (4.0, 5.0 + 6.0j))
        assert data["arr"][0, 0] == (7.0, 8.0)
    else:
        assert data["pair"].dtype == np.dtype([("1", "f8"), ("2", "f8")])
        assert data["nested"].dtype == np.dtype([("1", "f8"), ("2", [("1", "f8"), ("2", "c16")])])
        assert not data["nested"].dtype.hasobject
        assert data["pair"][0].tolist() == (1.0, 2.0)


def test_stanio_preserves_singleton_dimensions_and_one_row():
    names = "lp__,A.1.1.1,A.2.1.1,A.1.1.2,A.2.1.2,dummy.1.1.1"
    values = np.arange(6, dtype=np.float64)
    params = parse_header(names)
    assert params["A"].num_elts() == 4
    assert len(params["A"].columns()) == 4
    assert params["A"].extract_reshape(values).shape == (2, 1, 2)
    assert params["A"].extract_reshape(values[None, :]).shape == (1, 2, 1, 2)
    assert params["dummy"].extract_reshape(values).shape == (1, 1, 1)
    assert params["lp__"].extract_reshape(values).shape == ()


def test_stanio_tuple_array_of_matrices_structured_dtype():
    # Each tuple has a scalar and a 2x2 real matrix.
    header = "t.1:1,t.1:2.1.1,t.1:2.2.1,t.1:2.1.2,t.1:2.2.2"
    source = np.arange(1, 6, dtype=float)
    parsed = parse_header(header)
    out = parsed["t"].extract_reshape(source, object=False)
    assert out.shape == (1,)
    assert out.dtype == np.dtype([("1", "f8"), ("2", "(2,2)f8")])
    np.testing.assert_equal(out[0][1], [[2, 4], [3, 5]])


def test_stan_json_bool_complex_tuples_arrays_and_special_values(tmp_path):
    data = {
        "false": False,
        "true": np.bool_(True),
        "vector": np.arange(3, dtype=np.float64),
        "matrix": np.zeros((2, 0)),
        "complex": np.array([1 + 2j, 3 + 4j]),
        "tuple": (1, (2, 3 + 4j)),
        "none": None,
        "nan": float("nan"),
        "inf": [float("inf"), -float("inf")],
        "structured": np.array([(1, 2.0)], dtype="i4,f8"),
    }
    encoded = dump_stan_json(data)
    assert "NaN" in encoded
    assert "Infinity" in encoded
    parsed = json.loads(encoded)
    assert parsed["false"] == 0
    assert parsed["true"] == 1
    assert parsed["complex"] == [[1.0, 2.0], [3.0, 4.0]]
    assert parsed["tuple"] == {"1": 1, "2": {"1": 2, "2": [3.0, 4.0]}}
    assert parsed["structured"] == [{"1": 1, "2": 2.0}]
    assert parsed["matrix"] == [[], []]
    path = tmp_path / "input.json"
    write_stan_json(path, data)
    assert path.read_text(encoding="utf-8") == encoded


def test_stan_json_numpy_matrix_complex_and_pandas_frame(tmp_path):
    raw = np.arange(8, dtype=np.float64).reshape(2, 2, 2) / 10
    cmatrix = np.matrix(raw[..., 0] + 1j * raw[..., 1])
    assert json.loads(dump_stan_json({"z": cmatrix}))["z"] == raw.tolist()
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"foo": [1.0, 2.0], "bar": [3.0, 4.0]})
    path = tmp_path / "frame.json"
    write_stan_json(path, frame)
    assert json.loads(path.read_text())["foo"] == [1.0, 2.0]
    assert json.loads(path.read_text())["bar"] == [3.0, 4.0]


def test_stanio_4d_complex_array_fortran_order():
    """Complex values are interleaved and the first Stan index varies fastest."""
    dims = (2, 2, 2, 2)
    names = []
    flat = []
    for idx in np.ndindex(dims[::-1]):
        subscript = tuple(reversed(idx))
        suffix = ".".join(str(i + 1) for i in subscript)
        value = float(np.ravel_multi_index(subscript, dims, order="F") + 1)
        names.extend([f"z.{suffix}.real", f"z.{suffix}.imag"])
        flat.extend([value, value * 10])
    result = stan_variables(parse_header(",".join(names)), np.array(flat)[None, None, :])["z"]
    assert result.shape == (1, 1, *dims)
    for subscript in np.ndindex(dims):
        value = float(np.ravel_multi_index(subscript, dims, order="F") + 1)
        assert result[(0, 0, *subscript)] == value + 10j * value


@pytest.mark.parametrize("object_mode", [True, False])
def test_stanio_ultimate_tuple_shape_and_dtype(object_mode):
    """Match StanIO's deep tuple of vectors and matrices, including array dims."""
    names = []
    flat = []
    for outer_j in (1, 2, 3):
        for outer_i in (1, 2):
            stem = f"ultimate.{outer_i}.{outer_j}"
            for array_index in (1, 2):
                names.append(f"{stem}:1.{array_index}:1")
                flat.append(1.0)
                for inner_index in (1, 2):
                    names.append(f"{stem}:1.{array_index}:2.{inner_index}")
                    flat.append(float(inner_index + 1))
            for matrix_j in (1, 2, 3, 4, 5):
                for matrix_i in (1, 2, 3, 4):
                    names.append(f"{stem}:2.{matrix_i}.{matrix_j}")
                    flat.append(float(matrix_i + matrix_j))

    values = np.array(flat, dtype=np.float64)[None, None, :]
    result = stan_variables(parse_header(",".join(names)), values, object=object_mode)["ultimate"]
    assert result.shape == (1, 1, 2, 3)
    expected_type = np.dtype(
        [
            ("1", ([("1", "f8"), ("2", "(2,)f8")], (2,))),
            ("2", "(4,5)f8"),
        ]
    )
    if object_mode:
        first = result[0, 0, 0, 0]
        assert isinstance(first, tuple)
        assert first[0].shape == (2,)
        np.testing.assert_array_equal(first[0][0][1], [2.0, 3.0])
        assert first[1].shape == (4, 5)
    else:
        assert result.dtype == expected_type
        assert not result.dtype.hasobject
        assert result[0, 0, 0, 0][0].shape == (2,)
        assert result[0, 0, 0, 0][1].shape == (4, 5)


def test_stanio_special_json_float_tokens():
    data = {
        "a": np.array(
            [
                [-np.inf, np.inf, np.nan],
                [-float("inf"), float("inf"), float("nan")],
                [np.float32(-np.inf), np.float32(np.inf), np.float32(np.nan)],
                [1e200 * -1e200, 1e220 * 1e200, -np.nan],
            ]
        )
    }
    result = dump_stan_json(data)
    assert result.count("Infinity") == 8
    assert result.count("-Infinity") == 4
    assert result.count("NaN") == 4


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CSV headers differ: chain.csv", "Headers do not match"),
        ("chains have unequal selected draw counts: 10 vs 9", "must have the same shape"),
        ("chains have unequal draw counts after selection: 10 vs 9", "must have the same shape"),
        ("CSV record has 7 fields; expected 8", "must have the same shape"),
        ("expected 8 numeric values on line 50, got 7", "must have the same shape"),
    ],
)
def test_stanio_csv_compat_error_mapping(monkeypatch, message, expected):
    import sys
    from types import ModuleType

    from staniorust.csv import read_csv

    fake_api = ModuleType("staniorust._api")

    def fake_read_csv(_paths):
        raise ValueError(message)

    fake_api.read_csv = fake_read_csv
    monkeypatch.setitem(sys.modules, "staniorust._api", fake_api)
    with pytest.raises(ValueError, match=expected):
        read_csv("chain.csv")


@pytest.mark.parametrize(
    ("native_shape", "stanio_shape"),
    [
        ((1, 1, 9), (1, 9)),
        ((2, 1, 9), (2, 9)),
        ((1, 1000, 9), (1, 1000, 9)),
        ((2, 1000, 9), (2, 1000, 9)),
        ((1, 1000, 1), (1, 1000)),
        ((1, 1, 1), (1,)),
        ((3, 0, 9), (3, 0)),
    ],
)
def test_stanio_csv_numpy_loadtxt_shape_compatibility(monkeypatch, native_shape, stanio_shape):
    """The compatibility reader matches np.stack(np.loadtxt(...)), not the 3D API."""
    import sys
    from types import ModuleType

    from staniorust.csv import read_csv

    values = np.arange(np.prod(native_shape), dtype=np.float64).reshape(native_shape)
    fake_api = ModuleType("staniorust._api")

    def fake_read_csv(_paths):
        return "a,b", values

    fake_api.read_csv = fake_read_csv
    monkeypatch.setitem(sys.modules, "staniorust._api", fake_api)

    header, actual = read_csv("one_chain.csv")
    assert header == "a,b"
    assert actual.shape == stanio_shape
    np.testing.assert_array_equal(actual.ravel(), values.ravel())


def test_stanio_preserve_1d_single_optimization_row_from_csv(monkeypatch):
    """Regression for upstream test_reshape.py::test_preserve_1d."""
    import sys
    from types import ModuleType

    from staniorust.csv import read_csv

    names = (
        "A.1.1.1,A.2.1.1,A.1.1.2,A.2.1.2,A.1.1.3,A.2.1.3,"
        "dummy.1.1.1,dummy_tuple.1:1,dummy_tuple.1:2"
    )
    values = np.arange(9, dtype=np.float64).reshape(1, 1, -1)
    fake_api = ModuleType("staniorust._api")

    def fake_read_csv(_paths):
        return names, values

    fake_api.read_csv = fake_read_csv
    monkeypatch.setitem(sys.modules, "staniorust._api", fake_api)

    header, draws = read_csv("oned_opt.csv")
    params = parse_header(header)
    assert draws.shape == (1, 9)
    assert params["A"].extract_reshape(draws).shape == (1, 2, 1, 3)
    assert params["dummy"].extract_reshape(draws).shape == (1, 1, 1, 1)
    assert params["dummy_tuple"].extract_reshape(draws).shape == (1, 1)
    assert params["A"].extract_reshape(draws.squeeze()).shape == (2, 1, 3)
    assert params["dummy"].extract_reshape(draws.squeeze()).shape == (1, 1, 1)
    assert params["dummy_tuple"].extract_reshape(draws.squeeze()).shape == (1,)
