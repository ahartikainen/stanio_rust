"""High-level Python interface to the Rust-backed Stan numeric CSV reader."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

try:
    from . import _native
except ImportError as exc:
    raise ImportError(
        "staniorust's compiled Rust extension is missing; run 'maturin develop --release' "
        "in the stanio_rust project directory"
    ) from exc

Output = Literal["numpy", "pandas", "polars"]
Engine = Literal["auto", "mmap", "buffered"]
PathInput = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]


@dataclass(frozen=True, slots=True)
class StanChunk:
    """One contiguous chunk of draws from a single CmdStan chain.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame or polars.DataFrame
        Selected numerical values.
    chain : int
        One-based index into the input file list.
    draw_start : int
        Zero-based draw index within the current warmup/sampling phase.
    warmup : bool
        Whether the chunk contains saved warmup draws.
    columns : tuple[str, ...]
        Original flattened Stan CSV column names in selected order.
    """

    data: Any
    chain: int
    draw_start: int
    warmup: bool
    columns: tuple[str, ...]


def _paths(paths: PathInput) -> list[str]:
    """Normalize one or several path-like arguments."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    result = [os.fsdecode(path) for path in paths]
    if not result:
        raise ValueError("at least one CSV file is required")
    return result


def _selected(columns: Sequence[str] | None) -> list[str] | None:
    """Preserve the supplied selection without splitting string column names."""
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    result = list(columns)
    if not result:
        raise ValueError("columns cannot be empty")
    return result


def _cloud(path: str) -> bool:
    """Identify fsspec-compatible URIs without mistaking Windows drive letters."""
    parts = urlsplit(path)
    return bool(parts.scheme and len(parts.scheme) > 1 and parts.scheme != "file")


def _engine(path: str, engine: Engine) -> Literal["mmap", "buffered"]:
    """Use local immutable files with mmap and cloud files with buffered I/O."""
    if engine not in ("auto", "mmap", "buffered"):
        raise ValueError("engine must be 'auto', 'mmap', or 'buffered'")
    if _cloud(path):
        if engine == "mmap":
            raise ValueError("memory mapping is only possible for local paths")
        return "buffered"
    return "mmap" if engine == "auto" else engine


def _bounds(rows: slice | None) -> tuple[int, int | None]:
    """Translate a nonnegative, step-one row slice into Rust arguments."""
    if rows is None:
        return 0, None
    if not isinstance(rows, slice) or rows.step not in (None, 1):
        raise ValueError("rows must be a slice with step 1 (for example slice(100, 200))")
    start = 0 if rows.start is None else rows.start
    if start < 0 or (rows.stop is not None and rows.stop < 0):
        raise ValueError("negative row offsets are not supported")
    return start, rows.stop


def _max_bytes(limit: int | None) -> int | None:
    """Limit eager allocations by available *free* RAM, not installed RAM."""
    if limit is not None:
        if limit <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        return limit
    try:
        import psutil
    except ImportError:
        return None
    return max(1, int(psutil.virtual_memory().available * 0.65))


def _frame(
    values: dict[str, np.ndarray],
    output: Literal["pandas", "polars"],
    chain: int,
    draw_start: int,
    warmup: bool,
) -> Any:
    """Attach chain and draw metadata to Rust-owned numeric column vectors."""
    reserved = {"chain__", "draw__", "warmup__"}
    if reserved.intersection(values):
        raise ValueError("Stan CSV columns collide with internal metadata column names")
    n = len(next(iter(values.values()))) if values else 0
    data = {
        "chain__": np.full(n, chain, dtype=np.int32),
        "draw__": np.arange(draw_start, draw_start + n, dtype=np.int64),
        "warmup__": np.full(n, warmup),
        **values,
    }
    if output == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("install 'stanio_rust[pandas]' for pandas output") from exc
        return pd.DataFrame(data, copy=False)
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("install 'stanio_rust[polars]' for Polars output") from exc
    return pl.DataFrame(data)


def _frame_for_matrix(
    names: Sequence[str], array: np.ndarray, output: Literal["pandas", "polars"],
    chain: int, draw_start: int, warmup: bool,
) -> Any:
    """Wrap a row-major matrix when no columnar Rust path is available."""
    return _frame({name: array[:, i] for i, name in enumerate(names)}, output, chain, draw_start, warmup)


def _combine_arrays(chunks: Iterator[StanChunk], width: int, limit_bytes: int | None) -> np.ndarray:
    """Assemble a streaming fallback into one eager matrix if it fits RAM."""
    arrays: list[np.ndarray] = []
    total = 0
    for chunk in chunks:
        data = np.asarray(chunk.data)
        total += data.nbytes
        if limit_bytes is not None and total > limit_bytes:
            raise MemoryError("CSV output exceeds free-RAM budget; use iter_csv or summarize_csv")
        arrays.append(data)
    if not arrays:
        return np.empty((0, width), dtype=np.float64)
    return np.concatenate(arrays, axis=0)


def _read_numpy_buffered(
    paths: list[str], columns: list[str] | None, include_warmup: bool,
    rows: slice | None, limit_bytes: int | None,
) -> tuple[str, np.ndarray]:
    """Buffered fallback for cloud or mutable files, reading one chain at a time."""
    names: tuple[str, ...] | None = None
    arrays = []
    start, stop = _bounds(rows)
    for path in paths:
        if _cloud(path):
            from ._remote import _header, _open, _project

            with _open(path) as source:
                full, _, _ = _header(source)
            available, _ = _project(full, columns)
        else:
            available = _native.CsvStream(path, columns, include_warmup).columns
        if names is None:
            names = tuple(available)
        elif names != tuple(available):
            raise ValueError("selected CSV headers differ across chains")
        parts = iter_csv(path, columns=columns, include_warmup=include_warmup,
                         output="numpy", engine="buffered")
        offset = 0
        def selected_parts() -> Iterator[StanChunk]:
            nonlocal names, offset
            for part in parts:
                if names is None:
                    names = part.columns
                elif names != part.columns:
                    raise ValueError("CSV headers differ across input chains")
                length = len(part.data)
                lo = max(0, start - offset)
                hi = length if stop is None else min(length, stop - offset)
                offset += length
                if lo < hi:
                    yield StanChunk(part.data[lo:hi], part.chain, part.draw_start + lo,
                                    part.warmup, part.columns)
                if stop is not None and offset >= stop:
                    break
        arrays.append(_combine_arrays(selected_parts(), len(names), limit_bytes))
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"chains have unequal selected draw counts: {lengths}")
    total_bytes = sum(arr.nbytes for arr in arrays)
    if limit_bytes is not None and total_bytes * 2 > limit_bytes:
        raise MemoryError("stacking chains exceeds free-RAM budget; use iter_csv")
    return ",".join(names), np.stack(arrays, axis=0)


def read_csv(
    paths: PathInput,
    *,
    output: Output = "numpy",
    reshape: bool = False,
    columns: Sequence[str] | None = None,
    include_warmup: bool = False,
    rows: slice | None = None,
    engine: Engine = "auto",
    threads: int = 0,
    memory_limit_bytes: int | None = None,
) -> tuple[str, np.ndarray] | dict[str, np.ndarray] | Any:
    """Read one or several CmdStan numeric CSVs using the Rust extension.

    Parameters
    ----------
    paths : str, path-like, or sequence thereof
        Files in chain order, including cloud URIs through fsspec.
    output : {'numpy', 'pandas', 'polars'}, default 'numpy'
        NumPy returns ``(header_csv, float64 array[chains, draws, columns])``.
        DataFrames contain ``chain__``, ``draw__``, and ``warmup__`` columns.
    reshape : bool, default False
        Reconstruct dense Stan parameter arrays in Rust, returning a dictionary
        of NumPy arrays. Requires ``output='numpy'``.
    columns : sequence of str, optional
        Full Stan parameter names or individual flattened column names.
    include_warmup : bool, default False
        Include warmup draws saved by CmdStan.
    rows : slice, optional
        Zero-based row selection per chain; only step-one slices are supported.
    engine : {'auto', 'mmap', 'buffered'}, default 'auto'
        Auto maps local files and streams remote files through fsspec.
        The mmap path requires completed immutable files.
    threads : int, default 0
        Rayon workers, with zero selecting Rayon defaults. Ignored by the
        sequential buffered reader.
    memory_limit_bytes : int, optional
        Eager result RAM limit. By default 65% of currently free RAM when
        psutil is installed. Specify an explicit limit to override.

    Returns
    -------
    tuple[str, numpy.ndarray], dict[str, numpy.ndarray], or DataFrame
        Results contain float64 values; grouped variables use Stan indices.

    Notes
    -----
    Eager result buffers scale as rows * columns * 8 bytes. Prefer iter_csv,
    summarize_csv or read_csv_dask for large matrices.
    """
    files = _paths(paths)
    selected = _selected(columns)
    _bounds(rows)
    if threads < 0:
        raise ValueError("threads must be >= 0")
    if output not in ("numpy", "pandas", "polars"):
        raise ValueError("output must be numpy, pandas or polars")
    if reshape and output != "numpy":
        raise ValueError("reshape=True requires output='numpy'")
    limits = _max_bytes(memory_limit_bytes)
    if output == "numpy":
        if all(_engine(path, engine) == "mmap" for path in files):
            start, stop = _bounds(rows)
            names, values = _native.mmap_read_all(
                files, selected, include_warmup, threads, start, stop, limits
            )
            if reshape:
                return _native.reshape(names, values)
            return ",".join(names), values
        if any(_engine(path, engine) != "buffered" for path in files):
            raise ValueError("cannot mix mmap and remote buffered paths in one eager read")
        heading, values = _read_numpy_buffered(files, selected, include_warmup, rows, limits)
        if reshape:
            return _native.reshape(heading.split(","), values)
        return heading, values
    from itertools import chain

    frames = (part.data for part in iter_csv(
        files, chunksize=16_384, columns=selected, include_warmup=include_warmup,
        output=output, engine=engine, threads=threads, rows=rows,
    ))
    first = next(frames, None)
    if output == "pandas":
        import pandas as pd

        if first is None:
            names = open_csv(files[0], engine=engine).columns(selected)
            return _frame({name: np.empty(0) for name in names}, "pandas", 1, 0, False)
        return pd.concat(chain([first], frames), ignore_index=True, copy=False)
    import polars as pl

    if first is None:
        names = open_csv(files[0], engine=engine).columns(selected)
        return _frame({name: np.empty(0) for name in names}, "polars", 1, 0, False)
    return pl.concat(chain([first], frames), how="vertical", rechunk=False)


class StanReader:
    """Reusable, indexed reader for a completed local CmdStan CSV.

    Parameters
    ----------
    path : str
        Local CSV file, kept memory-mapped for the reader's lifetime.

    Notes
    -----
    Building a partition index scans newline positions but not numeric values.
    No index file is written. Do not modify or truncate an mmapped CSV.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._reader = _native.MappedCsv(path)

    @property
    def header(self) -> list[str]:
        """All flattened Stan column names."""
        return self._reader.header

    @property
    def metadata(self) -> dict[str, str]:
        """Initial CmdStan configuration comments."""
        return dict(self._reader.metadata)

    def nrows(self, include_warmup: bool = False) -> int:
        """Count retained numeric draws without decoding their values."""
        return self._reader.nrows(include_warmup)

    def columns(self, names: Sequence[str] | None = None) -> list[str]:
        """Resolve full parameter names against the flattened header."""
        return _resolve_names(self.header, names)

    def partitions(
        self, *, chunksize: int = 16_384, blocksize: int = 32 * 1024 * 1024,
        include_warmup: bool = False,
    ) -> list[tuple[int, int, int, int, int, bool]]:
        """Return byte-aligned partitions with row and warmup metadata."""
        return self._reader.partitions(include_warmup, chunksize, blocksize)

    def read(
        self, *, rows: slice | None = None, columns: Sequence[str] | None = None,
        include_warmup: bool = False, threads: int = 0, reshape: bool = False,
    ) -> tuple[str, np.ndarray] | dict[str, np.ndarray]:
        """Read the selected row range into one two-dimensional NumPy array."""
        start, stop = _bounds(rows)
        names, values = self._reader.read(_selected(columns), include_warmup,
                                          start, stop, threads)
        if reshape:
            return _native.reshape(names, values)
        return ",".join(names), values

    def summarize(
        self, *, columns: Sequence[str] | None = None,
        include_warmup: bool = False, threads: int = 0,
        quantiles: Sequence[float] | None = None,
        output: Literal["pandas", "dict"] = "pandas",
    ) -> Any:
        """Compute Rust-side online statistics without keeping draws in RAM."""
        return _summary_output(
            self._reader.summarize(_selected(columns), include_warmup,
                                   threads, list(quantiles) if quantiles is not None else None),
            quantiles, output,
        )


def _resolve_names(header: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    """Resolve individual labels and complete variable names in file order."""
    if requested is None:
        return list(header)
    selected = _selected(requested)
    assert selected is not None
    found = [name for name in header if any(
        name == base or name.startswith((base + ".", base + ":", base + "["))
        for base in selected
    )]
    for base in selected:
        if not any(name == base or name.startswith((base + ".", base + ":", base + "["))
                   for name in header):
            raise ValueError(f"unknown Stan column or variable: {base}")
    return found


def open_csv(path: str | os.PathLike[str], *, engine: Engine = "auto") -> StanReader:
    """Open and cache a local mmap index for repeated random-access queries.

    Parameters
    ----------
    path : str or path-like
        Completed local CmdStan CSV. Remote buffered files use read_csv
        and iter_csv, not a long-lived memory map.
    engine : {'auto', 'mmap', 'buffered'}, default 'auto'
        Only auto/mmap are valid for indexed local readers.

    Returns
    -------
    StanReader
        An indexed Rust-backed reader.
    """
    path_str = os.fsdecode(path)
    if _engine(path_str, engine) != "mmap":
        raise ValueError("open_csv requires a local immutable file; use iter_csv for buffered I/O")
    return StanReader(path_str)


def iter_csv(
    paths: PathInput,
    *,
    chunksize: int = 50_000,
    output: Output = "numpy",
    columns: Sequence[str] | None = None,
    include_warmup: bool = False,
    engine: Engine = "auto",
    threads: int = 0,
    rows: slice | None = None,
) -> Iterator[StanChunk]:
    """Read CmdStan CSV draws in bounded chunks.

    Parameters
    ----------
    paths : str or sequence of paths
        Local filenames or fsspec-compatible cloud URIs.
    chunksize : int, default 50000
        Maximum decoded rows returned per chunk.
    output : {'numpy', 'pandas', 'polars'}, default 'numpy'
        DataFrame output includes chain, draw and warmup metadata.
    columns : sequence of str, optional
        Projection of complete variables or individual flattened columns.
    include_warmup : bool, default False
        Include saved warmup draws.
    engine : {'auto', 'mmap', 'buffered'}, default 'auto'
        Mmap uses Rust row partitions; buffered uses sequential streams.
    threads : int, default 0
        Rust worker count for mmap array slices.
    rows : slice, optional
        Range of retained rows, separately for each chain.

    Yields
    ------
    StanChunk
        Each chunk contains data from exactly one chain and one phase.
    """
    if chunksize <= 0 or threads < 0:
        raise ValueError("chunksize must be positive and threads nonnegative")
    if output not in ("numpy", "pandas", "polars"):
        raise ValueError("output must be numpy, pandas or polars")
    selected = _selected(columns)
    start, stop = _bounds(rows)
    full_header = None
    for chain, path in enumerate(_paths(paths), 1):
        chosen = _engine(path, engine)
        if chosen == "mmap":
            reader = open_csv(path)
            if full_header is None:
                full_header = reader.header
            elif reader.header != full_header:
                raise ValueError(f"CSV headers differ across chains: {path}")
            names = tuple(reader.columns(selected))
            source_indices = [reader.header.index(name) for name in names]
            max_rows = max(1, min(chunksize, (32 * 1024 * 1024) // (8 * max(1, len(names)))))
            for begin, end, row_start, draw_start, count, warmup in reader.partitions(
                chunksize=max_rows, include_warmup=include_warmup,
            ):
                left = max(start, row_start)
                right = min(row_start + count, stop if stop is not None else row_start + count)
                if left >= right:
                    continue
                if output == "numpy" or left != row_start or right != row_start + count:
                    _, values = reader.read(
                        rows=slice(left, right), columns=selected,
                        include_warmup=include_warmup, threads=threads,
                    )
                    data = values if output == "numpy" else _frame_for_matrix(
                        names, values, output, chain, draw_start + left - row_start, warmup,
                    )
                else:
                    values = _native.mmap_read_partition(
                        path, begin, end, len(reader.header), source_indices, list(names)
                    )
                    data = _frame(values, output, chain, draw_start, warmup)
                yield StanChunk(data, chain, draw_start + left - row_start, warmup, names)
        elif _cloud(path):
            from ._remote import _header, _open, iter_remote

            with _open(path) as stream:
                remote_header, _, _ = _header(stream)
            if full_header is None:
                full_header = remote_header
            elif remote_header != full_header:
                raise ValueError(f"CSV headers differ across chains: {path}")
            yield from iter_remote(path, chain=chain, chunksize=chunksize,
                                   columns=selected, output=output,
                                   include_warmup=include_warmup, rows=rows)
        else:
            source = _native.CsvStream(path, selected, include_warmup)
            if full_header is None:
                full_header = source.header
            elif source.header != full_header:
                raise ValueError(f"CSV headers differ across chains: {path}")
            names = tuple(source.columns)
            bounded_rows = max(1, min(chunksize, (32 * 1024 * 1024) // (8 * max(1, len(names)))))
            offset = 0
            while (batch := source.next_chunk(bounded_rows, output != "numpy")) is not None:
                values, draw_start, warmup = batch
                n = len(next(iter(values.values()))) if isinstance(values, dict) else len(values)
                left, right = max(0, start - offset), n if stop is None else min(n, stop - offset)
                offset += n
                if left >= right:
                    if stop is not None and offset >= stop:
                        break
                    continue
                if isinstance(values, dict):
                    values = {name: array[left:right] for name, array in values.items()}
                else:
                    values = values[left:right]
                result = values if output == "numpy" else _frame(
                    values, output, chain, draw_start + left, warmup
                )
                yield StanChunk(result, chain, draw_start + left, warmup, names)
                if stop is not None and offset >= stop:
                    break


def _summary_output(data: dict, quantiles: Sequence[float] | None, output: str) -> Any:
    """Convert native Rust statistical vectors into a table or dictionary."""
    data = dict(data)
    for i, q in enumerate(quantiles or []):
        data[f"q{q:g}"] = data.pop(f"_q_{i}")
    if output == "dict":
        return data
    if output != "pandas":
        raise ValueError("output must be 'pandas' or 'dict'")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("install 'stanio_rust[pandas]' for statistics DataFrame") from exc
    return pd.DataFrame(data)


def summarize_csv(
    paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    include_warmup: bool = False,
    engine: Engine = "auto",
    threads: int = 0,
    quantiles: Sequence[float] | None = None,
    output: Literal["pandas", "dict"] = "pandas",
) -> Any:
    """Compute descriptive statistics by streaming values in Rust.

    Parameters
    ----------
    paths : str or sequence of paths
        One or multiple CmdStan chain files.
    columns : sequence of str, optional
        Numeric columns or complete parameter names to summarize.
    include_warmup : bool, default False
        Include saved warmup samples.
    engine : {'auto', 'mmap', 'buffered'}, default 'auto'
        Auto maps completed local files and streams remote files.
    threads : int, default 0
        Parallelize exact moments over mapped partitions.
    quantiles : sequence of float, optional
        Approximate (P²) quantiles between zero and one. For quantiles,
        Rust uses a sequential pass because P² sketches are not mergeable.
        Multi-chain approximate quantiles should be computed per chain.
    output : {'pandas', 'dict'}, default 'pandas'
        Statistics include count, nonfinite, mean, sample variance, min, max,
        and optional approximate quantiles. Nonfinite values are excluded from
        moments and quantiles but counted in ``nonfinite``.

    Returns
    -------
    pandas.DataFrame or dict
        Per-column statistics, not per-draw data.

    Notes
    -----
    No matrix of draws is materialized. Memory use is O(columns * threads)
    for exact moments, or O(columns * quantiles) for P² quantiles.
    """
    files = _paths(paths)
    selected = _selected(columns)
    qs = list(quantiles or [])
    if any(q < 0 or q > 1 or not np.isfinite(q) for q in qs):
        raise ValueError("all quantiles must lie in [0, 1]")
    if all(_engine(p, engine) == "mmap" for p in files):
        result = _native.mmap_summarize(files, selected, include_warmup, threads, qs or None)
        return _summary_output(result, qs, output)
    if qs:
        raise ValueError("buffered/cloud approximate quantiles are not supported yet")
    # Buffered fallback keeps only a fixed chunk of draws and one aggregate per column.
    # Accumulation is performed in Rust, not as a Python list of samples.
    from ._remote import summarize_buffered

    return _summary_output(
        summarize_buffered(files, selected, include_warmup), qs, output,
    )


def read_json(
    path: str | Path,
    *,
    arrays: Literal["list", "numpy"] = "list",
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse a Stan-compatible JSON object using Rust serde_json.

    Parameters
    ----------
    path : str or path-like
        Local path or fsspec-compatible URI.
    arrays : {'list', 'numpy'}, default 'list'
        Reconstruct rectangular numeric arrays as float64 NumPy arrays.
    keys : sequence of str, optional
        Only materialize named top-level members (other members are skipped
        by the Rust JSON deserializer).

    Returns
    -------
    dict
        JSON mapping with optional NumPy arrays.
    """
    if arrays not in ("list", "numpy"):
        raise ValueError("arrays must be 'list' or 'numpy'")
    path_text = os.fsdecode(path)
    if _cloud(path_text):
        try:
            import fsspec
        except ImportError as exc:
            raise ImportError("install 'stanio_rust[cloud]' for cloud URIs") from exc
        with fsspec.open(path_text, "rb") as source:
            return _native.read_json_bytes(source.read(), arrays == "numpy", _selected(keys))
    return _native.read_json(path_text, arrays == "numpy", _selected(keys))


def read_csv_dask(
    paths: PathInput,
    *,
    chunksize: int = 50_000,
    columns: Sequence[str] | None = None,
    include_warmup: bool = False,
    engine: Engine = "auto",
) -> Any:
    """Construct a lazy Dask DataFrame from independently readable byte ranges.

    Parameters
    ----------
    paths : str or sequence of paths
        Local CSV paths or fsspec URLs accessible to every Dask worker.
    chunksize : int, default 50000
        Maximum retained rows per partition.
    columns : sequence of str, optional
        Parameter name projection.
    include_warmup : bool, default False
        Include separately marked warmup samples.
    engine : {'auto', 'mmap', 'buffered'}, default 'auto'
        Local default uses mmap; remote files use buffered range requests.

    Returns
    -------
    dask.dataframe.DataFrame
        Lazy data with ``chain__``, ``draw__`` and ``warmup__`` metadata.
    """
    from ._dask import read_csv_dask as implementation

    return implementation(
        _paths(paths), chunksize=chunksize, columns=_selected(columns),
        include_warmup=include_warmup, engine=engine,
    )
