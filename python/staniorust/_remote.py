"""Bounded-memory fsspec byte streams and remote range-index support."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


from . import _native

_BLOCK_BYTES = 4 * 1024 * 1024


def _open(path: str):
    """Open a binary file through fsspec only when required by a remote URI."""
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError("install 'stanio_rust[cloud]' for cloud inputs") from exc
    return fsspec.open(path, mode="rb", block_size=_BLOCK_BYTES).open()


def _header(stream) -> tuple[list[str], int, int]:
    """Read the initial CmdStan comments and header without touching data rows."""
    warmup, thin, save_warmup = 0, 1, False
    method = "sample"
    while line := stream.readline():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith(b"#"):
            parts = raw[1:].split(b"=", 1)
            if len(parts) == 2:
                key = parts[0].strip().decode("utf-8")
                value = parts[1].strip().split(b" ", 1)[0].decode("utf-8")
                if key == "num_warmup":
                    warmup = int(value)
                elif key == "thin":
                    thin = max(1, int(value))
                elif key == "save_warmup":
                    save_warmup = value in ("1", "true")
                elif key == "method":
                    method = value
            continue
        columns = [name.strip() for name in raw.decode("utf-8").split(",")]
        if len(set(columns)) != len(columns) or not all(columns):
            raise ValueError("invalid or duplicate CSV headers")
        return columns, stream.tell(), (warmup + thin - 1) // thin if save_warmup and method == "sample" else 0
    raise ValueError("CmdStan CSV header was not found")


def _complete_blocks(stream, start: int, blocksize: int = _BLOCK_BYTES) -> Iterator[tuple[int, bytes]]:
    """Yield byte chunks ending at a newline, retaining only the trailing row."""
    unfinished = b""
    position = start
    while data := stream.read(blocksize):
        block = unfinished + data
        cut = block.rfind(b"\n")
        if cut < 0:
            unfinished = block
            continue
        end = cut + 1
        yield position, block[:end]
        position += end
        unfinished = block[end:]
    if unfinished:
        yield position, unfinished


def _project(header: Sequence[str], columns: Sequence[str] | None) -> tuple[list[str], list[int]]:
    """Select flattened indices, resolving complete Stan variable prefixes."""
    if columns is None:
        return list(header), list(range(len(header)))
    wanted = [columns] if isinstance(columns, str) else list(columns)
    if not wanted:
        raise ValueError("columns cannot be empty")
    selected = [i for i, col in enumerate(header) if any(
        col == name or col.startswith((name + ".", name + ":", name + "["))
        for name in wanted
    )]
    for name in wanted:
        if not any(col == name or col.startswith((name + ".", name + ":", name + "["))
                   for col in header):
            raise ValueError(f"unknown Stan column or variable: {name}")
    return [header[i] for i in selected], selected


def iter_remote(
    path: str, *, chain: int, chunksize: int, columns: Sequence[str] | None,
    output: str, include_warmup: bool, rows: slice | None,
) -> Iterator[Any]:
    """Parse file-like blocks in Rust without materializing remote files locally."""
    from ._api import StanChunk, _frame, _frame_for_matrix

    with _open(path) as stream:
        header, data_start, warmup_rows = _header(stream)
        names, indices = _project(header, columns)
        observed = 0
        emitted = 0
        requested_from = 0 if rows is None or rows.start is None else rows.start
        requested_to = None if rows is None else rows.stop
        warm_draw, sample_draw = 0, 0
        for _, block in _complete_blocks(stream, data_start):
            numeric = _native.decode_csv_block(block, len(header), indices)
            pos = 0
            while pos < len(numeric):
                is_warmup = observed < warmup_rows
                phase_available = warmup_rows - observed if is_warmup else len(numeric) - pos
                length = min(chunksize, len(numeric) - pos, phase_available)
                if length == 0:
                    raise RuntimeError("internal remote CSV phase tracking error")
                draw_start = warm_draw if is_warmup else sample_draw
                fragment = numeric[pos:pos + length]
                pos += length
                observed += length
                if is_warmup:
                    warm_draw += length
                else:
                    sample_draw += length
                if is_warmup and not include_warmup:
                    continue
                left = max(0, requested_from - emitted)
                right = length if requested_to is None else min(length, requested_to - emitted)
                emitted += length
                if left < right:
                    sub = fragment[left:right]
                    result = sub if output == "numpy" else _frame_for_matrix(
                        names, sub, output, chain, draw_start + left, is_warmup,
                    )
                    yield StanChunk(result, chain, draw_start + left, is_warmup, tuple(names))
                if requested_to is not None and emitted >= requested_to:
                    return


@dataclass(frozen=True, slots=True)
class RemotePartition:
    """Byte range and row metadata used to build remote Dask tasks."""

    start: int
    end: int
    row_start: int
    draw_start: int
    rows: int
    warmup: bool


def remote_index(
    path: str, *, chunksize: int, columns: Sequence[str] | None,
    include_warmup: bool,
) -> tuple[list[str], list[str], list[int], list[RemotePartition]]:
    """Find cloud byte ranges without decoding any numerical CSV values."""
    with _open(path) as stream:
        header, start, saved_warmup = _header(stream)
        names, indices = _project(header, columns)
        current: RemotePartition | None = None
        partitions = []
        observed = 0
        retained = 0
        warm = 0
        sample = 0
        for offset, block in _complete_blocks(stream, start):
            pos = 0
            while pos < len(block):
                cut = block.find(b"\n", pos)
                end = cut + 1 if cut >= 0 else len(block)
                line = block[pos:end].strip()
                begin = offset + pos
                pos = end
                if not line or line.startswith(b"#"):
                    continue
                is_warmup = observed < saved_warmup
                first_draw = warm if is_warmup else sample
                observed += 1
                if is_warmup:
                    warm += 1
                else:
                    sample += 1
                if is_warmup and not include_warmup:
                    continue
                if current is not None and (current.warmup != is_warmup or current.rows >= chunksize
                                            or current.end - current.start >= 32 * 1024 * 1024):
                    partitions.append(current)
                    current = None
                if current is None:
                    current = RemotePartition(begin, offset + pos, retained, first_draw, 0, is_warmup)
                current = RemotePartition(
                    current.start, offset + pos, current.row_start,
                    current.draw_start, current.rows + 1, current.warmup,
                )
                retained += 1
                if current.rows == chunksize:
                    partitions.append(current)
                    current = None
        if current is not None:
            partitions.append(current)
    return header, names, indices, partitions


def remote_partition(
    path: str, part: RemotePartition, names: list[str],
    indices: list[int], ncols: int, chain: int,
):
    """Fetch one exact remote byte range and parse it with Rust on a Dask worker."""
    from ._api import _frame_for_matrix

    with _open(path) as stream:
        stream.seek(part.start)
        data = stream.read(part.end - part.start)
    if len(data) != part.end - part.start:
        raise OSError("remote CSV changed after partition indexing")
    values = _native.decode_csv_block(data, ncols, indices)
    if len(values) != part.rows:
        raise ValueError("remote partition row count changed")
    return _frame_for_matrix(names, values, "pandas", chain, part.draw_start, part.warmup)


def summarize_buffered(
    paths: list[str], columns: list[str] | None, include_warmup: bool,
) -> dict:
    """Use a native online accumulator for local buffered and cloud streams."""
    from ._api import iter_csv

    aggregator = None
    expected = None
    for path in paths:
        for chunk in iter_csv(path, output="numpy", chunksize=16_384, columns=columns,
                              include_warmup=include_warmup, engine="buffered"):
            if expected is None:
                expected = chunk.columns
                aggregator = _native.OnlineStats(list(expected))
            elif chunk.columns != expected:
                raise ValueError("chain headers differ")
            aggregator.update(chunk.data)
    if aggregator is not None:
        return aggregator.result()
    # Completely empty file: load its header without loading any numeric data.
    first = paths[0]
    if "//" in first or first.startswith(("s3:", "gs:", "az:", "memory:")):
        with _open(first) as stream:
            names, _, _ = _header(stream)
        names, _ = _project(names, columns)
    else:
        names = _native.CsvStream(os.fspath(Path(first)), columns, include_warmup).columns
    return _native.OnlineStats(names).result()
