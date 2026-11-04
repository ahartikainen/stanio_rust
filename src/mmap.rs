//! Row-aligned parallel memory-mapped reader for completed CmdStan numeric CSVs.
//!
//! The mapping borrows OS-backed pages. Numeric values are parsed into separate
//! owned f64 buffers; a text CSV cannot be exposed as a zero-copy NumPy array.
//! Only map immutable, fully written files. See memmap2's safety documentation.

use std::fs::File;
use std::sync::Mutex;

use memchr::memchr;
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{invalid, selected_columns};

const PART_ROWS: usize = 16_384;
const PART_BYTES: usize = 32 * 1024 * 1024;

type PartitionTuple = (usize, usize, usize, usize, usize, bool);

#[derive(Clone, Debug)]
struct Part {
    start: usize,
    end: usize,
    row_start: usize,
    draw_start: usize,
    rows: usize,
    warmup: bool,
}

impl Part {
    fn tuple(&self) -> PartitionTuple {
        (
            self.start,
            self.end,
            self.row_start,
            self.draw_start,
            self.rows,
            self.warmup,
        )
    }
}

#[derive(Clone)]
struct Header {
    names: Vec<String>,
    start: usize,
    saved_warmup: usize,
    meta: Vec<(String, String)>,
}

fn lines(bytes: &[u8], from: usize, until: usize) -> impl Iterator<Item = (usize, usize, &[u8])> {
    let mut start = from;
    std::iter::from_fn(move || {
        if start >= until {
            return None;
        }
        let end = match memchr(b'\n', &bytes[start..until]) {
            Some(found) => start + found + 1,
            None => until,
        };
        let begin = start;
        start = end;
        Some((begin, end, trim(&bytes[begin..end])))
    })
}

fn trim(mut bytes: &[u8]) -> &[u8] {
    while matches!(bytes.first(), Some(b'\n' | b'\r' | b' ' | b'\t')) {
        bytes = &bytes[1..];
    }
    while matches!(bytes.last(), Some(b'\n' | b'\r' | b' ' | b'\t')) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn header(bytes: &[u8]) -> PyResult<Header> {
    let mut n_warmup = 0_usize;
    let mut thin = 1_usize;
    let mut save_warmup = false;
    let mut method = String::new();
    let mut meta = Vec::new();
    for (_, end, content) in lines(bytes, 0, bytes.len()) {
        if content.is_empty() {
            continue;
        }
        if content[0] == b'#' {
            let text = std::str::from_utf8(trim(&content[1..]))
                .map_err(|_| invalid("non-UTF8 CmdStan metadata"))?;
            if let Some((key, val)) = text.split_once('=') {
                let k = key.trim();
                let v = val.split_whitespace().next().unwrap_or("");
                meta.push((k.to_string(), v.to_string()));
                match k {
                    "num_warmup" => n_warmup = v.parse::<usize>().unwrap_or(0),
                    "thin" => thin = v.parse::<usize>().unwrap_or(1).max(1),
                    "save_warmup" => save_warmup = v == "1" || v == "true",
                    "method" => method = v.to_string(),
                    _ => {}
                }
            }
            continue;
        }
        let names = std::str::from_utf8(content)
            .map_err(|_| invalid("CSV header is not UTF-8"))?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();
        if names.is_empty() || names.iter().any(String::is_empty) {
            return Err(invalid("CSV header contains empty column names"));
        }
        if names.iter().collect::<std::collections::HashSet<_>>().len() != names.len() {
            return Err(invalid("CSV header contains duplicate column names"));
        }
        let saved_warmup = if save_warmup && (method.is_empty() || method == "sample") {
            n_warmup.div_ceil(thin)
        } else {
            0
        };
        return Ok(Header {
            names,
            start: end,
            saved_warmup,
            meta,
        });
    }
    Err(invalid("missing CmdStan numeric CSV header"))
}

fn parse_num(raw: &[u8]) -> PyResult<f64> {
    let bytes = trim(raw);
    let string = std::str::from_utf8(bytes).map_err(|_| invalid("invalid numeric UTF-8"))?;
    string
        .parse::<f64>()
        .map_err(|_| invalid(format!("invalid CmdStan floating-point value: {string:?}")))
}

fn parse_line(line: &[u8], projection: &[Option<usize>], row: &mut [f64]) -> PyResult<()> {
    let mut source_index = 0;
    let mut pos = 0;
    loop {
        let end = match memchr(b',', &line[pos..]) {
            Some(next) => pos + next,
            None => line.len(),
        };
        if source_index >= projection.len() {
            return Err(invalid("CSV record has more columns than its header"));
        }
        if let Some(col) = projection[source_index] {
            row[col] = parse_num(&line[pos..end])?;
        }
        source_index += 1;
        if end == line.len() {
            break;
        }
        pos = end + 1;
    }
    if source_index != projection.len() {
        return Err(invalid(format!(
            "CSV record has {source_index} fields; expected {}",
            projection.len()
        )));
    }
    Ok(())
}

fn selection(names: &[String], requested: Option<&[String]>) -> PyResult<(Vec<String>, Vec<Option<usize>>)> {
    let (names, _, projection) = selected_columns(names, requested)?;
    Ok((names, projection))
}

fn build_index(bytes: &[u8], hdr: &Header, include_warmup: bool, target_rows: usize, target_bytes: usize) -> Vec<Part> {
    let max_rows = target_rows.max(1);
    let max_bytes = target_bytes.max(1);
    let mut out = Vec::new();
    let mut active: Option<Part> = None;
    let mut warmup = hdr.saved_warmup > 0;
    let mut warmup_draw = 0;
    let mut sample_draw = 0;
    let mut selected_row = 0;
    for (start, end, line) in lines(bytes, hdr.start, bytes.len()) {
        if line.is_empty() {
            continue;
        }
        if line[0] == b'#' {
            if line.windows(b"Adaptation terminated".len()).any(|w| w == b"Adaptation terminated") {
                if let Some(part) = active.take() {
                    out.push(part);
                }
                warmup = false;
            }
            continue;
        }
        let was_warmup = warmup && warmup_draw < hdr.saved_warmup;
        let draw_start = if was_warmup { warmup_draw } else { sample_draw };
        if was_warmup {
            warmup_draw += 1;
            if warmup_draw >= hdr.saved_warmup {
                warmup = false;
            }
        } else {
            sample_draw += 1;
        }
        if was_warmup && !include_warmup {
            continue;
        }
        if let Some(ref part) = active {
            if part.warmup != was_warmup || part.rows >= max_rows || end - part.start > max_bytes {
                out.push(active.take().expect("active partition"));
            }
        }
        let part = active.get_or_insert_with(|| Part {
            start,
            end,
            row_start: selected_row,
            draw_start,
            rows: 0,
            warmup: was_warmup,
        });
        part.end = end;
        part.rows += 1;
        selected_row += 1;
        if part.rows == max_rows || part.end - part.start >= max_bytes || (was_warmup && !warmup) {
            out.push(active.take().expect("completed partition"));
        }
    }
    if let Some(part) = active {
        out.push(part);
    }
    out
}

fn parse_part(bytes: &[u8], part: &Part, projection: &[Option<usize>], width: usize, skip: usize, take: usize, result: &mut [f64]) -> PyResult<()> {
    if result.len() != take * width {
        return Err(invalid("invalid partition destination length"));
    }
    let mut seen = 0;
    let mut written = 0;
    for (_, _, line) in lines(bytes, part.start, part.end) {
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        if seen >= skip && written < take {
            parse_line(line, projection, &mut result[written * width..(written + 1) * width])?;
            written += 1;
        }
        seen += 1;
    }
    if seen != part.rows || written != take {
        return Err(invalid("CSV partition record count changed during reading"));
    }
    Ok(())
}

fn parallel_read_into(bytes: &[u8], parts: &[Part], projection: &[Option<usize>], width: usize, begin: usize, until: usize, threads: usize, output: &mut [f64]) -> PyResult<()> {
    let mut jobs = Vec::new();
    for part in parts {
        let lo = part.row_start.max(begin);
        let hi = (part.row_start + part.rows).min(until);
        if lo < hi {
            jobs.push((part, lo - part.row_start, hi - lo));
        }
    }
    let count = until.saturating_sub(begin);
    let size = count.checked_mul(width).ok_or_else(|| PyMemoryError::new_err("CSV result array is too large"))?;
    if output.len() != size {
        return Err(invalid("incorrect destination array size"));
    }
    let mut rem = output;
    let mut assigned = Vec::with_capacity(jobs.len());
    for (part, skip, take) in jobs {
        let len = take * width;
        let (head, tail) = std::mem::take(&mut rem).split_at_mut(len);
        rem = tail;
        assigned.push((part, skip, take, head));
    }
    let run = || {
        assigned
            .into_par_iter()
            .try_for_each(|(part, skip, take, destination)| {
                parse_part(bytes, part, projection, width, skip, take, destination)
            })
    };
    run_pool(threads, run)
}

fn parallel_read(bytes: &[u8], parts: &[Part], projection: &[Option<usize>], width: usize, begin: usize, until: usize, threads: usize) -> PyResult<Vec<f64>> {
    let size = until.saturating_sub(begin).checked_mul(width).ok_or_else(|| PyMemoryError::new_err("CSV result array is too large"))?;
    let mut output = vec![0.0; size];
    parallel_read_into(bytes, parts, projection, width, begin, until, threads, &mut output)?;
    Ok(output)
}

fn run_pool<T: Send, F: FnOnce() -> PyResult<T> + Send>(threads: usize, task: F) -> PyResult<T> {
    if threads == 0 {
        task()
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| invalid(format!("cannot start Rayon pool: {e}")))?;
        pool.install(task)
    }
}

#[pyclass(module = "staniorust._native")]
pub(crate) struct MappedCsv {
    mmap: Mmap,
    header: Header,
    index_post: Mutex<Option<Vec<Part>>>,
    index_all: Mutex<Option<Vec<Part>>>,
}

impl MappedCsv {
    fn open(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        if file.metadata().map_err(|e| PyIOError::new_err(e.to_string()))?.len() == 0 {
            return Err(invalid("empty CSV file"));
        }
        // SAFETY: the caller must only map completed, immutable files. mmap
        // cannot protect against other processes truncating this file.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let header = header(&mmap)?;
        Ok(Self {
            mmap,
            header,
            index_post: Mutex::new(None),
            index_all: Mutex::new(None),
        })
    }

    fn indexed(&self, warmup: bool) -> PyResult<Vec<Part>> {
        let mutex = if warmup { &self.index_all } else { &self.index_post };
        let mut guard = mutex.lock().map_err(|_| invalid("index lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(build_index(&self.mmap, &self.header, warmup, PART_ROWS, PART_BYTES));
        }
        Ok(guard.as_ref().expect("index initialized").clone())
    }

    fn extract(&self, projection: &[Option<usize>], width: usize, include_warmup: bool, start: usize, stop: Option<usize>, threads: usize) -> PyResult<(Vec<f64>, usize)> {
        let parts = self.indexed(include_warmup)?;
        let total = parts.last().map_or(0, |p| p.row_start + p.rows);
        let until = stop.unwrap_or(total).min(total);
        let begin = start.min(until);
        let out = parallel_read(&self.mmap, &parts, projection, width, begin, until, threads)?;
        Ok((out, until - begin))
    }
}

#[pymethods]
impl MappedCsv {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Self::open(&path)
    }

    #[getter]
    fn header(&self) -> Vec<String> {
        self.header.names.clone()
    }

    #[getter]
    fn metadata(&self) -> Vec<(String, String)> {
        self.header.meta.clone()
    }

    #[pyo3(signature = (include_warmup=false))]
    fn nrows(&self, py: Python<'_>, include_warmup: bool) -> PyResult<usize> {
        py.allow_threads(|| {
            let parts = self.indexed(include_warmup)?;
            Ok(parts.last().map_or(0, |p| p.row_start + p.rows))
        })
    }

    /// (byte_start, byte_end, retained_row_start, phase_draw_start, rows, warmup)
    #[pyo3(signature = (include_warmup=false, rows=16384, bytes_per_partition=33554432))]
    fn partitions(&self, py: Python<'_>, include_warmup: bool, rows: usize, bytes_per_partition: usize) -> PyResult<Vec<PartitionTuple>> {
        if rows == 0 || bytes_per_partition == 0 {
            return Err(invalid("partition rows and bytes_per_partition must be positive"));
        }
        py.allow_threads(|| {
            if rows == PART_ROWS && bytes_per_partition == PART_BYTES {
                return Ok(self.indexed(include_warmup)?.iter().map(Part::tuple).collect());
            }
            Ok(build_index(&self.mmap, &self.header, include_warmup, rows, bytes_per_partition)
                .iter().map(Part::tuple).collect())
        })
    }

    #[pyo3(signature = (columns=None, include_warmup=false, start=0, stop=None, threads=0))]
    fn read<'py>(&self, py: Python<'py>, columns: Option<Vec<String>>, include_warmup: bool, start: usize, stop: Option<usize>, threads: usize) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f64>>)> {
        let (names, projection) = selection(&self.header.names, columns.as_deref())?;
        let (values, rows) = py.allow_threads(|| self.extract(&projection, names.len(), include_warmup, start, stop, threads))?;
        let arr = Array2::from_shape_vec((rows, names.len()), values)
            .map_err(|e| invalid(e.to_string()))?;
        Ok((names, PyArray2::from_owned_array(py, arr)))
    }

    #[pyo3(signature = (columns=None, include_warmup=false, threads=0, quantiles=None))]
    fn summarize<'py>(&self, py: Python<'py>, columns: Option<Vec<String>>, include_warmup: bool, threads: usize, quantiles: Option<Vec<f64>>) -> PyResult<Bound<'py, PyDict>> {
        let (names, projection) = selection(&self.header.names, columns.as_deref())?;
        let q = quantiles.unwrap_or_default();
        validate_q(&q)?;
        let result = py.allow_threads(|| self.stats(&projection, names.len(), include_warmup, threads, &q))?;
        stats_to_python(py, &names, result)
    }
}

#[derive(Clone, Default)]
struct Stat {
    count: u64,
    nonfinite: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn add(&mut self, x: f64) {
        if !x.is_finite() {
            self.nonfinite += 1;
            return;
        }
        if self.count == 0 {
            self.min = x;
            self.max = x;
        } else {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            self.nonfinite += other.nonfinite;
            return;
        }
        if self.count == 0 {
            let missing = self.nonfinite;
            *self = other.clone();
            self.nonfinite += missing;
            return;
        }
        self.nonfinite += other.nonfinite;
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean += delta * other.count as f64 / total as f64;
        self.count = total;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

// Jain & Chlamtac's P-squared streaming quantile estimator (constant space).
// Approximate: unlike min/mean/variance, quantiles are NOT exact.
#[derive(Clone)]
struct P2 {
    probability: f64,
    initial: Vec<f64>,
    heights: [f64; 5],
    positions: [f64; 5],
    targets: [f64; 5],
    increments: [f64; 5],
}

impl P2 {
    fn new(p: f64) -> Self {
        Self {
            probability: p,
            initial: Vec::with_capacity(5),
            heights: [0.0; 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            targets: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            increments: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
        }
    }

    fn add(&mut self, x: f64) {
        if !x.is_finite() || self.probability == 0.0 || self.probability == 1.0 {
            return;
        }
        if self.initial.len() < 5 {
            self.initial.push(x);
            if self.initial.len() == 5 {
                self.initial.sort_by(f64::total_cmp);
                self.heights.copy_from_slice(&self.initial);
            }
            return;
        }
        let k = if x < self.heights[0] {
            self.heights[0] = x;
            0
        } else if x >= self.heights[4] {
            self.heights[4] = x;
            3
        } else {
            (0..4).find(|i| x < self.heights[i + 1]).unwrap_or(3)
        };
        for j in k + 1..5 {
            self.positions[j] += 1.0;
        }
        for j in 0..5 {
            self.targets[j] += self.increments[j];
        }
        for j in 1..4 {
            let d = self.targets[j] - self.positions[j];
            if (d >= 1.0 && self.positions[j + 1] - self.positions[j] > 1.0)
                || (d <= -1.0 && self.positions[j - 1] - self.positions[j] < -1.0)
            {
                let direction: isize = if d > 0.0 { 1 } else { -1 };
                let signed = direction as f64;
                let (p, h) = (&self.positions, &self.heights);
                let parabolic = h[j]
                    + signed / (p[j + 1] - p[j - 1])
                        * ((p[j] - p[j - 1] + signed) * (h[j + 1] - h[j]) / (p[j + 1] - p[j])
                            + (p[j + 1] - p[j] - signed) * (h[j] - h[j - 1])
                                / (p[j] - p[j - 1]));
                if self.heights[j - 1] < parabolic && parabolic < self.heights[j + 1] {
                    self.heights[j] = parabolic;
                } else {
                    let next = (j as isize + direction) as usize;
                    self.heights[j] += signed * (self.heights[next] - self.heights[j])
                        / (self.positions[next] - self.positions[j]);
                }
                self.positions[j] += signed;
            }
        }
    }

    fn estimate(&self, stat: &Stat) -> f64 {
        if stat.count == 0 {
            return f64::NAN;
        }
        if self.probability == 0.0 {
            return stat.min;
        }
        if self.probability == 1.0 {
            return stat.max;
        }
        if self.initial.len() < 5 {
            let mut initial = self.initial.clone();
            initial.sort_by(f64::total_cmp);
            let at = self.probability * (initial.len() - 1) as f64;
            let low = at.floor() as usize;
            let high = at.ceil() as usize;
            return initial[low] * (1.0 - at.fract()) + initial[high] * at.fract();
        }
        self.heights[2]
    }
}

fn visit_part<F>(bytes: &[u8], part: &Part, projection: &[Option<usize>], width: usize, scratch: &mut [f64], mut visitor: F) -> PyResult<()>
where
    F: FnMut(&[f64]),
{
    let mut found = 0;
    for (_, _, line) in lines(bytes, part.start, part.end) {
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        if scratch.len() != width {
            return Err(invalid("invalid stat scratch buffer"));
        }
        parse_line(line, projection, scratch)?;
        visitor(scratch);
        found += 1;
    }
    if found != part.rows {
        return Err(invalid("CSV changed during streaming statistics"));
    }
    Ok(())
}

fn validate_q(quantiles: &[f64]) -> PyResult<()> {
    if quantiles.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x)) {
        return Err(PyValueError::new_err("quantiles must all be between 0 and 1"));
    }
    Ok(())
}

impl MappedCsv {
    fn stats(&self, projection: &[Option<usize>], width: usize, include_warmup: bool, threads: usize, quantiles: &[f64]) -> PyResult<(Vec<Stat>, Vec<Vec<f64>>)> {
        let parts = self.indexed(include_warmup)?;
        if quantiles.is_empty() {
            let result = run_pool(threads, || {
                parts.par_iter()
                    .try_fold(
                        || (vec![Stat::default(); width], vec![0.0; width]),
                        |(mut acc, mut scratch), part| {
                            visit_part(&self.mmap, part, projection, width, &mut scratch, |row| {
                                for (stat, &value) in acc.iter_mut().zip(row) {
                                    stat.add(value);
                                }
                            })?;
                            Ok((acc, scratch))
                        },
                    )
                    .try_reduce(
                        || (vec![Stat::default(); width], vec![0.0; width]),
                        |(mut left, _), (right, _)| {
                            for (a, b) in left.iter_mut().zip(&right) {
                                a.merge(b);
                            }
                            Ok((left, vec![0.0; width]))
                        },
                    )
            })?;
            return Ok((result.0, Vec::new()));
        }
        // P² sketches are order dependent and not mergeable: use one pass in
        // file order for quantiles. Parallel exact stats remain available when
        // quantiles=None. This also caps memory at O(columns * quantiles).
        let mut stats = vec![Stat::default(); width];
        let mut sketches: Vec<Vec<P2>> = (0..width)
            .map(|_| quantiles.iter().copied().map(P2::new).collect())
            .collect();
        let mut scratch = vec![0.0; width];
        for part in &parts {
            visit_part(&self.mmap, part, projection, width, &mut scratch, |row| {
                for (col, &value) in row.iter().enumerate() {
                    stats[col].add(value);
                    for sketch in &mut sketches[col] {
                        sketch.add(value);
                    }
                }
            })?;
        }
        let values = (0..quantiles.len())
            .map(|q| (0..width).map(|col| sketches[col][q].estimate(&stats[col])).collect())
            .collect();
        Ok((stats, values))
    }
}

fn stats_to_python<'py>(py: Python<'py>, columns: &[String], result: (Vec<Stat>, Vec<Vec<f64>>)) -> PyResult<Bound<'py, PyDict>> {
    let (stats, quantiles) = result;
    let obj = PyDict::new(py);
    obj.set_item("column", columns)?;
    obj.set_item("count", stats.iter().map(|s| s.count).collect::<Vec<_>>())?;
    obj.set_item("nonfinite", stats.iter().map(|s| s.nonfinite).collect::<Vec<_>>())?;
    obj.set_item("mean", PyArray1::from_vec(py, stats.iter().map(|s| if s.count > 0 {s.mean} else {f64::NAN}).collect()))?;
    obj.set_item("variance", PyArray1::from_vec(py, stats.iter().map(|s| if s.count > 1 {s.m2 / (s.count - 1) as f64} else {f64::NAN}).collect()))?;
    obj.set_item("min", PyArray1::from_vec(py, stats.iter().map(|s| if s.count > 0 {s.min} else {f64::NAN}).collect()))?;
    obj.set_item("max", PyArray1::from_vec(py, stats.iter().map(|s| if s.count > 0 {s.max} else {f64::NAN}).collect()))?;
    for (i, q) in quantiles.into_iter().enumerate() {
        obj.set_item(format!("_q_{i}"), PyArray1::from_vec(py, q))?;
    }
    Ok(obj)
}

#[pyfunction]
#[pyo3(signature = (paths, columns=None, include_warmup=false, threads=0, start=0, stop=None, limit_bytes=None))]
fn mmap_read_all<'py>(py: Python<'py>, paths: Vec<String>, columns: Option<Vec<String>>, include_warmup: bool, threads: usize, start: usize, stop: Option<usize>, limit_bytes: Option<usize>) -> PyResult<(Vec<String>, Bound<'py, PyArray3<f64>>)> {
    if paths.is_empty() {
        return Err(invalid("at least one CSV file is required"));
    }
    let (names, values, draws) = py.allow_threads(|| {
        let mut sources = Vec::with_capacity(paths.len());
        let mut first_header = None;
        let mut first_names = Vec::new();
        let mut first_width = 0;
        let mut n_draws = None;
        for path in &paths {
            let source = MappedCsv::open(path)?;
            if let Some(ref header) = first_header {
                if header != &source.header.names {
                    return Err(invalid(format!("CSV headers differ: {path}")));
                }
            } else {
                first_header = Some(source.header.names.clone());
                let (names, projection) = selection(&source.header.names, columns.as_deref())?;
                first_width = projection.iter().filter(|x| x.is_some()).count();
                first_names = names;
            }
            let n = source.indexed(include_warmup)?.last().map_or(0, |p| p.row_start + p.rows);
            let to = stop.unwrap_or(n).min(n);
            let selected = to.saturating_sub(start);
            if let Some(expected) = n_draws {
                if expected != selected {
                    return Err(invalid(format!("chains have unequal selected draw counts: {expected} vs {selected} for {path}")));
                }
            } else {
                n_draws = Some(selected);
            }
            sources.push(source);
        }
        let draws = n_draws.unwrap_or(0_usize);
        let cells = paths.len().checked_mul(draws)
            .and_then(|x| x.checked_mul(first_width))
            .ok_or_else(|| PyMemoryError::new_err("CSV output shape overflows address space"))?;
        let output_bytes = cells.checked_mul(8)
            .ok_or_else(|| PyMemoryError::new_err("CSV output allocation overflows address space"))?;
        if limit_bytes.is_some_and(|limit| output_bytes > limit) {
            return Err(PyMemoryError::new_err(format!(
                "eager CSV output requires {output_bytes} bytes, exceeding the configured memory limit; use iter_csv or summarize_csv"
            )));
        }
        let mut all = vec![0.0; cells];
        let mut rest = all.as_mut_slice();
        let mut jobs = Vec::new();
        for source in &sources {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(draws * first_width);
            rest = tail;
            jobs.push((source, head));
        }
        let (_, projection) = selection(&sources[0].header.names, columns.as_deref())?;
        let run = || jobs.into_par_iter().try_for_each(|(source, dest)| {
            let parts = source.indexed(include_warmup)?;
            parallel_read_into(&source.mmap, &parts, &projection, first_width, start.min(start + draws), start + draws, 0, dest)
        });
        run_pool(threads, run)?;
        Ok((first_names, all, draws))
    })?;
    let arr = Array3::from_shape_vec((paths.len(), draws, names.len()), values)
        .map_err(|e| invalid(e.to_string()))?;
    Ok((names, PyArray3::from_owned_array(py, arr)))
}

#[pyfunction]
#[pyo3(signature = (paths, columns=None, include_warmup=false, threads=0, quantiles=None))]
fn mmap_summarize<'py>(py: Python<'py>, paths: Vec<String>, columns: Option<Vec<String>>, include_warmup: bool, threads: usize, quantiles: Option<Vec<f64>>) -> PyResult<Bound<'py, PyDict>> {
    if paths.is_empty() {
        return Err(invalid("at least one CSV file is required"));
    }
    let qs = quantiles.unwrap_or_default();
    validate_q(&qs)?;
    let (names, result) = py.allow_threads(|| -> PyResult<_> {
        let mut aggregate: Option<Vec<Stat>> = None;
        let mut quantile_output: Option<Vec<Vec<f64>>> = None;
        let mut first_header: Option<Vec<String>> = None;
        let mut selected_names = Vec::new();
        if !qs.is_empty() && paths.len() > 1 {
            // P² quantile sketches are order dependent and not mergeable.
            // Visit all chains in order with a single set of sketches.
            let mut stats: Vec<Stat> = Vec::new();
            let mut sketches: Vec<Vec<P2>> = Vec::new();
            for path in &paths {
                let file = MappedCsv::open(path)?;
                if let Some(ref first) = first_header {
                    if *first != file.header.names {
                        return Err(invalid(format!("CSV headers differ: {path}")));
                    }
                } else {
                    first_header = Some(file.header.names.clone());
                }
                let (names, projection) = selection(&file.header.names, columns.as_deref())?;
                selected_names = names;
                let width = selected_names.len();
                if stats.is_empty() {
                    stats = vec![Stat::default(); width];
                    sketches = (0..width)
                        .map(|_| qs.iter().copied().map(P2::new).collect())
                        .collect();
                }
                let mut scratch = vec![0.0; width];
                for part in file.indexed(include_warmup)? {
                    visit_part(&file.mmap, &part, &projection, width, &mut scratch, |row| {
                        for (col, &value) in row.iter().enumerate() {
                            stats[col].add(value);
                            for sketch in &mut sketches[col] {
                                sketch.add(value);
                            }
                        }
                    })?;
                }
            }
            let estimates = (0..qs.len())
                .map(|q| (0..stats.len())
                    .map(|col| sketches[col][q].estimate(&stats[col]))
                    .collect())
                .collect();
            return Ok((selected_names, (stats, estimates)));
        }
        for path in &paths {
            let file = MappedCsv::open(path)?;
            if let Some(ref first) = first_header {
                if *first != file.header.names {
                    return Err(invalid(format!("CSV headers differ: {path}")));
                }
            } else {
                first_header = Some(file.header.names.clone());
            }
            let (names, projection) = selection(&file.header.names, columns.as_deref())?;
            selected_names = names;
            let (local, q) = file.stats(&projection, selected_names.len(), include_warmup, threads, &qs)?;
            if let Some(ref mut agg) = aggregate {
                for (total, part) in agg.iter_mut().zip(&local) {
                    total.merge(part);
                }
            } else {
                aggregate = Some(local);
            }
            if !q.is_empty() {
                quantile_output = Some(q);
            }
        }
        Ok((selected_names, (aggregate.unwrap_or_default(), quantile_output.unwrap_or_default())))
    })?;
    stats_to_python(py, &names, result)
}

#[pyfunction]
fn mmap_read_partition<'py>(py: Python<'py>, path: String, start: usize, end: usize, ncols: usize, indices: Vec<usize>, names: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
    let result = py.allow_threads(|| -> PyResult<Vec<Vec<f64>>> {
        let file = File::open(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let size = file.metadata().map_err(|e| PyIOError::new_err(e.to_string()))?.len() as usize;
        if end < start || end > size || indices.len() != names.len() || ncols == 0 {
            return Err(invalid("invalid partition byte bounds or column indices"));
        }
        // SAFETY: the input file must not be modified while being mapped.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut projection = vec![None; ncols];
        for (dst, src) in indices.iter().copied().enumerate() {
            if src >= ncols || projection[src].is_some() {
                return Err(invalid("invalid or duplicate selected column indices"));
            }
            projection[src] = Some(dst);
        }
        let mut values = vec![Vec::<f64>::new(); names.len()];
        let mut scratch = vec![0.0; names.len()];
        for (_, _, line) in lines(&map, start, end) {
            if line.is_empty() || line[0] == b'#' {
                continue;
            }
            parse_line(line, &projection, &mut scratch)?;
            for (column, &value) in values.iter_mut().zip(&scratch) {
                column.push(value);
            }
        }
        Ok(values)
    })?;
    let output = PyDict::new(py);
    for (name, values) in names.iter().zip(result) {
        output.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok(output)
}


/// Rust parsing of one complete-line byte block supplied by fsspec / buffered I/O.
/// Comments are skipped, and output values are C-contiguous float64.
#[pyfunction]
fn decode_csv_block<'py>(py: Python<'py>, bytes: Vec<u8>, ncols: usize, indices: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (values, rows) = py.allow_threads(|| -> PyResult<_> {
        if ncols == 0 || indices.is_empty() {
            return Err(invalid("CSV header and projection must be nonempty"));
        }
        let mut projection = vec![None; ncols];
        for (dst, &src) in indices.iter().enumerate() {
            if src >= ncols || projection[src].is_some() {
                return Err(invalid("invalid selected column indexes"));
            }
            projection[src] = Some(dst);
        }
        let mut values = Vec::<f64>::new();
        let mut rows = 0;
        for (_, _, line) in lines(&bytes, 0, bytes.len()) {
            if line.is_empty() || line[0] == b'#' {
                continue;
            }
            let begin = values.len();
            values.resize(begin + indices.len(), 0.0);
            parse_line(line, &projection, &mut values[begin..])?;
            rows += 1;
        }
        Ok((values, rows))
    })?;
    let arr = Array2::from_shape_vec((rows, indices.len()), values)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Rust Welford online reduction over successive bounded NumPy chunks.
#[pyclass(module = "staniorust._native")]
struct OnlineStats {
    names: Vec<String>,
    stats: Mutex<Vec<Stat>>,
}

#[pymethods]
impl OnlineStats {
    #[new]
    fn new(names: Vec<String>) -> Self {
        Self { stats: Mutex::new(vec![Stat::default(); names.len()]), names }
    }

    fn update(&self, py: Python<'_>, values: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let array = values.as_array();
        if array.ncols() != self.names.len() {
            return Err(invalid("online accumulator columns do not match the input array"));
        }
        let bytes = array.as_slice().ok_or_else(|| invalid("statistics input must be contiguous"))?;
        py.allow_threads(|| {
            let mut stats = self.stats.lock().map_err(|_| invalid("statistics lock poisoned"))?;
            for row in bytes.chunks_exact(stats.len().max(1)) {
                for (s, &value) in stats.iter_mut().zip(row) {
                    s.add(value);
                }
            }
            Ok(())
        })
    }

    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.stats.lock().map_err(|_| invalid("statistics lock poisoned"))?.clone();
        stats_to_python(py, &self.names, (result, Vec::new()))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MappedCsv>()?;
    m.add_class::<OnlineStats>()?;
    m.add_function(wrap_pyfunction!(decode_csv_block, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_read_all, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_summarize, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_read_partition, m)?)?;
    Ok(())
}
