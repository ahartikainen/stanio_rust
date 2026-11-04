//! Rust hot path for CmdStan numeric CSV and Stan JSON.
//! Paths are local, seekable files. Reading never writes beside the input files.

mod mmap;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyString};
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde::Deserializer;
use serde_json::Value;
use std::fmt;

type Partition = (u64, u64, usize, usize, bool);

fn invalid(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn io_error(error: std::io::Error) -> PyErr {
    PyIOError::new_err(error.to_string())
}

fn selected_columns(
    full: &[String],
    requested: Option<&[String]>,
) -> PyResult<(Vec<String>, Vec<usize>, Vec<Option<usize>>)> {
    let mut keep = vec![false; full.len()];
    match requested {
        None => keep.fill(true),
        Some(names) => {
            if names.is_empty() {
                return Err(invalid("columns cannot be empty"));
            }
            for name in names {
                let found = full.iter().enumerate().filter(|(_, col)| {
                    *col == name
                        || (!name.contains('.')
                            && !name.contains('[')
                            && (col.starts_with(&format!("{name}."))
                                || col.starts_with(&format!("{name}["))
                                || col.starts_with(&format!("{name}:"))))
                });
                let mut matched = false;
                for (idx, _) in found {
                    keep[idx] = true;
                    matched = true;
                }
                if !matched {
                    return Err(invalid(format!("unknown Stan column or variable: {name}")));
                }
            }
        }
    }
    let indices: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(idx, yes)| yes.then_some(idx))
        .collect();
    let names = indices.iter().map(|idx| full[*idx].clone()).collect();
    let mut map = vec![None; full.len()];
    for (out_idx, idx) in indices.iter().enumerate() {
        map[*idx] = Some(out_idx);
    }
    Ok((names, indices, map))
}

struct CsvCore {
    reader: BufReader<File>,
    header: Vec<String>,
    columns: Vec<String>,
    indices: Vec<usize>,
    index_map: Vec<Option<usize>>,
    include_warmup: bool,
    warmup: bool,
    warmup_rows: usize,
    draw: usize,
    offset: u64,
    line_number: usize,
}

impl CsvCore {
    fn open(path: &str, selected: Option<&[String]>, include_warmup: bool) -> PyResult<Self> {
        let file = File::open(path).map_err(io_error)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let mut line = String::new();
        let mut offset = 0_u64;
        let mut line_number = 0_usize;
        let mut n_warmup = 0_usize;
        let mut thin = 1_usize;
        let mut save_warmup = false;
        let mut method = String::new();
        let header = loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(io_error)?;
            if n == 0 {
                return Err(invalid(format!("missing CmdStan CSV header in {path}")));
            }
            offset += n as u64;
            line_number += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(comment) = trimmed.strip_prefix('#') {
                if let Some((key, raw_value)) = comment.trim().split_once('=') {
                    let key = key.trim();
                    let value = raw_value.split_whitespace().next().unwrap_or("");
                    match key {
                        "method" => method = value.to_string(),
                        "num_warmup" => n_warmup = value.parse().unwrap_or(0),
                        "thin" => thin = value.parse::<usize>().unwrap_or(1).max(1),
                        "save_warmup" => save_warmup = value == "1" || value == "true",
                        _ => {}
                    }
                }
                continue;
            }
            break trimmed
                .split(',')
                .map(|name| name.trim().to_string())
                .collect::<Vec<_>>();
        };
        if header.is_empty() || header.iter().any(String::is_empty) {
            return Err(invalid("invalid or empty CSV column header"));
        }
        let (columns, indices, index_map) = selected_columns(&header, selected)?;
        // CmdStan counts iterations from zero; the first and every Nth iteration is saved.
        let warmup_rows = if save_warmup && (method.is_empty() || method == "sample") {
            n_warmup.div_ceil(thin)
        } else {
            0
        };
        Ok(Self {
            reader,
            header,
            columns,
            indices,
            index_map,
            include_warmup,
            warmup: warmup_rows > 0,
            warmup_rows,
            draw: 0,
            offset,
            line_number,
        })
    }

    fn end_warmup(&mut self) {
        if self.warmup {
            self.warmup = false;
            self.draw = 0;
        }
    }

    fn counted_row(&mut self) {
        self.draw += 1;
        if self.warmup && self.draw >= self.warmup_rows {
            self.end_warmup();
        }
    }

    fn read_line(&mut self, line: &mut String) -> PyResult<usize> {
        line.clear();
        let n = self.reader.read_line(line).map_err(io_error)?;
        if n != 0 {
            self.offset += n as u64;
            self.line_number += 1;
        }
        Ok(n)
    }
}

enum BatchData {
    Rows(Vec<f64>),
    Columns(Vec<Vec<f64>>),
}

impl BatchData {
    fn new(ncol: usize, rows: usize, columnar: bool) -> Self {
        if columnar {
            Self::Columns((0..ncol).map(|_| Vec::with_capacity(rows)).collect())
        } else {
            Self::Rows(Vec::with_capacity(rows.saturating_mul(ncol)))
        }
    }

    fn push_line(
        &mut self,
        line: &str,
        map: &[Option<usize>],
        line_number: usize,
    ) -> PyResult<()> {
        let mut seen = 0;
        for (src_idx, field) in line.split(',').enumerate() {
            if src_idx >= map.len() {
                return Err(invalid(format!("too many values on line {line_number}")));
            }
            if let Some(dst_idx) = map[src_idx] {
                let number = field.trim().parse::<f64>().map_err(|_| {
                    invalid(format!(
                        "invalid float '{}' at line {line_number}, column {}",
                        field.trim(),
                        src_idx + 1
                    ))
                })?;
                match self {
                    Self::Rows(values) => values.push(number),
                    Self::Columns(columns) => columns[dst_idx].push(number),
                }
            }
            seen += 1;
        }
        if seen != map.len() {
            return Err(invalid(format!(
                "expected {} numeric values on line {line_number}, got {seen}",
                map.len()
            )));
        }
        Ok(())
    }

    fn into_python<'py>(
        self,
        py: Python<'py>,
        names: &[String],
        nrows: usize,
    ) -> PyResult<Py<PyAny>> {
        match self {
            Self::Rows(values) => {
                let data = Array2::from_shape_vec((nrows, names.len()), values)
                    .map_err(|err| invalid(err.to_string()))?;
                Ok(PyArray2::from_owned_array(py, data).into_any().unbind())
            }
            Self::Columns(columns) => {
                let result = PyDict::new(py);
                for (name, values) in names.iter().zip(columns) {
                    result.set_item(name, PyArray1::from_vec(py, values))?;
                }
                Ok(result.into_any().unbind())
            }
        }
    }
}

struct RawBatch {
    values: BatchData,
    nrows: usize,
    start_draw: usize,
    warmup: bool,
}

fn collect_batch(core: &mut CsvCore, max_rows: usize, columnar: bool) -> PyResult<Option<RawBatch>> {
    if max_rows == 0 {
        return Err(invalid("chunksize must be positive"));
    }
    let mut values = BatchData::new(core.columns.len(), max_rows, columnar);
    let mut nrows = 0;
    let mut start_draw = 0;
    let mut batch_warmup = core.warmup;
    let mut line = String::new();
    loop {
        if nrows >= max_rows {
            break;
        }
        let n = core.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            if trimmed.contains("Adaptation terminated") {
                core.end_warmup();
                if nrows > 0 && batch_warmup {
                    break;
                }
            }
            continue;
        }
        if core.warmup && !core.include_warmup {
            core.counted_row();
            continue;
        }
        if nrows == 0 {
            start_draw = core.draw;
            batch_warmup = core.warmup;
        }
        if core.warmup != batch_warmup {
            // State transitions happen after a row; this is defensive.
            return Err(invalid("warmup phase changed unexpectedly"));
        }
        values.push_line(trimmed, &core.index_map, core.line_number)?;
        nrows += 1;
        core.counted_row();
        if batch_warmup && !core.warmup {
            break;
        }
    }
    Ok((nrows > 0).then_some(RawBatch {
        values,
        nrows,
        start_draw,
        warmup: batch_warmup,
    }))
}

#[pyclass(module = "staniorust._native")]
struct CsvStream {
    core: CsvCore,
}

#[pymethods]
impl CsvStream {
    #[new]
    #[pyo3(signature = (path, columns=None, include_warmup=false))]
    fn new(path: String, columns: Option<Vec<String>>, include_warmup: bool) -> PyResult<Self> {
        Ok(Self {
            core: CsvCore::open(&path, columns.as_deref(), include_warmup)?,
        })
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.core.columns.clone()
    }

    #[getter]
    fn header(&self) -> Vec<String> {
        self.core.header.clone()
    }

    /// Returns (ndarray or {column: ndarray}, first 0-based draw index, is_warmup).
    fn next_chunk(
        &mut self,
        py: Python<'_>,
        chunksize: usize,
        columnar: bool,
    ) -> PyResult<Option<(Py<PyAny>, usize, bool)>> {
        let batch = py.allow_threads(|| collect_batch(&mut self.core, chunksize, columnar))?;
        match batch {
            None => Ok(None),
            Some(batch) => Ok(Some((
                batch
                    .values
                    .into_python(py, &self.core.columns, batch.nrows)?,
                batch.start_draw,
                batch.warmup,
            ))),
        }
    }
}

/// StanIO-style eager array with shape (chains, draws, columns).
#[pyfunction]
#[pyo3(signature = (paths, columns=None, include_warmup=false))]
fn read_all<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    columns: Option<Vec<String>>,
    include_warmup: bool,
) -> PyResult<(Vec<String>, Bound<'py, PyArray3<f64>>)> {
    if paths.is_empty() {
        return Err(invalid("at least one file is required"));
    }
    let (names, data, rows) = py.allow_threads(|| -> PyResult<(Vec<String>, Vec<f64>, usize)> {
        let mut first_header: Option<Vec<String>> = None;
        let mut selected_names = Vec::new();
        let mut all = Vec::new();
        let mut common_rows = None;
        for path in &paths {
            let mut core = CsvCore::open(path, columns.as_deref(), include_warmup)?;
            if let Some(ref header) = first_header {
                if *header != core.header {
                    return Err(invalid(format!("CSV headers differ: {path}")));
                }
            } else {
                first_header = Some(core.header.clone());
                selected_names = core.columns.clone();
            }
            let mut nrows = 0;
            while let Some(batch) = collect_batch(&mut core, 65_536, false)? {
                if let BatchData::Rows(values) = batch.values {
                    all.extend(values);
                }
                nrows += batch.nrows;
            }
            match common_rows {
                None => common_rows = Some(nrows),
                Some(prev) if prev == nrows => {}
                Some(prev) => {
                    return Err(invalid(format!(
                        "chains have unequal draw counts ({prev} versus {nrows} in {path}); use iter_csv"
                    )));
                }
            }
        }
        Ok((selected_names, all, common_rows.unwrap_or(0)))
    })?;
    let output = Array3::from_shape_vec((paths.len(), rows, names.len()), data)
        .map_err(|err| invalid(err.to_string()))?;
    Ok((names, PyArray3::from_owned_array(py, output)))
}

/// Scan only line offsets; do not parse numeric cells. Used for independent Dask tasks.
#[pyfunction]
#[pyo3(signature = (path, chunksize, columns=None, include_warmup=false))]
fn index_csv(
    py: Python<'_>,
    path: String,
    chunksize: usize,
    columns: Option<Vec<String>>,
    include_warmup: bool,
) -> PyResult<(Vec<String>, Vec<String>, Vec<usize>, Vec<Partition>)> {
    py.allow_threads(|| {
        if chunksize == 0 {
            return Err(invalid("chunksize must be positive"));
        }
        let mut core = CsvCore::open(&path, columns.as_deref(), include_warmup)?;
        let mut partitions = Vec::new();
        let mut begin = None;
        let mut count = 0_usize;
        let mut first_draw = 0_usize;
        let mut warmup = false;
        let mut line = String::new();
        loop {
            let line_start = core.offset;
            if core.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('#') {
                if trimmed.contains("Adaptation terminated") {
                    if let Some(start) = begin.take() {
                        partitions.push((start, line_start, first_draw, count, warmup));
                        count = 0;
                    }
                    core.end_warmup();
                }
                continue;
            }
            let row_phase = core.warmup;
            if include_warmup || !row_phase {
                if begin.is_none() {
                    begin = Some(line_start);
                    first_draw = core.draw;
                    warmup = row_phase;
                }
                count += 1;
            }
            core.counted_row();
            if let Some(start) = begin {
                if count >= chunksize || (row_phase && !core.warmup) {
                    partitions.push((start, core.offset, first_draw, count, warmup));
                    begin = None;
                    count = 0;
                }
            }
        }
        if let Some(start) = begin {
            partitions.push((start, core.offset, first_draw, count, warmup));
        }
        Ok((core.header, core.columns, core.indices, partitions))
    })
}

/// Read a byte-range that starts and ends on row boundaries, returning NumPy columns.
#[pyfunction]
fn read_partition<'py>(
    py: Python<'py>,
    path: String,
    start: u64,
    end: u64,
    ncols: usize,
    indices: Vec<usize>,
    names: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = py.allow_threads(|| -> PyResult<BatchData> {
        if end < start || indices.len() != names.len() {
            return Err(invalid("invalid CSV partition offsets or columns"));
        }
        let mut map = vec![None; ncols];
        for (dst, &src) in indices.iter().enumerate() {
            if src >= ncols || map[src].is_some() {
                return Err(invalid("invalid or duplicated selected column index"));
            }
            map[src] = Some(dst);
        }
        let mut file = File::open(path).map_err(io_error)?;
        file.seek(SeekFrom::Start(start)).map_err(io_error)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let mut columns = BatchData::new(names.len(), 1024, true);
        let mut position = start;
        let mut line_number = 0;
        let mut line = String::new();
        while position < end {
            line.clear();
            let n = reader.read_line(&mut line).map_err(io_error)?;
            if n == 0 {
                return Err(invalid("CSV file changed since partition offsets were indexed"));
            }
            position += n as u64;
            line_number += 1;
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                columns.push_line(trimmed, &map, line_number)?;
            }
        }
        if position != end {
            return Err(invalid("CSV partition ended inside a line; file may have changed"));
        }
        Ok(columns)
    })?;
    // The batches are column-wise, so each Python column owns one Rust Vec.
    let output = PyDict::new(py);
    if let BatchData::Columns(items) = columns {
        for (name, values) in names.iter().zip(items) {
            output.set_item(name, PyArray1::from_vec(py, values))?;
        }
    }
    Ok(output)
}

#[derive(Default)]
struct Group {
    name: String,
    entries: Vec<(usize, Vec<usize>)>,
}

fn decode_name(name: &str) -> PyResult<(String, Vec<usize>)> {
    // Tuple slots and complex components are kept under their own names,
    // while any numeric indices on a tuple slot are still reshaped.
    if name.ends_with(".real") || name.ends_with(".imag") {
        return Ok((name.to_string(), Vec::new()));
    }
    if let Some((base, rest)) = name.split_once(':') {
        if let Some((slot, suffix)) = rest.split_once('.') {
            if let Some(indices) = suffix.split('.').map(|s| s.parse::<usize>().ok().filter(|n| *n > 0)).collect::<Option<Vec<_>>>() {
                return Ok((format!("{base}:{slot}"), indices));
            }
        }
        return Ok((name.to_string(), Vec::new()));
    }
    if let Some((base, _)) = name.split_once('.') {
        let parts: Vec<&str> = name.split('.').skip(1).collect();
        if let Some(indices) = parts
            .iter()
            .map(|part| part.parse::<usize>().ok().filter(|n| *n > 0))
            .collect::<Option<Vec<_>>>()
        {
            return Ok((base.to_string(), indices));
        }
    }
    if let Some(bracket) = name.find('[') {
        if name.ends_with(']') {
            let parts = &name[bracket + 1..name.len() - 1];
            if let Some(indices) = parts
                .split(',')
                .map(|part| part.trim().parse::<usize>().ok().filter(|n| *n > 0))
                .collect::<Option<Vec<_>>>()
            {
                return Ok((name[..bracket].to_string(), indices));
            }
        }
    }
    Ok((name.to_string(), Vec::new()))
}

/// Reconstruct dense parameter shapes from the actual 1-based header indices in Rust.
#[pyfunction]
fn reshape<'py>(
    py: Python<'py>,
    names: Vec<String>,
    source: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let view = source.as_array();
    let shape = view.shape();
    if shape.is_empty() || *shape.last().unwrap_or(&0) != names.len() {
        return Err(invalid("last array axis must match the number of CSV columns"));
    }
    let raw = view
        .as_slice()
        .ok_or_else(|| invalid("reshaping requires a C-contiguous float64 array"))?;
    let prefix = &shape[..shape.len() - 1];
    let rows = prefix.iter().product::<usize>();
    let width = names.len();
    let mut groups: Vec<Group> = Vec::new();
    let mut positions = HashMap::<String, usize>::new();
    for (src_idx, name) in names.iter().enumerate() {
        let (base, indices) = decode_name(name)?;
        let group_idx = match positions.get(&base) {
            Some(&idx) => idx,
            None => {
                let idx = groups.len();
                groups.push(Group {
                    name: base.clone(),
                    ..Group::default()
                });
                positions.insert(base, idx);
                idx
            }
        };
        groups[group_idx].entries.push((src_idx, indices));
    }
    let result = PyDict::new(py);
    for group in groups {
        let rank = group.entries[0].1.len();
        if group.entries.iter().any(|(_, idx)| idx.len() != rank) {
            return Err(invalid(format!("mixed scalar and indexed columns: {}", group.name)));
        }
        let mut dims = vec![0; rank];
        for (_, indices) in &group.entries {
            for (axis, &index) in indices.iter().enumerate() {
                dims[axis] = dims[axis].max(index);
            }
        }
        let width_param = dims.iter().product::<usize>();
        if group.entries.len() != width_param {
            return Err(invalid(format!(
                "incomplete or non-rectangular variable '{}': selected {} cells, expected {width_param}",
                group.name,
                group.entries.len()
            )));
        }
        let mut src_to_dst = Vec::with_capacity(group.entries.len());
        let mut seen = vec![false; width_param];
        for (src, indices) in group.entries {
            let flat = indices
                .iter()
                .zip(&dims)
                .fold(0_usize, |acc, (&index, &size)| acc * size + (index - 1));
            if seen[flat] {
                return Err(invalid(format!("duplicate indexed element of {}", group.name)));
            }
            seen[flat] = true;
            src_to_dst.push((src, flat));
        }
        let mut output = vec![0.0; rows * width_param];
        for row in 0..rows {
            for &(src, dst) in &src_to_dst {
                output[row * width_param + dst] = raw[row * width + src];
            }
        }
        let mut result_shape = prefix.to_vec();
        result_shape.extend(&dims);
        let matrix = ArrayD::from_shape_vec(IxDyn(&result_shape), output)
            .map_err(|err| invalid(err.to_string()))?;
        result.set_item(group.name, PyArrayDyn::from_owned_array(py, matrix))?;
    }
    Ok(result)
}

fn flatten_numbers(value: &Value, depth: usize, dims: &mut Vec<usize>, out: &mut Vec<f64>) -> bool {
    match value {
        Value::Array(items) => {
            if dims.len() == depth {
                dims.push(items.len());
            } else if dims[depth] != items.len() {
                return false;
            }
            items
                .iter()
                .all(|item| flatten_numbers(item, depth + 1, dims, out))
        }
        Value::Number(number) if depth == dims.len() => {
            if let Some(num) = number.as_f64() {
                out.push(num);
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

fn json_value_to_python(py: Python<'_>, value: &Value, numeric_arrays: bool) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(flag) => Ok((*flag).into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            } else if let Some(value) = number.as_u64() {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(number
                    .as_f64()
                    .ok_or_else(|| invalid("invalid JSON number"))?
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            }
        }
        Value::String(text) => Ok(PyString::new(py, text).into_any().unbind()),
        Value::Array(items) => {
            if numeric_arrays {
                let mut dims = Vec::new();
                let mut values = Vec::new();
                if flatten_numbers(value, 0, &mut dims, &mut values) {
                    let data = ArrayD::from_shape_vec(IxDyn(&dims), values)
                        .map_err(|err| invalid(err.to_string()))?;
                    return Ok(PyArrayDyn::from_owned_array(py, data).into_any().unbind());
                }
            }
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_python(py, item, numeric_arrays)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(fields) => {
            let dictionary = PyDict::new(py);
            for (key, value) in fields {
                dictionary.set_item(key, json_value_to_python(py, value, numeric_arrays)?)?;
            }
            Ok(dictionary.into_any().unbind())
        }
    }
}


struct JsonSelection {
    keys: Option<HashSet<String>>,
}

impl<'de> DeserializeSeed<'de> for JsonSelection {
    type Value = serde_json::Map<String, Value>;

    fn deserialize<D: Deserializer<'de>>(self, de: D) -> Result<Self::Value, D::Error> {
        de.deserialize_map(JsonSelectionVisitor { keys: self.keys })
    }
}

struct JsonSelectionVisitor {
    keys: Option<HashSet<String>>,
}

impl<'de> Visitor<'de> for JsonSelectionVisitor {
    type Value = serde_json::Map<String, Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a Stan-compatible JSON object")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if self.keys.as_ref().is_none_or(|keys| keys.contains(&key)) {
                values.insert(key, map.next_value::<Value>()?);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(values)
    }
}

fn parse_selected_json<'a, R: std::io::Read>(reader: R, keys: Option<&'a [String]>) -> PyResult<serde_json::Map<String, Value>> {
    let selection = JsonSelection {
        keys: keys.map(|items| items.iter().cloned().collect()),
    };
    let mut decoder = serde_json::Deserializer::from_reader(reader);
    let result = selection.deserialize(&mut decoder)
        .map_err(|e| invalid(format!("invalid Stan JSON: {e}")))?;
    decoder.end().map_err(|e| invalid(format!("extra JSON content: {e}")))?;
    if let Some(requested) = keys {
        for key in requested {
            if !result.contains_key(key) {
                return Err(invalid(format!("unknown JSON key: {key}")));
            }
        }
    }
    Ok(result)
}

fn jsonify<'py>(py: Python<'py>, root: serde_json::Map<String, Value>, numeric_arrays: bool) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (key, value) in root {
        output.set_item(key, json_value_to_python(py, &value, numeric_arrays)?)?;
    }
    Ok(output)
}

/// Read Stan input JSON. Unselected top-level keys are not materialized.
#[pyfunction]
#[pyo3(signature = (path, numeric_arrays=false, keys=None))]
fn read_json<'py>(py: Python<'py>, path: String, numeric_arrays: bool, keys: Option<Vec<String>>) -> PyResult<Bound<'py, PyDict>> {
    let root = py.allow_threads(|| -> PyResult<_> {
        let file = File::open(&path).map_err(io_error)?;
        parse_selected_json(BufReader::with_capacity(256 * 1024, file), keys.as_deref())
    })?;
    jsonify(py, root, numeric_arrays)
}

/// Read JSON from cloud-fetched bytes without constructing Python JSON objects.
#[pyfunction]
#[pyo3(signature = (bytes, numeric_arrays=false, keys=None))]
fn read_json_bytes<'py>(py: Python<'py>, bytes: Vec<u8>, numeric_arrays: bool, keys: Option<Vec<String>>) -> PyResult<Bound<'py, PyDict>> {
    let root = py.allow_threads(|| parse_selected_json(bytes.as_slice(), keys.as_deref()))?;
    jsonify(py, root, numeric_arrays)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    mmap::register(m)?;
    m.add_class::<CsvStream>()?;
    m.add_function(wrap_pyfunction!(read_all, m)?)?;
    m.add_function(wrap_pyfunction!(index_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_partition, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(read_json, m)?)?;
    m.add_function(wrap_pyfunction!(read_json_bytes, m)?)?;
    Ok(())
}
