//  Copyright (C) 2025 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::io::{self};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyList, PyUnicode},
};
use serde::de::{IgnoredAny, Deserialize};
use serde_json::{self, Deserializer};


pub fn read_skipping_ws(mut reader: impl io::Read) -> io::Result<u8> {
    loop {
        let mut byte = 0u8;
        reader.read_exact(std::slice::from_mut(&mut byte))?;
        if !byte.is_ascii_whitespace() {
            return Ok(byte);
        }
    }
}

pub fn invalid_data(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn stream_error(error: &str, offset: u64) -> io::Error {
    let msg = format!("[Parse error, offset={}] {}", offset, error);
    invalid_data(msg.as_str())
}

pub fn parse_serde_json_value_from_page<R>(
    reader: &mut R,
    offset: u64,
    size: u64,
) -> Result<serde_json::Value, io::Error>
where
    R: io::Read + io::Seek,
{
    reader.seek(io::SeekFrom::Start(offset))?;

    let mut buf = vec![0u8; size as usize];
    let _ = reader.read(buf.as_mut_slice())?;

    let img_dict_str = String::from_utf8(buf).ok().ok_or(invalid_data(
        format!("Cannot read offset: {} and size: {}", offset, size).as_str(),
    ))?;

    serde_json::from_str(img_dict_str.as_str())
        .ok()
        .ok_or(invalid_data(
            format!("Cannot parse to dict offset: {} and size: {}", offset, size).as_str(),
        ))
}

pub fn parse_serde_json_value(
    reader: impl io::Read + io::Seek,
) -> Result<serde_json::Value, io::Error> {
    let de = serde_json::Deserializer::from_reader(reader);
    let mut stream = de.into_iter::<serde_json::Value>();
    match stream.next().unwrap() {
        Ok(x) => Ok(x),
        Err(e) => {
            let cur_pos = stream.byte_offset();
            let msg = format!("Parse error: {} at pos: {}", e, cur_pos);
            Err(invalid_data(msg.as_str()))
        }
    }
}

pub fn skip_serde_json_value<R: io::Read>(reader: &mut R) -> Result<(), std::io::Error> {
    // Skip the next JSON value in the stream
    let mut de = Deserializer::from_reader(reader);
    match IgnoredAny::deserialize(&mut de) {
        Ok(_) => Ok(()),
        Err(_) => Err(io::Error::new(io::ErrorKind::InvalidData, "Failed to skip JSON value")),
    }
}

pub fn convert_to_py_object(value: &serde_json::Value, py: Python<'_>) -> PyResult<PyObject> {
    if value.is_array() {
        let list = PyList::empty(py);

        for child in value.as_array().unwrap() {
            list.append(convert_to_py_object(child, py)?)?;
        }

        return Ok(list.into());
    } else if value.is_object() {
        let dict = PyDict::new(py);

        for (key, child) in value.as_object().unwrap().iter() {
            let child = convert_to_py_object(child, py)?;
            dict.set_item(key, child)?;
        }

        return Ok(dict.into());
    } else if value.is_boolean() {
        return Ok(PyBool::new(py, value.as_bool().unwrap()).into());
    } else if value.is_f64() {
        return Ok(PyFloat::new(py, value.as_f64().unwrap()).into());
    } else if value.is_i64() {
        return Ok(value.as_i64().unwrap().to_object(py));
    } else if value.is_u64() {
        return Ok(value.as_u64().unwrap().to_object(py));
    } else if value.is_string() {
        return Ok(PyUnicode::new(py, value.as_str().unwrap()).into());
    } else if value.is_null() {
        return Ok(py.None());
    } else {
        return Err(PyValueError::new_err("Unknown value type"));
    }
}
