//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

pub mod coco_page_mapper;
pub mod datum_page_mapper;
pub mod json_section_page_mapper;
mod page_mapper;
mod page_maps;
mod utils;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::coco_page_mapper::CocoPageMapper;
use crate::datum_page_mapper::DatumPageMapper;
use crate::json_section_page_mapper::JsonSectionPageMapper;

/// Datumaro Rust API
#[pymodule]
#[pyo3(name = "rust_api")]
fn rust_api(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CocoPageMapper>()?;
    m.add_class::<DatumPageMapper>()?;
    m.add_class::<JsonSectionPageMapper>()?;
    Ok(())
}
