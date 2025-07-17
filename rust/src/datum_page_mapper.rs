//  Copyright (C) 2025 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::{
    fs::File,
    io::{self, BufReader, Read, Seek},
    path::Path,
    str::FromStr,
};
use strum::EnumString;

use crate::{
    page_mapper::{JsonPageMapper, ParsedJsonSection},
    page_maps::{ImgPageMap, JsonDict},
    utils::{convert_to_py_object, invalid_data, parse_serde_json_value, read_skipping_ws},
};
use pyo3::prelude::*;
use serde_json::json;
#[derive(EnumString, Debug)]
#[strum(ascii_case_insensitive, serialize_all = "snake_case")]
pub enum DatumJsonSection {
    DmFormatVersion(String),
    MediaType(i64),
    Infos(JsonDict),
    Categories(JsonDict),
    Items(ImgPageMap<String>),
}

impl ParsedJsonSection for DatumJsonSection {
    fn parse(
        buf_key: String,
        mut reader: impl Read + Seek,
    ) -> Result<Box<DatumJsonSection>, io::Error> {
        match DatumJsonSection::from_str(buf_key.as_str()) {
            Ok(curr_key) => {
                while let Ok(c) = read_skipping_ws(&mut reader) {
                    if c == b':' {
                        break;
                    }
                }
                match curr_key {
                    DatumJsonSection::DmFormatVersion(_) => {
                        let v = parse_serde_json_value(reader)?
                            .as_str()
                            .ok_or(invalid_data(
                                "Cannot parse datumaro format version from the json file",
                            ))?
                            .to_string();
                        Ok(Box::new(DatumJsonSection::DmFormatVersion(v)))
                    }
                    DatumJsonSection::MediaType(_) => {
                        let v = parse_serde_json_value(reader)?
                            .as_i64()
                            .ok_or(invalid_data("Cannot parse media type from the json file"))?;
                        Ok(Box::new(DatumJsonSection::MediaType(v)))
                    }
                    DatumJsonSection::Infos(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(DatumJsonSection::Infos(v)))
                    }
                    DatumJsonSection::Categories(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(DatumJsonSection::Categories(v)))
                    }
                    DatumJsonSection::Items(_) => {
                        let v = ImgPageMap::from_reader(reader)?;
                        Ok(Box::new(DatumJsonSection::Items(v)))
                    }
                }
            }
            Err(e) => {
                let cur_pos = reader.stream_position()?;
                let msg = format!("Unknown key: {} at pos: {}", e, cur_pos);
                Err(invalid_data(msg.as_str()))
            }
        }
    }
}

#[derive(Debug)]
pub struct DatumPageMapperImpl {
    dm_format_version: Option<String>,
    media_type: Option<i64>,
    infos: JsonDict,
    categories: JsonDict,
    items: ImgPageMap<String>,
}

impl JsonPageMapper<DatumJsonSection> for DatumPageMapperImpl {}

impl DatumPageMapperImpl {
    pub fn dm_format_version(&self) -> &Option<String> {
        return &self.dm_format_version;
    }
    pub fn media_type(&self) -> &Option<i64> {
        return &self.media_type;
    }
    pub fn infos(&self) -> &JsonDict {
        return &self.infos;
    }
    pub fn categories(&self) -> &JsonDict {
        return &self.categories;
    }
    pub fn get_img_ids(&self) -> &Vec<String> {
        self.items.ids()
    }
    pub fn get_item_dict(
        &self,
        img_id: &String,
        mut reader: impl Read + Seek,
    ) -> Result<JsonDict, io::Error> {
        self.items.get_dict(&mut reader, img_id)
    }

    pub fn new(mut reader: impl Read + Seek) -> Result<Self, io::Error> {
        let sections = Self::parse_json(&mut reader, true)?;

        let mut dm_format_version = None;
        let mut media_type = None;
        let mut infos = None;
        let mut categories = None;
        let mut items = None;

        for section in sections {
            match *section {
                DatumJsonSection::DmFormatVersion(v) => {
                    dm_format_version = Some(v);
                }
                DatumJsonSection::MediaType(v) => {
                    media_type = Some(v);
                }
                DatumJsonSection::Infos(v) => {
                    infos = Some(v);
                }
                DatumJsonSection::Categories(v) => {
                    categories = Some(v);
                }
                DatumJsonSection::Items(v) => {
                    items = Some(v);
                }
            }
        }
        let infos = infos.unwrap_or(json!({}));
        let categories = categories.ok_or(invalid_data("Cannot find the categories section."))?;
        let items = items.ok_or(invalid_data("Cannot find the items section."))?;

        Ok(DatumPageMapperImpl {
            dm_format_version,
            media_type,
            infos,
            categories,
            items,
        })
    }
}

#[pyclass]
pub struct DatumPageMapper {
    reader: BufReader<File>,
    mapper: DatumPageMapperImpl,
}

#[pymethods]
impl DatumPageMapper {
    #[new]
    fn py_new(path: String) -> PyResult<Self> {
        let file = File::open(Path::new(&path))?;
        let mut reader = BufReader::new(file);
        let mapper = DatumPageMapperImpl::new(&mut reader)?;

        Ok(DatumPageMapper { reader, mapper })
    }

    fn dm_format_version(self_: PyRef<Self>) -> Option<String> {
        self_.mapper.dm_format_version().clone()
    }

    fn media_type(self_: PyRef<Self>) -> Option<i64> {
        self_.mapper.media_type().clone()
    }

    fn infos(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.infos(), self_.py())
    }

    fn categories(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.categories(), self_.py())
    }

    fn get_item_dict(&mut self, py: Python<'_>, img_id: String) -> PyResult<PyObject> {
        let item_dict = self.mapper.get_item_dict(&img_id, &mut self.reader)?;
        Ok(convert_to_py_object(&item_dict, py)?)
    }

    fn get_img_ids(&self) -> Vec<String> {
        self.mapper.get_img_ids().to_owned()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.mapper.get_img_ids().len())
    }
}
