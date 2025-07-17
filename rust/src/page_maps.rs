//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use crate::utils::{
    invalid_data, parse_serde_json_value_from_page, read_skipping_ws, stream_error,
};
use std::{
    collections::HashMap,
    io::{self},
};

fn is_empty_list(mut reader: impl io::Read + io::Seek) -> Result<(bool, u64), io::Error> {
    let curr_pos = reader.stream_position()?;
    let mut empty_list_str = [0u8; 2];
    for i in 0..2 {
        if let Ok(c) = read_skipping_ws(&mut reader) {
            empty_list_str[i] = c;
        }
    }

    if empty_list_str == "[]".as_bytes() {
        Ok((true, curr_pos))
    } else {
        Ok((false, curr_pos))
    }
}

pub type JsonDict = serde_json::Value;

#[derive(Debug)]
pub struct ImgPage {
    pub offset: u64,
    pub size: u32,
}

pub trait ItemPageMapKeyTrait:
    std::cmp::Eq + std::hash::Hash + std::clone::Clone + std::fmt::Display
{
    fn get_id(parsed_map: HashMap<String, JsonDict>, offset: u64) -> Result<Self, io::Error>
    where
        Self: Sized;
}

impl ItemPageMapKeyTrait for i64 {
    fn get_id(parsed_map: HashMap<String, JsonDict>, offset: u64) -> Result<Self, io::Error> {
        parsed_map
            .get("id")
            .ok_or(stream_error("Cannot find an image id", offset))?
            .as_i64()
            .ok_or(stream_error("The image id is not an integer.", offset))
    }
}

impl ItemPageMapKeyTrait for String {
    fn get_id(parsed_map: HashMap<String, JsonDict>, offset: u64) -> Result<Self, io::Error> {
        Ok(parsed_map
            .get("id")
            .ok_or(stream_error("Cannot find an image id", offset))?
            .as_str()
            .ok_or(stream_error("The image id is not an integer.", offset))?
            .to_string())
    }
}

#[derive(Debug)]
pub struct ImgPageMap<T>
where
    T: ItemPageMapKeyTrait,
{
    ids: Vec<T>,
    pages: HashMap<T, ImgPage>,
}

impl<T> ImgPageMap<T>
where
    T: ItemPageMapKeyTrait,
{
    pub fn get_dict<R>(&self, reader: &mut R, img_id: &T) -> Result<JsonDict, io::Error>
    where
        R: io::Read + io::Seek,
    {
        match self.pages.get(img_id) {
            Some(page) => parse_serde_json_value_from_page(reader, page.offset, page.size as u64),
            None => Err(invalid_data(
                format!("Image id: {} is not on the page map", img_id).as_str(),
            )),
        }
    }

    pub fn push(&mut self, img_id: T, page: ImgPage) {
        self.ids.push(img_id.clone());
        self.pages.insert(img_id.clone(), page);
    }

    pub fn from_reader(mut reader: impl io::Read + io::Seek) -> Result<ImgPageMap<T>, io::Error> {
        let mut page_map = ImgPageMap::default();

        let (empty, rewind_pos) = is_empty_list(&mut reader)?;

        if empty {
            return Ok(page_map);
        } else {
            reader.seek(io::SeekFrom::Start(rewind_pos))?;
        }

        while let Ok(c) = read_skipping_ws(&mut reader) {
            match c {
                b'[' | b',' => {
                    let curr_pos = reader.stream_position()?;
                    let de = serde_json::Deserializer::from_reader(&mut reader);
                    let mut stream = de.into_iter::<HashMap<String, serde_json::Value>>();
                    let offset = curr_pos + stream.byte_offset() as u64;

                    match stream.next().ok_or_else(|| stream_error("Unexpected end of stream", offset))? {
                        Ok(parsed_map) => {
                            let id = ItemPageMapKeyTrait::get_id(parsed_map, offset)?;

                            let size = (curr_pos + stream.byte_offset() as u64 - offset) as u32;
                            page_map.push(id, ImgPage { offset, size });
                        }
                        Err(e) => {
                            return Err(stream_error(e.to_string().as_str(), offset));
                        }
                    }
                }
                b']' => break,
                _ => {}
            }
        }
        Ok(page_map)
    }

    pub fn ids(&self) -> &Vec<T> {
        return &self.ids;
    }
}

impl<T> IntoIterator for ImgPageMap<T>
where
    T: ItemPageMapKeyTrait,
{
    type Item = (T, ImgPage);

    type IntoIter = <HashMap<T, ImgPage> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.pages.into_iter()
    }
}

impl<T> Default for ImgPageMap<T>
where
    T: ItemPageMapKeyTrait,
{
    fn default() -> Self {
        Self {
            ids: Vec::with_capacity(0),
            pages: HashMap::with_capacity(0),
        }
    }
}

#[derive(Debug)]
pub struct AnnPage {
    pub id: i64,
    pub offset: u64,
    pub size: u32,
    pub ptr: usize,
}

#[derive(Debug)]
pub struct AnnPageMap {
    pages: Vec<AnnPage>,
    head_pointers: HashMap<i64, usize>,
}

impl AnnPageMap {
    pub fn get_anns<R>(&self, reader: &mut R, img_id: i64) -> Result<Vec<JsonDict>, io::Error>
    where
        R: io::Read + io::Seek,
    {
        let curr_ptr = self.head_pointers.get(&img_id);

        match curr_ptr {
            Some(head) => {
                let mut anns = vec![];
                let mut ptr = *head;

                loop {
                    if ptr == usize::MAX {
                        break;
                    }
                    let page = &self.pages[ptr];
                    ptr = page.ptr;

                    match parse_serde_json_value_from_page(reader, page.offset, page.size as u64) {
                        Ok(v) => {
                            anns.push(v);
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }

                Ok(anns)
            }
            None => Ok(vec![]),
        }
    }

    pub fn push(&mut self, ann_id: i64, img_id: i64, offset: u64, size: u32) {
        let lookup = self.head_pointers.get(&img_id);

        let mut ptr = usize::MAX;
        if let Some(idx) = lookup {
            ptr = *idx;
        }
        let new_head_idx = self.pages.len();
        self.pages.push(AnnPage {
            id: ann_id,
            offset,
            size,
            ptr,
        });
        self.head_pointers.insert(img_id, new_head_idx);
    }

    pub fn from_reader(mut reader: impl io::Read + io::Seek) -> Result<AnnPageMap, io::Error> {
        let mut page_map = AnnPageMap::default();

        let (empty, rewind_pos) = is_empty_list(&mut reader)?;

        if empty {
            return Ok(page_map);
        } else {
            reader.seek(io::SeekFrom::Start(rewind_pos))?;
        }

        let mut missing_ann_id = 0;

        while let Ok(c) = read_skipping_ws(&mut reader) {
            match c {
                b'[' | b',' => {
                    let curr_pos = reader.stream_position()?;
                    let de = serde_json::Deserializer::from_reader(&mut reader);
                    let mut stream = de.into_iter::<HashMap<String, serde_json::Value>>();
                    let offset = curr_pos + stream.byte_offset() as u64;

                    match stream.next().ok_or_else(|| stream_error("Unexpected end of stream", offset))? {
                        Ok(parsed_map) => {
                            let ann_id = if let Some(v) = parsed_map.get("id") {
                                v.as_i64().ok_or(stream_error(
                                    "The annotation id is not an integer.",
                                    offset,
                                ))?
                            } else {
                                let new_id = missing_ann_id;
                                missing_ann_id += 1;
                                new_id
                            };

                            let img_id = parsed_map
                                .get("image_id")
                                .ok_or(stream_error("Cannot find an image id", offset))?
                                .as_i64()
                                .ok_or(stream_error("The image id is not an integer.", offset))?;

                            let size = (curr_pos + stream.byte_offset() as u64 - offset) as u32;
                            page_map.push(ann_id, img_id, offset, size);
                        }
                        Err(e) => {
                            return Err(stream_error(e.to_string().as_str(), offset));
                        }
                    }
                }
                b']' => break,
                _ => {}
            }
        }
        Ok(page_map)
    }
}

impl Default for AnnPageMap {
    fn default() -> Self {
        Self {
            pages: Vec::with_capacity(0),
            head_pointers: HashMap::with_capacity(0),
        }
    }
}
