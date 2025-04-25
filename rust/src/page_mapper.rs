//  Copyright (C) 2025 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use crate::utils::{invalid_data, read_skipping_ws, skip_serde_json_value};
use std::io::{Error, Read, Seek};

pub trait ParsedJsonSection: Sized {
    fn parse(buf_key: String, reader: impl Read + Seek) -> Result<Box<Self>, Error>;
}

pub trait JsonPageMapper<T>: Sized
where
    T: ParsedJsonSection,
{
    fn parse_json(mut reader: impl Read + Seek, strict: bool) -> Result<Vec<Box<T>>, Error> {
        let mut brace_level = 0;
        let mut json_sections = Vec::new();

        while let Ok(c) = read_skipping_ws(&mut reader) {
            match c {
                b'{' => brace_level += 1,
                b'"' => {
                    let mut buf_key = Vec::new();
                    while let Ok(c) = read_skipping_ws(&mut reader) {
                        if c == b'"' {
                            break;
                        }
                        buf_key.push(c);
                    }
                    match String::from_utf8(buf_key.clone()) {
                        Ok(key) => {
                            match T::parse(key.clone(), &mut reader) {
                                Ok(section) => {
                                    json_sections.push(section);
                                }
                                Err(e) => {
                                    if strict {
                                        return Err(e);
                                    } else {
                                        // Skip unknown section
                                        eprintln!("Skipping unknown section key: \"{}\"", key);
                                        read_skipping_ws(&mut reader)?;
                                        let _ = skip_serde_json_value(&mut reader);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let cur_pos = reader.stream_position()?;
                            let msg = format!(
                                "Section key buffer, {:?} is invalid at pos: {}. {}",
                                buf_key, cur_pos, e
                            );
                            let err = invalid_data(msg.as_str());
                            return Err(err);
                        }
                    }
                }
                b',' => {
                    continue;
                }
                b'}' => {
                    brace_level -= 1;
                    if brace_level == 0 {
                        break;
                    }
                }
                _ => {
                    let cur_pos = reader.stream_position()?;
                    let msg = format!("{} is invalid character at pos: {}", c, cur_pos);
                    let err = invalid_data(msg.as_str());
                    return Err(err);
                }
            }
        }
        Ok(json_sections)
    }
}
