//  Copyright (C) 2025 Intel Corporation
//
//  SPDX-License-Identifier: MIT

mod test_helpers;

use datumaro_rust_api::coco_page_mapper::CocoPageMapperImpl;
use test_helpers::prepare_reader;

#[test]
fn test_instance() {
    const EXAMPLE: &str = r#"
    {
        "licenses":[{"name":"test_instance()","id":0,"url":""}],
        "info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},
        "categories":[
            {"id":1,"name":"a","supercategory":""},
            {"id":2,"name":"b","supercategory":""},
            {"id":4,"name":"c","supercategory":""}
        ],
        "images":[
            {"id":5,"width":10,"height":5,"file_name":"a.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0},
            {"id":6,"width":10,"height":5,"file_name":"b.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0}
        ],
        "annotations":[
            {"id":1,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
            {"id":2,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
            {"id":3,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
            {"id":4,"image_id":6,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
            {"id":5,"image_id":6,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0}
        ]
    }"#;

    let (tempfile, mut reader) = prepare_reader(EXAMPLE);
    let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

    println!("{:?}", coco_page_mapper);

    for img_id in [5, 6] {
        let item = coco_page_mapper
            .get_item_dict(&img_id, &mut reader)
            .unwrap();

        assert_eq!(item["id"].as_i64(), Some(img_id));

        let anns = coco_page_mapper.get_anns_dict(img_id, &mut reader).unwrap();
        assert!(anns.len() > 0);

        for ann in anns {
            assert_eq!(ann["image_id"].as_i64(), Some(img_id));
        }
    }
}

#[test]
fn test_unknown_keys() {
    // Tests if the dataset can be parsed when the JSON contains additional unknown keys
    const EXAMPLE: &str = r#"
    {
        "licenses":[{"name":"","id":0,"url":""}],
        "info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},
        "categories":[]
        "images":[],
        "annotations":[]
        "unknown_key": "unknown_value",
    }"#;

    let (tempfile, mut reader) = prepare_reader(EXAMPLE);
    let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

    println!("{:?}", coco_page_mapper);
}

#[test]
fn test_image_info_default() {
    const EXAMPLE: &str = r#"
    {"licenses": [{"name": "", "id": 0, "url": ""}], "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}, "categories": [], "images": [{"id": 1, "width": 2, "height": 4, "file_name": "1.jpg", "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}], "annotations": []}
    "#;

    let (tempfile, mut reader) = prepare_reader(EXAMPLE);
    let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

    println!("{:?}", coco_page_mapper);
}

#[test]
fn test_panoptic_has_no_ann_id() {
    const EXAMPLE: &str = r#"
    {"licenses":[{"name":"","id":0,"url":""}],"info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},"categories":[{"id":1,"name":"0","supercategory":"","isthing":0},{"id":2,"name":"1","supercategory":"","isthing":0},{"id":3,"name":"2","supercategory":"","isthing":0},{"id":4,"name":"3","supercategory":"","isthing":0},{"id":5,"name":"4","supercategory":"","isthing":0},{"id":6,"name":"5","supercategory":"","isthing":0},{"id":7,"name":"6","supercategory":"","isthing":0},{"id":8,"name":"7","supercategory":"","isthing":0},{"id":9,"name":"8","supercategory":"","isthing":0},{"id":10,"name":"9","supercategory":"","isthing":0}],"images":[{"id":1,"width":4,"height":4,"file_name":"1.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0}],"annotations":[{"image_id":1,"file_name":"1.png","segments_info":[{"id":3,"category_id":5,"area":5.0,"bbox":[1.0,0.0,2.0,2.0],"iscrowd":0}]}]}
    "#;

    let (tempfile, mut reader) = prepare_reader(EXAMPLE);
    let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

    println!("{:?}", coco_page_mapper);
}
