# Datumaro Format

So far, in the field of computer vision, there are various tasks such as classification, detection,
and segmentation, as well as pose estimation and visual tracking, and public data is used by providing
a format suitable for each task. Even within the same segmentation task, some data formats provide
annotation information as polygons, while others provide mask form. In order to ensure compatibility
with different tasks and formats, we provide a novel Datumaro format with `.json` ([Datumaro](./formats/datumaro)) or `.datum` ([DatumaroBinary](./formats/datumaro_binary)) extensions.

A variety of metadata can be stored in the datumaro format. First of all, `dm_format_version` field
is provided for backward compatibility to help with data version tracing and various metadata can be
added to the `info` field. For example, you can record task types such as detection and segmentation,
or record data creation time. Labels and attributes can be saved in the `categories` field, and mask
colormap information can be saved. In addition, in the datumaro format, in order to respond to
hierarchical classification or multi-label classification tasks, `label_group` is provided to record
whether or not enabling multiple selection between labels in a group and the `parent` is provided to
specify the parent label for each label. Finally, in the `item` field, we can write the annotation
information for each media id, and additionally write the data path and data size.

Here is the example of `json` annotation file:

```json
{
    "dm_format_version": "1.0",
    "info": {
        "task": "anomaly_detection",
        "creation time": "2023.4.1"
    },
    "categories": {
        "label": {
            "labels": [
                {
                    "name": "Normal",
                    "parent": "",
                    "attributes": []
                },
                {
                    "name": "Anomalous",
                    "parent": "",
                    "attributes": []
                }
            ],
            "label_groups": [
                {
                    "name": "Label",
                    "group_type": "exclusive",
                    "labels": [
                        "Anomalous",
                        "Normal"
                    ]
                }
            ],
            "attributes": []
        },
        "mask": {
            "colormap": [
                {
                    "label_id": 1,
                    "r": 255,
                    "g": 255,
                    "b": 255
                }
            ]
        }
    },
    "items": [
        {
            "id": "good_001",
            "annotations": [
                {
                    "id": 0,
                    "type": "label",
                    "attributes": {},
                    "group": 0,
                    "label_id": 0
                }
            ],
            "image": {
                "path": "good_001.jpg",
                "size": [
                    900,
                    900
                ]
            }
        },
        {
            "id": "broken_small_001",
            "annotations": [
                {
                    "id": 0,
                    "type": "bbox",
                    "attributes": {},
                    "group": 0,
                    "label_id": 1,
                    "z_order": 0,
                    "bbox": [
                        350.8999938964844,
                        151.3899993896484,
                        275.1399841308594,
                        126.4900054931640
                    ]
                }
            ],
            "image": {
                "path": "broken_small_001.jpg",
                "size": [
                    900,
                    900
                ]
            }
        },
    ]
}
```

A Datumaro format directory have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── dataset_meta.json # a list of non-format labels (optional)
├── images/
│   ├── train/  # directory with training images
│   |    ├── img001.png
│   |    ├── img002.png
│   |    └── ...
│   ├── val/  # directory with validation images
│   |    ├── img001.png
│   |    ├── img002.png
│   |    └── ...
│   └── ...
├── videos/  # directory to store video files
└── annotations/
    ├── train.json  # annotation file with training data
    ├── val.json  # annotation file with validation data
    └── ...
```
