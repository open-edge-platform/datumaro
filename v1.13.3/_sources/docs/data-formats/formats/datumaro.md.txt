# Datumaro

## Format specification

Datumaro format is [Datumaro](https://github.com/open-edge-platform/datumaro)'s own data format.
It aims to cover all media types and annotation types in Datumaro as possible.
Therefore, if you do not want information loss when re-importing your dataset by [Datumaro](https://github.com/open-edge-platform/datumaro), we recommend exporting your dataset using the Datumaro format.
In addition, you can directly use the Datumaro format for the model training using [OpenVINO™ Training Extensions](https://github.com/openvinotoolkit/training_extensions).

Supported media types:

- `Image`
- `PointCloud`
- `Video`
- `VideoFrame`

Supported annotation types:

- `Label`
- `Mask`
- `PolyLine`
- `Polygon`
- `Bbox`
- `Points`
- `Caption`
- `Cuboid3d`
- `Ellipse`

Supported annotation attributes:

- No restrictions

## Convert Datumaro dataset

A Datumaro dataset can be converted in the following way:

```console
datum convert --input-format datumaro --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

A Datumaro dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->

```
└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── images/
    │   ├── <subset_name_1>/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── <subset_name_2> /
    │       ├── <image_name1.ext>
    │       ├── <image_name2.ext>
    │       └── ...
    ├── videos/  # directory to store video files
    │   ├── <subset_name_1>/
    │   │   ├── <video_name1.ext>
    │   │   ├── <video_name2.ext>
    │   │   └── ...
    │   └── <subset_name_2> /
    │       ├── <video_name1.ext>
    │       ├── <video_name2.ext>
    │       └── ...
    └── annotations/
        ├── <subset_name_1>.json
        ├── <subset_name_2>.json
        └── ...
```

Note that the subset name shouldn't contain path separators.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

## Export to other formats

It can convert Datumaro dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a Datumaro dataset to other dataset formats
using CLI:

- Convert a dataset from Datumaro format to VOC format:

```console
datum convert --input-format datumaro --input-path <path/to/dataset> \
    --output-format voc --output-dir <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to Datumaro

There are several ways to convert a dataset to Datumaro format:

```console
# converting to Datumaro format from other format
datum convert --input-format voc --input-path <path/to/dataset> \
    --output-format datumaro --output-dir <output/dir> -- --save-media
```

Extra options for exporting to Datumaro format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/tree/develop/tests/unit/data_formats/datumaro/test_datumaro_format.py)
