# Datumaro

## Format specification

Datumaro format is [Datumaro](https://github.com/openvinotoolkit/datumaro)'s own data format.
It aims to cover all media types and annotation types in Datumaro as possible.
Therefore, if you do not want information loss when re-importing your dataset by [Datumaro](https://github.com/openvinotoolkit/datumaro), we recommend exporting your dataset using the Datumaro format.
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

## Import Datumaro dataset

A Datumaro project with a Datumaro source can be created in the following way:

```console
datum project create
datum project import --format datumaro <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

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

If your dataset is not following the above directory structure,
it cannot detect and import your dataset as the Datumaro format properly.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

It can convert Datumaro dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a Datumaro dataset to other dataset formats
using CLI:

- Export a dataset from Datumaro format to VOC format:

```console
datum project create
datum project import -f datumaro <path/to/dataset>
datum project export -f voc -o <output/dir>
```

or

```console
datum convert -if datumaro -i <path/to/dataset> -f voc -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to Datumaro

There are several ways to convert a dataset to Datumaro format:

- Export a dataset from an existing project to Datumaro format:
```console
# export dataset into Datumaro format from existing project
datum project export -p <path/to/project> -f datumaro -o <output/dir> \
    -- --save-media
```

- Convert a dataset from VOC format to Datumaro format:
```console
# converting to Datumaro format from other format
datum convert -if voc -i <path/to/dataset> \
    -f datumaro -o <output/dir> -- --save-media
```

Extra options for exporting to Datumaro format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/data_formats/datumaro/test_datumaro_format.py)
