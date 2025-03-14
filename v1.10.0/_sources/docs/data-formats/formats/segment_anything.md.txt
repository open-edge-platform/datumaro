# Segment Anything

## Format specification

SA-1B format specification is available [here](https://github.com/facebookresearch/segment-anything#dataset).

The SA-1B is a dataset consisting of 1 billion segmentation masks model generated.
The model is known as [SAM](https://ai.facebook.com/research/publications/segment-anything/) from FAIR and
it can produce reliable and accurate segmentation masks given user prompts,
such as points, boxes, or natural sentences.

Supported media types:
- `Image`

Supported annotation types:
- `Bbox`
- `Mask`
- `Polygon`
- `Ellipse`

Supported annotation attributes:
- `predicted_iou`
- `stability_score`
- `crop_box`
- `point_coords`

## Import SA-1B dataset

The SA-1B dataset is available [here](https://ai.facebook.com/datasets/segment-anything-downloads/)

A Datumaro project with a SA-1B source can be created in the following way:

```console
datum project create
datum project import --format segment_anything <path/to/dataset>
```

It is possible to specify the project name and the project directory. Run
`datum project create --help` for more information.

An SA-1B dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->

```
└─ Dataset/
    ├── <name1.ext>
    ├── <name1>.json
    ├── <name2.ext>
    ├── <name2>.json
    └── ...
```

If your dataset is not following the above directory structure,
it cannot detect and import your dataset as the SA-1B format properly.

To make sure that the selected dataset has been added to the project, you can
run `datum project pinfo`, which will display the project information.

## Export to other formats

It can convert the dataset into any other formats [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task.

There are several ways to convert a Datumaro dataset to other dataset formats
using CLI:

- Export a dataset from Datumaro format to VOC format:

```console
datum project create
datum project import -f segment_anything <path/to/dataset>
datum project export -f coco -o <output/dir>
```

or

```console
datum convert -if segment_anything -i <path/to/dataset> -f coco -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'segment_anything')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to SA-1B

**Please note that exporting to SA-1B format would drop label information in annotations due to the nature of the format.**

There are several ways to convert a dataset to Segement Anything format:

- Export a dataset from an existing project to Segement Anything format:
```console
# export dataset into Segement Anything format from existing project
datum project export -p <path/to/project> -f segment_anything -o <output/dir> \
    -- --save-media
```

- Convert a dataset from COCO format to Segement Anything format:
```console
# converting to segment_anything format from other format
datum convert -if coco -i <path/to/dataset> \
    -f segment_anything -o <output/dir> -- --save-media
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/data_formats/test_segment_anything_format.py)
