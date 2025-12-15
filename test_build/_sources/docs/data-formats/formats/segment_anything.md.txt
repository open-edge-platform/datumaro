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

## Convert SA-1B dataset

The SA-1B dataset is available [here](https://ai.facebook.com/datasets/segment-anything-downloads/)

A SA-1B dataset can be converted in the following way:

```console
datum convert --input-format segment_anything --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

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
it cannot detect and convert your dataset as the SA-1B format properly.

## Export to other formats

It can convert the dataset into any other formats [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task.

There are several ways to convert a SA-1B dataset to other dataset formats
using CLI:

- Convert a dataset from SA-1B format to COCO format:

```console
datum convert --input-format segment_anything --input-path <path/to/dataset> \
    --output-format coco --output-dir <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'segment_anything')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to SA-1B

**Please note that exporting to SA-1B format would drop label information in annotations due to the nature of the format.**

There are several ways to convert a dataset to Segment Anything format:

- Convert a dataset from COCO format to Segment Anything format:
```console
# converting to segment_anything format from other format
datum convert --input-format coco --input-path <path/to/dataset> \
    --output-format segment_anything --output-dir <output/dir> -- --save-media
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/blob/develop/tests/unit/data_formats/test_segment_anything_format.py)
