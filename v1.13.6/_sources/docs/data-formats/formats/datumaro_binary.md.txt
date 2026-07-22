# DatumaroBinary

## Format specification

DatumaroBinary format is [Datumaro](https://github.com/open-edge-platform/datumaro)'s own data format as same as [Datumaro format](./datumaro.md).
Basically, it provides the same function as [Datumaro format](./datumaro.md),
but the difference is that the annotation file is not JSON but binary format.
Including changes in the file format, DatumaroBinary provides two key features compared to the datumaro format:

1. [Efficient storage cost](#efficient-storage-cost)
2. [Multi-processing import and export](#multi-processing-import-and-export)

### Efficient storage cost

While JSON annotation file in the [Datumaro format](./datumaro.md) has the advantage of being easily viewable using any text viewer,
the DatumaroBinary format takes up significantly less storage space since it is schemaless and stores data in binary form.
To demonstrate the storage cost-effectiveness of DatumaroBinary,
we conducted an experiment to compare the annotation file sizes of three dataset formats:
COCO (JSON), Datumaro (JSON), and DatumaroBinary (binary).
The table below shows the sizes of each format:

| Format | COCO (JSON) | Datumaro (JSON) | DatumaroBinary (binary) |
| :----: | :---------: | :-------------: | :---------------------: |
|  Size  |    468Mb    |     1046Mb      |          301Mb          |

This table shows that DatumaroBinary reduces the size of annotation files to **64.3% (COCO) and 28.8% (Datumaro).**

For this experiment, we used the training and validation annotation files of [2017 COCO instance segmentation task](https://cocodataset.org):

```console
Dataset/
├── images/
│   ├── train/
│   │   ├── <image_name1.ext>
│   │   └── ...
│   └── val/
│       ├── <image_name1.ext>
│       └── ...
├── videos/  # directory to store video files
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Multi-processing import and export

The DatumaroBinary format stores annotation file data as several blobs by sharding, making it easy to export and import using multi-processing. Therefore, this format is suitable to accelerate export and import performance by utilizing multiple cores. You can enable this feature from the CLI or Python API. In the CLI, add `--num-workers #` extra argument, while in the Python API, use extra parameters such as `num_workers=#`. You can check examples of both methods: [Convert datasets with multi-processing](#convert-datasets-with-multi-processing) and [Export datasets with multi-processing](#export-datasets-with-multi-processing).

### Usage for model training

You can directly use the DatumaroBinary format for the model training using [OpenVINO™ Training Extensions](https://github.com/openvinotoolkit/training_extensions).

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

## Convert DatumaroBinary dataset

A DatumaroBinary dataset can be converted in the following way:

```console
datum convert --input-format datumaro_binary --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

A DatumaroBinary dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->

```
└─ Dataset/
    ├── dataset_meta.json   # a list of custom labels (optional)
    ├── images/
    │   ├── <subset_name_1>/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── <subset_name_2> /
    │       ├── <image_name1.ext>
    │       ├── <image_name2.ext>
    │       └── ...
    ├── videos/
    └── annotations/
        ├── <subset_name_1>.datum
        ├── <subset_name_2>.datum
        └── ...
```

Note that the subset name shouldn't contain path separators.

If your dataset is not following the above directory structure,
it cannot be detected and converted properly.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

### Convert datasets with multi-processing

Using CLI

```console
# Convert DatumaroBinary format dataset with 4 multi-processing workers
datum convert --input-format datumaro_binary --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir> \
    -- --num-workers 4
```

or using Python API

```python
import datumaro as dm

# Import DatumaroBinary format dataset with 4 multi-processing workers
dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro_binary', num_workers=4)
```

Extra options for importing DatumaroBinary format:

- `--num-workers NUM_WORKERS` allow to multi-processing for the import. If num_workers = 0, do not use multiprocessing (default: 0).

## Export to other formats

It can convert DatumaroBinary dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a DatumaroBinary dataset to other dataset formats
using CLI:

- Convert a dataset from DatumaroBinary to VOC format:

```console
datum convert --input-format datumaro_binary --input-path <path/to/dataset> \
    --output-format voc --output-dir <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro_binary')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to DatumaroBinary

There are several ways to convert a dataset to DatumaroBinary format:

- Convert a dataset from VOC format to DatumaroBinary:

```console
# converting to DatumaroBinary format from other format
datum convert --input-format voc --input-path <path/to/dataset> \
    --output-format datumaro_binary --output-dir <output/dir> -- --save-media
```

## Export datasets with multi-processing

Using CLI

```console
# Convert dataset to DatumaroBinary with 4 multi-processing workers
datum convert --input-format <input-format> --input-path <path/to/dataset> \
    --output-format datumaro_binary --output-dir <output/dir> \
    -- --save-media --num-workers 4
```

or using Python API

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', '<dataset-format>')
# Export dataset into DatumaroBinary with 4 multi-processing workers
dataset.export('save_dir', 'datumaro_binary', save_media=True, num_workers=4)
```

Extra options for exporting to DatumaroBinary format:

- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--num-workers NUM_WORKERS` allow to multi-processing for the export. If num_workers = 0, do not use multiprocessing (default: 0).

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/tree/develop/tests/unit/data_formats/datumaro/test_datumaro_format.py)
