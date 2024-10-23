# DatumaroBinary

## Format specification

DatumaroBinary format is [Datumaro](https://github.com/openvinotoolkit/datumaro)'s own data format as same as [Datumaro format](./datumaro.md).
Basically, it provides the same function as [Datumaro format](./datumaro.md),
but the difference is that the annotation file is not JSON but binary format.
Including changes in the file format, DatumaroBinary provides three key features compared to the datumaro format:

1. [Efficient storage cost](#efficient-storage-cost)
2. [Dataset encryption](#dataset-encryption)
3. [Multi-processing import and export](#multi-processing-import-and-export)

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

### Dataset encryption

Another advantage of the DatumaroBinary format is that it supports dataset encryption. If your dataset is hijacked by a potential attacker and you are concerned that your intellectual properties may be damaged, you can use this feature to protect your dataset from attackers. Enabling the dataset encryption feature allows you to encrypt both annotations and media or only the annotations. If you export the dataset to DatumaroBinary format with encryption, the secret key is automatically generated at the same time. You must keep this secret key separate from the exported dataset. This is because the secret key should be needed to read the exported dataset. Therefore, you have to be careful not to lose the secret key. If you would like to see an example of dataset encryption using Datumaro's Python API, please see [here](https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/09_encrypt_dataset.ipynb). For CLI usage of encryption, please see [Import encrypted datasets](#import-encrypted-datasets) and [Export datasets with encryption](#export-datasets-with-encryption) sections.

### Multi-processing import and export

The DatumaroBinary format stores annotation file data as several blobs by sharding, making it easy to export and import using multi-processing. Therefore, this format is suitable to accelerate export and import performance by utilizing multiple cores. You can enable this feature from the CLI or Python API. In the CLI, add `--num-workers #` extra argument, while in the Python API, use extra parameters such as `num_workers=#`. You can check examples of both methods: [Import datasets with multi-processing](#import-datasets-with-multi-processing) and [Export datasets with multi-processing](#export-datasets-with-multi-processing).

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

## Import DatumaroBinary dataset

A Datumaro project with a DatumaroBinary source can be created in the following way:

```console
datum project create
datum project import --format datumaro_binary <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

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
it cannot detect and import your dataset as the DatumaroBinary format properly.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

### Import encrypted datasets

If you want to import the dataset with encryption, please give `--encryption-key <secret-key>` to the optional arguments:

```console
datum project create
datum project import --format datumaro_binary <path/to/dataset> -- --encryption-key <secret-key>
```

`<secret-key>` is a 50-bytes long base64 encoded string prefixed with `datum-`. It is auto-generated in `<output/dir>/secret_key.txt` when the dataset is exported to DatumaroBinary format with `--encryption` option. You must have a correct `<secret-key>` to import the dataset encrypted by Datumaro.

### Import datasets with multi-processing

Using CLI

```console
# Import DatumaroBinary format dataset with 4 multi-processing workers
datum project create
datum project import --format datumaro_binary <path/to/dataset> -- --num-workers 4
```

or using Python API

```python
import datumaro as dm

# Import DatumaroBinary format dataset with 4 multi-processing workers
dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro_binary', num_workers=4)
```

Extra options for importing DatumaroBinary format:

- `--encryption-key ENCRYPTION_KEY` is a secret key required to import an encrypted dataset. If an incorrect key is provided, the dataset cannot be imported. If your dataset does not require encryption, you can ignore this argument.
- `--num-workers NUM_WORKERS` allow to multi-processing for the import. If num_workers = 0, do not use multiprocessing (default: 0).

## Export to other formats

It can convert DatumaroBinary dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a DatumaroBinary dataset to other dataset formats
using CLI:

- Export a dataset from DatumaroBinary to VOC format:

```console
datum project create
datum project import -f datumaro_binary <path/to/dataset>
datum project export -f voc -o <output/dir>
```

or

```console
datum convert -if datumaro_binary -i <path/to/dataset> -f voc -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro_binary')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to DatumaroBinary

There are several ways to convert a dataset to DatumaroBinary format:

- Export a dataset from an existing project to DatumaroBinary:

```console
# export dataset into DatumaroBinary format from existing project
datum project export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media
```

- Convert a dataset from VOC format to DatumaroBinary:

```console
# converting to DatumaroBinary format from other format
datum convert -if voc -i <path/to/dataset> \
    -f datumaro_binary -o <output/dir> -- --save-media
```

## Export datasets with encryption

If you want to encrypt your dataset, please add ``--encryption` directive to your command:

```console
# export dataset into DatumaroBinary format from existing project
datum project export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media --encryption
```

> Note:
> Please keep your secret key file seperate from the exported dataset. The secret key file is in `<output/dir>/secret_key.txt`. You should have this secret key whenever you want to import your dataset. **We are not responsible for you permanently losing access to your datasets if you lose this secret key.**

If you want to encrypt the annotation files only, not the media files, please add `--no-media-encryption` in addition to ``--encryption` directive to your command:

```console
# export dataset into DatumaroBinary format from existing project
datum project export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media --encryption --no-media-encryption
```

## Export datasets with multi-processing

Using CLI

```console
# Export dataset into DatumaroBinary with 4 multi-processing workers
datum project export -p <path/to/project> -f datumaro_binary -o <output/dir> \
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
- `--encryption` allow to encrypt your dataset with the auto-generated secret key.
- `--num-workers NUM_WORKERS` allow to multi-processing for the export. If num_workers = 0, do not use multiprocessing (default: 0).

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/data_formats/datumaro/test_datumaro_format.py)
