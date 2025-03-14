# VoTT JSON

## Format specification

[VoTT](https://github.com/microsoft/VoTT) (Visual Object Tagging Tool) is
an open source annotation tool released by Microsoft.
[VoTT JSON](https://roboflow.com/formats/vott-json) is the format used by VoTT
when the user exports a project and selects "VoTT JSON" as the export format.

Supported annotation types:
- `Bbox`

## Import VoTT dataset

A Datumaro project with a VoTT JSON source can be created in the following way:

```bash
datum project create
datum project import --format vott_json <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
import datumaro as dm

vott_json_dataset = dm.Dataset.import_from('<path/to/dataset>', 'vott_json')
```

VoTT JSON dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── dataset_meta.json # a list of custom labels (optional)
├── img0001.jpg
├── img0002.jpg
├── img0003.jpg
├── img0004.jpg
├── ...
├── test-export.json
├── train-export.json
└── ...
```

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

## Export to other formats

Datumaro can convert a VoTT JSON dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports bounding boxes.

There are several ways to convert a VoTT JSON dataset to other dataset
formats using CLI:

```bash
datum project create
datum project import -f vott_json <path/to/dataset>
datum project export -f voc -o ./save_dir -- --save-media
```
or
``` bash
datum convert -if vott_json -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'vott_json')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[VoTT JSON tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_vott_json_format.py).
