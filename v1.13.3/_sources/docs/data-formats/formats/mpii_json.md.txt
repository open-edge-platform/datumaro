# MPII Human Pose JSON

## Format specification

The original MPII Human Pose Dataset is available
[here](http://human-pose.mpi-inf.mpg.de).

Supported annotation types:
- `Bbox`
- `Points`

Supported attributes:
- `center` (a list with two coordinates of the center point
  of the object)
- `scale` (float)

## Convert MPII Human Pose Dataset (JSON)

An MPII Human Pose Dataset (JSON) can be converted in the following way:

```bash
datum convert --input-format mpii_json --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

It is also possible to convert the dataset using Python API:

```python
import datumaro as dm

mpii_dataset = dm.Dataset.import_from('<path/to/dataset>', 'mpii_json')
```

MPII Human Pose Dataset (JSON) directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── jnt_visible.npy # optional
├── mpii_annotations.json
├── mpii_headboxes.npy # optional
├── mpii_pos_gt.npy # optional
├── 000000001.jpg
├── 000000002.jpg
├── 000000003.jpg
└── ...
```

## Export to other formats

Datumaro can convert an MPII Human Pose Dataset (JSON) into
any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports bounding boxes or points.

There are several ways to convert an MPII Human Pose Dataset (JSON)
to other dataset formats using CLI:

``` bash
datum convert -if mpii_json -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mpii_json')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/blob/develop/tests/unit/test_mpii_json_format.py)
