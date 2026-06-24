# MPII Human Pose

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

## Convert MPII Human Pose Dataset

A Datumaro dataset can be converted in the following way:

```bash
datum convert -if mpii -i <path/to/dataset> -o <output/dir>
```

It is also possible to convert the dataset using Python API:

```python
import datumaro as dm

mpii_dataset = dm.Dataset.import_from('<path/to/dataset>', 'mpii')
```

MPII Human Pose Dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── mpii_human_pose_v1_u12_1.mat
├── 000000001.jpg
├── 000000002.jpg
├── 000000003.jpg
└── ...
```

## Export to other formats

Datumaro can convert an MPII Human Pose Dataset into
any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports bounding boxes or points.

There are several ways to convert an MPII Human Pose Dataset
to other dataset formats using CLI:

```bash
datum convert -if mpii -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mpii')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/blob/develop/tests/unit/test_mpii_format.py)
