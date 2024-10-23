# BraTS

## Format specification

The original BraTS dataset is available
[here](https://www.med.upenn.edu/sbia/brats2017/data.html).
The BraTS data provided since BraTS'17 differs significantly from the data
provided during the previous BraTS challenges (i.e., 2016 and backwards).
Datumaro supports BraTS'17-20.

Supported annotation types:
- `Mask`

## Import BraTS dataset

A Datumaro project with a BraTS source can be created in the following way:

```bash
datum project create
datum project import --format brats <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro.components.dataset import Dataset

brats_dataset = Dataset.import_from('<path/to/dataset>', 'brats')
```

BraTS dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── imagesTr
│   │── <img1>.nii.gz
│   │── <img2>.nii.gz
│   └── ...
├── imagesTs
│   │── <img3>.nii.gz
│   │── <img4>.nii.gz
│   └── ...
├── labels
└── labelsTr
    │── <img1>.nii.gz
    │── <img2>.nii.gz
    └── ...
```

The data in Datumaro is stored as multi-frame images (set of 2D images).
Annotated images are stored as masks for each 2d image separately
with an `image_id` attribute.

## Export to other formats

Datumaro can convert a BraTS dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports segmentation masks.

There are several ways to convert a BraTS dataset to other dataset
formats using CLI:

```bash
datum project create
datum project import -f brats <path/to/dataset>
datum project export -f voc -o <output/dir> -- --save-media
```
or
``` bash
datum convert -if brats -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-media
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'brats')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_brats_format.py)
