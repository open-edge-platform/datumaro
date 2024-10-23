# CIFAR

## Format specification

CIFAR format specification is available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Supported annotation types:
- `Label`

Datumaro supports Python version CIFAR-10/100.
The difference between CIFAR-10 and CIFAR-100 is how labels are stored
in the meta files (`batches.meta` or `meta`) and in the annotation files.
The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image
comes with a "fine" label (the class to which it belongs) and a "coarse" label
(the superclass to which it belongs). In CIFAR-10 there are no superclasses.

CIFAR formats contain 32 x 32 images. As an extension, Datumaro supports
reading and writing of arbitrary-sized images.

## Import CIFAR dataset

The CIFAR dataset is available for free download:

- [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz):
  CIFAR-10 python version
- [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz):
  CIFAR-100 python version

A Datumaro project with a CIFAR source can be created in the following way:

``` bash
datum project create
datum project import --format cifar <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

CIFAR-10 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-format labels (optional)
    ├── batches.meta
    ├── <subset_name1>
    ├── <subset_name2>
    └── ...
```

CIFAR-100 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-format labels (optional)
    ├── meta
    ├── <subset_name1>
    ├── <subset_name2>
    └── ...
```

Dataset files use the [Pickle](https://docs.python.org/3/library/pickle.html)
data format.

Meta files:

<!--lint disable fenced-code-flag-->
```
CIFAR-10:
    num_cases_per_batch: 1000
    label_names: list of strings (['airplane', 'automobile', 'bird', ...])
    num_vis: 3072

CIFAR-100:
    fine_label_names: list of strings (['apple', 'aquarium_fish', ...])
    coarse_label_names: list of strings (['aquatic_mammals', 'fish', ...])
```

Annotation files:

<!--lint disable fenced-code-flag-->
```
Common:
    'batch_label': 'training batch 1 of <N>'
    'data': numpy.ndarray of uint8, layout N x C x H x W
    'filenames': list of strings

    If images have non-default size (32x32) (Datumaro extension):
        'image_sizes': list of (H, W) tuples

CIFAR-10:
    'labels': list of strings

CIFAR-100:
    'fine_labels': list of integers
    'coarse_labels': list of integers
```

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

## Export to other formats

Datumaro can convert a CIFAR dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports the classification task (e.g. MNIST, ImageNet, PascalVOC, etc.)

There are several ways to convert a CIFAR dataset to other dataset
formats using CLI:

``` bash
datum project create
datum project import -f cifar <path/to/cifar>
datum project export -f imagenet -o <output/dir>
```
or
``` bash
datum convert -if cifar -i <path/to/dataset> \
    -f imagenet -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'cifar')
dataset.export('save_dir', 'imagenet', save_media=True)
```

## Export to CIFAR

There are several ways to convert a dataset to CIFAR format:

``` bash
# export dataset into CIFAR format from existing project
datum project export -p <path/to/project> -f cifar -o <output/dir> \
    -- --save-media
```
``` bash
# converting to CIFAR format from other format
datum convert -if imagenet -i <path/to/dataset> \
    -f cifar -o <output/dir> -- --save-media
```

Extra options for exporting to CIFAR format:

- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting the dataset (by default `.png`)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)

The format (CIFAR-10 or CIFAR-100) in which the dataset will be
exported depends on the presence of superclasses in the `LabelCategories`.

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the CIFAR format in particular. Follow the [user manual](../../user-manual/how_to_use_datumaro)
to get more information about these operations.

There are several examples of using Datumaro operations to solve
particular problems with CIFAR dataset:

### Example 1. How to create a custom CIFAR-like dataset

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(id=0, image=np.ones((32, 32, 3)),
        annotations=[dm.Label(3)]
    ),
    dm.DatasetItem(id=1, image=np.ones((32, 32, 3)),
        annotations=[dm.Label(8)]
    )
], categories=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'])

dataset.export('./dataset', format='cifar')
```

### Example 2. How to filter and convert a CIFAR dataset to ImageNet

Convert a CIFAR dataset to ImageNet format, keep only images with the
`dog` class present:

``` bash
# Download CIFAR-10 dataset:
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
datum convert --input-format cifar --input-path <path/to/cifar> \
              --output-format imagenet \
              --filter '/item[annotation/label="dog"]'
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_cifar_format.py)
