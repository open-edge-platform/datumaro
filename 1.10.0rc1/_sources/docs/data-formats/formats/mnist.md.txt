# MNIST

## Format specification

MNIST format specification is available [here](http://yann.lecun.com/exdb/mnist/).

Fashion MNIST format specification is available [here](https://github.com/zalandoresearch/fashion-mnist).

MNIST in CSV  format specification is available [here](https://pjreddie.com/projects/mnist-in-csv/).

The dataset has several data formats available. Datumaro supports the
binary (Python pickle) format and the CSV variant. Each data format is covered
by a separate Datumaro format.

Supported formats:
- Binary (Python pickle) - `mnist`
- CSV - `mnist_csv`

Supported annotation types:
- `Label`

The format only supports single channel 28 x 28 images.

## Import MNIST dataset

The MNIST dataset is available for free download:

- [train-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz):
  training set images
- [train-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz):
  training set labels
- [t10k-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz):
  test set images
- [t10k-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz):
  test set labels

The Fashion MNIST dataset is available for free download:

- [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz):
  training set images
- [train-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz):
  training set labels
- [t10k-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz):
  test set images
- [t10k-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz):
  test set labels

The MNIST in CSV dataset is available for free download:

- [mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)
- [mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)

A Datumaro project with a MNIST source can be created in the following way:

``` bash
datum project create
datum project import --format mnist <path/to/dataset>
datum project import --format mnist_csv <path/to/dataset>
```

MNIST dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-format labels (optional)
    ├── labels.txt # a list of non-digit labels  in other format (optional)
    ├── t10k-images-idx3-ubyte.gz
    ├── t10k-labels-idx1-ubyte.gz
    ├── train-images-idx3-ubyte.gz
    └── train-labels-idx1-ubyte.gz
```

MNIST in CSV dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-format labels (optional)
    ├── labels.txt # a list of non-digit labels  in other format (optional)
    ├── mnist_test.csv # it should have the following pattern. mnist_{<subset-name>}.csv
    └── mnist_train.csv
```

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file)
and `labels.txt`.
If the `dataset_meta.json` is not represented in the dataset, then
`labels.txt` will be imported if possible.

For example, `labels.txt` for Fashion MNIST the following contents:

<!--lint disable fenced-code-flag-->
```
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

## Export to other formats

Datumaro can convert a MNIST dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the classification task (e.g. CIFAR-10/100, ImageNet, PascalVOC,
etc.)

There are several ways to convert a MNIST dataset to other dataset formats:

``` bash
datum project create
datum project import -f mnist <path/to/mnist>
datum project export -f imagenet -o <output/dir>
```
or
``` bash
datum convert -if mnist -i <path/to/mnist> -f imagenet -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mnist')
dataset.export('save_dir', 'imagenet', save_media=True)
```

These steps also will work for MNIST in CSV, if you use `mnist_csv`
instead of `mnist`.

## Export to MNIST

There are several ways to convert a dataset to MNIST format:

``` bash
# export dataset into MNIST format from existing project
datum project export -p <path/to/project> -f mnist -o <output/dir> \
    -- --save-media
```
``` bash
# converting to MNIST format from other format
datum convert -if imagenet -i <path/to/dataset> \
    -f mnist -o <output/dir> -- --save-media
```

Extra options for exporting to MNIST format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting dataset (by default `.png`)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)

These commands also work for MNIST in CSV if you use `mnist_csv` instead of `mnist`.

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the MNIST format in particular. Follow the [user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

There are several examples of using Datumaro operations to solve
particular problems with MNIST dataset:

### Example 1. How to create a custom MNIST-like dataset

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(id=0, image=np.ones((28, 28)),
        annotations=[dm.Label(2)]
    ),
    dm.DatasetItem(id=1, image=np.ones((28, 28)),
        annotations=[dm.Label(7)]
    )
], categories=[str(label) for label in range(10)])

dataset.export('./dataset', format='mnist')
```

### Example 2. How to filter and convert a MNIST dataset to ImageNet

Convert MNIST dataset to ImageNet format, keep only images with `3` class
presented:

``` bash
# Download MNIST dataset:
# https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
# https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
datum convert --input-format mnist --input-path <path/to/mnist> \
              --output-format imagenet \
              --filter '/item[annotation/label="3"]'
```

Examples of using this format from the code can be found in
the [binary format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_mnist_format.py) and [csv format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_mnist_csv_format.py)
