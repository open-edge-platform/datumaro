# Open Images

## Format specification

A description of the Open Images Dataset (OID) format is available
[here](https://storage.googleapis.com/openimages/web/download.html).
Datumaro supports versions 4, 5 and 6.

Supported annotation types:
- `Label` (human-verified image-level labels)
- `Bbox` (bounding boxes)
- `Mask` (segmentation masks)

Supported annotation attributes:
- Labels
  - `score` (read/write, float).
    The confidence level from 0 to 1.
    A score of 0 indicates that
    the image does not contain objects of the corresponding class.

- Bounding boxes
  - `score` (read/write, float).
    The confidence level from 0 to 1.
    In the original dataset this is always equal to 1,
    but custom datasets may be created with arbitrary values.
  - `occluded` (read/write, boolean).
    Whether the object is occluded by another object.
  - `truncated` (read/write, boolean).
    Whether the object extends beyond the boundary of the image.
  - `is_group_of` (read/write, boolean).
    Whether the object represents a group of objects of the same class.
  - `is_depiction` (read/write, boolean).
    Whether the object is a depiction (such as a drawing)
    rather than a real object.
  - `is_inside` (read/write, boolean).
    Whether the object is seen from the inside.

- Masks
  - `box_id` (read/write, string).
    An identifier for the bounding box associated with the mask.
  - `predicted_iou` (read/write, float).
    Predicted IoU value with respect to the ground truth.

## Import Open Images dataset

The Open Images dataset is available for free download.

See the [`open-images-dataset` GitHub repository](https://github.com/cvdfoundation/open-images-dataset)
for information on how to download the images.

Datumaro also requires the image description files,
which can be downloaded from the following URLs:

- [complete set](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv)
- [train set](https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv)
- [validation set](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv)
- [test set](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv)

In addition, the following metadata file must be present
in the `annotations` directory:

- [class descriptions](https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv)

You can optionally download the following additional metadata file:

- [class hierarchy](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)

Annotations can be downloaded from the following URLs:

- [train image labels](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv)
- [validation image labels](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv)
- [test image labels](https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv)
- [train bounding boxes](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv)
- [validation bounding boxes](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv)
- [test bounding boxes](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv)
- [train segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv)
- train segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-f.zip)
- [validation segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv)
- validation segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-f.zip)
- [test segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv)
- test segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-f.zip)

All annotation files are optional,
except that if the mask metadata files for a given subset are downloaded,
all corresponding images must be downloaded as well, and vice versa.

A Datumaro project with an OID source can be created in the following way:

``` bash
datum project create
datum project import --format open_images <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

Open Images dataset directory should have the following structure:

```
└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── annotations/
    │   └── bbox_labels_600_hierarchy.json
    │   └── image_ids_and_rotation.csv  # optional
    │   └── oidv6-class-descriptions.csv
    │   └── *-annotations-bbox.csv
    │   └── *-annotations-human-imagelabels.csv
    │   └── *-annotations-object-segmentation.csv
    ├── images/
    |   ├── test/
    |   │   ├── <image_name1.jpg>
    |   │   ├── <image_name2.jpg>
    |   │   └── ...
    |   ├── train/
    |   │   ├── <image_name1.jpg>
    |   │   ├── <image_name2.jpg>
    |   │   └── ...
    |   └── validation/
    |       ├── <image_name1.jpg>
    |       ├── <image_name2.jpg>
    |       └── ...
    └── masks/
        ├── test/
        │   ├── <mask_name1.png>
        │   ├── <mask_name2.png>
        │   └── ...
        ├── train/
        │   ├── <mask_name1.png>
        │   ├── <mask_name2.png>
        │   └── ...
        └── validation/
            ├── <mask_name1.png>
            ├── <mask_name2.png>
            └── ...
```

The mask images must be extracted from the ZIP archives linked above.

To use per-subset image description files instead of `image_ids_and_rotation.csv`,
place them in the `annotations` subdirectory.
The `annotations` directory is optional and you can store all annotation files
in the root of input path.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

### Creating an image metadata file

To load bounding box and segmentation mask annotations,
Datumaro needs to know the sizes of the corresponding images.
By default, it will determine these sizes by loading each image from disk,
which requires the images to be present and makes the loading process slow.

If you want to load the aforementioned annotations on a machine where
the images are not available,
or just to speed up the dataset loading process,
you can extract the image size information in advance
and record it in an image metadata file.
This file must be placed at `annotations/images.meta`,
and must contain one line per image, with the following structure:

``` bash
<ID> <height> <width>
```

Where `<ID>` is the file name of the image without the extension,
and `<height>` and `<width>` are the dimensions of that image.
`<ID>` may be quoted with either single or double quotes.

The image metadata file, if present, will be used to determine the image
sizes without loading the images themselves.

Here's one way to create the `images.meta` file using ImageMagick,
assuming that the images are present on the current machine:

```bash
# run this from the dataset directory
find images -name '*.jpg' -exec \
    identify -format '"%[basename]" %[height] %[width]\n' {} + \
    > annotations/images.meta
```

## Export to other formats

Datumaro can convert OID into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports image-level labels.
There are several ways to convert OID to other dataset formats:

``` bash
datum project create
datum project import -f open_images <path/to/open_images>
datum project export -f cvat -o <output/dir>
```
or
``` bash
datum convert -if open_images -i <path/to/open_images> -f cvat -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'open_images')
dataset.export('save_dir', 'cvat', save_media=True)
```

## Export to Open Images

There are several ways to convert an existing dataset to the Open Images format:

``` bash
# export dataset into Open Images format from existing project
datum project export -p <path/to/project> -f open_images -o <output/dir> \
  -- --save_media
```
``` bash
# convert a dataset in another format to the Open Images format
datum convert -if imagenet -i <path/to/dataset> \
    -f open_images -o <output/dir> \
    -- --save-media
```

Extra options for exporting to the Open Images format:
- `--save-media` - save media files when exporting the dataset
  (by default, `False`)
- `--image-ext IMAGE_EXT` - save image files with the specified extension
  when exporting the dataset (by default, uses the original extension
  or `.jpg` if there isn't one)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Open Images format in particular. Follow the
[user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

Here are a few examples of using Datumaro operations to solve
particular problems with the Open Images dataset:

### Example 1. Load the Open Images dataset and convert to the CVAT format

```bash
datum project create -o project
datum project import -p project -f open_images ./open-images-dataset/
datum stats -p project
datum project export -p project -f cvat -- --save-media
```

### Example 2. Create a custom OID-like dataset

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(
        id='0000000000000001',
        image=np.ones((1, 5, 3)),
        subset='validation',
        annotations=[
            dm.Label(0, attributes={'score': 1}),
            dm.Label(1, attributes={'score': 0}),
        ],
    ),
], categories=['/m/0', '/m/1'])

dataset.export('./dataset', format='open_images')
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_open_images_format.py).
