# KITTI

## Format specification

The KITTI dataset has many annotations for different tasks. Datumaro supports
only a few of them.

Supported tasks / formats:
- [Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark) - `kitti_detection`
  The format specification is available in `README.md` [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip).
- [Segmentation](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) - `kitti_segmentation`
  The format specification is available in `README.md` [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_semantics.zip).
- Raw 3D / Velodyne Points - described [here](./kitti_raw.md)

Supported annotation types:
- `Bbox` (object detection)
- `Mask` (segmentation)

Supported annotation attributes:
- `truncated` (boolean) - indicates that the bounding box specified for
  the object does not correspond to the full extent of the object
- `occluded` (boolean) - indicates that a significant portion of the object
  within the bounding box is occluded by another object
- `score` (float) - indicates confidence in detection

## Import KITTI dataset

The KITTI left color images for object detection are available [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).
The KITTI object detection labels are available [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip).
The KITTI segmentation dataset is available [here](http://www.cvlibs.net/download.php?file=data_semantics.zip).

A Datumaro project with a KITTI source can be created in the following way:

``` bash
datum project create
datum project import --format kitti <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

KITTI detection dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── testing/
    │   └── image_2/
    │       ├── <name_1>.<img_ext>
    │       ├── <name_2>.<img_ext>
    │       └── ...
    └── training/
        ├── image_2/ # left color camera images
        │   ├── <name_1>.<img_ext>
        │   ├── <name_2>.<img_ext>
        │   └── ...
        └─── label_2/ # left color camera label files
            ├── <name_1>.txt
            ├── <name_2>.txt
            └── ...
```

KITTI segmentation dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-format labels (optional)
    ├── label_colors.txt # optional, color map for non-original segmentation labels
    ├── testing/
    │   └── image_2/
    │       ├── <name_1>.<img_ext>
    │       ├── <name_2>.<img_ext>
    │       └── ...
    └── training/
        ├── image_2/ # left color camera images
        │   ├── <name_1>.<img_ext>
        │   ├── <name_2>.<img_ext>
        │   └── ...
        ├── label_2/ # left color camera label files
        │   ├── <name_1>.txt
        │   ├── <name_2>.txt
        │   └── ...
        ├── instance/ # instance segmentation masks
        │   ├── <name_1>.png
        │   ├── <name_2>.png
        │   └── ...
        ├── semantic/ # semantic segmentation masks (labels are encoded by its id)
        │   ├── <name_1>.png
        │   ├── <name_2>.png
        │   └── ...
        └── semantic_rgb/ # semantic segmentation masks (labels are encoded by its color)
            ├── <name_1>.png
            ├── <name_2>.png
            └── ...
```

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file)
and `label_colors.txt`.
If the `dataset_meta.json` is not represented in the dataset, then
`label_colors.txt` will be imported if possible.

You can import a dataset for specific tasks
of KITTI dataset instead of the whole dataset,
for example:

``` bash
datum project import --format kitti_detection <path/to/dataset>
```

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a KITTI dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. segmentation annotations can be
saved in `Cityscapes` format, but not as `COCO keypoints`.

There are several ways to convert a KITTI dataset to other dataset formats:

``` bash
datum project create
datum project import -f kitti <path/to/kitti>
datum project export -f cityscapes -o <output/dir>
```
or
``` bash
datum convert -if kitti -i <path/to/kitti> -f cityscapes -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'kitti')
dataset.export('save_dir', 'cityscapes', save_media=True)
```

## Export to KITTI

There are several ways to convert a dataset to KITTI format:

``` bash
# export dataset into KITTI format from existing project
datum project export -p <path/to/project> -f kitti -o <output/dir> \
    -- --save-media
```
``` bash
# converting to KITTI format from other format
datum convert -if cityscapes -i <path/to/dataset> \
    -f kitti -o <output/dir> -- --save-media
```

Extra options for exporting to KITTI format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
- `--apply-colormap APPLY_COLORMAP` allow to use colormap for class masks
  (in folder `semantic_rgb`, by default `True`)
- `--label_map` allow to define a custom colormap. Example:

``` bash
# mycolormap.txt :
# 0 0 255 sky
# 255 0 0 person
#...
datum project export -f kitti -- --label-map mycolormap.txt

```
or you can use original kitti colomap:
``` bash
datum project export -f kitti -- --label-map kitti
```
- `--tasks TASKS` allow to specify tasks for export dataset,
by default Datumaro uses all tasks. Example:

```bash
datum project export -f kitti -- --tasks detection
```
- `--allow-attributes ALLOW_ATTRIBUTES` allow export of attributes
(by default `True`).

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the KITTI format in particular. Follow the
[user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

There are several examples of using Datumaro operations to solve
particular problems with KITTI dataset:

### Example 1. How to load an original KITTI dataset and convert to Cityscapes

```bash
datum project create -o project
datum project import -p project -f kitti ./KITTI/
datum stats -p project
datum project export -p project -f cityscapes -- --save-media
```

### Example 2. How to create a custom KITTI-like dataset

```python
import numpy as np
import datumaro as dm

import datumaro.plugins.kitti_format as KITTI

label_map = {}
label_map['background'] = (0, 0, 0)
label_map['label_1'] = (1, 2, 3)
label_map['label_2'] = (3, 2, 1)
categories = KITTI.make_kitti_categories(label_map)

dataset = dm.Dataset.from_iterable([
  dm.DatasetItem(id=1,
    image=np.ones((1, 5, 3)),
    annotations=[
      dm.Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1, id=0,
        attributes={'is_crowd': False}),
      dm.Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=0,
        attributes={'is_crowd': False}),
    ]
  ),
], categories=categories)

dataset.export('./dataset', format='kitti')
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_kitti_format.py)
