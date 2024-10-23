# COCO

## Format specification

COCO format specification is available [here](https://cocodataset.org/#format-data).

The dataset has annotations for multiple tasks. Each task has its own format
in Datumaro, and there is also a combined `coco` format, which includes all
the available tasks. The sub-formats have the same options as the "main"
format and only limit the set of annotation files they work with. To work with
multiple formats, use the corresponding option of the `coco` format.

Supported tasks / formats:
- The combined format - `coco`
- [Image Captioning](https://cocodataset.org/#captions-2015) - `coco_caption`
- [Object Detection](https://cocodataset.org/#detection-2020) - `coco_instances`
- [Panoptic Segmentation](https://cocodataset.org/#panoptic-2020) - `coco_panoptic`
- [Keypoint Detection](https://cocodataset.org/#keypoints-2020) - `coco_person_keypoints`
- [Stuff Segmentation](https://cocodataset.org/#stuff-2019) - `coco_stuff`
- Image Info - `coco_image_info`
- Image classification (Datumaro extension) - `coco_labels`. The format is like
  Object Detection, but uses only `category_id` and `score` annotation fields.

Supported annotation types (depending on the task):
- `Caption` (captions)
- `Label` (label, Datumaro extension)
- `Bbox` (instances, person keypoints)
- `Polygon` (instances, person keypoints)
- `Mask` (instances, person keypoints, panoptic, stuff)
- `Points` (person keypoints)

Supported annotation attributes:
- `is_crowd` (boolean; on `bbox`, `polygon` and `mask` annotations) - Indicates
  that the annotation covers multiple instances of the same class.
- `score` (number; range \[0; 1\]) - Indicates the confidence in this
  annotation. Ground truth annotations always have 1.
- arbitrary attributes (string/number) - A Datumaro extension. Stored
  in the `attributes` section of the annotation descriptor.

## Import COCO dataset

The COCO dataset is available for free download:

Images:
- [train images](http://images.cocodataset.org/zips/train2017.zip)
- [val images](http://images.cocodataset.org/zips/val2017.zip)
- [test images](http://images.cocodataset.org/zips/test2017.zip)
- [unlabeled images](http://images.cocodataset.org/zips/unlabeled2017.zip)

Annotations:
- [captions](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [image_info](http://images.cocodataset.org/annotations/image_info_test2017.zip)
- [instances](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
- [person_keypoints](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [stuff](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)

A Datumaro project with a COCO source can be created in the following way:

``` bash
datum project create
datum project import --format coco <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

Extra options for adding a source in the COCO format:

- `--keep-original-category-ids`: Add dummy label categories so that
 category indexes in the imported data source correspond to the category IDs
 in the original annotation file.

A COCO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── images/
    │   ├── train/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── val/
    │       ├── <image_name1.ext>
    │       ├── <image_name2.ext>
    │       └── ...
    └── annotations/
        ├── <task>_<subset_name>.json
        └── ...
```

For the panoptic task, a dataset directory should have the following structure:

<a id="dir-structure"></a>
<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── images/
    │   ├── train/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   ├── val/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    └── annotations/
        ├── panoptic_train/
        │   ├── <image_name1.ext>
        │   ├── <image_name2.ext>
        │   └── ...
        ├── panoptic_train.json
        ├── panoptic_val/
        │   ├── <image_name1.ext>
        │   ├── <image_name2.ext>
        │   └── ...
        └── panoptic_val.json
```

Annotation files must have the names like `<task_name>_<subset_name>.json`.
The year is treated as a part of the subset name.
If the annotation file name does't match this pattern, use one of the
task-specific formats instead of plain `coco`: `coco_captions`,
`coco_image_info`, `coco_instances`, `coco_labels`, `coco_panoptic`,
`coco_person_keypoints`, `coco_stuff`. In this case all items of the
dataset will be added to the `default` subset.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

You can import a dataset for one or several tasks
instead of the whole dataset. This option also allows to import annotation
files with non-default names. For example:

``` bash
datum project create
datum project import --format coco_stuff -r <relpath/to/stuff.json> <path/to/dataset>
```

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

Notes:
- COCO categories can have any integer ids, however, Datumaro will count
  annotation category id 0 as "not specified". This does not contradict
  the original annotations, because they have category indices starting from 1.

## Export to other formats

Datumaro can convert COCO dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a COCO dataset to other dataset formats
using CLI:

``` bash
datum project create
datum project import -f coco <path/to/coco>
datum project export -f voc -o <output/dir>
```
or
``` bash
datum convert -if coco -i <path/to/coco> -f voc -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'coco')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to COCO

There are several ways to convert a dataset to COCO format:

``` bash
# export dataset into COCO format from existing project
datum project export -p <path/to/project> -f coco -o <output/dir> \
    -- --save-media
```
``` bash
# converting to COCO format from other format
datum convert -if voc -i <path/to/dataset> \
    -f coco -o <output/dir> -- --save-media
```

Extra options for exporting to COCO format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.jpg`, if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
- `--segmentation-mode MODE` allow to specify save mode for instance
  segmentation:
  - 'guess': guess the mode for each instance
    (using 'is_crowd' attribute as hint)
  - 'polygons': save polygons (merge and convert masks, prefer polygons)
  - 'mask': save masks (merge and convert polygons, prefer masks)
(by default `guess`)
- `--crop-covered` allow to crop covered segments so that background objects
  segmentation was more accurate (by default `False`)
- `--allow-attributes ALLOW_ATTRIBUTES` allow export of attributes
  (by default `True`). The parameter enables or disables writing
  the custom annotation attributes to the "attributes" annotation
  field. This field is an extension to the original COCO format
- `--reindex REINDEX` allow to assign new indices to images and annotations,
  useful to avoid merge conflicts (by default `False`).
  This option allows to control if the images and
  annotations must be given new indices. It can be useful, when
  you want to preserve the original indices in the produced dataset.
  Consider having this option enabled when converting from other formats
  or merging datasets to avoid conflicts
- `--merge-images` allow to save all images into a single directory
  (by default `False`). The parameter controls the output directory for
  images. When enabled, the dataset images are saved into a single
  directory, otherwise they are saved in separate directories by subsets.
- `--tasks TASKS` allow to specify tasks for export dataset,
  by default Datumaro uses all tasks. Example:

```bash
datum project create
datum project import -f coco <path/to/dataset>
datum project export -f coco -- --tasks instances,stuff
```

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the COCO format in particular. Follow the
[user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

There are several examples of using Datumaro operations to solve
particular problems with a COCO dataset:

### Example 1. How to load an original panoptic COCO dataset and convert to Pascal VOC

```bash
datum project create -o project
datum project import -p project -f coco_panoptic ./COCO/annotations/panoptic_val2017.json
datum stats -p project
datum project export -p project -f voc -- --save-media
```

### Example 2. How to create custom COCO-like dataset

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable([
  dm.DatasetItem(id='000000000001',
    image=np.ones((1, 5, 3)),
    subset='val',
    attributes={'id': 40},
    annotations=[
      dm.Mask(image=np.array([[0, 0, 1, 1, 0]]), label=3,
        id=7, group=7, attributes={'is_crowd': False}),
      dm.Mask(image=np.array([[0, 1, 0, 0, 1]]), label=1,
        id=20, group=20, attributes={'is_crowd': True}),
    ]
  ),
], categories=['a', 'b', 'c', 'd'])

dataset.export('./dataset', format='coco_panoptic')
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_coco_format.py)

## COCO from Roboflow

The datasets available for download from [Roboflow](https://roboflow.com/) have the exact same JSON format with MS COCO.
However, it has a different directory structure than [MS COCO](#dir-structure):

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── train
    │   ├── <image_name1.ext>
    │   ├── ...
    │   └── _annotations.coco.json
    └── val
        ├── <image_name1.ext>
        ├── ...
        └── _annotations.coco.json
```

As you can see, in the root directory there are subdirectories representing a subset (e.g. `train`), which contain the JSON annotations file `_annotations.coco.json`.
Datumaro supports importing this COCO format dataset available from Roboflow by using `coco_roboflow` data format directive.

### Import using CLI

``` bash
datum project create
datum project import --format roboflow_coco <path/to/dataset>
```

### Import using Python API

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'roboflow_coco')
```
