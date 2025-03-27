# Pascal VOC

## Format specification

Pascal VOC format specification is available
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf).

The dataset has annotations for multiple tasks. Each task has its own format
in Datumaro, and there is also a combined `voc` format, which includes all
the available tasks. The sub-formats have the same options as the "main"
format and only limit the set of annotation files they work with. To work with
multiple formats, use the corresponding option of the `voc` format.

Supported tasks / formats:
- The combined format - `voc`
- Image classification - `voc_classification`
- Object detection - `voc_detection`
- Action classification - `voc_action`
- Class and instance segmentation - `voc_segmentation`
- Person layout detection - `voc_layout`

Supported annotation types:
- `Label` (classification)
- `Bbox` (detection, action detection and person layout)
- `Mask` (segmentation)

Supported annotation attributes:
- `occluded` (boolean) - indicates that a significant portion of the
  object within the bounding box is occluded by another object
- `truncated` (boolean) - indicates that the bounding box specified for
  the object does not correspond to the full extent of the object
- `difficult` (boolean) - indicates that the object is considered difficult
  to recognize
- action attributes (boolean) - `jumping`, `reading` and
  [others](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/actionexamples/index.html).
  Indicate that the object does the corresponding action.
- arbitrary attributes (string/number) - A Datumaro extension. Stored
  in the `attributes` section of the annotation `xml` file. Available for
  bbox annotations only.

## Import Pascal VOC dataset

The Pascal VOC dataset is available for free download
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

A Datumaro project with a Pascal VOC source can be created in the following way:

``` bash
datum project create
datum project import --format voc <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

Pascal VOC dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
   ├── dataset_meta.json # a list of non-Pascal labels (optional)
   ├── labelmap.txt # or a list of non-Pascal labels in other format (optional)
   │
   ├── Annotations/
   │     ├── ann1.xml # Pascal VOC format annotation file
   │     ├── ann2.xml
   │     └── ...
   ├── JPEGImages/
   │    ├── img1.jpg
   │    ├── img2.jpg
   │    └── ...
   ├── SegmentationClass/ # directory with semantic segmentation masks
   │    ├── img1.png
   │    ├── img2.png
   │    └── ...
   ├── SegmentationObject/ # directory with instance segmentation masks
   │    ├── img1.png
   │    ├── img2.png
   │    └── ...
   │
   └── ImageSets/
        ├── Main/ # directory with list of images for detection and classification task
        │   ├── test.txt  # list of image names in test subset  (without extension)
        |   ├── train.txt # list of image names in train subset (without extension)
        |   └── ...
        ├── Layout/ # directory with list of images for person layout task
        │   ├── test.txt
        |   ├── train.txt
        |   └── ...
        ├── Action/ # directory with list of images for action classification task
        │   ├── test.txt
        |   ├── train.txt
        |   └── ...
        └── Segmentation/ # directory with list of images for segmentation task
            ├── test.txt
            ├── train.txt
            └── ...
```

The `ImageSets` directory should contain at least one of the directories:
`Main`, `Layout`, `Action`, `Segmentation`.
These directories contain `.txt` files with a list of images in a subset,
the subset name is the same as the `.txt` file name. Subset names can be
arbitrary.

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file)
and `labelmap.txt`.
If the `dataset_meta.json` is not represented in the dataset, then
`labelmap.txt` will be imported if possible.

In `labelmap.txt` you can define custom color map and non-pascal labels,
for example:

```
# label_map [label : color_rgb : parts : actions]
helicopter:::
elephant:0,124,134:head,ear,foot:
```
It is also possible to import grayscale (1-channel) PNG masks.
For grayscale masks provide a list of labels with the number of lines
equal to the maximum color index on images. The lines must be in the
right order so that line index is equal to the color index. Lines can
have arbitrary, but different, colors. If there are gaps in the used
color indices in the annotations, they must be filled with arbitrary
dummy labels. Example:

```
car:0,128,0:: # color index 0
aeroplane:10,10,128:: # color index 1
_dummy2:2,2,2:: # filler for color index 2
_dummy3:3,3,3:: # filler for color index 3
boat:108,0,100:: # color index 3
...
_dummy198:198,198,198:: # filler for color index 198
_dummy199:199,199,199:: # filler for color index 199
the_last_label:12,28,0:: # color index 200
```

You can import dataset for specific tasks
of Pascal VOC dataset instead of the whole dataset,
for example:

``` bash
datum project import -f voc_detection -r ImageSets/Main/train.txt <path/to/dataset>
```

To make sure that the selected dataset has been added to the project, you
can run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a Pascal VOC dataset into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. image classification annotations can be
saved in `ImageNet` format, but not as `COCO keypoints`.

There are several ways to convert a Pascal VOC dataset to other dataset formats:

``` bash
datum project create
datum project import -f voc <path/to/voc>
datum project export -f coco -o <output/dir>
```
or
``` bash
datum convert -if voc -i <path/to/voc> -f coco -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'voc')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to Pascal VOC

There are several ways to convert an existing dataset to Pascal VOC format:

``` bash
# export dataset into Pascal VOC format (classification) from existing project
datum project export -p <path/to/project> -f voc -o <output/dir> -- --tasks classification
```
``` bash
# converting to Pascal VOC format from other format
datum convert -if imagenet -i <path/to/dataset> \
    -f voc -o <output/dir> \
    -- --label_map voc --save-media
```

Extra options for exporting to Pascal VOC format:
- `--save-media` - allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` - allow to specify image extension
  for exporting dataset (by default use original or `.jpg` if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
- `--apply-colormap APPLY_COLORMAP` - allow to use colormap for class
  and instance masks (by default `True`)
- `--allow-attributes ALLOW_ATTRIBUTES` - allow export of attributes
  (by default `True`)
- `--keep-empty KEEP_EMPTY` - write subset lists even if they are empty
  (by default `False`)
- `--tasks TASKS` - allow to specify tasks for export dataset,
  by default Datumaro uses all tasks. Example:

```bash
datum project export -f voc -- --tasks detection,classification
```

- `--label_map PATH` - allows to define a custom colormap. Example:

``` bash
# mycolormap.txt [label : color_rgb : parts : actions]:
# cat:0,0,255::
# person:255,0,0:head:
datum project export -f voc_segmentation -- --label-map mycolormap.txt
```
or you can use original voc colormap:
``` bash
datum project export -f voc_segmentation -- --label-map voc
```

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Pascal VOC format in particular. Follow
[user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with Pascal VOC dataset:

### Example 1. How to prepare an original dataset for training
In this example, preparing the original dataset to train the semantic
segmentation model includes:
loading,
checking duplicate images,
setting the number of images,
splitting into subsets,
export the result to Pascal VOC format.

```bash
datum project create -o project
datum project import -p project -f voc_segmentation ./VOC2012/ImageSets/Segmentation/trainval.txt
datum stats -p project # check statisctics.json -> repeated images
datum transform -p project -t ndr -- -w trainval -k 2500
datum filter -p project -e '/item[subset="trainval"]'
datum transform -p project -t random_split -- -s train:.8 -s val:.2
datum project export -p project -f voc -- --label-map voc --save-media
```

### Example 2. How to create a custom dataset

```python
import datumaro as dm

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(id='image1', image=dm.Image.from_file(path='image1.jpg', size=(10, 20)),
        annotations=[
            dm.Label(3),
            dm.Bbox(1.0, 1.0, 10.0, 8.0, label=0, attributes={'difficult': True, 'running': True}),
            dm.Polygon([1, 2, 3, 2, 4, 4], label=2, attributes={'occluded': True}),
            dm.Polygon([6, 7, 8, 8, 9, 7, 9, 6], label=2),
        ]
    ),
], categories=['person', 'sky', 'water', 'lion'])

dataset.transform('polygons_to_masks')
dataset.export('./mydataset', format='voc', label_map='my_labelmap.txt')
```

`my_labelmap.txt` has the following contents:

```
# label:color_rgb:parts:actions
person:0,0,255:hand,foot:jumping,running
sky:128,0,0::
water:0,128,0::
lion:255,128,0::
```

### Example 3. Load, filter and convert from code
Load Pascal VOC dataset, and export train subset with items
which has `jumping` attribute:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('./VOC2012', format='voc')

train_dataset = dataset.get_subset('train').as_dataset()

def only_jumping(item):
    for ann in item.annotations:
        if ann.attributes.get('jumping'):
            return True
    return False

train_dataset.select(only_jumping)

train_dataset.export('./jumping_label_me', format='label_me', save_media=True)
```

### Example 4. Get information about items in Pascal VOC 2012 dataset for segmentation task

```python
import datumaro as dm

dataset = dm.Dataset.import_from('./VOC2012', format='voc')

def has_mask(item):
    for ann in item.annotations:
        if ann.type == dm.AnnotationType.mask:
            return True
    return False

dataset.select(has_mask)

print("Pascal VOC 2012 has %s images for segmentation task:" % len(dataset))
for subset_name, subset in dataset.subsets().items():
    for item in subset:
        print(item.id, subset_name, end=";")
```

After executing this code, we can see that there are 5826 images
in Pascal VOC 2012 has for segmentation task and this result is the same as the
[official documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html)

Examples of using this format from the code can be found in
[tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_voc_format.py)
