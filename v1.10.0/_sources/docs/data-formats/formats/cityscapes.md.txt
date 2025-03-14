# Cityscapes

## Format specification

Cityscapes format overview is available [here](https://www.cityscapes-dataset.com/dataset-overview/).

Cityscapes format specification is available [here](https://github.com/mcordts/cityscapesScripts#the-cityscapes-dataset).

Supported annotation types:
- `Masks`

Supported annotation attributes:
- `is_crowd` (boolean). Specifies if the annotation label can
  distinguish between different instances.
  If `False`, the annotation `id` field encodes the instance id.

## Import Cityscapes dataset

The Cityscapes dataset is available for free [download](https://www.cityscapes-dataset.com/downloads/).

A Datumaro project with a Cityscapes source can be created in the following way:

``` bash
datum project create
datum project import --format cityscapes <path/to/dataset>
```

Cityscapes dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of non-Cityscapes labels (optional)
    ├── label_colors.txt # a list of non-Cityscapes labels in other format (optional)
    ├── leftImg8bit/
    │   ├── <split: train,val, ...>
    │   │   ├── {city1}
    │   │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_leftImg8bit.png
    │   │   │   └── ...
    │   │   ├── {city2}
    │   │   └── ...
    └── gtFine/
        ├── <split: train,val, ...>
        │   ├── {city1}
        │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_color.png
        │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_instanceIds.png
        │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_labelIds.png
        │   │   └── ...
        │   ├── {city2}
        │   └── ...
        └── ...
```

Annotated files description:
1. `*_leftImg8bit.png` - left images in 8-bit LDR format
1. `*_color.png` - class labels encoded by its color
1. `*_labelIds.png` - class labels are encoded by its index
1. `*_instanceIds.png` - class and instance labels encoded by an instance ID.
  The pixel values encode class and the individual instance: the integer part
  of a division by 1000 of each ID provides class ID, the remainder
  is the instance ID. If a certain annotation describes multiple instances,
  then the pixels have the regular ID of that class

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file)
and `label_colors.txt`.
If the `dataset_meta.json` is not represented in the dataset, then
`label_colors.txt` will be imported if possible.

In `label_colors.txt` you can define custom color map and non-cityscapes labels,
for example:

```
# label_colors [color_rgb name]
0 124 134 elephant
```

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a Cityscapes dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the segmentation task (e.g. PascalVOC, CamVID, etc.)

There are several ways to convert a Cityscapes dataset to other dataset
formats using CLI:

``` bash
datum project create
datum project import -f cityscapes <path/to/cityscapes>
datum project export -f voc -o <output/dir>
```
or
``` bash
datum convert -if cityscapes -i <path/to/cityscapes> \
    -f voc -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'cityscapes')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to Cityscapes

There are several ways to convert a dataset to Cityscapes format:

``` bash
# export dataset into Cityscapes format from existing project
datum project export -p <path/to/project> -f cityscapes -o <output/dir> \
    -- --save-media
```
``` bash
# converting to Cityscapes format from other format
datum convert -if voc -i <path/to/dataset> \
    -f cityscapes -o <output/dir> -- --save-media
```

Extra options for exporting to Cityscapes format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
- `--label_map` allow to define a custom colormap. Example:

``` bash
# mycolormap.txt :
# 0 0 255 sky
# 255 0 0 person
#...
datum project export -f cityscapes -- --label-map mycolormap.txt
```
or you can use original cityscapes colomap:
``` bash
datum project export -f cityscapes -- --label-map cityscapes
```

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Cityscapes format in particular. Follow the
[user manual](../../user-manual/how_to_use_datumaro/)
to get more information about these operations.

There are several examples of using Datumaro operations to solve
particular problems with a Cityscapes dataset:

### Example 1. Load the original Cityscapes dataset and convert to Pascal VOC

```bash
datum project create -o project
datum project import -p project -f cityscapes ./Cityscapes/
datum stats -p project
datum project export -p project -o dataset/ -f voc -- --save-media
```

### Example 2. Create a custom Cityscapes-like dataset

```python
from collections import OrderedDict

import numpy as np
import datumaro as dm
import datumaro.plugins.cityscapes_format as Cityscapes

label_map = OrderedDict()
label_map['background'] = (0, 0, 0)
label_map['label_1'] = (1, 2, 3)
label_map['label_2'] = (3, 2, 1)
categories = Cityscapes.make_cityscapes_categories(label_map)

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(id=1,
        image=np.ones((1, 5, 3)),
        annotations=[
            dm.Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1),
            dm.Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=2,
                attributes={'is_crowd': False}),
        ]
    ),
], categories=categories)

dataset.export('./dataset', format='cityscapes')
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_cityscapes_format.py)
