# MVTec AD

## Format specification

The MVTec AD format specification is available
[here](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf).

The dataset has annotations for detecting abnormal pixels through binary masks
and it turns into bounding boxes or abnormal labels for supporting classification,
detection, and segmentation tasks. The MVTec AD dataset is composed of training data only
for `good` category without any annotation and testing data for both `good` and multiple
`defective` categories with masks. The dataset contains total 15 kinds of objects or textures.

Supported tasks / formats:
- The combined format - `mvtec`
- Image classification - `mvtec_classification`
- Object detection - `mvtec_detection`
- Instance segmentation - `mvtec_segmentation`

Supported annotation types:
- `Label` (classification)
- `Bbox` (detection)
- `Mask` (segmentation)

## Import MVTec AD dataset

The MVTec AD dataset is available for free download
[here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

A Datumaro project with a MVTec AD source can be created in the following way:

``` bash
datum project create
datum project import --format mvtec_segmentation <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

The MVTec AD dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/Category
   ├── train/
   │   ├── good/ # directory with list of good images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   ├── test/
   │   ├── good/ # directory with list of good images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   │   ├── defective1/ # directory with list of defective images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   │   ├── defective2/ # directory with list of defective images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   └── ground_truth/ # directory with semantic segmentation masks
       ├── defective1/ # directory with list of defective images for detection and segmentation task
       │   ├── img1_mask.png
       |   ├── img2_mask.png
       |   └── ...
       ├── defective2/ # directory with list of defective images for detection and segmentation task
       │   ├── img1_mask.png
       |   ├── img2_mask.png
       |   └── ...
```

To make sure that the selected dataset has been added to the project, you
can run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a MVTec AD dataset into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g., image classification annotations can be
saved in `ImageNet` format, but not as `COCO keypoints`.

There are several ways to convert a MVTec AD dataset to other dataset formats:

``` bash
datum project create
datum project import -f mvtec <path/to/mvtec>
datum project export -f coco -o <output/dir>
```
or
``` bash
datum convert -if mvtec -i <path/to/mvtec> -f coco -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/mvtec>', 'mvtec')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to MVTec AD format

There are several ways to convert an existing dataset to Pascal VOC format:

``` bash
# export dataset into MVTec AD format (classification) from existing project
datum project export -p <path/to/project> -f mvtec -o <output/dir> -- --tasks classification
```
``` bash
# converting to MVTec AD format from other format
datum convert -if imagenet -i <path/to/dataset> \
    -f mvtec -o <output/dir> \
    -- \
    --save-media
```

Extra options for exporting to MVTec AD format:
- `--save-media` - allow to export dataset with saving media files
  (by default `False`)
- `--tasks TASKS` - allow to specify tasks for export dataset,
  by default Datumaro uses all tasks.

```bash
datum project export -f mvtec -- --tasks detection,classification
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_mvtec_format.py).
