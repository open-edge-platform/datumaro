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

## Convert MVTec AD dataset

The MVTec AD dataset is available for free download
[here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

A MVTec AD dataset can be converted in the following way:

``` bash
datum convert --input-format mvtec_segmentation --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

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

## Export to other formats

Datumaro can convert a MVTec AD dataset into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g., image classification annotations can be
saved in `ImageNet` format, but not as `COCO keypoints`.

There are several ways to convert a MVTec AD dataset to other dataset formats:

``` bash
datum convert --input-format mvtec --input-path <path/to/mvtec> \
    --output-format coco --output-dir <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/mvtec>', 'mvtec')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to MVTec AD format

There are several ways to convert an existing dataset to MVTec AD format:

``` bash
# converting to MVTec AD format from other format
datum convert --input-format imagenet --input-path <path/to/dataset> \
    --output-format mvtec --output-dir <output/dir> \
    -- \
    --save-media
```

Extra options for exporting to MVTec AD format:
- `--save-media` - allow to export dataset with saving media files
  (by default `False`)
- `--tasks TASKS` - allow to specify tasks for export dataset,
  by default Datumaro uses all tasks.

```bash
datum convert --input-format <source-format> --input-path <path/to/dataset> \
    --output-format mvtec --output-dir <output/dir> -- --tasks detection,classification
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/blob/develop/tests/unit/test_mvtec_format.py).
