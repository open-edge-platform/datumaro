# MMDetection COCO

## Format specification

[MMDetection](https://mmdetection.readthedocs.io/en/latest/) is a training framework for object detection and instance segmentation tasks, providing a modular and flexible architecture that supports various state-of-the-art models, datasets, and training techniques. MMDetection has gained popularity in the research community for its comprehensive features and ease of use in developing and benchmarking object detection algorithms.
MMDetection specifies their COCO format [here](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

Most of available tasks or formats are similar to the [original COCO format](./formats/coco), while only the image directories are separated with respect to subsets.
In this document, we just describe the directory structure of MMDetection COCO format as per [here](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).
MMDetection COCO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── <subset_name>/
    │   ├── <image_name1.ext>
    │   ├── <image_name2.ext>
    │   └── ...
    ├── <subset_name>/
    │   ├── <image_name1.ext>
    │   ├── <image_name2.ext>
    │   └── ...
    └── annotations/
        ├── instances_<subset_name>.json
        └── ...
```

### Import using CLI

``` bash
datum project create
datum project import --format mmdet_coco <path/to/dataset>
```

### Import using Python API

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mmdet_coco')
```
