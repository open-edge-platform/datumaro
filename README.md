# Dataset Management Framework (Datumaro)

[![Build status](https://github.com/open-edge-platform/datumaro/actions/workflows/health_check.yml/badge.svg)](https://github.com/open-edge-platform/datumaro/actions/workflows/health_check.yml)
[![codecov](https://codecov.io/gh/open-edge-platform/datumaro/branch/develop/graph/badge.svg?token=FG25VU096Q)](https://codecov.io/gh/open-edge-platform/datumaro)
[![Downloads](https://static.pepy.tech/badge/datumaro)](https://pepy.tech/project/datumaro)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/open-edge-platform/datumaro/badge)](https://scorecard.dev/viewer/?uri=github.com/open-edge-platform/datumaro)

A framework and CLI tool to build, transform, and analyze datasets.

<!--lint disable fenced-code-flag-->

```
VOC dataset                                  ---> Annotation tool
     +                                     /
COCO dataset -----> Datumaro ---> dataset ------> Model training
     +                                     \
CVAT annotations                             ---> Publication, statistics etc.
```

<!--lint enable fenced-code-flag-->

- [Getting started](https://open-edge-platform.github.io/datumaro/latest/docs/get-started/quick-start-guide)
- [Level Up](https://open-edge-platform.github.io/datumaro/latest/docs/level-up/basic_skills)
- [Features](#features)
- [User manual](https://open-edge-platform.github.io/datumaro/latest/docs/user-manual/how_to_use_datumaro)
- [Developer manual](https://open-edge-platform.github.io/datumaro/latest/docs/reference/datumaro_module)
- [Contributing](#contributing)

## Features

[(Back to top)](#dataset-management-framework-datumaro)

- Dataset reading, writing, conversion in any direction.

  - [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) (`classification`)
  - [Cityscapes](https://www.cityscapes-dataset.com/)
  - [COCO](http://cocodataset.org/#format-data) (`image_info`, `instances`, `person_keypoints`,
    `captions`, `labels`, `panoptic`, `stuff`)
  - [CVAT](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/)
  - [ImageNet](http://image-net.org/)
  - [Kitti](http://www.cvlibs.net/datasets/kitti/index.php) (`segmentation`, `detection`,
    `3D raw` / `velodyne points`)
  - [LabelMe](http://labelme.csail.mit.edu/Release3.0)
  - [LFW](http://vis-www.cs.umass.edu/lfw/) (`classification`, `person re-identification`,
    `landmarks`)
  - [MNIST](http://yann.lecun.com/exdb/mnist/) (`classification`)
  - [Open Images](https://storage.googleapis.com/openimages/web/download.html)
  - [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
    (`classification`, `detection`, `segmentation`, `action_classification`, `person_layout`)
  - [TF Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
    (`bboxes`, `masks`)
  - [YOLO](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data) (`bboxes`)

  Other formats and documentation for them can be found [here](https://open-edge-platform.github.io/datumaro/latest/docs/data-formats/formats).

- Dataset building
  - Merging multiple datasets into one
  - Dataset filtering by a custom criteria:
    - remove polygons of a certain class
    - remove images without annotations of a specific class
    - remove `occluded` annotations from images
    - keep only vertically-oriented images
    - remove small area bounding boxes from annotations
  - Annotation conversions, for instance:
    - polygons to instance masks and vice-versa
    - apply a custom colormap for mask annotations
    - rename or remove dataset labels
  - Splitting a dataset into multiple subsets like `train`, `val`, and `test`:
    - random split
    - task-specific splits based on annotations,
      which keep initial label and attribute distributions
      - for classification task, based on labels
      - for detection task, based on bboxes
      - for re-identification task, based on labels,
        avoiding having same IDs in training and test splits
- Dataset quality checking
  - Simple checking for errors
  - Comparison with model inference
  - Merging and comparison of multiple datasets
  - Annotation validation based on the task type(classification, etc)
- Dataset comparison
- Dataset statistics (image mean and std, annotation statistics)

> Check
> [the design document](https://open-edge-platform.github.io/datumaro/latest/docs/explanation/architecture)
> for a full list of features.
> Check
> [the user manual](https://open-edge-platform.github.io/datumaro/latest/docs/user-manual/how_to_use_datumaro)
> for usage instructions.

## Contributing

[(Back to top)](#dataset-management-framework-datumaro)

Feel free to
[open an Issue](https://github.com/open-edge-platform/datumaro/issues/new), if you
think something needs to be changed. You are welcome to participate in
development, instructions are available in our
[contribution guide](https://github.com/open-edge-platform/datumaro/blob/develop/contributing.md).
