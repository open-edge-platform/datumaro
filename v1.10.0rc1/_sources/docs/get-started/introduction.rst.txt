Introduction
############

**Datumaro** is a framework and CLI tool to build, transform, and analyze datasets.

- a tool to build composite datasets and iterate over them
- a tool to create and maintain datasets

  - Version control of annotations and images
  - Publication (with removal of sensitive information)
  - Editing
  - Joining and splitting
  - Exporting, format changing
  - Image preprocessing
- a dataset storage
- a tool to debug datasets

  - A network can be used to generate informative data subsets (e.g., with false-positives) to be analyzed further

Key Features
------------

Datumaro supports the following features:

- Dataset reading, writing, conversion in any direction.

  - `CIFAR-10/100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ (`classification`)
  - `Cityscapes <https://www.cityscapes-dataset.com/>`_
  - `COCO <http://cocodataset.org/#format-data>`_ (`image_info`, `instances`, `person_keypoints`,
    `captions`, `labels`, `panoptic`, `stuff`)
  - `CVAT <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/xml_format>`_
  - `ImageNet <http://image-net.org/>`_
  - `Kitti <http://www.cvlibs.net/datasets/kitti/index.php>`_ (`segmentation`, `detection`,
    `3D raw` / `velodyne points`)
  - `LabelMe <http://labelme.csail.mit.edu/Release3.0>`_
  - `LFW <http://vis-www.cs.umass.edu/lfw/>`_ (`classification`, `person re-identification`,
    `landmarks`)
  - `MNIST <http://yann.lecun.com/exdb/mnist/>`_ (`classification`)
  - `Open Images <https://storage.googleapis.com/openimages/web/download.html>`_
  - `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html>`_
    (`classification`, `detection`, `segmentation`, `action_classification`, `person_layout`)
  - `TF Detection API <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md>`_
    (`bboxes`, `masks`)
  - `YOLO <https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data>`_ (`bboxes`)

  Other formats and documentation for them can be found `here <https://openvinotoolkit.github.io/datumaro/latest/docs/data-formats/formats/index.html>`_.
- Dataset building

  - Merging multiple datasets into one
  - Dataset filtering by a custom criteria:

    - remove polygons of a certain class
    - remove images without annotations of a specific class
    - remove ``occluded`` annotations from images
    - keep only vertically-oriented images
    - remove small area bounding boxes from annotations

  - Annotation conversions, for instance:

    - polygons to instance masks and vice-versa
    - apply a custom colormap for mask annotations
    - rename or remove dataset labels

  - Splitting a dataset into multiple subsets like ``train``, ``val``, and ``test``:

    - random split
    - task-specific splits based on annotations, which keep initial label and attribute distributions

      - for classification task, based on labels
      - for detection task, based on bboxes
      - for re-identification task, based on labels, avoiding having same IDs in training and test splits

  - Sampling a dataset

    - analyzes inference result from the given dataset
      and selects the ``best`` and the ``least amount of`` samples for annotation.
    - Select the sample that best suits model training.

      - sampling with Entropy based algorithm

- Dataset quality checking

  - Simple checking for errors
  - Comparison with model inference
  - Merging and comparison of multiple datasets
  - Annotation validation based on the task type(classification, etc)

- Dataset comparison
- Dataset statistics (image mean and std, annotation statistics)
- Model integration

  - Inference (OpenVINO, Caffe, PyTorch, TensorFlow, MxNet, etc.)
  - Explainable AI (`RISE algorithm <https://arxiv.org/abs/1806.07421>`_)

    - RISE for classification
    - RISE for object detection
