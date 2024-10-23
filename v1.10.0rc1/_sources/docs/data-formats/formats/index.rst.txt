Supported Data Formats
######################

.. toctree::
   :maxdepth: 1
   :hidden:

   ade20k2017
   ade20k2020
   align_celeba
   arrow
   ava_action
   brats
   brats_numpy
   celeba
   cifar
   cityscapes
   coco
   common_semantic_segmentation
   common_super_resolution
   cvat
   datumaro_binary
   datumaro
   dota
   icdar
   image_zip
   imagenet
   kaggle
   kinetics
   kitti
   kitti_raw
   labelme
   lfw
   mapillary_vistas
   market1501
   mars
   mmdet
   mnist
   mot
   mots
   mpii_json
   mpii
   mvtec
   nyu_depth_v2
   open_images
   pascal_voc
   roboflow
   segment_anything
   sly_pointcloud
   synthia
   tabular
   vgg_face2
   video
   vott_csv
   vott_json
   wider_face
   yolo
   yolo_ultralytics

* ADE20k (v2017) (import-only)
   * `Format specification <https://www.kaggle.com/soumikrakshit/ade20k>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/ade20k2017_dataset>`_
   * `Format documentation <./ade20k2017.md>`_
* ADE20k (v2020) (import-only)
   * `Format specification <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/ade20k2020_dataset>`_
   * `Format documentation <ade20k2020.md>`_
* Align CelebA (``classification``, ``landmarks``) (import-only)
   * `Format specification <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/align_celeba_dataset>`_
   * `Format documentation <align_celeba.md>`_
* BraTS (``segmentation``) (import-only)
   * `Format specification <https://www.med.upenn.edu/sbia/brats2018/data.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/brats_dataset>`_
   * `Format documentation <brats.md>`_
* BraTS Numpy (``detection``, ``segmentation``) (import-only)
   * `Format specification <https://www.med.upenn.edu/sbia/brats2018/data.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/brats_numpy_dataset>`_
   * `Format documentation <brats_numpy.md>`_
* CamVid (``segmentation``)
   * `Format specification <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/camvid_dataset>`_
* CelebA (``classification``, ``detection``, ``landmarks``) (import-only)
   * `Format specification <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/celeba_dataset>`_
   * `Format documentation <celeba.md>`_
* CIFAR-10/100 (``classification``)
   * `Format specification <https://www.cs.toronto.edu/~kriz/cifar.html>`_
   * `Dataset example CIFAR-10 <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cifar10_dataset>`_
   * `Dataset example CIFAR-100 <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cifar100_dataset>`_
   * `Format documentation <cifar.md>`_
* Cityscapes (``segmentation``)
   * `Format specification <https://www.cityscapes-dataset.com/dataset-overview/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cityscapes_dataset>`_
   * `Format documentation <cityscapes.md>`_
* Common Semantic Segmentation (``segmentation``) (import-only)
   * `Format specification <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/annotation_converters/README.md>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/common_semantic_segmentation_dataset>`_
   * `Format documentation <common_semantic_segmentation.md>`_
* Common Super Resolution
   * `Format specification <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/annotation_converters/README.md>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/common_super_resolution_dataset>`_
   * `Format documentation <common_super_resolution.md>`_
* CVAT (`for images`, `for video` (import-only))
   * `Format specification <https://opencv.github.io/cvat/docs/manual/advanced/xml_format>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cvat_dataset>`_
   * `Format documentation <cvat.md>`_
* DOTA (``detection_rotated``)
   * `Format specification <https://captain-whu.github.io/DOTA/dataset.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/dota_dataset>`_
   * `Format documentation <dota.md>`_
* ICDAR13/15 (``word recognition``, ``text localization``, ``text segmentation``)
   * `Format specification <https://rrc.cvc.uab.es/?ch=2>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/icdar_dataset>`_
* ImageNet (``classification``, ``detection``)
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/imagenet_dataset>`_
   * `Dataset example (txt for classification) <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/imagenet_txt_dataset>`_
   * Detection format is the same as in PASCAL VOC
   * `Format documentation <imagenet.md>`_
* Kaggle (``classification``, ``detection``, ``segmentation``) (import-only)
   * `Dataset examples <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kaggle_datasets>`_
   * `Format documentation <kaggle.md>`_
* KITTI (``segmentation``, ``detection``)
   * `Format specification <http://www.cvlibs.net/datasets/kitti/index.php>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kitti_dataset>`_
   * `Format documentation <kitti.md>`_
* KITTI 3D (``raw``, ``tracklets``, ``velodyne points``)
   * `Format specification <http://www.cvlibs.net/datasets/kitti/raw_data.php>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kitti_dataset/kitti_raw>`_
   * `Format documentation <kitti_raw.md>`_
* Kinetics 400/600/700
   * `Format specification <https://www.deepmind.com/open-source/kinetics>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kinetics_dataset>`_
   * `Format documentation <kinetics.md>`_
* LabelMe (``labels``, ``boxes``, ``masks``)
   * `Format specification <http://labelme.csail.mit.edu/Release3.0>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/labelme_dataset>`_
   * `Format documentation <labelme.md>`_
* LFW (``classification``, ``person re-identification``, ``landmarks``)
   * `Format specification <http://vis-www.cs.umass.edu/lfw/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/lfw_dataset>`_
   * `Format documentation <lfw.md>`_
* Mapillary Vistas (``segmentation``) (import-only)
   * `Format specification <https://www.mapillary.com/dataset/vistas>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mapillary_vistas_dataset>`_
   * `Format documentation <mapillary_vistas.md>`_
* Market-1501 (``person re-identification``)
   * `Format specification <https://www.aitribune.com/dataset/2018051063>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/market1501_dataset>`_
* MARS (import-only)
   * `Format specification <https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mars_dataset>`_
   * `Format documentation <mars.md>`_
* MMDet-COCO (``detection``, ``segmentation``)
   * `Format specification <https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/coco_dataset/mmdet_coco>`_
   * `Format documentation <mmdet.md>`_
* MNIST (``classification``)
   * `Format specification <http://yann.lecun.com/exdb/mnist/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mnist_dataset>`_
   * `Format documentation <mnist.md>`_
* MNIST in CSV (``classification``)
   * `Format specification <https://pjreddie.com/projects/mnist-in-csv/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mnist_csv_dataset>`_
   * `Format documentation <mnist.md>`_
* MOT sequences
   * `Format specification <https://arxiv.org/pdf/1906.04567.pdf>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mot_dataset>`_
   * `Format documentation <mot.md>`_
* MOTS (png)
   * `Format specification <https://www.vision.rwth-aachen.de/page/mots>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mots_dataset>`_
   * `Format documentation <mots.md>`_
* MPII Human Pose (``detection``, ``pose estimation``) (import-only)
   * `Format specification <http://human-pose.mpi-inf.mpg.de>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mpii_dataset>`_
   * `Format documentation <mpii.md>`_
* MPII Human Pose JSON (``detection``, ``pose estimation``) (import-only)
   * `Format specification <http://human-pose.mpi-inf.mpg.de>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mpii_json_dataset>`_
   * `Format documentation <mpii_json.md>`_
* MS COCO (``image info``, ``instances``, ``person keypoints``, ``captions``, ``labels``, ``panoptic``, ``stuff``)
   * `Format specification <http://cocodataset.org/#format-data>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/coco_dataset>`_
   * ``labels`` are our extension - like `instances` with only `category_id`
   * `Format documentation <coco.md>`_
* Roboflow (import-only)
   * `Format specification <https://roboflow.com/formats/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/roboflow_dataset>`_
   * `Format documentation <roboflow.md>`_
* NYU Depth Dataset V2 (``depth estimation``) (import-only)
   * `Format specification <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/nyu_depth_v2_dataset>`_
   * `Format documentation <nyu_depth_v2.md>`_
* OpenImages (``classification``, ``detection``, ``segmentation``)
   * `Format specification <https://storage.googleapis.com/openimages/web/download.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/open_images_dataset>`_
   * `Format documentation <open_images.md>`_
* PASCAL VOC (``classification``, ``detection``, ``segmentation``, ``action classification``, ``person layout``)
   * `Format specification <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/voc_dataset>`_
   * `Format documentation <pascal_voc.md>`_
* Segment Anything (a.k.a SA-1B) (``detection``, ``segmentation``)
   * `Format specification <https://github.com/facebookresearch/segment-anything#dataset>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/segment_anything_dataset>`_
   * `Format documentation <segment_anything.md>`_
* Supervisely (``pointcloud``)
   * `Format specification <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/sly_pointcloud_dataset>`_
   * `Format documentation <sly_pointcloud.md>`_
* SYNTHIA (``segmentation``) (import-only)
   * `Format specification <https://synthia-dataset.net/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/synthia_dataset>`_
   * `Format documentation <synthia.md>`_
* Tabular (``classification``, ``regression``) (import/export only)
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/tabular_dataset/adopt-a-buddy>`_
   * `Format documentation <tabular.md>`_
* TF Detection API (``bboxes``, ``masks``)
   * Format specifications: `[bboxes] <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md>`_, `[masks] <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/tf_detection_api_dataset>`_
* VGGFace2 (``landmarks``, ``bboxes``)
   * `Format specification <https://github.com/ox-vgg/vgg_face2>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vgg_face2_dataset>`_
   * `Format documentation <vgg_face2.md>`_
* VoTT CSV (``detection``) (import-only)
   * `Format specification <https://github.com/microsoft/VoTT>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vott_csv_dataset>`_
   * `Format documentation <vott_csv.md>`_
* VoTT JSON (``detection``) (import-only)
   * `Format specification <https://github.com/microsoft/VoTT>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vott_json_dataset>`_
   * `Format documentation <vott_json.md>`_
* WIDERFace (``bboxes``)
   * `Format specification <http://shuoyang1213.me/WIDERFACE/>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/widerface_dataset>`_
   * `Format documentation <wider_face.md>`_
* YOLO (``bboxes``)
   * `Format specification <https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data>`_
   * `Dataset example <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/yolo_dataset>`_
   * `Format documentation <yolo.md>`_
* YOLO-Ultralytics (``bboxes``)
   * `Format specification <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml>`_
   * `Dataset example <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml>`_
   * `Format documentation <yolo_ultralytics.md>`_

Supported Annotation Types
##########################

* Labels
* Bounding Boxes
* Polygons
* Polylines
* (Segmentation) Masks
* (Key-) Points
* Captions
* 3D cuboids
* Super Resolution Annotation
* Depth Annotation
* Ellipses
* Hash Keys

Datumaro does not separate datasets by tasks like classification, detection, segmentation, etc.
Instead, datasets can have any annotations. When a dataset is exported in a specific format,
only relevant annotations are exported.

Dataset Meta Info File
######################

It is possible to use classes that are not original to the format.
To do this, use ``dataset_meta.json``.

.. code-block:: json

  {
    "label_map": {"0": "background", "1": "car", "2": "person"},
    "segmentation_colors": [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
    "background_label": "0"
  }

- ``label_map`` is a dictionary where the class ID is the key and
  the class name is the value.
- ``segmentation_colors`` is a list of channel-wise values for each class.
  This is only necessary for the segmentation task.
- ``background_label`` is a background label ID in the dataset.
