# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[Unreleased\]

### New features
- Convert Cuboid2D annotation to/from 3D data
  (<https://github.com/openvinotoolkit/datumaro/pull/1639>)
- Add label groups for hierarchical classification in ImageNet
  (<https://github.com/openvinotoolkit/datumaro/pull/1645>)

### Enhancements
- Enhance 'id_from_image_name' transform to ensure each identifier is unique
  (<https://github.com/openvinotoolkit/datumaro/pull/1635>)
- Optimize path assignment to handle point cloud in JSON without images
  (<https://github.com/openvinotoolkit/datumaro/pull/1643>)

### Bug fixes
- Fix assertion to compare hashkeys against expected value
  (<https://github.com/openvinotoolkit/datumaro/pull/1641>)

## Q4 2024 Release 1.10.0

### New features
- Support KITTI 3D format
  (<https://github.com/openvinotoolkit/datumaro/pull/1619>, <https://github.com/openvinotoolkit/datumaro/pull/1621>)
- Add PseudoLabeling transform for unlabeled dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/1594>)

### Enhancements
- Raise an appropriate error when exporting a datumaro dataset if its subset name contains path separators.
  (<https://github.com/openvinotoolkit/datumaro/pull/1615>)
- Update docs for transform plugins
  (<https://github.com/openvinotoolkit/datumaro/pull/1599>)
- Update ov ir model for explorer openvino launcher with CLIP ViT-L/14@336px model
  (<https://github.com/openvinotoolkit/datumaro/pull/1603>)
- Optimize path assignment to handle point cloud in JSON without images
  (<https://github.com/openvinotoolkit/datumaro/pull/1643>)
- Set TabularTransform to process clean transform in parallel
  (<https://github.com/openvinotoolkit/datumaro/pull/1648>)

### Bug fixes
- Fix datumaro format to load visibility information from Points annotations
  (<https://github.com/openvinotoolkit/datumaro/pull/1644>)

## Q4 2024 Release 1.9.1
### Enhancements
- Support multiple labels for kaggle format
  (<https://github.com/openvinotoolkit/datumaro/pull/1607>)
- Use DataFrame.map instead of DataFrame.applymap
  (<https://github.com/openvinotoolkit/datumaro/pull/1613>)

### Bug fixes
- Fix StreamDataset merging when importing in eager mode
  (<https://github.com/openvinotoolkit/datumaro/pull/1609>)

## Q3 2024 Release 1.9.0
### New features
- Add a new CLI command: datum format
  (<https://github.com/openvinotoolkit/datumaro/pull/1570>)
- Add a new Cuboid2D annotation type
  (<https://github.com/openvinotoolkit/datumaro/pull/1601>)
- Support language dataset for DmTorchDataset
  (<https://github.com/openvinotoolkit/datumaro/pull/1592>)

### Enhancements
- Change _Shape to Shape and add comments for subclasses of Shape
  (<https://github.com/openvinotoolkit/datumaro/pull/1568>)
- Fix `kitti_raw` importer and exporter for dimensions (height, width, length) in meters
  (<https://github.com/openvinotoolkit/datumaro/pull/1596>)

### Bug fixes
- Fix KITTI-3D importer and exporter
  (<https://github.com/openvinotoolkit/datumaro/pull/1596>)

## Q3 2024 Release 1.8.0
### New features
- Add TabularValidator
  (<https://github.com/openvinotoolkit/datumaro/pull/1498>)
- Add Clean Transform for tabular data type
  (<https://github.com/openvinotoolkit/datumaro/pull/1520>)

### Enhancements
- Set label name with parents to avoid duplicates for AstypeAnnotations
  (<https://github.com/openvinotoolkit/datumaro/pull/1492>)
- Pass Keyword Argument to TabularDataBase
  (<https://github.com/openvinotoolkit/datumaro/pull/1522>)
- Support hierarchical structure for ImageNet dataset format
  (<https://github.com/openvinotoolkit/datumaro/pull/1528>)
- Enable dtype argument when calling media.data
  (<https://github.com/openvinotoolkit/datumaro/pull/1546>)

### Bug fixes
- Preserve end_frame information of a video when it is zero.
  (<https://github.com/openvinotoolkit/datumaro/pull/1541>)
- Changed the Datumaro format to ensure exported videos have relative paths and to prevent the same video from being overwritten.
  (<https://github.com/openvinotoolkit/datumaro/pull/1547>)

## Q2 2024 Release 1.7.0
### New features
- Support 'Video' media type in datumaro format
  (<https://github.com/openvinotoolkit/datumaro/pull/1491>)
- Add ann_types property for dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/1422>, <https://github.com/openvinotoolkit/datumaro/pull/1479>)
- Add AnnotationType.rotated_bbox for oriented object detection
  (<https://github.com/openvinotoolkit/datumaro/pull/1459>)
- Add DOTA data format for oriented object detection task
  (<https://github.com/openvinotoolkit/datumaro/pull/1475>)
- Add AstypeAnnotations Transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1484>)
- Enhance DatasetItem annotations for semantic segmentation model training use case
  (<https://github.com/openvinotoolkit/datumaro/pull/1503>)
- Add TabularValidator
  (<https://github.com/openvinotoolkit/datumaro/pull/1498>)
- Add Clean Transform for tabular data type
  (<https://github.com/openvinotoolkit/datumaro/pull/1520>)
- Add notebook for data handling of kaggle dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/1534>)

### Enhancements
- Fix ambiguous COCO format detector
  (<https://github.com/openvinotoolkit/datumaro/pull/1442>)
- Get target information for tabular dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/1471>)
- Add ExtractedMask and update importers who can use it to use it
  (<https://github.com/openvinotoolkit/datumaro/pull/1480>)
- Improve PIL and COLOR_BGR context image decode performance
  (<https://github.com/openvinotoolkit/datumaro/pull/1501>)
- Improve get_area() of Polygon through Shoelace formula
  (<https://github.com/openvinotoolkit/datumaro/pull/1507>)
- Improve _Shape point converter
  (<https://github.com/openvinotoolkit/datumaro/pull/1508>)

### Bug fixes
- Split the video directory into subsets to avoid overwriting
  (<https://github.com/openvinotoolkit/datumaro/pull/1485>)
- Doc update to replace --save-images is replaced with --save-media
  (<https://github.com/openvinotoolkit/datumaro/pull/1514>)

## May 2024 Release 1.6.1
### Enhancements
- Prevent AcLauncher for OpenVINO 2024.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1450>)

### Bug fixes
- Modify lxml dependency constraint
  (<https://github.com/openvinotoolkit/datumaro/pull/1460>)
- Fix CLI error occurring when installed with default option only
  (<https://github.com/openvinotoolkit/datumaro/issues/1444>, <https://github.com/openvinotoolkit/datumaro/pull/1454>)
- Relax Pillow dependency constraint
  (<https://github.com/openvinotoolkit/datumaro/pull/1436>)
- Modify Numpy dependency constraint
  (<https://github.com/openvinotoolkit/datumaro/pull/1435>)
- Relax old pandas version constraint
  (<https://github.com/openvinotoolkit/datumaro/pull/1467>)

## Apr. 2024 Release 1.6.0
### New features
- Changed supported Python version range (>=3.9, <=3.11)
  (<https://github.com/openvinotoolkit/datumaro/pull/1269>)
- Support MMDetection COCO format
  (<https://github.com/openvinotoolkit/datumaro/pull/1213>)
- Develop JsonSectionPageMapper in Rust API
  (<https://github.com/openvinotoolkit/datumaro/pull/1224>)
- Add Filtering via User-Provided Python Functions
  (<https://github.com/openvinotoolkit/datumaro/pull/1230>, <https://github.com/openvinotoolkit/datumaro/pull/1233>)
- Remove supporting MacOS platform
  (<https://github.com/openvinotoolkit/datumaro/pull/1235>)
- Support Kaggle image data (`KaggleImageCsvBase`, `KaggleImageTxtBase`, `KaggleImageMaskBase`, `KaggleVocBase`, `KaggleYoloBase`)
  (<https://github.com/openvinotoolkit/datumaro/pull/1240>)
- Add `__getitem__()` for random accessing with O(1) time complexity
  (<https://github.com/openvinotoolkit/datumaro/pull/1247>)
- Add Data-aware Anchor Generator
  (<https://github.com/openvinotoolkit/datumaro/pull/1251>)
- Support bounding box import within Kaggle extractors and add `KaggleCocoBase`
  (<https://github.com/openvinotoolkit/datumaro/pull/1273>)

### Enhancements
- Optimize Python import to make CLI entrypoint faster
  (<https://github.com/openvinotoolkit/datumaro/pull/1182>)
- Add ImageColorScale context manager
  (<https://github.com/openvinotoolkit/datumaro/pull/1194>)
- Enhance visualizer to toggle plot title visibility
  (<https://github.com/openvinotoolkit/datumaro/pull/1228>)
- Enhance Datumaro data format detect() to be memory-bounded and performant
  (<https://github.com/openvinotoolkit/datumaro/pull/1229>)
- Change RoIImage and MosaicImage to have np.uint8 dtype as default
  (<https://github.com/openvinotoolkit/datumaro/pull/1245>)
- Enable image backend and color channel format to be selectable
  (<https://github.com/openvinotoolkit/datumaro/pull/1246>)
- Boost up `CityscapesBase` and `KaggleImageMaskBase` by dropping `np.unique`
  (<https://github.com/openvinotoolkit/datumaro/pull/1261>)
- Enhance RISE algortihm for explainable AI
  (<https://github.com/openvinotoolkit/datumaro/pull/1263>)
- Enhance explore unit test to use real dataset from ImageNet
  (<https://github.com/openvinotoolkit/datumaro/pull/1266>)
- Fix each method of the comparator to be used separately
  (<https://github.com/openvinotoolkit/datumaro/pull/1290>)
- Bump ONNX version to 1.16.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1376>)
- Print the color channel format (RGB) for datum stats command
  (<https://github.com/openvinotoolkit/datumaro/pull/1389>)
- Add ignore_index argument to Mask.as_class_mask() and Mask.as_instance_mask()
  (<https://github.com/openvinotoolkit/datumaro/pull/1409>)

### Bug fixes
- Fix wrong example of Datumaro dataset creation in document
  (<https://github.com/openvinotoolkit/datumaro/pull/1195>)
- Fix wrong command to install datumaro from github
  (<https://github.com/openvinotoolkit/datumaro/pull/1202>, <https://github.com/openvinotoolkit/datumaro/pull/1207>)
- Update document to correct wrong `datum project import` command and add filtering example to filter out items containing annotations.
  (<https://github.com/openvinotoolkit/datumaro/pull/1210>)
- Fix label compare of distance method
  (<https://github.com/openvinotoolkit/datumaro/pull/1205>)
- Fix Datumaro visualizer's import errors after introducing lazy import
  (<https://github.com/openvinotoolkit/datumaro/pull/1220>)
- Fix broken link to supported formats in readme
  (<https://github.com/openvinotoolkit/datumaro/pull/1221>)
- Fix Kinetics data format to have media data
  (<https://github.com/openvinotoolkit/datumaro/pull/1223>)
- Handling undefined labels at the annotation statistics
  (<https://github.com/openvinotoolkit/datumaro/pull/1232>)
- Add unit test for item rename
  (<https://github.com/openvinotoolkit/datumaro/pull/1237>)
- Fix a bug in the previous behavior when importing nested datasets in the project
  (<https://github.com/openvinotoolkit/datumaro/pull/1243>)
- Fix Kaggle importer when adding duplicated labels
  (<https://github.com/openvinotoolkit/datumaro/pull/1244>)
- Fix input tensor shape in model interpreter for OpenVINO 2023.3
  (<https://github.com/openvinotoolkit/datumaro/pull/1251>)
- Add default value for target in prune cli
  (<https://github.com/openvinotoolkit/datumaro/pull/1253>)
- Remove deprecated MediaManager
  (<https://github.com/openvinotoolkit/datumaro/pull/1262>)
- Fix explore command without project
  (<https://github.com/openvinotoolkit/datumaro/pull/1271>)
- Fix enable COCO to import only bboxes
  (<https://github.com/openvinotoolkit/datumaro/pull/1360>)
- Fix resize transform for RleMask annotation
- (<https://github.com/openvinotoolkit/datumaro/pull/1361>)
- Fix import YOLO variants from extractor when `urls` is not specified
  (<https://github.com/openvinotoolkit/datumaro/pull/1362>)

## Jan. 2024 Release 1.5.2
### Enhancements
- Add memory bounded datumaro data format detect to release 1.5.1
  (<https://github.com/openvinotoolkit/datumaro/pull/1241>)
- Bump version string to 1.5.2
  (<https://github.com/openvinotoolkit/datumaro/pull/1249>)
- Remove Protobuf version limitation (<4)
  (<https://github.com/openvinotoolkit/datumaro/pull/1248>)

## Nov. 2023 Release 1.5.1
### Enhancements
- Enhance Datumaro data format stream importer performance
  (<https://github.com/openvinotoolkit/datumaro/pull/1153>)
- Change image default dtype from float32 to uint8
  (<https://github.com/openvinotoolkit/datumaro/pull/1175>)
- Add comparison level-up doc
  (<https://github.com/openvinotoolkit/datumaro/pull/1174>)
- Add ImportError to catch GitPython import error
  (<https://github.com/openvinotoolkit/datumaro/pull/1174>)

### Bug fixes
- Modify the draw function in the visualizer not to raise an error for unsupported annotation types.
  (<https://github.com/openvinotoolkit/datumaro/pull/1180>)
- Correct explore path in the related document.
  (<https://github.com/openvinotoolkit/datumaro/pull/1176>)
- Fix errata in the voc document. Color values in the labelmap.txt should be separated by commas, not colons.
  (<https://github.com/openvinotoolkit/datumaro/pull/1162>)
- Fix hyperlink errors in the document
  (<https://github.com/openvinotoolkit/datumaro/pull/1159>, <https://github.com/openvinotoolkit/datumaro/pull/1161>)
- Fix memory unbounded Arrow data format export/import
  (<https://github.com/openvinotoolkit/datumaro/pull/1169>)
- Update CVAT format doc to bypass warning
  (<https://github.com/openvinotoolkit/datumaro/pull/1183>)

## 15/09/2023 - Release 1.5.0
### New features
- Add SAMAutomaticMaskGeneration transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1168>)
- Add tabular data import/export
  (<https://github.com/openvinotoolkit/datumaro/pull/1089>)
- Support video annotation import/export
  (<https://github.com/openvinotoolkit/datumaro/pull/1124>)
- Add multiframework (PyTorch, Tensorflow) converter
  (<https://github.com/openvinotoolkit/datumaro/pull/1125>)
- Add SAM OVMS and Triton server Docker image builders
  (<https://github.com/openvinotoolkit/datumaro/pull/1129>)
- Add SAMBboxToInstanceMask transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1133>, <https://github.com/openvinotoolkit/datumaro/pull/1134>)
- Add ConfigurableValidator
  (<https://github.com/openvinotoolkit/datumaro/pull/1142>)

### Enhancements
- Enhance `ClassificationValidator` for multi-label classification datasets with `label_groups`
  (<https://github.com/openvinotoolkit/datumaro/pull/1116>)
- Replace Roboflow `xml.etree` with `defusedxml`
  (<https://github.com/openvinotoolkit/datumaro/pull/1117>)
- Define `GroupType` with `IntEnum` for, where `0` is `EXCLUSIVE`
  (<https://github.com/openvinotoolkit/datumaro/pull/1116>)
- Add Rust API to optimize COCOPageMapper performance
  (<https://github.com/openvinotoolkit/datumaro/pull/1120>)
- Support a dictionary input in addition to a single image input for the model launcher to support Segment Anything Model
  (<https://github.com/openvinotoolkit/datumaro/pull/1133>)
- Remove deprecates announced to be removed in 1.5.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1140>)
- Add multi-threading option to ModelTransform and SAMBboxToInstanceMask
  (<https://github.com/openvinotoolkit/datumaro/pull/1145>, <https://github.com/openvinotoolkit/datumaro/pull/1149>)

### Bug fixes
- Coco exporter can export annotations even if there is no media, except for mask annotations which require media info.
  (<https://github.com/openvinotoolkit/datumaro/issues/1147>)(<https://github.com/openvinotoolkit/datumaro/pull/1158>)
- Fix bugs for Tile transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1123>)
- Disable Roboflow Tfrecord format when Tensorflow is not installed
  (<https://github.com/openvinotoolkit/datumaro/pull/1130>)
- Raise VcsAlreadyExists error if vcs directory exists
  (<https://github.com/openvinotoolkit/datumaro/pull/1138>)

## 27/07/2023 - Release 1.4.1
### Bug fixes
- Report errors for COCO (stream) and Datumaro importers
  (<https://github.com/openvinotoolkit/datumaro/pull/1110>)

## 21/07/2023 - Release 1.4.0
### New features
- Add documentation and notebook example for Prune API
  (<https://github.com/openvinotoolkit/datumaro/pull/1070>)
- Changed supported Python version range (>=3.8, <=3.11)
  (<https://github.com/openvinotoolkit/datumaro/pull/1083>)
- Migrate OpenVINO v2023.0.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1036>)
- Add Roboflow data format support (COCO JSON, Pascal VOC XML, YOLOv5-PyTorch, YOLOv7-PyTorch, YOLOv8, YOLOv5 Oriented Bounding Boxes, Multiclass CSV, TFRecord, CreateML JSON)
  (<https://github.com/openvinotoolkit/datumaro/pull/1044>)
- Add MissingAnnotationDetection transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1049>, <https://github.com/openvinotoolkit/datumaro/pull/1063>, <https://github.com/openvinotoolkit/datumaro/pull/1064>)
- Add OVMSLauncher
  (<https://github.com/openvinotoolkit/datumaro/pull/1056>)
- Add Prune API
  (<https://github.com/openvinotoolkit/datumaro/pull/1058>)
- Add TritonLauncher
  (<https://github.com/openvinotoolkit/datumaro/pull/1059>)
- Migrate DVC v3.0.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1072>)
- Stream dataset import/export
  (<https://github.com/openvinotoolkit/datumaro/pull/1077>, <https://github.com/openvinotoolkit/datumaro/pull/1081>, <https://github.com/openvinotoolkit/datumaro/pull/1082>, <https://github.com/openvinotoolkit/datumaro/pull/1091>, <https://github.com/openvinotoolkit/datumaro/pull/1093>, <https://github.com/openvinotoolkit/datumaro/pull/1098>, <https://github.com/openvinotoolkit/datumaro/pull/1102>)
- Support mask annotations for CVAT data format
  (<https://github.com/openvinotoolkit/datumaro/pull/1078>)

### Enhancements
- Support list query for explorer
  (<https://github.com/openvinotoolkit/datumaro/pull/1087>)
- update contributing.md
  (<https://github.com/openvinotoolkit/datumaro/pull/1094>)
- Update 3rd-party.txt for release 1.4.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1099>)
- Give notice that the deprecation works will be done in datumaro==1.5.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1085>)
- Unify COCO, Datumaro, VOC, YOLO importer/exporter progress reporter descriptions
  (<https://github.com/openvinotoolkit/datumaro/pull/1100>)
- Enhance import performance for built-in plugins
  (<https://github.com/openvinotoolkit/datumaro/pull/1031>)
- Change default dtype of load_image() to np.uint8
  (<https://github.com/openvinotoolkit/datumaro/pull/1041>)
- Add OTX ATSS detector model interpreter & refactor interfaces
  (<https://github.com/openvinotoolkit/datumaro/pull/1047>)
- Refactor Launcher and ModelInterpreter
  (<https://github.com/openvinotoolkit/datumaro/pull/1055>)
- Add CVAT data format document
  (<https://github.com/openvinotoolkit/datumaro/pull/1060>)
- Reduce peak memory usage when importing COCO and Datumaro formats
  (<https://github.com/openvinotoolkit/datumaro/pull/1061>)
- Enhance the error message for datum stats to be more user friendly
  (<https://github.com/openvinotoolkit/datumaro/pull/1069>)
- Refactor dataset.py to seperate DatasetStorage
  (<https://github.com/openvinotoolkit/datumaro/pull/1073>)

### Bug fixes
- Create cache dir under only writable filesystem
  (<https://github.com/openvinotoolkit/datumaro/pull/1088>)
- Fix: Dataset infos() can be broken if a transform not redefining infos() is stacked on the top
  (<https://github.com/openvinotoolkit/datumaro/pull/1101>)
- Fix warnings in test_visualizer.py
  (<https://github.com/openvinotoolkit/datumaro/pull/1039>)
- Fix LabelMe data format
  (<https://github.com/openvinotoolkit/datumaro/pull/1053>)
- Prevent installing protobuf>=4
  (<https://github.com/openvinotoolkit/datumaro/pull/1054>)
- Fix UnionMerge
  (<https://github.com/openvinotoolkit/datumaro/pull/1086>)

## 26/05/2023 - Release 1.3.2
### Enhancements
- Let CocoBase continue even if an InvalidAnnotationError is raised
  (<https://github.com/openvinotoolkit/datumaro/pull/1050>)

### Bug fixes
- Install dvc version to 2.x
  (<https://github.com/openvinotoolkit/datumaro/pull/1048>)
- Replace np.append() in Validator
  (<https://github.com/openvinotoolkit/datumaro/pull/1050>)

## 26/05/2023 - Release 1.3.1
### Bug fixes
- Fix Cityscapes format mis-detection
  (<https://github.com/openvinotoolkit/datumaro/pull/1029>)

## 25/05/2023 - Release 1.3.0
### New features
- Add CocoRoboflowImporter
  (<https://github.com/openvinotoolkit/datumaro/pull/976>, <https://github.com/openvinotoolkit/datumaro/pull/1000>)
- Add SynthiaSfImporter and SynthiaAlImporter
  (<https://github.com/openvinotoolkit/datumaro/pull/987>)
- Add intermediate skill docs for filter
  (<https://github.com/openvinotoolkit/datumaro/pull/996>)
- Add VocInstanceSegmentationImporter and VocInstanceSegmentationExporter
  (<https://github.com/openvinotoolkit/datumaro/pull/997>)
- Add Segment Anything data format support
  (<https://github.com/openvinotoolkit/datumaro/pull/1005>, <https://github.com/openvinotoolkit/datumaro/pull/1009>)
- Add Correct transformation
  (<https://github.com/openvinotoolkit/datumaro/pull/1006>)
- Implement ReindexAnnotations transform
  (<https://github.com/openvinotoolkit/datumaro/pull/1008>)
- Add notebook examples for importing/exporting detection and segmentation data
  (<https://github.com/openvinotoolkit/datumaro/pull/1020>, <https://github.com/openvinotoolkit/datumaro/pull/1023>)
- Update CLI from diff to compare, add TableComparator
  (<https://github.com/openvinotoolkit/datumaro/pull/1012>)

### Enhancements
- Use autosummary for fully-automatic Python module docs generation
  (<https://github.com/openvinotoolkit/datumaro/pull/973>)
- Enrich stack trace for better user experience when importing
  (<https://github.com/openvinotoolkit/datumaro/pull/992>)
- Save and load hashkey for explorer
  (<https://github.com/openvinotoolkit/datumaro/pull/981>)
  (<https://github.com/openvinotoolkit/datumaro/pull/1003>)
- Add MOT and MOTS data format docs
  (<https://github.com/openvinotoolkit/datumaro/pull/999>)
- Improve RemoveAnnotations to remove specific annotations with ids
  (<https://github.com/openvinotoolkit/datumaro/pull/1004>)
- Add Jupyter notebook example of noisy label detection for detection tasks
  (<https://github.com/openvinotoolkit/datumaro/pull/1011>)

### Bug fixes
- Fix Mapillary Vistas data format
  (<https://github.com/openvinotoolkit/datumaro/pull/977>)
- Fix `bytes` property returning `None` if function is given to `data`
  (<https://github.com/openvinotoolkit/datumaro/pull/978>)
- Fix Synthia-Rand data format
  (<https://github.com/openvinotoolkit/datumaro/pull/987>)
- Fix `person_layout` categories and `action_classification` attributes in imported Pascal-VOC dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/997>)
- Drop a malformed transform from StackedTransform automatically
  (<https://github.com/openvinotoolkit/datumaro/pull/1001>)
- Fix `Cityscapes` to drop `ImgsFine` directory
  (<https://github.com/openvinotoolkit/datumaro/pull/1023>)

## 04/05/2023 - Release 1.2.1
### Bug fixes
- Fix project level CVAT for images format import
  (<https://github.com/openvinotoolkit/datumaro/pull/980>)
- Fix an info message when using the convert CLI command with no args.input_format
  (<https://github.com/openvinotoolkit/datumaro/pull/982>)
- Fix media contents not returning bytes in arrow format
  (<https://github.com/openvinotoolkit/datumaro/pull/986>)

## 20/04/2023 - Release 1.2.0
### New features
- Add Skill Up section to documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/920>, <https://github.com/openvinotoolkit/datumaro/pull/933>, <https://github.com/openvinotoolkit/datumaro/pull/935>, <https://github.com/openvinotoolkit/datumaro/pull/945>, <https://github.com/openvinotoolkit/datumaro/pull/949>, <https://github.com/openvinotoolkit/datumaro/pull/953>, <https://github.com/openvinotoolkit/datumaro/pull/959>, <https://github.com/openvinotoolkit/datumaro/pull/960>, <https://github.com/openvinotoolkit/datumaro/pull/967>)
- Add LossDynamicsAnalyzer for noisy label detection
  (<https://github.com/openvinotoolkit/datumaro/pull/928>)
- Add Apache Arrow format support
  (<https://github.com/openvinotoolkit/datumaro/pull/931>, <https://github.com/openvinotoolkit/datumaro/pull/948>)
- Add sort transform
  (<https://github.com/openvinotoolkit/datumaro/pull/931>)

### Enhancements
- Add multiprocessing to DatumaroBinaryBase
  (<https://github.com/openvinotoolkit/datumaro/pull/897>)
- Refactor merge code
  (<https://github.com/openvinotoolkit/datumaro/pull/901>, <https://github.com/openvinotoolkit/datumaro/pull/906>)
- Refactor download CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/909>)
- Refactor CLI commands w/ and w/o project
  (<https://github.com/openvinotoolkit/datumaro/pull/910>, <https://github.com/openvinotoolkit/datumaro/pull/952>)
- Refactor Media to be initialized from explicit sources
  (<https://github.com/openvinotoolkit/datumaro/pull/911> <https://github.com/openvinotoolkit/datumaro/pull/921>, <https://github.com/openvinotoolkit/datumaro/pull/944>)
- Refactor hl_ops.py
  (<https://github.com/openvinotoolkit/datumaro/pull/912>)
- Add tfds:uc_merced and tfds:eurosat download
  (<https://github.com/openvinotoolkit/datumaro/pull/914>)
- Migrate documentation framework to Sphinx
  (<https://github.com/openvinotoolkit/datumaro/pull/917>, <https://github.com/openvinotoolkit/datumaro/pull/922>, <https://github.com/openvinotoolkit/datumaro/pull/947>, <https://github.com/openvinotoolkit/datumaro/pull/954>, <https://github.com/openvinotoolkit/datumaro/pull/958>, <https://github.com/openvinotoolkit/datumaro/pull/961>, <https://github.com/openvinotoolkit/datumaro/pull/962>, <https://github.com/openvinotoolkit/datumaro/pull/963>, <https://github.com/openvinotoolkit/datumaro/pull/964>, <https://github.com/openvinotoolkit/datumaro/pull/965>, <https://github.com/openvinotoolkit/datumaro/pull/969>)
- Update merge tutorial for real life usecase
  (<https://github.com/openvinotoolkit/datumaro/pull/930>)
- Abbreviate "detect-format" to "detect" for prettifying
  (<https://github.com/openvinotoolkit/datumaro/pull/951>)

### Bug fixes
- Add UserWarning if an invalid media_type comes to image statistics computation
  (<https://github.com/openvinotoolkit/datumaro/pull/891>)
- Fix negated `is_encrypted`
  (<https://github.com/openvinotoolkit/datumaro/pull/907>)
- Save extra images of PointCloud when exporting to datumaro format
  (<https://github.com/openvinotoolkit/datumaro/pull/918>)
- Fix log issue when importing celeba and align celeba dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/919>)

## 28/03/2023 - Release 1.1.1
### Bug fixes
- Fix to not export absolute media path in Datumaro and DatumaroBinary formats
  (<https://github.com/openvinotoolkit/datumaro/pull/896>)
- Change pypi_publish.yml to publish_sdist_to_pypi.yml
  (<https://github.com/openvinotoolkit/datumaro/pull/895>)

## 23/03/2023 - Release 1.1.0
### New features
- Add with_subset_dirs decorator (Add ImagenetWithSubsetDirsImporter)
  (<https://github.com/openvinotoolkit/datumaro/pull/816>)
- Add CommonSemanticSegmentationWithSubsetDirsImporter
  (<https://github.com/openvinotoolkit/datumaro/pull/826>)
- Add DatumaroBinary format
  (<https://github.com/openvinotoolkit/datumaro/pull/828>, <https://github.com/openvinotoolkit/datumaro/pull/829>, <https://github.com/openvinotoolkit/datumaro/pull/830>, <https://github.com/openvinotoolkit/datumaro/pull/831>, <https://github.com/openvinotoolkit/datumaro/pull/880>, <https://github.com/openvinotoolkit/datumaro/pull/883>)
- Add Explorer CLI documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/838>)
- Add version to dataset exported as datumaro format
  (<https://github.com/openvinotoolkit/datumaro/pull/842>)
- Add Ava action data format support
  (<https://github.com/openvinotoolkit/datumaro/pull/847>)
- Add Shift Analyzer (both covariate and label shifts)
  (<https://github.com/openvinotoolkit/datumaro/pull/855>)
- Add YOLO Loose format
  (<https://github.com/openvinotoolkit/datumaro/pull/856>)
- Add Ultralytics YOLO format
  (<https://github.com/openvinotoolkit/datumaro/pull/859>)

### Enhancements
- Refactor Datumaro format code and test code
  (<https://github.com/openvinotoolkit/datumaro/pull/824>)
- Add publish to PyPI Github action
  (<https://github.com/openvinotoolkit/datumaro/pull/867>)
- Add --no-media-encryption option
  (<https://github.com/openvinotoolkit/datumaro/pull/875>)

### Bug fixes
- Fix image filenames and anomaly mask appearance in MVTec exporter
  (<https://github.com/openvinotoolkit/datumaro/pull/835>)
- Fix CIFAR10 and 100 detect function
  (<https://github.com/openvinotoolkit/datumaro/pull/836>)
- Fix celeba and align_celeba detect function
  (<https://github.com/openvinotoolkit/datumaro/pull/837>)
- Choose the top priority detect format for all directory depths
  (<https://github.com/openvinotoolkit/datumaro/pull/839>)
- Fix MVTec format detect function
  (<https://github.com/openvinotoolkit/datumaro/pull/843>)
- Fix wrong `__len__()` of Subset when the item is removed
  (<https://github.com/openvinotoolkit/datumaro/pull/854>)
- Fix mask visualization bug
  (<https://github.com/openvinotoolkit/datumaro/pull/860>)
- Fix detect unit tests to test false negatives as well
  (<https://github.com/openvinotoolkit/datumaro/pull/868>)

## 24/02/2023 - Release v1.0.0
### New features
- Add Data Explorer
  (<https://github.com/openvinotoolkit/datumaro/pull/773>)
- Add Ellipse annotation type
  (<https://github.com/openvinotoolkit/datumaro/pull/807>)
- Add MVTec anomaly data support
  (<https://github.com/openvinotoolkit/datumaro/pull/810>)

### Enhancements
- Refactor existing tests
  (<https://github.com/openvinotoolkit/datumaro/pull/803>)
- Raise ImportError on importing malformed COCO directory
  (<https://github.com/openvinotoolkit/datumaro/pull/812>)
- Remove the duplicated and cyclical category context in documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/822>)

### Bug fixes
- Fix for importing CVAT image 1.1 data format exported to project level
  (<https://github.com/openvinotoolkit/datumaro/pull/795>)
- Fix a problem on setting log-level via CLI
  (<https://github.com/openvinotoolkit/datumaro/pull/800>)
- Fix code format with the latest black==23.1.0
  (<https://github.com/openvinotoolkit/datumaro/pull/802>)
- Fix [Explain command cannot find the model (#721)](https://github.com/openvinotoolkit/datumaro/issues/721)  (<https://github.com/openvinotoolkit/datumaro/pull/804>)
- Fix a problem found on model remove CLI command
  (<https://github.com/openvinotoolkit/datumaro/pull/805>)

## 27/01/2023 - Release v0.5.0
### New features
- Add Tile transformation
  (<https://github.com/openvinotoolkit/datumaro/pull/790>)
- Add Video keyframe extraction
  (<https://github.com/openvinotoolkit/datumaro/pull/791>)
- Add TileTransform documentation and Jupyter notebook example
  (<https://github.com/openvinotoolkit/datumaro/pull/794>)
- Add MergeTile transformation
  (<https://github.com/openvinotoolkit/datumaro/pull/796>)

### Enhancements
- Improved mask_to_rle performance
  (<https://github.com/openvinotoolkit/datumaro/pull/770>)

### Deprecated
- N/A

### Removed
- N/A

### Bug fixes
- Fix MacOS CI failures
  (<https://github.com/openvinotoolkit/datumaro/pull/789>)
- Fix auto-documentation for the data_format plugins
  (<https://github.com/openvinotoolkit/datumaro/pull/793>)

### Security
- Add security.md file for the SDL
  (<https://github.com/openvinotoolkit/datumaro/pull/798>)

## 06/12/2022 - Release v0.4.0.1
### New features
- Support for exclusive of labels with LabelGroup
  (<https://github.com/openvinotoolkit/datumaro/pull/742>)
- Jupyter samples
  - Introducing how to merge datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/738>)
  - Introducing how to visualize dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/747>)
  - Introducing how to filter dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/748>)
  - Introducing how to transform dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/759>)
- Visualization Python API
  - Bbox feature
    (<https://github.com/openvinotoolkit/datumaro/pull/744>)
  - Label, Points, Polygon, PolyLine, and Caption visualization features
    (<https://github.com/openvinotoolkit/datumaro/pull/746>)
  - Mask, SuperResolution, Depth visualization features
    (<https://github.com/openvinotoolkit/datumaro/pull/747>)
- Documentation for Python API
  (<https://github.com/openvinotoolkit/datumaro/pull/753>)
  - dataset handler, visualizer, filter descriptions
    (<https://github.com/openvinotoolkit/datumaro/pull/761>)
- `__repr__` for Dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/750>)
- Support for exporting as CVAT video format
  (<https://github.com/openvinotoolkit/datumaro/pull/757>)
- CodeCov coverage reporting feature to CI/CD
  (<https://github.com/openvinotoolkit/datumaro/pull/756>)
- Jupyter notebook example rendering to documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/758>)
- An interface to manipulate 'infos' to store the dataset meta-info
  (<https://github.com/openvinotoolkit/datumaro/pull/767>)
- 'bbox' annotation when importing a COCO dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/772>)

### Enhancements
- Wrap title text according to its plot width
  (<https://github.com/openvinotoolkit/datumaro/pull/769>)
- Get list of subsets and support only Image media type in visualizer
  (<https://github.com/openvinotoolkit/datumaro/pull/768>)

### Deprecated
- N/A

### Removed
- N/A

### Bug fixes
- Correcting static type checking
  (<https://github.com/openvinotoolkit/datumaro/pull/743>)
- Fixing a VOC dataset export when a label contains 'space'
  (<https://github.com/openvinotoolkit/datumaro/pull/771>)

### Security
- N/A

## 06/09/2022 - Release v0.3.1
### New features
- Support for custom media types, new `PointCloud` media type,
  `DatasetItem.media` and `.media_as(type)` members
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- \[API\] A way to request dataset and extractor media type with `media_type`
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- BraTS format (import-only) (.npy and .nii.gz), new `MultiframeImage`
  media type (<https://github.com/openvinotoolkit/datumaro/pull/628>)
- Common Semantic Segmentation dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/685>)
- An option to disable `data/` prefix inclusion in YOLO export
  (<https://github.com/openvinotoolkit/datumaro/pull/689>)
- New command `describe-downloads` to print information about downloadable datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/678>)
- Detection for Cityscapes format
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Maximum recursion `--depth` parameter for `detect-dataset` CLI command
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- An option to save a single subset in the `download` command
  (<https://github.com/openvinotoolkit/datumaro/pull/697>)
- Common Super Resolution dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/700>)
- Kinetics 400/600/700 dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/706>)
- NYU Depth Dataset V2 format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/712>)

### Enhancements
- `env.detect_dataset()` now returns a list of detected formats at all recursion levels
  instead of just the lowest one
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Open Images: allowed to store annotations file in root path as well
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Improved parsing error messages in COCO, VOC and YOLO formats
  (<https://github.com/openvinotoolkit/datumaro/pull/684>,
   <https://github.com/openvinotoolkit/datumaro/pull/686>,
   <https://github.com/openvinotoolkit/datumaro/pull/687>)
- YOLO format now supports almost any subset names, except `backup`, `names` and `classes`
  (instead of just `train` and `valid`). The reserved names now raise an error on exporting.
  (<https://github.com/openvinotoolkit/datumaro/pull/688>)

### Deprecated
- `--save-images` is replaced with `--save-media` in CLI and converter API
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- \[API\] `image`, `point_cloud` and `related_images` of `DatasetItem` are
  replaced with `media` and `media_as(type)` members and c-tor parameters
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)

### Removed
- N/A

### Bug fixes
- Detection for LFW format
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Adding depth value of image when dataset is exported in VOC format
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)
- Adding to handle the numerical labels in task chains properly
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)
- Fixing the issue that annotations inside another annotation (polygon)
  are duplicated during import for VOC format
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)

### Security
- N/A

## 21/02/2022 - Release v0.3
### New features
- Ability to import a video as frames with the `video_frames` format and
  to split a video into frames with the `datum util split_video` command
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- `--subset` parameter in the `image_dir` format
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- `MediaManager` API to control loaded media resources at runtime
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Command to detect the format of a dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/576>)
- More comfortable access to library API via `import datumaro`
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)
- CLI command-like free functions (`export`, `transform`, ...)
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)
- Reading specific annotation files for train dataset in Cityscapes
  (<https://github.com/openvinotoolkit/datumaro/pull/632>)
- Random sampling transforms (`random_sampler`, `label_random_sampler`)
  to create smaller datasets from bigger ones
  (<https://github.com/openvinotoolkit/datumaro/pull/636>,
   <https://github.com/openvinotoolkit/datumaro/pull/640>)
- API to report dataset import and export progress;
  API to report dataset import and export errors and take action (skip, fail)
  (supported in COCO, VOC and YOLO formats)
  (<https://github.com/openvinotoolkit/datumaro/pull/650>)
- Support for downloading the ImageNetV2 and COCO datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/653>,
   <https://github.com/openvinotoolkit/datumaro/pull/659>)
- A way for formats to signal that they don't support detection
  (<https://github.com/openvinotoolkit/datumaro/pull/665>)
- Removal transforms to remove items/annoations/attributes from dataset
  (`remove_items`, `remove_annotations`, `remove_attributes`)
  (<https://github.com/openvinotoolkit/datumaro/pull/670>)

### Enhancements
- Allowed direct file paths in `datum import`. Such sources are imported like
  when the `rpath` parameter is specified, however, only the selected path
  is copied into the project
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Improved `stats` performance, added new filtering parameters,
  image stats (`unique`, `repeated`) moved to the `dataset` section,
  removed `mean` and `std` from the `dataset` section
  (<https://github.com/openvinotoolkit/datumaro/pull/621>)
- Allowed `Image` creation from just `size` info
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Added image search in VOC XML-based subformats
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Added image path equality checks in simple merge, when applicable
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Supported saving box attributes when downloading the TFDS version of VOC
  (<https://github.com/openvinotoolkit/datumaro/pull/668>)
- Switched to a `pyproject.toml`-based build
  (<https://github.com/openvinotoolkit/datumaro/pull/671>)

### Deprecated
- TBD

### Removed
- Official support of Python 3.6 (due to it's EOL)
  (<https://github.com/openvinotoolkit/datumaro/pull/617>)
- Backward compatibility annotation symbols in `components.extractor`
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)

### Bug fixes
- Prohibited calling `add`, `import` and `export` commands without a project
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Calling `make_dataset` on empty project tree now produces the error properly
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Saving (overwriting) a dataset in a project when rpath is used
  (<https://github.com/openvinotoolkit/datumaro/pull/613>)
- Output image extension preserving in the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/606>)
- Memory overuse in the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/607>)
- Invalid image pixels produced by the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/618>)
- Numeric warnings that sometimes occurred in `stats` command
  (e.g. <https://github.com/openvinotoolkit/datumaro/issues/607>)
  (<https://github.com/openvinotoolkit/datumaro/pull/621>)
- Added missing item attribute merging in simple merge
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Inability to disambiguate VOC from LabelMe in some cases
  (<https://github.com/openvinotoolkit/datumaro/issues/658>)

### Security
- TBD

## 28/01/2022 - Release v0.2.3
### New features
- Command to download public datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/582>)
- Extension autodetection in `ByteImage`
  (<https://github.com/openvinotoolkit/datumaro/pull/595>)
- MPII Human Pose Dataset (import-only) (.mat and .json)
  (<https://github.com/openvinotoolkit/datumaro/pull/584>)
- MARS format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/585>)

### Enhancements
- The `pycocotools` dependency lower bound is raised to `2.0.4`.
  (<https://github.com/openvinotoolkit/datumaro/pull/449>)
- `smooth_line` from `datumaro.util.annotation_util` - the function
  is renamed to `approximate_line` and has updated interface
  (<https://github.com/openvinotoolkit/datumaro/pull/592>)

### Deprecated
- Python 3.6 support

### Removed
- TBD

### Bug fixes
- Fails in multimerge when lines are not approximated and when there are no
  label categories (<https://github.com/openvinotoolkit/datumaro/pull/592>)
- Cannot convert LabelMe dataset, that has no subsets
  (<https://github.com/openvinotoolkit/datumaro/pull/600>)

### Security
- TBD

## 24/12/2021 - Release v0.2.2
### New features
- Video reading API
  (<https://github.com/openvinotoolkit/datumaro/pull/521>)
- Python API documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/526>)
- Mapillary Vistas dataset format (Import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/537>)
- Datumaro can now be installed on Windows on Python 3.9
  (<https://github.com/openvinotoolkit/datumaro/pull/547>)
- Import for SYNTHIA dataset format
  (<https://github.com/openvinotoolkit/datumaro/pull/532>)
- Support of `score` attribute in KITTI detetion
  (<https://github.com/openvinotoolkit/datumaro/pull/571>)
- Support for Accuracy Checker dataset meta files in formats
  (<https://github.com/openvinotoolkit/datumaro/pull/553>,
  <https://github.com/openvinotoolkit/datumaro/pull/569>,
  <https://github.com/openvinotoolkit/datumaro/pull/575>)
- Import for VoTT dataset format
  (<https://github.com/openvinotoolkit/datumaro/pull/573>)
- Image resizing transform
  (<https://github.com/openvinotoolkit/datumaro/pull/581>)

### Enhancements
- The following formats can now be detected unambiguously:
  `ade20k2017`, `ade20k2020`, `camvid`, `coco`, `cvat`, `datumaro`,
  `icdar_text_localization`, `icdar_text_segmentation`,
  `icdar_word_recognition`, `imagenet_txt`, `kitti_raw`, `label_me`, `lfw`,
  `mot_seq`, `open_images`, `vgg_face2`, `voc`, `widerface`, `yolo`
  (<https://github.com/openvinotoolkit/datumaro/pull/531>,
  <https://github.com/openvinotoolkit/datumaro/pull/536>,
  <https://github.com/openvinotoolkit/datumaro/pull/550>,
  <https://github.com/openvinotoolkit/datumaro/pull/557>,
  <https://github.com/openvinotoolkit/datumaro/pull/558>)
- Allowed Pytest-native tests
  (<https://github.com/openvinotoolkit/datumaro/pull/563>)
- Allowed export options in the `datum merge` command
  (<https://github.com/openvinotoolkit/datumaro/pull/545>)

### Deprecated
- Using `Image`, `ByteImage` from `datumaro.util.image` - these classes
  are moved to `datumaro.components.media`
  (<https://github.com/openvinotoolkit/datumaro/pull/538>)

### Removed
- Equality comparison support between `datumaro.components.media.Image`
  and `numpy.ndarray`
  (<https://github.com/openvinotoolkit/datumaro/pull/568>)

### Bug fixes
- Bug #560: import issue with MOT dataset when using seqinfo.ini file
  (<https://github.com/openvinotoolkit/datumaro/pull/564>)
- Empty lines in VOC subset lists are not ignored
  (<https://github.com/openvinotoolkit/datumaro/pull/587>)

### Security
- TBD

## 16/11/2021 - Release v0.2.1
### New features
- Import for CelebA dataset format.
  (<https://github.com/openvinotoolkit/datumaro/pull/484>)

### Enhancements
- File `people.txt` became optional in LFW
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- File `image_ids_and_rotation.csv` became optional Open Images
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- Allowed underscores (`_`) in subset names in COCO
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- Allowed annotation files with arbitrary names in COCO
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- The `icdar_text_localization` format is no longer detected in every directory
  (<https://github.com/openvinotoolkit/datumaro/pull/531>)
- Updated `pycocotools` version to 2.0.2
  (<https://github.com/openvinotoolkit/datumaro/pull/534>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Unhandled exception when a file is specified as the source for a COCO or
  MOTS dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/530>)
- Exporting dataset without `color` attribute into the
  `icdar_text_segmentation` format
  (<https://github.com/openvinotoolkit/datumaro/pull/556>)
### Security
- TBD

## 14/10/2021 - Release v0.2
### New features
- A new installation target: `pip install datumaro[default]`, which should
  be used by default. The simple `datumaro` is supposed for library users.
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Dataset and project versioning capabilities (Git-like)
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- "dataset revpath" concept in CLI, allowing to pass a dataset path with
  the dataset format in `diff`, `merge`, `explain` and `info` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `import`, `remove`, `commit`, `checkout`, `log`, `status`, `info` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `Coco*Extractor` classes now have an option to preserve label IDs from the
  original annotation file
  (<https://github.com/openvinotoolkit/datumaro/pull/453>)
- `patch` CLI command to patch datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- `ProjectLabels` transform to change dataset labels for merging etc.
  (<https://github.com/openvinotoolkit/datumaro/pull/401>,
   <https://github.com/openvinotoolkit/datumaro/pull/478>)
- Support for custom labels in the KITTI detection format
  (<https://github.com/openvinotoolkit/datumaro/pull/481>)
- Type annotations and docs for Annotation classes
  (<https://github.com/openvinotoolkit/datumaro/pull/493>)
- Options to control label loading behavior in `imagenet_txt` import
  (<https://github.com/openvinotoolkit/datumaro/pull/434>,
  <https://github.com/openvinotoolkit/datumaro/pull/489>)

### Enhancements
- A project can contain and manage multiple datasets instead of a single one.
  CLI operations can be applied to the whole project, or to separate datasets.
  Datasets are modified inplace, by default
  (<https://github.com/openvinotoolkit/datumaro/issues/328>)
- CLI help for builtin plugins doesn't require project
  (<https://github.com/openvinotoolkit/datumaro/issues/328>)
- Annotation-related classes were moved into a new module,
  `datumaro.components.annotation`
  (<https://github.com/openvinotoolkit/datumaro/pull/439>)
- Rollback utilities replaced with Scope utilities
  (<https://github.com/openvinotoolkit/datumaro/pull/444>)
- The `Project` class from `datumaro.components` is changed completely
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `diff` and `ediff` are joined into a single `diff` CLI command
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Projects use new file layout, incompatible with old projects.
  An old project can be updated with `datum project migrate`
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Inheriting `CliPlugin` is not required in plugin classes
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `Importer`s do not create `Project`s anymore and just return a list of
  extractor configurations
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)

### Deprecated
- TBD

### Removed
- `import`, `project merge` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Support for project hierarchies. A project cannot be a source anymore
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Project cannot have independent internal dataset anymore. All the project
  data must be stored in the project data sources
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `datumaro_project` format
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Unused `path` field of `DatasetItem`
  (<https://github.com/openvinotoolkit/datumaro/pull/455>)

### Bug fixes
- Deprecation warning in `open_images_format.py`
  (<https://github.com/openvinotoolkit/datumaro/pull/440>)
- `lazy_image` returning unrelated data sometimes
  (<https://github.com/openvinotoolkit/datumaro/issues/409>)
- Invalid call to `pycocotools.mask.iou`
  (<https://github.com/openvinotoolkit/datumaro/pull/450>)
- Importing of Open Images datasets without image data
  (<https://github.com/openvinotoolkit/datumaro/pull/463>)
- Return value type in `Dataset.is_modified`
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- Remapping of secondary categories in `RemapLabels`
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- VOC dataset patching for classification and segmentation tasks
  (<https://github.com/openvinotoolkit/datumaro/pull/478>)
- Exported mask label ids in KITTI segmentation
  (<https://github.com/openvinotoolkit/datumaro/pull/481>)
- Missing `label` for `Points` read in the LFW format
  (<https://github.com/openvinotoolkit/datumaro/pull/494>)

### Security
- TBD

## 24/08/2021 - Release v0.1.11
### New features
- The Open Images format now supports bounding box
  and segmentation mask annotations
  (<https://github.com/openvinotoolkit/datumaro/pull/352>,
  <https://github.com/openvinotoolkit/datumaro/pull/388>).
- Bounding boxes values decrement transform (<https://github.com/openvinotoolkit/datumaro/pull/366>)
- Improved error reporting in `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/386>)
- Support ADE20K format (import only) (<https://github.com/openvinotoolkit/datumaro/pull/400>)
- Documentation website at <https://openvinotoolkit.github.io/datumaro> (<https://github.com/openvinotoolkit/datumaro/pull/420>)

### Enhancements
- Datumaro no longer depends on scikit-image
  (<https://github.com/openvinotoolkit/datumaro/pull/379>)
- `Dataset` remembers export options on saving / exporting for the first time (<https://github.com/openvinotoolkit/datumaro/pull/386>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Application of `remap_labels` to dataset categories of different length (<https://github.com/openvinotoolkit/datumaro/issues/314>)
- Patching of datasets in formats (<https://github.com/openvinotoolkit/datumaro/issues/348>)
- Improved Cityscapes export performance (<https://github.com/openvinotoolkit/datumaro/pull/367>)
- Incorrect format of `*_labelIds.png` in Cityscapes export (<https://github.com/openvinotoolkit/datumaro/issues/325>, <https://github.com/openvinotoolkit/datumaro/issues/342>)
- Item id in ImageNet format (<https://github.com/openvinotoolkit/datumaro/pull/371>)
- Double quotes for ICDAR Word Recognition (<https://github.com/openvinotoolkit/datumaro/pull/375>)
- Wrong display of builtin formats in CLI (<https://github.com/openvinotoolkit/datumaro/issues/332>)
- Non utf-8 encoding of annotation files in Market-1501 export (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Import of ICDAR, PASCAL VOC and VGGFace2 images from subdirectories on WIndows
  (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Saving of images with Unicode paths on Windows (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Calling `ProjectDataset.transform()` with a string argument (<https://github.com/openvinotoolkit/datumaro/issues/402>)
- Attributes casting for CVAT format (<https://github.com/openvinotoolkit/datumaro/pull/403>)
- Loading of custom project plugins (<https://github.com/openvinotoolkit/datumaro/issues/404>)
- Reading, writing anno file and saving name of the subset for test subset
  (<https://github.com/openvinotoolkit/datumaro/pull/447>)

### Security
- Fixed unsafe unpickling in CIFAR import (<https://github.com/openvinotoolkit/datumaro/pull/362>)

## 14/07/2021 - Release v0.1.10
### New features
- Support for import/export zip archives with images (<https://github.com/openvinotoolkit/datumaro/pull/273>)
- Subformat importers for VOC and COCO (<https://github.com/openvinotoolkit/datumaro/pull/281>)
- Support for KITTI dataset segmentation and detection format (<https://github.com/openvinotoolkit/datumaro/pull/282>)
- Updated YOLO format user manual (<https://github.com/openvinotoolkit/datumaro/pull/295>)
- `ItemTransform` class, which describes item-wise dataset `Transform`s (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- `keep-empty` export parameter in VOC format (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- A base class for dataset validation plugins (<https://github.com/openvinotoolkit/datumaro/pull/299>)
- Partial support for the Open Images format;
  only images and image-level labels can be read/written
  (<https://github.com/openvinotoolkit/datumaro/pull/291>,
  <https://github.com/openvinotoolkit/datumaro/pull/315>).
- Support for Supervisely Point Cloud dataset format (<https://github.com/openvinotoolkit/datumaro/pull/245>, <https://github.com/openvinotoolkit/datumaro/pull/353>)
- Support for KITTI Raw / Velodyne Points dataset format (<https://github.com/openvinotoolkit/datumaro/pull/245>)
- Support for CIFAR-100 and documentation for CIFAR-10/100 (<https://github.com/openvinotoolkit/datumaro/pull/301>)

### Enhancements
- Tensorflow AVX check is made optional in API and disabled by default (<https://github.com/openvinotoolkit/datumaro/pull/305>)
- Extensions for images in ImageNet_txt are now mandatory (<https://github.com/openvinotoolkit/datumaro/pull/302>)
- Several dependencies now have lower bounds (<https://github.com/openvinotoolkit/datumaro/pull/308>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Incorrect image layout on saving and a problem with ecoding on loading (<https://github.com/openvinotoolkit/datumaro/pull/284>)
- An error when XPath filter is applied to the dataset or its subset (<https://github.com/openvinotoolkit/datumaro/issues/259>)
- Tracking of `Dataset` changes done by transforms (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- Improved CLI startup time in several cases (<https://github.com/openvinotoolkit/datumaro/pull/306>)

### Security
- Known issue: loading CIFAR can result in arbitrary code execution (<https://github.com/openvinotoolkit/datumaro/issues/327>)

## 03/06/2021 - Release v0.1.9
### New features
- Support for escaping in attribute values in LabelMe format (<https://github.com/openvinotoolkit/datumaro/issues/49>)
- Support for Segmentation Splitting (<https://github.com/openvinotoolkit/datumaro/pull/223>)
- Support for CIFAR-10/100 dataset format (<https://github.com/openvinotoolkit/datumaro/pull/225>, <https://github.com/openvinotoolkit/datumaro/pull/243>)
- Support for COCO panoptic and stuff format (<https://github.com/openvinotoolkit/datumaro/pull/210>)
- Documentation file and integration tests for Pascal VOC format (<https://github.com/openvinotoolkit/datumaro/pull/228>)
- Support for MNIST and MNIST in CSV dataset formats (<https://github.com/openvinotoolkit/datumaro/pull/234>)
- Documentation file for COCO format (<https://github.com/openvinotoolkit/datumaro/pull/241>)
- Documentation file and integration tests for YOLO format (<https://github.com/openvinotoolkit/datumaro/pull/246>)
- Support for Cityscapes dataset format (<https://github.com/openvinotoolkit/datumaro/pull/249>)
- Support for Validator configurable threshold (<https://github.com/openvinotoolkit/datumaro/pull/250>)

### Enhancements
- LabelMe format saves dataset items with their relative paths by subsets
  without changing names (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed arbitrary subset count and names in classification and detection
  splitters (<https://github.com/openvinotoolkit/datumaro/pull/207>)
- Annotation-less dataset elements are now participate in subset splitting (<https://github.com/openvinotoolkit/datumaro/pull/211>)
- Classification task in LFW dataset format (<https://github.com/openvinotoolkit/datumaro/pull/222>)
- Testing is now performed with pytest instead of unittest (<https://github.com/openvinotoolkit/datumaro/pull/248>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Added support for auto-merging (joining) of datasets with no labels and
  having labels (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed explicit label removal in `remap_labels` transform (<https://github.com/openvinotoolkit/datumaro/pull/203>)
- Image extension in CVAT format export (<https://github.com/openvinotoolkit/datumaro/pull/214>)
- Added a label "face" for bounding boxes in Wider Face (<https://github.com/openvinotoolkit/datumaro/pull/215>)
- Allowed adding "difficult", "truncated", "occluded" attributes when
  converting to Pascal VOC if these attributes are not present (<https://github.com/openvinotoolkit/datumaro/pull/216>)
- Empty lines in YOLO annotations are ignored (<https://github.com/openvinotoolkit/datumaro/pull/221>)
- Export in VOC format when no image info is available (<https://github.com/openvinotoolkit/datumaro/pull/239>)
- Fixed saving attribute in WiderFace extractor (<https://github.com/openvinotoolkit/datumaro/pull/251>)

### Security
- TBD

## 31/03/2021 - Release v0.1.8
### New features
- TBD

### Enhancements
- Added an option to allow undeclared annotation attributes in CVAT format
  export (<https://github.com/openvinotoolkit/datumaro/pull/192>)
- COCO exports images in separate dirs by subsets. Added an option to control
  this (<https://github.com/openvinotoolkit/datumaro/pull/195>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Instance masks of `background` class no more introduce an instance (<https://github.com/openvinotoolkit/datumaro/pull/188>)
- Added support for label attributes in Datumaro format (<https://github.com/openvinotoolkit/datumaro/pull/192>)

### Security
- TBD

## 24/03/2021 - Release v0.1.7
### New features
- OpenVINO plugin examples (<https://github.com/openvinotoolkit/datumaro/pull/159>)
- Dataset validation for classification and detection datasets (<https://github.com/openvinotoolkit/datumaro/pull/160>)
- Arbitrary image extensions in formats (import and export) (<https://github.com/openvinotoolkit/datumaro/issues/166>)
- Ability to set a custom subset name for an imported dataset (<https://github.com/openvinotoolkit/datumaro/issues/166>)
- CLI support for NDR(<https://github.com/openvinotoolkit/datumaro/pull/178>)

### Enhancements
- Common ICDAR format is split into 3 sub-formats (<https://github.com/openvinotoolkit/datumaro/pull/174>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- The ability to work with file names containing Cyrillic and spaces (<https://github.com/openvinotoolkit/datumaro/pull/148>)
- Image reading and saving in ICDAR formats (<https://github.com/openvinotoolkit/datumaro/pull/174>)
- Unnecessary image loading on dataset saving (<https://github.com/openvinotoolkit/datumaro/pull/176>)
- Allowed spaces in ICDAR captions (<https://github.com/openvinotoolkit/datumaro/pull/182>)
- Saving of masks in VOC when masks are not requested (<https://github.com/openvinotoolkit/datumaro/pull/184>)

### Security
- TBD

## 03/02/2021 - Release v0.1.6.1 (hotfix)
### New features
- TBD

### Enhancements
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Images with no annotations are exported again in VOC formats (<https://github.com/openvinotoolkit/datumaro/pull/123>)
- Inference result for only one output layer in OpenVINO launcher (<https://github.com/openvinotoolkit/datumaro/pull/125>)

### Security
- TBD

## 02/26/2021 - Release v0.1.6
### New features
- `Icdar13/15` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/96>)
- Laziness, source caching, tracking of changes and partial updating for `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/102>)
- `Market-1501` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/108>)
- `LFW` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/110>)
- Support of polygons' and masks' confusion matrices and mismathing classes in
  `diff` command (<https://github.com/openvinotoolkit/datumaro/pull/117>)
- Add near duplicate image removal plugin (<https://github.com/openvinotoolkit/datumaro/pull/113>)
- Sampler Plugin that analyzes inference result from the given dataset and
  selects samples for annotation(<https://github.com/openvinotoolkit/datumaro/pull/115>)

### Enhancements
- OpenVINO model launcher is updated for OpenVINO r2021.1 (<https://github.com/openvinotoolkit/datumaro/pull/100>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- High memory consumption and low performance of mask import/export, #53 (<https://github.com/openvinotoolkit/datumaro/pull/101>)
- Masks, covered by class 0 (background), should be exported with holes inside
(<https://github.com/openvinotoolkit/datumaro/pull/104>)
- `diff` command invocation problem with missing class methods (<https://github.com/openvinotoolkit/datumaro/pull/117>)

### Security
- TBD

## 01/23/2021 - Release v0.1.5
### New features
- `WiderFace` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/65>, <https://github.com/openvinotoolkit/datumaro/pull/90>)
- Function to transform annotations to labels (<https://github.com/openvinotoolkit/datumaro/pull/66>)
- Dataset splits for classification, detection and re-id tasks (<https://github.com/openvinotoolkit/datumaro/pull/68>, <https://github.com/openvinotoolkit/datumaro/pull/81>)
- `VGGFace2` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/69>, <https://github.com/openvinotoolkit/datumaro/pull/82>)
- Unique image count statistic (<https://github.com/openvinotoolkit/datumaro/pull/87>)
- Installation with pip by name `datumaro`

### Enhancements
- `Dataset` class extended with new operations: `save`, `load`, `export`, `import_from`, `detect`, `run_model` (<https://github.com/openvinotoolkit/datumaro/pull/71>)
- Allowed importing `Extractor`-only defined formats
  (in `Project.import_from`, `dataset.import_from` and CLI/`project import`) (<https://github.com/openvinotoolkit/datumaro/pull/71>)
- `datum project ...` commands replaced with `datum ...` commands (<https://github.com/openvinotoolkit/datumaro/pull/84>)
- Supported more image formats in `ImageNet` extractors (<https://github.com/openvinotoolkit/datumaro/pull/85>)
- Allowed adding `Importer`-defined formats as project sources (`source add`) (<https://github.com/openvinotoolkit/datumaro/pull/86>)
- Added max search depth in `ImageDir` format and importers (<https://github.com/openvinotoolkit/datumaro/pull/86>)

### Deprecated
- `datum project ...` CLI context (<https://github.com/openvinotoolkit/datumaro/pull/84>)

### Removed
- TBD

### Bug fixes
- Allow plugins inherited from `Extractor` (instead of only `SourceExtractor`)
  (<https://github.com/openvinotoolkit/datumaro/pull/70>)
- Windows installation with `pip` for `pycocotools` (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- `YOLO` extractor path matching on Windows (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- Fixed inplace file copying when saving images (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed `labelmap` parameter type checking in `VOC` converter (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed model copying on addition in CLI (<https://github.com/openvinotoolkit/datumaro/pull/94>)

### Security
- TBD

## 12/10/2020 - Release v0.1.4
### New features
- `CamVid` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/57>)
- Ability to install `opencv-python-headless` dependency with `DATUMARO_HEADLESS=1`
  environment variable instead of `opencv-python` (<https://github.com/openvinotoolkit/datumaro/pull/62>)

### Enhancements
- Allow empty supercategory in COCO (<https://github.com/openvinotoolkit/datumaro/pull/54>)
- Allow Pascal VOC to search in subdirectories (<https://github.com/openvinotoolkit/datumaro/pull/50>)

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- TBD

### Security
- TBD

## 10/28/2020 - Release v0.1.3
### New features
- `ImageNet` and `ImageNetTxt` dataset formats (<https://github.com/openvinotoolkit/datumaro/pull/41>)

### Enhancements
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- Default `label-map` parameter value for VOC converter (<https://github.com/openvinotoolkit/datumaro/pull/34>)
- Randomness of random split transform (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- `Transform.subsets()` method (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- Supported unknown image formats in TF Detection API converter (<https://github.com/openvinotoolkit/datumaro/pull/40>)
- Supported empty attribute values in CVAT extractor (<https://github.com/openvinotoolkit/datumaro/pull/45>)

### Security
- TBD

## 10/05/2020 - Release v0.1.2
### New features
- `ByteImage` class to represent encoded images in memory and avoid recoding
  on save (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Enhancements
- Implementation of format plugins simplified (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- `default` is now a default subset name, instead of `None`. The values are
  interchangeable. (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- Improved performance of transforms (<https://github.com/openvinotoolkit/datumaro/pull/22>)

### Deprecated
- TBD

### Removed
- `image/depth` value from VOC export (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Bug fixes
- Zero division errors in dataset statistics (<https://github.com/openvinotoolkit/datumaro/pull/31>)

### Security
- TBD

## 09/24/2020 - Release v0.1.1
### New features
- `reindex` option in COCO and CVAT converters (<https://github.com/openvinotoolkit/datumaro/pull/18>)
- Support for relative paths in LabelMe format (<https://github.com/openvinotoolkit/datumaro/pull/19>)
- MOTS png mask format support (<https://github.com/openvinotoolkit/datumaro/21>)

### Enhancements
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- TBD

### Security
- TBD

## 09/10/2020 - Release v0.1.0
### New features
- Initial release

## Template
```
## [Unreleased]
### New features
- TBD

### Enhancements
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Bug fixes
- TBD

### Security
- TBD
```
