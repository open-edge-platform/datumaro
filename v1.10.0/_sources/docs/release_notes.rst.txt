Release Notes
#############

.. toctree::
   :maxdepth: 1

v1.10.0 (2024 Q4)

New features
^^^^^^^^^^^^
- Support KITTI 3D format
- Add PseudoLabeling transform for unlabeled dataset

Enhancements
^^^^^^^^^^^^
- Raise an appropriate error when exporting a datumaro dataset if its subset name contains path separators.
- Update docs for transform plugins
- Update ov ir model for explorer openvino launcher with CLIP ViT-L/14@336px model
- Optimize path assignment to handle point cloud in JSON without images
- Set TabularTransform to process clean transform in parallel

Bug fixes
^^^^^^^^^
- Fix datumaro format to load visibility information from Points annotations

v1.9.1 (2024 Q3)
----------------

Enhancements
^^^^^^^^^^^^
- Support multiple labels for kaggle format
- Use DataFrame.map instead of DataFrame.applymap

Bug fixes
^^^^^^^^^
- Fix StreamDataset merging when importing in eager mode

v1.9.0 (2024 Q3)
----------------

New features
^^^^^^^^^^^^
- Add a new CLI command: datum format
- Support language dataset for DmTorchDataset

Enhancements
^^^^^^^^^^^^
- Change _Shape to Shape and add comments for subclasses of Shape

Bug fixes
^^^^^^^^^
- Fix KITTI-3D importer and exporter

v1.8.0 (2024 Q3)
----------------

New features
^^^^^^^^^^^^
- Add TabularValidator
- Add Clean Transform for tabular data type

Enhancements
^^^^^^^^^^^^
- Set label name with parents to avoid duplicates for AstypeAnnotations
- Pass Keyword Argument to TabularDataBase

Bug fixes
^^^^^^^^^
- Preserve end_frame information of a video when it is zero.
- Changed the Datumaro format to ensure exported videos have relative paths and to prevent the same video from being overwritten.

v1.7.0 (2024 Q2)
----------------

New features
^^^^^^^^^^^^
- Add ann_types property for dataset
- Add AnnotationType.rotated_bbox for oriented object detection
- Add DOTA data format for oriented object detection task
- Add AstypeAnnotations Transform

Enhancements
^^^^^^^^^^^^
- Fix ambiguous COCO format detector
- Get target information for tabular dataset
- Add ExtractedMask and update importers who can use it to use it

v1.6.1 (2024.05)
----------------

Enhancements
^^^^^^^^^^^^
- Prevent AcLauncher for OpenVINO 2024.0

Bug fixes
^^^^^^^^^
- Modify lxml dependency constraint
- Fix CLI error occurring when installed with default option only
- Relax Pillow dependency constraint
- Modify Numpy dependency constraint
- Relax old pandas version constraint

v1.6.0 (2024.04)
----------------

New features
^^^^^^^^^^^^
- Changed supported Python version range (>=3.9, <=3.11)
- Support MMDetection COCO format
- Develop JsonSectionPageMapper in Rust API
- Add Filtering via User-Provided Python Functions
- Remove supporting MacOS platform
- Support Kaggle image data (`KaggleImageCsvBase`, `KaggleImageTxtBase`, `KaggleImageMaskBase`, `KaggleVocBase`, `KaggleYoloBase`)
- Add `__getitem__()` for random accessing with O(1) time complexity
- Add Data-aware Anchor Generator
- Support bounding box import within Kaggle extractors and add `KaggleCocoBase`

Enhancements
^^^^^^^^^^^^
- Optimize Python import to make CLI entrypoint faster
- Add ImageColorScale context manager
- Enhance visualizer to toggle plot title visibility
- Enhance Datumaro data format detect() to be memory-bounded and performant
- Change RoIImage and MosaicImage to have np.uint8 dtype as default
- Enable image backend and color channel format to be selectable
- Boost up `CityscapesBase` and `KaggleImageMaskBase` by dropping `np.unique`
- Enhance RISE algortihm for explainable AI
- Enhance explore unit test to use real dataset from ImageNet
- Fix each method of the comparator to be used separately

Bug fixes
^^^^^^^^^
- Fix wrong example of Datumaro dataset creation in document
- Fix wrong command to install datumaro from github
- Update document to correct wrong `datum project import` command and add filtering example to filter out items containing annotations.
- Fix label compare of distance method
- Fix Datumaro visualizer's import errors after introducing lazy import
- Fix broken link to supported formats in readme
- Fix Kinetics data format to have media data
- Handling undefined labels at the annotation statistics
- Add unit test for item rename
- Fix a bug in the previous behavior when importing nested datasets in the project
- Fix Kaggle importer when adding duplicated labels
- Fix input tensor shape in model interpreter for OpenVINO 2023.3
- Add default value for target in prune cli
- Remove deprecated MediaManager
- Fix explore command without project

v1.5.2 (2024.01)
----------------

Enhancements
^^^^^^^^^^^^

- Add memory bounded datumaro data format detect
- Remove Protobuf version limitation (<4)

v1.5.1 (2023.11)
----------------

Enhancements
^^^^^^^^^^^^
- Enhance Datumaro data format stream importer performance
- Change image default dtype from float32 to uint8
- Add comparison level-up doc
- Add ImportError to catch GitPython import error

Bug fixes
^^^^^^^^^
- Modify the draw function in the visualizer not to raise an error for unsupported annotation types.
- Correct explore path in the related document.
- Fix errata in the voc document. Color values in the labelmap.txt should be separated by commas, not colons.
- Fix hyperlink errors in the document.
- Fix memory unbounded Arrow data format export/import.
- Update CVAT format doc to bypass warning.

v1.5.0 (2023.09)
----------------

New features
^^^^^^^^^^^^
- Add tabular data import/export
- Support video annotation import/export
- Add multiframework (PyTorch, Tensorflow) converter
- Add SAM OVMS and Triton server Docker image builders
- Add SAMBboxToInstanceMask transform
- Add ConfigurableValidator

Enhancements
^^^^^^^^^^^^
- Enhance `ClassificationValidator` for multi-label classification datasets with `label_groups`
- Replace Roboflow `xml.etree` with `defusedxml`
- Define `GroupType` with `IntEnum` for, where `0` is `EXCLUSIVE`
- Add Rust API to optimize COCOPageMapper performance
- Support a dictionary input in addition to a single image input for the model launcher to support Segment Anything Model
- Remove deprecates announced to be removed in 1.5.0
- Add multi-threading option to ModelTransform and SAMBboxToInstanceMask

Bug fixes
^^^^^^^^^
- Fix bugs for Tile transform
- Disable Roboflow Tfrecord format when Tensorflow is not installed
- Raise VcsAlreadyExists error if vcs directory exists

v1.4.1 (2023.07)
----------------

Bug fixes
^^^^^^^^^
- Report errors for COCO (stream) and Datumaro importers

v1.4.0 (2023.07)
----------------

New features
^^^^^^^^^^^^
- Add documentation and notebook example for Prune API
- Changed supported Python version range (>=3.8, <=3.11)
- Migrate OpenVINO v2023.0.0
- Add Roboflow data format support (COCO JSON, Pascal VOC XML, YOLOv5-PyTorch, YOLOv7-PyTorch, YOLOv8, YOLOv5 Oriented Bounding Boxes, Multiclass CSV, TFRecord, CreateML JSON)
- Add MissingAnnotationDetection transform
- Add OVMSLauncher
- Add Prune API
- Add TritonLauncher
- Migrate DVC v3.0.0
- Stream dataset import/export
- Support mask annotations for CVAT data format

Enhancements
^^^^^^^^^^^^
- Support list query for explorer
- update contributing.md
- Update 3rd-party.txt for release 1.4.0
- Give notice that the deprecation works will be done in datumaro==1.5.0
- Unify COCO, Datumaro, VOC, YOLO importer/exporter progress reporter descriptions
- Enhance import performance for built-in plugins
- Change default dtype of load_image() to np.uint8
- Add OTX ATSS detector model interpreter & refactor interfaces
- Refactor Launcher and ModelInterpreter
- Add CVAT data format document
- Reduce peak memory usage when importing COCO and Datumaro formats
- Enhance the error message for datum stats to be more user friendly
- Refactor dataset.py to seperate DatasetStorage

Bug fixes
^^^^^^^^^
- Create cache dir under only writable filesystem
- Fix: Dataset infos() can be broken if a transform not redefining infos() is stacked on the top
- Fix warnings in test_visualizer.py
- Fix LabelMe data format
- Prevent installing protobuf>=4
- Fix UnionMerge

v1.3.2 (2023.06)
----------------

Enhancements
^^^^^^^^^^^^
- Let CocoBase continue even if an InvalidAnnotationError is raised

Bug fixes
^^^^^^^^^
- Install dvc version to 2.x
- Replace np.append() in Validator

v1.3.1 (2023.05)
----------------

Bug fixes
^^^^^^^^^
- Fix `Cityscapes` format mis-detection problem

v1.3.0 (2023.05)
----------------

New features
^^^^^^^^^^^^
- Add `CocoRoboflowImporter`
- Add `SynthiaSfImporter` and `SynthiaAlImporter`
- Add intermediate skill document for filter
- Add `VocInstanceSegmentationImporter` and `VocInstanceSegmentationExporter`
- Add `Segment Anything` data format support
- Add `Correct` transformation
- Add `ReindexAnnotations` transform

Enhancements
^^^^^^^^^^^^
- Use autosummary for fully-automatic Python module docs generation
- Enrich stack trace for better user experience when importing
- Save and load `hashkey` for explorer
- Add `MOT` and `MOTS` data format documents
- Improve `RemoveAnnotations` to remove specific annotations with ids
- Add Jupyter notebook example of noisy label detection for detection tasks
- Add Juypter notebook examples for importing/exporting detection and segmentation data

Bug fixes
^^^^^^^^^
- Fix `Mapillary Vistas` data format
- Fix `bytes` property returning None if function is given to data
- Fix `Synthia-Rand` data format
- Fix `person_layout` categories and `action_classification` attributes in imported Pascal-VOC dataset
- Drop a malformed transform from `StackedTransform` automatically
- Fix `Cityscapes` to drop `ImgsFine` directory

v1.2.1 (2023.05)
----------------

Bug fixes
^^^^^^^^^
- Fix project level CVAT for images format import
- Fix an info message when using the convert CLI command with no args.input_format
- Fix media contents not returning bytes in arrow format

v1.2.0 (2023.04)
----------------

New features
^^^^^^^^^^^^
- Add Skill Up section to documentation
- Add LossDynamicsAnalyzer for noisy label detection
- Add Apache Arrow format support
- Add sort transform

Enhancements
^^^^^^^^^^^^
- Add multiprocessing to DatumaroBinaryBase
- Refactor merge code
- Refactor download CLI commands
- Refactor CLI commands w/ and w/o project
- Refactor Media to be initialized from explicit sources
- Refactor hl_ops.py
- Add tfds:uc_merced and tfds:eurosat download
- Migrate documentation framework to Sphinx
- Update merge tutorial for real life usecase
- Abbreviate "detect-format" to "detect" for prettifying

Bug fixes
^^^^^^^^^
- Add UserWarning if an invalid media_type comes to image statistics computation
- Fix negated `is_encrypted`
- Save extra images of PointCloud when exporting to datumaro format
- Fix log issue when importing celeba and align celeba dataset

v1.1.0 (2023.03)
----------------

New features
^^^^^^^^^^^^
- Add with_subset_dirs decorator (Add ImagenetWithSubsetDirsImporter)
- Add CommonSemanticSegmentationWithSubsetDirsImporter
- Add DatumaroBinary format
- Add Searcher CLI documentation
- Add version to dataset exported as datumaro format
- Add Ava action data format support
- Add Shift Analyzer (both covariate and label shifts)
- Add YOLO Loose format
- Add Ultralytics YOLO format

Enhancements
^^^^^^^^^^^^
- Refactor Datumaro format code and test code

Bug fixes
^^^^^^^^^
- Fix image filenames and anomaly mask appearance in MVTec exporter
- Fix CIFAR10 and 100 detect function
- Fix celeba and align_celeba detect function
- Choose the top priority detect format for all directory depths
- Fix MVTec format detect function
- Fix wrong `__len__()` of Subset when the item is removed
- Fix mask visualization bug

v1.0.0 (2023.02)
----------------

New features
^^^^^^^^^^^^
- Add Data Explorer
- Add Ellipse annotation type
- Add MVTec anomaly data support

Enhancements
^^^^^^^^^^^^
- Refactor existing tests
- Raise ImportError on importing malformed COCO directory
- Remove the duplicated and cyclical category context in documentation

Bug fixes
^^^^^^^^^
- Fix for importing CVAT image 1.1 data format exported to project level
- Fix a problem on setting log-level via CLI
- Fix code format with the latest black==23.1.0
- Fix 'Explain command cannot find the model'
- Fix a problem found on model remove CLI command

.. note::
   About the release of the developed version can be read in the `CHANGELOG.md <https://github.com/openvinotoolkit/datumaro/blob/develop/CHANGELOG.md>`_ of the develop branch.
