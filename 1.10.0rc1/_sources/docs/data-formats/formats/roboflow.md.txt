# Roboflow
## Format specification
[Roboflow](https://universe.roboflow.com/) provides a range of services and tools to assist with various aspects of computer vision and machine learning projects.
These services are aimed at simplifying the process of data importing, annotating, and training models for tasks like image classification, object detection, segmentation, and more.
Datumaro supports various [Roboflow formats](https://roboflow.com/formats) so that can make it easier for users to import and work with datasets that have been prepared and annotated using Roboflow tools.
You can enjoy some examples [here](https://universe.roboflow.com/joseph-nelson/bccd/dataset/4).

Supported annotation formats:
- `COCO JSON`
- `VOC XML`
- `YOLOv5, YOLOv6, YOLOv7, YOLOv8 TXT`
- `TFRecord`
- `CreateML JSON`
- `YOLOv5 Oriented Bounding Boxes TXT`
- `Multiclass Classification CSV`

## Import Roboflow dataset
A Datumaro project with a Roboflow source can be created in the following way:

```bash
datum project create
datum project import --format roboflow_coco <path/to/dataset>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'roboflow_coco')
```

### Roboflow COCO JSON
#### Directory structure
<!--lint disable fenced-code-flag-->
```
coco/
├── train                       # Subset directory
│   ├── _annotations.coco.json  # Annotation file
│   ├── train_img1.jpg          # Image file
│   ├── train_img2.jpg          # Image file
│   └── ...
├── valid                       # Subset directory
│   ├── _annotations.coco.json  # Annotation file
│   ├── valid_img1.jpg          # Image file
│   └── ...
└── test                        # Subset directory
│   ├── _annotations.coco.json  # Annotation file
│   ├── test_img1.jpg           # Image file
│   └── ...
```
#### Annotation JSON file
The example of `_annotations.coco.json` is given by EXAMPLE tap of [Roboflow COCO JSON](https://roboflow.com/formats/coco-json).

### Roboflow VOC XML
#### Directory structure
<!--lint disable fenced-code-flag-->
```
voc/
├── train                       # Subset directory
│   ├── train_img1.jpg          # Image file
│   ├── train_img1.xml          # Annotation file
│   ├── train_img2.jpg          # Image file
│   ├── train_img2.xml          # Annotation file
│   └── ...
├── valid                       # Subset directory
│   ├── valid_img1.jpg          # Image file
│   ├── valid_img1.xml          # Annotation file
│   └── ...
└── test                        # Subset directory
│   ├── test_img1.jpg           # Image file
│   ├── test_img1.xml           # Annotation file
│   └── ...
```
#### Annotation XML file
The example of `{*}.xml` is given by EXAMPLE tap of [Roboflow VOC XML](https://roboflow.com/formats/pascal-voc-xml).

### Roboflow YOLOv5, YOLOv7, YOLOv8 TXT
#### Directory structure
<!--lint disable fenced-code-flag-->
```
yolo/
├── data.yaml                   # YAML meta file (required)
├── train                       # Subset directory
│   ├── images                  # Image directory
│   │   ├── train_img1.jpg      # Image file
│   │   ├── train_img2.jpg      # Image file
│   │   └── ...
│   └── labels                  # Label directory
│       ├── train_img1.txt      # Annotation file
│       ├── train_img2.txt      # Annotation file
│       └── ...
├── valid                       # Subset directory
│   ├── images                  # Image directory
│   │   ├── valid_img1.jpg      # Image file
│   │   └── ...
│   └── labels                  # Label directory
│       ├── valid_img1.txt      # Annotation file
│       └── ...
└── test                        # Subset directory
    ├── images                  # Image directory
    │   ├── test_img1.jpg       # Image file
    │   └── ...
    └── labels                  # Label directory
        ├── test_img1.txt       # Annotation file
        └── ...
```
#### Annotation TXT file
The example of `{*}.txt` is given by EXAMPLE tap of [Roboflow YOLO TXT](https://roboflow.com/formats/yolov8-pytorch-txt).

### Roboflow MT-YOLOv6 TXT
#### Directory structure
<!--lint disable fenced-code-flag-->
```
mt-yolov6/
├── data.yaml                   # YAML meta file (required)
├── images                      # Image directory
│   ├── train                   # Subset directory
│   │   ├── train_img1.jpg      # Image file
│   │   ├── train_img2.jpg      # Image file
│   │   └── ...
│   ├── valid                   # Subset directory
│   │   ├── valid_img1.jpg      # Image file
│   │   └── ...
│   └── test                    # Subset directory
│       ├── test_img1.jpg       # Image file
│       └── ...
└── labels                      # Label directory
    ├── train                   # Subset directory
    │   ├── train_img1.txt      # Annotation file
    │   ├── train_img2.txt      # Annotation file
    │   └── ...
    ├── valid                   # Subset directory
    │   ├── test_img1.txt       # Annotation file
    │   └── ...
    └── test                    # Subset directory
        ├── test_img1.txt       # Annotation file
        └── ...
```
#### Annotation TXT file
The example of `{*}.txt` is given by EXAMPLE tap of [Roboflow MT-YOLOv6 TXT](https://roboflow.com/formats/mt-yolov6).

### Roboflow Tensorflow TFRecord
#### Directory structure
<!--lint disable fenced-code-flag-->
```
tfrecord/
├── train                       # Subset directory
│   ├── label_map.pbtxt         # Label map file (label names and ids)
│   └── sample.tfrecord         # Tfrecord file
├── valid                       # Subset directory
│   ├── label_map.pbtxt         # Label map file (label names and ids)
│   └── sample.tfrecord         # Tfrecord file
└── test                        # Subset directory
    ├── label_map.pbtxt         # Label map file (label names and ids)
    └── sample.tfrecord         # Tfrecord file
```

### Roboflow CreateML JSON
#### Directory structure
<!--lint disable fenced-code-flag-->
```
createml/
├── train                           # Subset directory
│   ├── _annotations.createml.json  # Annotation file
│   ├── train_img1.jpg              # Image file
│   ├── train_img2.jpg              # Image file
│   └── ...
├── valid                           # Subset directory
│   ├── _annotations.createml.json  # Annotation file
│   ├── valid_img1.jpg              # Image file
│   └── ...
└── test                            # Subset directory
│   ├── _annotations.createml.json  # Annotation file
│   ├── test_img1.jpg               # Image file
│   └── ...
```
#### Annotation JSON file
The example of `_annotations.createml.json` is given by EXAMPLE tap of [Roboflow CreateML JSON](https://roboflow.com/formats/createml-json).

### Roboflow YOLOv5 Oriented Bounding Boxes
#### Directory structure
<!--lint disable fenced-code-flag-->
```
yolov5-obb/
├── data.yaml                   # YAML meta file (required)
├── train                       # Subset directory
│   ├── images                  # Image directory
│   │   ├── train_img1.jpg      # Image file
│   │   ├── train_img2.jpg      # Image file
│   │   └── ...
│   └── labelTxt                # Label directory
│       ├── train_img1.txt      # Annotation file
│       ├── train_img2.txt      # Annotation file
│       └── ...
├── valid                       # Subset directory
│   ├── images                  # Image directory
│   │   ├── valid_img1.jpg      # Image file
│   │   └── ...
│   └── labelTxt                # Label directory
│       ├── valid_img1.txt      # Annotation file
│       └── ...
└── test                        # Subset directory
    ├── images                  # Image directory
    │   ├── test_img1.jpg       # Image file
    │   └── ...
    └── labelTxt                # Label directory
        ├── test_img1.txt       # Annotation file
        └── ...
```
#### Annotation TXT file
The example of `{*}.txt` is given by EXAMPLE tap of [Roboflow YOLOv5-OBB TXT](https://roboflow.com/formats/yolov5-obb).

### Roboflow Multiclass Classification CSV
#### Directory structure
<!--lint disable fenced-code-flag-->
```
multiclass/
├── data.yaml                   # YAML meta file (required)
├── train                       # Subset directory
│   ├── _classes.csv            # Annotation file
│   ├── train_img1.jpg          # Image file
│   ├── train_img2.jpg          # Image file
│   └── ...
├── valid                       # Subset directory
│   ├── _classes.csv            # Annotation file
│   ├── valid_img1.jpg          # Image file
│   └── ...
└── test                        # Subset directory
    ├── _classes.csv            # Annotation file
    ├── test_img1.jpg           # Image file
    └── ...
```
#### Annotation CSV file
The example of `_classes.csv` is given by EXAMPLE tap of [Roboflow Multiclass CSV](https://roboflow.com/formats/multiclass-classification-csv).
