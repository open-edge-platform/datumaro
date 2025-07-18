# DOTA
## Format specification
[DOTA](https://captain-whu.github.io/DOTA/dataset.html) is a collection of 2K aerial images for a rotated object detection task.
Each objects are annotated with 4 coordinates for representing oriented bounding boxes, a label among 15 classes (baseball-diamond, basketball-court, bridge, ground-track-field, harbor, helicopter, large-vehicle, plane, roundabout, ship, small-vehicle, soccer-ball-field, storage-tank, swimming-pool, tennis-court) and a difficulty.

## Import DOTA dataset
A Datumaro project with a DOTA source can be created in the following way:

```bash
datum project create
datum project import --format dota <path/to/dataset>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'dota')
```

## Directory structure
<!--lint disable fenced-code-flag-->
```
dota/
├── train                     # Subset directory
│   ├── images
│   │   ├── img1.jpg          # Image file
│   │   ├── img2.jpg          # Image file
│   │   └── ...
│   ├── labelTxt
│   │   ├── img1.txt          # Annotation file
│   │   ├── img2.txt          # Annotation file
│   │   └── ...
├── val                       # Subset directory
│   ├── images
│   │   ├── img3.jpg          # Image file
│   │   ├── img4.jpg          # Image file
│   │   └── ...
│   ├── labelTxt
│   │   ├── img3.txt          # Annotation file
│   │   ├── img4.txt          # Annotation file
│   │   └── ...
└── ...
```
## Annotation Txt file
The example of `<image_id>.txt` is given by [DOTA annotation format](https://captain-whu.github.io/DOTA/dataset.html).
