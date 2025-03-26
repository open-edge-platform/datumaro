# YOLO-Ultralytics

## Format specification

The YOLO-Ultralytics dataset format is used for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), developed by [Ultralytics](https://ultralytics.com/). An example for this format is available [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml). This format shares the same annotation bounding box text file format with [YOLO](./yolo.md#bounding-box-annotation-text-file). However, it requires a YAML meta file where `train`, `val`, and `test` (optional) subsets are specified.

Supported annotation types:
- `Bounding boxes`

YOLO-Ultralytics format doesn't support attributes for annotations.

The format only supports three subset names: `train`, `val`, and `test` (optional).

> Note, the YOLO-Ultralytics trainer does not expect any subset names,
  except `train`, `val`, and `test` (optional). If there is any other subset name in your project,
  Datumaro raises an exception when you export the dataset to the YOLO-Ultralytics format.

## Import YOLO dataset

A Datumaro project with a YOLO source can be created in the following way:

```bash
datum project create
datum project import --format yolo <path/to/dataset>
```

### Directory structure

YOLO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
yolo-ultralytics/
├── dataset_meta.json   # a list of non-format labels (optional)
├── data.yaml           # YAML meta file (required)
├── train.txt           # Train image file list (required)
├── val.txt             # Validation image file list (required)
├── test.txt            # Test image file list (optional)
├── images              # Image directory
│   ├── train           # (required)
│   │   ├── img1.jpg    # Image file
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val             # (required)
│   └── test            # (optional)
└── labels              # Label directory
    ├── train           # (required)
    │   ├── img1.txt    # Bounding box label file (Its name must be paired with the image)
    │   ├── img2.txt
    │   └── ...
    ├── val             # (required)
    └── test            # (optional)
```

#### Meta file

- `data.yaml` should have the following content:
```yaml
test: test.txt # (optional)
train: train.txt
val: val.txt
names:
  0: <label_name_1>
  1: <label_name_1>
  ...
```

#### Subset files

- Files `train.txt`, `val.txt`, and `test.txt` (optional) should have the following structure:
```
./images/<subset-name>/<image-file-name-1.jpg>
./images/<subset-name>/<image-file-name-2.jpg>
...
```

#### Bounding box annotation text file

- Files in directories `labels/<subset-name>` should contain information about labeled bounding boxes
for images:
```
# image1.txt:
# <label_index> <x_center> <y_center> <width> <height>
0 0.250000 0.400000 0.300000 0.400000
3 0.600000 0.400000 0.400000 0.266667
```
Here `x_center`, `y_center`, `width`, and `height` are relative to the image's
width and height. The `x_center` and `y_center` are center of rectangle
(are not top-left corner).

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

## Export to YOLO-Ultralytics format

Datumaro can convert [any other image dataset format](/docs/data-formats/formats/index.rst) which has bounding box annotations into YOLO-Ultralytics format.
After the successful conversion, you can train your own detector with the exported dataset and  [Ultralytics YOLOv8 trainer](https://github.com/ultralytics/ultralytics).

> Note, if you want to see the end-to-end Jupyter-notebook example from the dataset conversion to the training, please see this [link](https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/08_e2e_example_yolo_ultralytics_trainer.ipynb).

There are several ways to convert other dataset formats to the YOLO-Ultralytics format:

```bash
datum project create
datum project add -f <any-other-dataset-format> <path/to/dataset/>
datum project export -f yolo_ultralytics -o <output/dir> -- --save-media
```
or
```bash
datum convert -if <any-other-dataset-format> -i <path/to/dataset> \
              -f yolo_ultralytics -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', '<any-other-dataset-format>')
dataset.export('save_dir', 'yolo_ultralytics', save_media=True)
```

> Note, we recommend you to turn on `--save-media` (CLI) or `save_media=True` (Python API) option. This is because without this option, you would have to manually copy and paste the image into the appropriate location in the exported dataset directory. Enabling this option will save your manual effort.
