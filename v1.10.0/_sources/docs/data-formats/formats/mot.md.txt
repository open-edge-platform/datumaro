# Multiple Object Tracking (MOT)

## Format specification

The [Multiple Object Tracking (MOT) challenge dataset](https://arxiv.org/pdf/1906.04567.pdf) provides bounding box tracking data for multiple objects within image sequences.

Supported annotation types:
- `Bbox` (object detection)

Supported annotation attributes:
- `track_id` (int) - Unique ID assigned to an object within a trajectory
- `visibility` (float) - Visibility ratio of each bounding box due to occlusion by another static or moving object, or due to image border cropping
- `occluded` (boolean) - `True` if `visibility < occlusion_threshold`, otherwise `False`
- `ignored` (boolean) - `True` if the confidence score of bounding box is zero, otherwise `False`
- `score` (float) - Confidence score of bounding box

## Import MOT dataset

You can download the MOT challenge dataset [here](https://motchallenge.net).

A Datumaro project with the MOT challange source can be created in the following way:

``` bash
datum project create
datum project import --format mot <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

The MOT challenge dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
  ├── gt
  │   ├── gt.txt
  │   └── labels.txt
  ├── img1
  │   ├── <name_1>.<img_ext>
  │   ├── <name_2>.<img_ext>
  │   └── ...
  └── seqinfo.ini (optional)
```

`seqinfo.ini` is provided by the MOT challange dataset but it is optional in Datumaro.
It includes `imdir` field which is the name of directory having image files.
If this file is given, Datumaro will find the image files from the directory written in the `imdir` field.

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert the MOT challange dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. object detection annotations can be
saved in `COCO instances` format, but not as `COCO keypoints`.

There are several ways to convert the MOT dataset to other dataset formats:

``` bash
datum project create
datum project import -f mot <path/to/mot>
datum project export -f coco_instances -o <output/dir>
```
or
``` bash
datum convert -if mot -i <path/to/mot> -f coco_instances -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mot')
dataset.export('save_dir', 'coco_instances', save_media=True)
```

Extra options for importing to the MOT format:
- `occlusion_threshold` determines the `occluded` boolean attribution of a bounding box.
If `visibility < occlusion_threshold`, the bounding box will have `occluded=True`, otherwise it will have `occluded=False`.
The default value is  `occlusion_threshold=0.0`.

## Export to MOT

There are several ways to convert a dataset to the MOT format:

``` bash
# export dataset into MOT format from existing project
datum project export -p <path/to/project> -f mot -o <output/dir> \
    -- --save-media
```
``` bash
# converting to MOT format from other format
datum convert -if coco_instances -i <path/to/dataset> \
    -f mot -o <output/dir> -- --save-media
```

Extra options for exporting to the MOT format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_mot_format.py)
