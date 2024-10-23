# Multiple Object Tracking and Segmentation (MOTS)

## Format specification

The [Multiple Object Tracking and Segmentation (MOTS) challenge dataset](https://arxiv.org/pdf/1906.04567.pdf) provides a pixel-level segmentation masks for multiple objects within image sequences.
There are two format specifications according to the representation of segmentation masks: 1) PNG format and 2) TXT format.
The PNG format represents a segmentation mask as a PNG file with a 16-bits single color channel.
On the other hand, the TXT format uses run-length encoding (RLE) for a segmentation mask.
Datumaro currently only supports the PNG format, not the TXT format.

Supported annotation types:
- `Mask` (segmentation)

Supported annotation attributes:
- `track_id` (int) - Unique ID assigned to an object within a trajectory

## Import MOTS dataset

You can download the PNG format of MOTS challange dataset [here](https://www.vision.rwth-aachen.de/page/mots).

A Datumaro project with the MOTS challange source can be created in the following way:

``` bash
datum project create
datum project import --format mots <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

The MOTS challange dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
  ├── train
  │   ├── images
  │   │   ├── <name_1>.<img_ext>
  │   │   ├── <name_2>.<img_ext>
  │   │   └── ...
  │   └── instances
  │       ├── <name_1>.png
  │       ├── <name_2>.png
  │       ├── ...
  │       └── labels.txt
  └── val
      ├── images
      │   └── ...     # Same as above
      └── instances
          └── ...     # Same as above
```

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert the MOTS challange dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. segmentation annotations can be
saved in `Cityscapes` format, but not as `COCO keypoints`.

There are several ways to convert a MOTS dataset to other dataset formats:

``` bash
datum project create
datum project import -f mots <path/to/mots>
datum project export -f coco_instances -o <output/dir>
```
or
``` bash
datum convert -if mots -i <path/to/mots> -f coco_instances -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'mots')
dataset.export('save_dir', 'cityscapes', save_media=True)
```

## Export to MOTS

There are several ways to convert a dataset to MOTS format:

``` bash
# export dataset into MOTS format from existing project
datum project export -p <path/to/project> -f mots -o <output/dir> \
    -- --save-media
```
``` bash
# converting to MOTS format from other format
datum convert -if cityscapes -i <path/to/dataset> \
    -f mots -o <output/dir> -- --save-media
```

Extra options for exporting to MOTS format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_mots_format.py)
