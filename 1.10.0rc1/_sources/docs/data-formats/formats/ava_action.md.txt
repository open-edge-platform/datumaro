# AVA Action

## Format specification

The AVA action format specification is available
[here](https://arxiv.org/pdf/1705.08421.pdf).

The dataset has annotations for recognizing an action per instance from video frames
like visual tracking task. Specifically, the AVA action dataset contains frame indices,
bounding box cooridnates, actions, and tracking ids in the annotation file. The action
categories are described in `ava_action_list_v2.2.pbtxt`. For the ease use for object
detection, the AVA action dataset provides the bounding box proposals from `Faster R-CNN`.

Supported task / format:
- Object detection - `ava`

Supported annotation types:
- `Bbox` (detection)

## Import AVA action dataset

The AVA action dataset is available for free download
[here](https://research.google.com/ava/download.html#ava_actions_download).

A Datumaro project with a AVA action source can be created in the following way:

``` bash
datum project create
datum project import --format ava <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

The AVA action dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset
   ├── frames/
   │   ├── video0/ # directory with list of frames extracted from video0
   │   │   ├── img1.jpg
   │   |   ├── img2.jpg
   │   |   └── ...
   │   ├── video1/ # directory with list of frames extracted from video1
   │   │   ├── img1.jpg
   │   |   ├── img2.jpg
   │   |   └── ...
   └── annotations/
       ├── ava_action_list_v2.2.pbtxt # list of action categories
       ├── ava_train_v2.2.csv # annotations for training data
       ├── ava_val_v2.2.csv # annotations for validation data
       ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl # region proposals for training data
       ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl # region proposals for validation data
       └── ...
```

To make sure that the selected dataset has been added to the project, you
can run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a AVA action dataset into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g., AVA action annotations can be converted to `COCO detection`.

There are several ways to convert a AVA action dataset to other dataset formats:

``` bash
datum project create
datum project import -f ava <path/to/ava>
datum project export -f coco -o <output/dir>
```
or
``` bash
datum convert -if ava -i <path/to/ava> -f coco -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/ava>', 'ava')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export to AVA action format

There are several ways to convert an existing dataset to AVA action format:

``` bash
# export dataset into AVA action format (detection) from existing project
datum project export -p <path/to/project> -f ava -o <output/dir> --
```
``` bash
# converting to AVA action format from other format
datum convert -if imagenet -i <path/to/dataset> \
    -f ava -o <output/dir> \
    -- \
    --save-media
```

Extra options for exporting to AVA action format:
- `--save-media` - allow to export dataset with saving media files
  (by default `False`).

```bash
datum project export -f ava -- --extra-option-1 --extra-option-2 ...
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_ava_format.py).
