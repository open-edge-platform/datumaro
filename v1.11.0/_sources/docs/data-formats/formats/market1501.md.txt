# Market-1501

## Format specification

Market-1501 is a dataset for person re-identification task, link
for downloading this dataset is available
[here](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).

Supported items attributes:
- `person_id` (str): four-digit number that represent ID of pedestrian;
- `camera_id` (int): one-digit number that represent ID of camera that took
  the image (original dataset has totally 6 cameras);
- `track_id` (int): one-digit number that represent ID of the track with
  the particular pedestrian, this attribute matches with `sequence_id`
  in the original dataset;
- `frame_id` (int): six-digit number, that mean number of
  frame within this track. For the tracks, their names are accumulated
  for each ID, but for frames, they start from "0001" in each track;
- `bbox_id` (int): two-digit number, that mean number of
  bounding bbox that was selected for that image
  (see the
  [original docs](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
  for more info).

These item attributes decodes into the image name with such convention:
```
0000_c1s1_000000_00.jpg
```
- first four digits indicate the `person_id`;
- digit after `c` indicates the `camera_id`;
- digit after `s` indicate the `track_id`;
- six digits after `s1_` indicate the `frame_id`;
- the last two digits before `.jpg` indicate the `bbox_id`.

## Import Market-1501 dataset

Importing of Market-1501 dataset into the Datumaro project:
```bash
datum project create
datum project import -f market1501 <path_to_market1501>
```
See more information about adding datasets to the project in the
[docs](../../command-reference/context/sources.md#add-dataset).

Or you can import Market-1501 using Python API:

```python
import datumaro as dm
dataset = dm.Dataset.import_from('<path_to_dataset>', 'market1501')
```

For successful importing the Market-1501 dataset, the directory with it
should has the following structure:

```
market1501_dataset/
├── query # optional directory with query image
│   ├── 0001_c1s1_001051_00.jpg
│   ├── 0002_c1s1_001051_00.jpg
│   ├── ...
├── bounding_box_<subset_name1>
│   ├── 0003_c1s1_001051_00.jpg
│   ├── 0003_c2s1_001054_01.jpg
│   ├── 0004_c1s1_001051_00.jpg
│   ├── ...
├── bounding_box_<subset_name2>
│   ├── 0005_c1s1_001051_00.jpg
│   ├── 0006_c1s1_001051_00.jpg
│   ├── ...
├── ...
```

## Export dataset to the Market-1501 format

With Datumaro you can export dataset, that has `person_id` item attribute,
to the Market-1501 format, example:

```bash
# Converting MARS dataset into the Market-1501
datum convert -if mars -i ./mars_dataset \
    -f market1501 -o ./output_dir
```
``` bash
# Export dataaset to the Market-1501 format through the Datumaro project:
datum project create
datum project add -f mars ../mars
datum project export -f market1501 -o ./output_dir -- --save-media --image-ext png
```

> Note: if your dataset contains only person_id attributes Datumaro
> will assign default values for other attributes (camera_id, track_id, bbox_id)
> and increment frame_id for collisions.

Available extra export options for Market-1501 dataset format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original)
