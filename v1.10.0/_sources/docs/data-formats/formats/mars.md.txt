# MARS

## Format specification

MARS is a dataset for the motion analysis and person identification task.
MARS dataset is available for downloading
[here](https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html)

Supported types of annotations:
- `Bbox`

Required attributes:
- `person_id` (str): four-digit number that represent ID of pedestrian;
- `camera_id` (int): one-digit number that represent ID of camera that took
  the image (original dataset has totally 6 cameras);
- `track_id` (int): four-digit number that represent ID of the track with
  the particular pedestrian;
- `frame_id` (int): three-digit number, that mean number of
  frame within this track. For the tracks, their names are accumulated
  for each ID, but for frames, they start from "0001" in each track.

## Import MARS dataset

Use these instructions to import MARS dataset into Datumaro project:

```bash
datum project create
datum project add -f mars ./dataset
```

> Note: the directory with dataset should be subdirectory of the
> project directory.

```
mars_dataset
├── <bbox_subset_name1>
│   ├── 0001 # directory with images of pedestrian with id 0001
│   │   ├── 0001C1T0001F001.jpg
│   │   ├── 0001C1T0001F002.jpg
│   │   ├── ...
│   ├── 0002 # directory with images of pedestrian with id 0002
│   │   ├── 0002C1T0001F001.jpg
│   │   ├── 0002C1T0001F001.jpg
│   │   ├── ...
│   ├── 0000 # distractors images, which negatively affect retrieval accuracy.
│   │   ├── 0000C1T0001F001.jpg
│   │   ├── 0000C1T0001F001.jpg
│   │   ├── ...
│   ├── 00-1 # junk images which do not affect retrieval accuracy
│   │   ├── 00-1C1T0001F001.jpg
│   │   ├── 00-1C1T0001F001.jpg
│   │   ├── ...
├── <bbox_subset_name2>
│   ├── ...
├── ...
```

All images in MARS dataset follow a strict convention of naming:
```
xxxxCxTxxxxFxxx.jpg
```
- the first four digits indicate the pedestrian's number;
- digit after `C` indicates the camera id;
- four digits after `T` indicate the track id for this pedestrian;
- three digits after `F` indicate the frame id with this track.

> Note: there are two specific pedestrian IDs 0000 and 00-1
> which indicate distracting images and unwanted images respectively.
