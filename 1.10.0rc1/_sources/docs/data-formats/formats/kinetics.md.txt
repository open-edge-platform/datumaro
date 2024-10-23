# Kinetics

## Format specification

Kinetics 400/600/700 is a video datasets for action recognition task.
Dataset is available for downloading
[here](https://www.deepmind.com/open-source/kinetics)

Supported media type:
- `Video`

Supported type of annotations:
- `Label`

Supported attributes for labels:
- `time_start` (integer) - time (in seconds) of the start of recognized action
- `time_end` (integer) - time (in seconds) of the end of recognized action

## Import Kinetics dataset

A Datumaro project with a Kinetics dataset can be created
in the following way using CLI:

```
datum project create
datum project import -f kinetics <path_to_dataset>
```

Or using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset>', format='kinetics')
```

```
├── test.csv
├── train.json
├── train
│   ├── <name_of_video_1_with_yt_id>.avi # extension of video could be other
│   ├── <name_of_video_2_with_yt_id>.avi
│   ├── ...
└── test
    ├── <name_of_video_100_with_yt_id>.avi # extension of video could be other
    ├── <name_of_video_101_with_yt_id>.avi
    ├── ...
```

Kinetics dataset has two equivalent annotation file formats: `.csv` and
`.json`. Datumaro supports both, but in case when two annotation files have
same names but different extensions Datumaro will use `.csv`.

> Note: name of each video file must contain youtube_id of this video,
> that specified in annotation file. And to speed up the import, you can leave
> only the youtube_id in the video filename.

See the full list of supported video extensions [here](/docs/data-formats/media_formats).
