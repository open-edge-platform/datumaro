# Video

## Format specification
There are many kinds of video extensions as listed up
[here](https://github.com/open-edge-platform/datumaro/blob/develop/src/datumaro/plugins/data_formats/video.py).

Datumaro can import a video into video frames by adjusting the start frame, end frame,
and step size. Furthermore, with a `video_keyframes` format, Datumaro can extract
keyframes by comparing zero-mean normalized cross correlation (ZNCC) metric between
successive frames as following [here](https://www.sciencedirect.com/science/article/pii/S1047320312000223).
Plus, Datumaro provides the options for choosing image extension and name patterns
for efficient data management from multiple videos.

## Convert video

A Datumaro dataset can be converted from video in the following way:

```
datum convert -if video_frames -i <path_to_video> -o <output/dir>
```

Load video through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_video>', format='video_frames')
```

Datumaro has few import options for `video_frames` format, to apply them
use the `--` after the main command argument.
Note that a video has a closed interval of [`start-frame`, `end-frame`].

`video_frames` import options:
- `--subset` (string) - The name of the subset for the produced
  dataset items (default: none)
- `-p, --name-pattern` (string) - Name pattern for the produced
  images (default: `%06d`)
- `-s, --step` (integer) - Frame step (default: 1)
- `-b, --start-frame` (integer) - Starting frame (default: 0)
- `-e, --end-frame` (integer) - Finishing frame (default: none)
- `-h, --help` - Print the help message and exit

Usage:

``` console
datum convert -if video_frames [-h] [-i INPUT] [-o OUTPUT]
  [--step STEP] [--start-frame START_FRAME] [--end-frame END_FRAME]
```

Example: convert a video into frames, use each 30th frame:
```console
datum convert -if video_frames -i video.mp4 -o video-frames -- --step 30
```

Example: convert a video into frames, save as 'frame_xxxxxx.png' files:
```console
datum convert -if video_frames -i video.mp4 -o video-frames -- --image-ext=.png --name-pattern='frame_%%06d'
```

Example: convert a video into keyframes:
```console
datum convert -if video_keyframes -i video.mp4 -o video-frames
```
