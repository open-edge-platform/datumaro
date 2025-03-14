# Video

## Format specification
There are many kinds of video extensions as listed up
[here](https://github.com/openvinotoolkit/datumaro/blob/develop/datumaro/plugins/data_formats/video.py).

Datumaro can import a video into video frames by adjusting the start frame, end frame,
and step size. Furthermore, with a `video_keyframes` format, Datumaro can extract
keyframes by comparing zero-mean normalized cross correlation (ZNCC) metric between
successive frames as following [here](https://www.sciencedirect.com/science/article/pii/S1047320312000223).
Plus, Datumaro provides the options for choosing image extension and name patterns
for efficient data management from multiple videos.

## Import video

A Datumaro project with a video frames can be created
in the following way:

```
datum project create
datum project import -f video_frames <path_to_video>
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
datum project import -f video_frames [-h] [-n NAME] [-r RPATH] [-p PROJECT_DIR]
  [--no-check] [-s STEP] [-b START_FRAME] [-e END_FRAME] url ...
```

Example: import a video into frames, use each 30th frame:
```console
datum project import -f video_frames video.mp4 -o video-frames --step 30
```

Example: import a video into frames, save as 'frame_xxxxxx.png' files:
```console
datum project import -f video_frames video.mp4 -o video-frames --image-ext=.png --name-pattern='frame_%%06d'
```

Example: import a video into keyframes:
```console
datum project import -f video_keyframes video.mp4 -o video-frames
```
