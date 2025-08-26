# Util

## Utilities

### Split video into frames

Splits a video into separate frames and saves them in a directory.
After the splitting, the images can be converted to various dataset formats using
the [`convert` command](convert.md) and the `image_dir` format.

This command is useful for making a dataset from a video file.
Unlike direct video reading during model training, which can produce
different results if the system environment changes, this command
allows to split the video into frames and use them instead, making
the dataset reproducible and stable.

This command provides different options like setting the frame step
(the `-s/--step` option), file name pattern (`-n/--name-pattern`),
starting (`-b/--start-frame`) and finishing (`-e/--end-frame`) frame etc.
Note that starting and finishing frames denote a closed interval [`start-frame`, `end-frame`].

Note that this command is equivalent to the following workflow:
```bash
# Split video, then convert to desired format
datumaro util split_video video.mp4 -o frames_dir -- <params>
datumaro convert -i frames_dir -if image_dir -f <desired_format> -o <output_dir>
```

Usage:

``` bash
datum util split_video [-h] -i SRC_PATH [-o DST_DIR] [--overwrite]
  [-n NAME_PATTERN] [-s STEP] [-b START_FRAME] [-e END_FRAME] [-x IMAGE_EXT]
```

Parameters:
- `-i, --input-path` (string) - Path to the video file
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty
- `-n, --name-pattern` (string) - Name pattern for the produced
  images (default: `%06d`)
- `-s, --step` (integer) - Frame step (default: 1)
- `-b, --start-frame` (integer) - Starting frame (default: 0)
- `-e, --end-frame` (integer) - Finishing frame (default: none)
- `-x, --image-ext` (string) Output image extension (default: `.jpg`)
- `-h, --help` - Print the help message and exit

Example: split a video into frames, use each 30-rd frame:
```bash
datum util split_video -i video.mp4 -o video.mp4-frames --step 30
```

Example: split a video into frames, save as 'frame_xxxxxx.png' files:
```bash
datum util split_video -i video.mp4 --image-ext=.png --name-pattern='frame_%%06d'
```

Example: split a video, add frames and annotations into dataset, export as YOLO:
```bash
datumaro util split_video -i video.avi -o video-frames
datumaro merge video-frames:image_dir annotations.json:coco_instances -f yolo -o output_dataset -- --save-media
```
