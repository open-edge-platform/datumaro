# Supported Media Formats

Datumaro supports the following media types:
- 2D RGB(A) images
- Videos
- KITTI Point Clouds
- Tabular file (csv format)

To create an unlabelled dataset from an arbitrary directory with images use
`image_dir` and `image_zip` formats:

::::{tab-set}
:::{tab-item} CLI
``` bash
# Convert directory with images to a dataset format
datum convert -i </path/to/directory/containing/images> -if image_dir -f datumaro -o <output_dir>
```
:::
:::{tab-item} Python
```python
from datumaro import Dataset

dataset = Dataset.import_from('/path/to/directory/containing/images', 'image_dir')
```
:::
::::

This will search for images in the directory recursively and add
them as dataset entries with names like `<subdir1>/<subsubdir1>/<image_name1>`.
The list of formats matches the list of supported image formats in OpenCV:
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

Once there is a `Dataset` instance, its items can be split into subsets,
renamed, filtered, joined with annotations, exported in various formats etc.

To import frames from a video, you can split the video into frames with
the [`split_video` command](../command-reference/util.md#split-video-into-frames)
and then use the `image_dir` format described above. In more complex cases,
consider using [FFmpeg](https://ffmpeg.org/) and other tools for
video processing.

Alternatively, you can use the `video_frames` format directly:

> **Note**, however, that it can produce different results if the system
> environment changes. If you want to obtain reproducible results, consider
> splitting the video into frames by any method.

::::{tab-set}
:::{tab-item} CLI
``` bash
# Convert video frames to a dataset format
datum convert -i </path/to/video> -if video_frames -f datumaro -o <output_dir>
```
:::
:::{tab-item} Python
```python
from datumaro import Dataset

dataset = Dataset.import_from('/path/to/video', 'video_frames')
```
:::
::::

Datumaro supports the following video formats:
```
.3gp, .3g2, .asf, .wmv, .avi, .divx, .evo, .f4v, .flv, .mkv, .mk3d,
.mp4, .mpg, .mpeg, .m2p, .ps, .ts, .m2ts, .mxf, .ogg, .ogv, .ogx,
.mov, .qt, .rmvb, .vob, .webm
```

Also, Daturamo supports a tabular format.
A tabular dataset can be a single `.csv` file or a folder contains `.csv` files.

::::{tab-set}
:::{tab-item} CLI
``` bash
# Convert tabular data to a dataset format
datum convert -i </path/to/tabular> -if tabular -f datumaro -o <output_dir>
```
:::
:::{tab-item} Python
```python
from datumaro import Dataset

dataset = Dataset.import_from('/path/to/tabular', 'tabular')
```
:::
::::
