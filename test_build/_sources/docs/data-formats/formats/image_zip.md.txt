# Image zip

## Format specification

The image zip format allows to export/import unannotated datasets
with images to/from a zip archive. The format doesn't support any
annotations or attributes.

## Convert Image zip dataset

There are several ways to convert unannotated datasets with Datumaro:

- From an existing archive:

```bash
datum convert --input-format image_zip --input-path ./images.zip \
    --output-format <desired_format> --output-dir <output/dir>
```

- From a directory with zip archives. Datumaro will convert images from
  all zip files in the directory:

```bash
datum convert --input-format image_zip --input-path ./foo \
    --output-format <desired_format> --output-dir <output/dir>
```

The directory with zip archives must have the following structure:

```
└── foo/
    ├── archive1.zip/
    |   ├── image_1.jpg
    |   ├── image_2.png
    |   ├── subdir/
    |   |   ├── image_3.jpg
    |   |   └── ...
    |   └── ...
    ├── archive2.zip/
    |   ├── image_101.jpg
    |   ├── image_102.jpg
    |   └── ...
    ...
```

Images in the archives must have a supported extension,
follow the [media format](/docs/data-formats/media_formats/) to see the supported
extensions.

## Export to other formats

Datumaro can convert image zip dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
For example:

```bash
datum convert -if image_zip -i ./images.zip -f coco -o ./new_dir -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'image_zip')
dataset.export('save_dir', 'coco', save_media=True)
```

## Export an unannotated dataset to a zip archive

Example: converting images from a VOC dataset to zip archives:
```bash
datum convert -if voc -i ./VOC2012 -f image_zip -o ./output_dir -- --name voc_images.zip
```

Extra options for exporting to image_zip format:
- `--save-media` allow to export dataset with saving media files
  (default: `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting dataset (default: use original or `.jpg`, if none)
- `--name` name of output zipfile (default: `default.zip`)
- `--compression` allow to specify archive compression method.
  Available methods:
  `ZIP_STORED`, `ZIP_DEFLATED`, `ZIP_BZIP2`, `ZIP_LZMA` (default: `ZIP_STORED`).
  Follow [zip documentation](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)
  for more information.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/tree/develop/tests/test_image_zip_format.py)
