# CVAT

## Format Specification

CVAT format is a dedicated data format provided by the [CVAT.ai](https://www.cvat.ai) annotation tool.
It is formatted using XML (eXtensible Markup Language). For detailed information about the XML file specification, please refer to [this link](https://opencv.github.io/cvat/docs/manual/advanced/xml_format).
CVAT format includes two data formats for different purposes:

- [CVAT for video](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/#interpolation):
  This format is used for video tasks. It treats a track as the atomic unit of an annotation object.
  Each track corresponds to an object that can be present in multiple frames.
  Therefore, each annotation is a child of a track and includes a frame ID to indicate the associated time.

- [CVAT for images](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/#annotation):
  This format is used for image tasks. It maintains a list of images.
  For each image, it includes annotations derived from objects within the image.
  This format is more similar to other commonly used data formats in computer vision tasks.

Supported annotation types:
- `Bbox`
- `Label`
- `Points`
- `Polygon`
- `PolyLine`
- `Mask`

Supported annotation attributes:
- It supports any arbitrary boolean, floating number, or string attribute.

## Convert CVAT dataset

A CVAT dataset can be converted in the following way:

``` bash
datum convert --input-format cvat --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

A CVAT dataset directory should have the following structure:

- Exported from the CVAT project: If the dataset is exported from the CVAT project, the image files are grouped into subsets and they are located under the `images` directory.

  <!--lint disable fenced-code-flag-->
  ```console
  └─ Dataset/
      ├── dataset_meta.json # a list of custom labels (optional)
      ├── annotations.xml
      └── images/
          ├── <subset-1>
          │   ├── <image_name1.ext>
          │   ├── <image_name1.ext>
          │   └── ...
          ├── ...
          └── <subset-n>
              ├── <image_name1.ext>
              └── ...
  ```

- Exported from the CVAT task: If the dataset is exported from the CVAT task, there is only one subset in the task. Therefore, there is no subset sub-directory in the `images` directory.

  <!--lint disable fenced-code-flag-->
  ```console
  └─ Dataset/
      ├── dataset_meta.json # a list of custom labels (optional)
      ├── annotations.xml
      └── images/
          ├── <image_name1.ext>
          ├── <image_name1.ext>
          └── ...
  ```

The annotation file must have the name like `annotations.xml` in the root directory.
The image files exist in the `images` directory. There are sub-directory according to their subset information.
However, this is only provided if the dataset is exported from the CVAT project.
If the dataset is exported from the CVAT task, all images are directly under the `images` directory without subset information.
To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

## Export to other formats

Datumaro can convert CVAT dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a CVAT dataset to other dataset formats
using CLI:

``` bash
datum convert --input-format cvat --input-path <path/to/dataset> \
    --output-format voc --output-dir <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'cvat')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to CVAT

There are several ways to convert a dataset to CVAT format:

``` bash
# converting to CVAT format from other format
datum convert --input-format voc --input-path <path/to/dataset> \
    --output-format cvat --output-dir <output/dir> -- --save-media
```

Extra options for exporting to CVAT format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.jpg`, if none)
- `--save-dataset-meta` allow to export dataset with saving dataset meta
  file (by default `False`)
- `--reindex` assign new indices to frames
- `--allow-undeclared-attrs` write annotation attributes even if they are not present in the input dataset metainfo

When performing `convert` to CVAT format, you may encounter a warning message like the following:
```bash
skipping undeclared attribute 'is_crowd' for label '<label>' (allow with --allow-undeclared-attrs option)
```
In such cases, you can bypass this warning by using the `--allow-undeclared-attrs` option as follows:
```bash
datum convert --input-format <source-format> --input-path <path/to/dataset> \
    --output-format cvat --output-dir <output/dir> -- --allow-undeclared-attrs
```
This allows you to proceed with the export while bypassing the warning.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/tree/develop/tests/unit/test_cvat_format.py)
