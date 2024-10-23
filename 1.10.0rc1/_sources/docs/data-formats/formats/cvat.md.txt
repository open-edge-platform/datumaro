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

## Import CVAT dataset

A Datumaro project with a CVAT source can be created in the following way:

``` bash
datum project create
datum project import --format cvat <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum project create --help` for more information.

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

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert CVAT dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a CVAT dataset to other dataset formats
using CLI:

``` bash
datum project create
datum project import -f cvat <path/to/dataset>
datum project export -f voc -o <output/dir>
```
or
``` bash
datum convert -if cvat -i <path/to/dataset> -f voc -o <output/dir>
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
# export dataset into CVAT format from existing project
datum project export -p <path/to/project> -f cvat -o <output/dir> \
    -- --save-media
```
``` bash
# converting to CVAT format from other format
datum convert -if voc -i <path/to/dataset> \
    -f cvat -o <output/dir> -- --save-media
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
In such cases, you can bypass this warning by using the `export` command as follows:
```bash
datum project export -o <output/dir> -p <path/to/project> -f cvat -- --allow-undeclared-attrs
```
This allows you to proceed with the export while bypassing the warning.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/test_cvat_format.py)
