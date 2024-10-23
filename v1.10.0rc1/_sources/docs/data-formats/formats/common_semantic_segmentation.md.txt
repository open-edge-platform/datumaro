# Common Semantic Segmentation

## Format specification

CSS format specification is available [here](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/annotation_converters/README.md#supported-converters).

Supported annotation types:
- `Masks`

## Import Common Semantic Segmentation dataset

A Datumaro project with a CSS source can be created in the following way:

``` bash
datum project create
datum project import --format common_semantic_segmentation <path/to/dataset>
```

Extra import options:
- `--image-prefix IMAGE_PREFIX` allow to import dataset with custom image prefix
  (by default '')
- `--mask-prefix MASK_PREFIX` allow to import dataset with custom mask prefix
  (by default '')

CSS dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of labels
    ├── images/
    │   ├── <img1>.png
    │   ├── <img2>.png
    │   └── ...
    └── masks/
        ├── <img1>.png
        ├── <img2>.png
        └── ...
```

To describe classes and colors, you should use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a CSS dataset into any other format [Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to formats
that support the segmentation task (e.g. PASCAL VOC, CamVid, Cityscapes, etc.)

There are several ways to convert a CSS dataset to other dataset
formats using CLI:

``` bash
datum project create
datum project import -f common_semantic_segmentation <path/to/dataset>
datum project export -f voc -o <output/dir>
```
or
``` bash
datum convert -if common_semantic_segmentation -i <path/to/dataset> \
    -f cityscapes -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'common_semantic_segmentation')
dataset.export('save_dir', 'camvid', save_media=True)
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/data_formats/test_common_semantic_segmentation_format.py)
