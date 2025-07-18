# ICDAR

## Format specification
ICDAR is a dataset for text recognition task, it's available
for download [here](https://rrc.cvc.uab.es/). There is exists
two most popular version of this dataset: ICDAR13 and ICDAR15,
Datumaro supports both of them.

Original dataset contains the following subformats:
- ICDAR word recognition;
- ICDAR text localization;
- ICDAR text segmentation.

Supported types of annotations:
- ICDAR word recognition
  - `Caption`
- ICDAR text localization
  - `Polygon`, `Bbox`
- ICDAR text segmentation
  - `Mask`

Supported attributes:
- ICDAR text localization
  - `text`: transcription of text is inside a `Polygon`/`Bbox`.
- ICDAR text segmentation
  - `index`: identifier of the annotation object, which is encoded in the mask
    and coincides with the line number in which the description
    of this object is written;
  - `text`: transcription of text is inside a `Mask`;
  - `color`: RGB values of the color corresponding text in the mask image
    (three numbers separated by space);
  - `center`: coordinates of the center of text
    (two numbers separated by space).

## Import ICDAR dataset

There is few ways to import ICDAR dataset with Datumaro:
- Through the Datumaro project
``` bash
datum project create
datum project import -f icdar_text_localization <text_localization_dataset>
datum project import -f icdar_text_segmentation <text_segmentation_dataset>
datum project import -f icdar_word_recognition <word_recognition_dataset>
```
- With Python API
```python
import datumaro as dm
data1 = dm.Dataset.import_from('text_localization_path', 'icdar_text_localization')
data2 = dm.Dataset.import_from('text_segmentation_path', 'icdar_text_segmentation')
data3 = dm.Dataset.import_from('word_recognition_path', 'icdar_word_recognition')
```
Dataset with ICDAR dataset should have the following structure:

For `icdar_word_recognition`
```
<dataset_path>/
├── <subset_name_1>
│   ├── gt.txt
│   └── images
│       ├── word_1.png
│       ├── word_2.png
│       ├── ...
├── <subset_name_2>
├── ...
```
For `icdar_text_localization`
```
<dataset_path>/
├── <subset_name_1>
│   ├── gt_img_1.txt
│   ├── gt_img_2.txt
│   ├── ...
│   └── images
│       ├── img_1.png
│       ├── img_2.png
│       ├── ...
├── <subset_name_2>
│   ├── ...
├── ...
```
For `icdar_text_segmentation`
```
<dataset_path>/
├── <subset_name_1>
│   ├── image_1_GT.bmp # mask for image_1
│   ├── image_1_GT.txt # description of mask objects on the image_1
│   ├── image_2_GT.bmp
│   ├── image_2_GT.txt
│   ├── ...
│   └── images
│       ├── image_1.png
│       ├── image_2.png
│       ├── ...
├── <subset_name_2>
│   ├── ...
├── ...
```
See more information about adding datasets to the project in the
[docs](../../command-reference/context/sources.md#add-dataset).

## Export to other formats
Datumaro can convert ICDAR dataset into any other format
[Datumaro supports](../../data-formats/formats/index.rst). Examples:
``` bash
# converting ICDAR text segmentation dataset into the VOC with `convert` command
datum convert -if icdar_text_segmentation -i source_dataset \
    -f voc -o export_dir -- --save-media
```
``` bash
# converting ICDAR text localization into the LabelMe through Datumaro project
datum project create
datum project import -f icdar_text_localization source_dataset
datum project export -f label_me -o ./export_dir -- --save-media
```
> Note: some formats have extra export options. For particular format see the
> [docs](/docs/data-formats/formats/index.rst) to get information about it.

With Datumaro you can also convert your dataset to one of the ICDAR formats,
but to get expected result, the source dataset should contain required
attributes, described in previous section.
> Note: in case with `icdar_text_segmentation` format, if your dataset contains
> masks without attribute `color` then it will be generated automatically.

Available extra export options for ICDAR dataset formats:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original)
