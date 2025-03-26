# Kaggle

## Format specification

[Kaggle](https://www.kaggle.com/) provides more than 2800 computer vision datasets for the public good and fair competition.
All datasets are available for downloading at [here](https://www.kaggle.com/datasets?tags=13207-Computer+Vision).
However, since Kaggle doesn't enforce community to follow specific rule for dataset uploads, it is more natural to explore a dataset directoy structure by manual.
So it eventually requires to take some time for importing those datasets into their machine learning codes.
Therefore, Datumaro is providing an ability to import them through Datumaro Python APIs.

Supported type of annotations:
- `Label` (classification)
- `Bbox` (object detection)
- `Mask` (segmentation)

## Import Kaggle Image CSV dataset

Indeed, Kaggle doesn't have any specific directory structure, and Datumaro hence requires more user-aided arguments for importing.
For `kaggle_image_csv` format, we want to have one `csv` file and `image_directory` as shown below.

```
├── <annotation_file>.csv
└── <image_directory>
    ├── <name_of_image_1>.jpg # extension of video could be other
    ├── <name_of_image_2>.jpg
    └── ...
```

Here `<annotation_file>.csv` contains media name and annotation information such as label or box coordinates as

```
media_name_in_image_directory,label,...
<name_of_image_1>,<category_1>,...
<name_of_image_2>,<category_2>,...
...
```

A Datumaro dataset with a Kaggle dataset can be created in the following way in Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_image_directory>', format='kaggle_image_csv', ann_file='<path_to_csv_file>', columns={"media": "column_name_of_media", "label": "column_name_of_label"})
```
At this time, it's essential to specify the column names for media and label such as `dm.Dataset.import_from(..., columns={"media": "column_name_of_media", "label": "column_name_of_label"})`

## Import Kaggle Image Txt dataset

Another `kaggle_image_txt` format replaces only `columns` with an order of information in `.txt`.
For instance, dataset can be created by

```python
dataset = dm.Dataset.import_from('<path_to_image_directory>', format='kaggle_image_txt', ann_file='<path_to_txt_file>', columns={"media": 0, "label": 1})
```

## Import Kaggle Image Mask dataset

For segmentation tasks, `kaggle_image_mask` requires to have a directory for mask images as following Python API.
```python
dataset = dm.Dataset.import_from('<path_to_image_directory>', format='kaggle_image_mask', mask_path='<path_to_mask_directory>')
```

## Import Kaggle VOC and Kaggle YOLO datasets

Sometimes, communities upload their annotation files for each images with VOC (`xml`) and YOLO (`txt`) formats thanks to its popularity.
But, they violate the directory sturcture of the original Pascal-VOC and YOLO described in [VOC](./pascal_voc.md) and [YOLO](./yolo.md), respectively.
For these cases, we provide `kaggle_voc` and `kaggle_yolo` formats by specifying the path to annotation files as below.

```python
dataset = dm.Dataset.import_from('<path_to_image_directory>', format='kaggle_voc', ann_path='<path_to_annotation_directory>')
```
or
```python
dataset = dm.Dataset.import_from('<path_to_image_directory>', format='kaggle_yolo', ann_path='<path_to_annotation_directory>')
```

Please refer to [here](https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/20_kaggle_data_import.ipynb) for various practices.
