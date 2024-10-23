# Vgg Face2 CSV

## Format specification

Vgg Face 2 is a dataset for face-recognition task,
the repository with some information and sample data of Vgg Face 2 is available
[here](https://github.com/ox-vgg/vgg_face2)

Supported types of annotations:
- `Bbox`
- `Points`
- `Label`

Format doesn't support any attributes for annotations objects.

## Import Vgg Face2 dataset

A Datumaro project with a Vgg Face 2 dataset can be created
in the following way:

```
datum project create
datum project import -f vgg_face2 <path_to_dataset>
```

> Note: if you use `datum project import` then <path_to_dataset> should not be a
> subdirectory of directory with Datumaro project, see more information about
> it in the [docs](../../command-reference/context/sources.md#add-dataset).

And you can also load Vgg Face 2 through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset>', format='vgg_face2')
```

For successful importing of Vgg Face2 face the input directory with dataset
should has the following structure:

```
vgg_face2_dataset/
├── labels.txt # labels mapping
├── bb_landmark
│   ├── loose_bb_test.csv  # information about bounding boxes for test subset
│   ├── loose_bb_train.csv
│   ├── loose_bb_<any_other_subset_name>.csv
│   ├── loose_landmark_test.csv # landmark points information for test subset
│   ├── loose_landmark_train.csv
│   └── loose_landmark_<any_other_subset_name>.csv
├── test
│   ├── n000001 # directory with images for n000001 label
│   │   ├── 0001_01.jpg
│   │   ├── 0001_02.jpg
│   │   ├── ...
│   ├── n000002 # directory with images for n000002 label
│   │   ├── 0002_01.jpg
│   │   ├── 0003_01.jpg
│   │   ├── ...
│   ├── ...
├── train
│   ├── n000004
│   │   ├── 0004_01.jpg
│   │   ├── 0004_02.jpg
│   │   ├── ...
│   ├── ...
└── <any_other_subset_name>
    ├── ...
```

## Export Vgg Face2 dataset

Datumaro can convert a Vgg Face2 dataset into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).
There is few examples how to do it:

```
# Using `convert` command
datum convert -if vgg_face2 -i <path_to_vgg_face2> \
    -f voc -o <output_dir> -- --save-media

# Using Datumaro project
datum project create
datum project import -f vgg_face2 <path_to_vgg_face2>
datum project export -f yolo -o <output_dir>
```

> Note: to get the expected result from the conversion, the output format
> should support the same types of annotations (one or more) as Vgg Face2
> (`Bbox`, `Points`, `Label`)

And also you can convert your Vgg Face2 dataset using Python API

```python
import datumaro as dm

vgg_face2_dataset = dm.Dataset.import_from('<path_to_dataset', format='vgg_face2')

vgg_face2_dataset.export('<output_dir>', format='open_images', save_media=True)
```

> Note: some formats have extra export options. For particular format see the
> [docs](/docs/data-formats/formats/index.rst) to get information about it.

## Export dataset to the Vgg Face2 format

If you have dataset in some format and want to convert this dataset
into the Vgg Face2, ensure that this dataset contains `Bbox` or/and `Points`
or/and `Label` and use Datumaro to perform conversion.
There is few examples:

```
# Using convert command
datum convert -if wider_face -i <path_to_wider> \
    -f vgg_face2 -o <output_dir>

# Using Datumaro project
datum project create
datum project import -f wider_face <path_to_wider>
datum project export -f vgg_face2 -o <output_dir> -- --save-media --image-ext '.png'
```

> Note: `vgg_face2` format supports only one `Bbox` per image

Extra options for exporting to Vgg Face2 format:

- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting the dataset (by default `.png`)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
