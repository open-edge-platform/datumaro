# ImageNet

## Format specification
ImageNet is one of the most popular datasets for image classification task,
this dataset is available for downloading
[here](https://image-net.org/download.php)

Supported types of annotations:
- `Label`

Format doesn't support any attributes for annotations objects.

The original ImageNet dataset contains about 1.2M images and information
about class name for each image. Datumaro supports two versions of ImageNet
format: `imagenet` and `imagenet_txt`. The `imagenet_txt` format assumes storing
information about the class of the image in `*.txt` files. And `imagenet` format
assumes storing information about the class of the image in the name of
directory where is this image stored.

## Import ImageNet dataset

A Datumaro project with a ImageNet dataset can be created
in the following way:

```
datum project create
datum project import -f imagenet <path_to_dataset>
# or
datum project import -f imagenet_txt <path_to_dataset>
```

> Note: if you use `datum project import` then <path_to_dataset> should not be a
> subdirectory of directory with Datumaro project, see more information about
> it in the [docs](../../command-reference/context/sources.md#add-dataset).

Load ImageNet dataset through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset>', format='imagenet_txt')
# or
dataset = dm.Dataset.import_from('<path_to_dataset>', format='imagenet')
```

For successful importing of ImageNet dataset the input directory with dataset
should has the following structure:

- `imagenet` format:
  <!--lint disable fenced-code-flag-->
  ```
  imagenet_dataset/
  ├── label_0
  │   ├── <image_name_0>.jpg
  │   ├── <image_name_1>.jpg
  │   ├── <image_name_2>.jpg
  │   ├── ...
  ├── label_1
  │   ├── <image_name_0>.jpg
  │   ├── <image_name_1>.jpg
  │   ├── <image_name_2>.jpg
  │   ├── ...
  ├── ...
  ```
- `imagenet_txt` format:
  <!--lint disable fenced-code-flag-->
  ```
  imagenet_txt_dataset/
  ├── images # directory with images
  │   ├── <image_name_0>.jpg
  │   ├── <image_name_1>.jpg
  │   ├── <image_name_2>.jpg
  │   ├── ...
  ├── synsets.txt # optional, list of labels
  └── train.txt   # list of pairs (image_name, label)
  ```

> Note: if you don't have synsets file then Datumaro will automatically generate
> classes with a name pattern `class-<i>`.

Datumaro has few import options for `imagenet_txt` format, to apply them
use the `--` after the main command argument.

`imagenet_txt` import options:
- `--labels` {`file`, `generate`}: allow to specify where to get label
  descriptions from (use `file` to load from the file specified
  by `--labels-file`; `generate` to create generic ones)
- `--labels-file` allow to specify path to the file with label descriptions
  ("synsets.txt")

## Export ImageNet dataset

Datumaro can convert ImageNet into any other format
[Datumaro supports](/docs/data-formats/formats/index.rst).
To get the expected result, convert the dataset to a format
that supports `Label` annotation objects.

```
# Using `convert` command
datum convert -if imagenet -i <path_to_imagenet> \
    -f voc -o <output_dir> -- --save-media

# Using Datumaro project
datum project create
datum project import -f imagenet_txt <path_to_imagenet> -- --labels generate
datum project export -f open_images -o <output_dir>
```

And also you can convert your ImageNet dataset using Python API

```python
import datumaro as dm

imagenet_dataset = dm.Dataset.import_from('<path_to_dataset', format='imagenet')

imagenet_dataset.export('<output_dir>', format='vgg_face2', save_media=True)
```

> Note: some formats have extra export options. For particular format see the
> [docs](/docs/data-formats/formats/index.rst) to get information about it.

## Export dataset to the ImageNet format

If your dataset contains `Label` for images and you want to convert this
dataset into the ImagetNet format, you can use Datumaro for it:

```
# Using convert command
datum convert -if open_images -i <path_to_oid> \
    -f imagenet_txt -o <output_dir> -- --save-media --save-dataset-meta

# Using Datumaro project
datum project create
datum project import -f open_images <path_to_oid>
datum project export -f imagenet -o <output_dir>
```

Extra options for exporting to ImageNet formats:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting the dataset (by default `.png`)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
