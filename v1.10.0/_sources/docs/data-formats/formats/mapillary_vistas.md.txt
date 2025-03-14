# Mapillary Vistas

## Format specification

Mapillary Vistas dataset homepage is available
[here](https://www.mapillary.com/dataset/vistas).
After registration the dataset will be available for downloading.
The specification for this format contains in the root directory of original
dataset.

Supported annotation types:
    - `Mask` (class, instances, panoptic)
    - `Polygon`

Supported atttibutes:
    - `is_crowd`(boolean; on panoptic `mask`): Indicates that the annotation
    covers multiple instances of the same class.

## Import Mapillary Vistas dataset

Use these instructions to import Mapillary Vistas dataset into Datumaro project:

```bash
datum project create
datum project add -f mapillary_vistas ./dataset
```

> Note: the directory with dataset should be subdirectory of the
> project directory.
<!-- markdownlint-disable-line MD028 -->
> Note: there is no opportunity to import both instance and panoptic
> masks for one dataset.

Use one of subformats (`mapillary_vistas_instances`, `mapillary_vistas_panoptic`),
if your dataset contains both panoptic and instance masks:
```bash
datum project add -f mapillary_vistas_instances ./dataset
```
or
``` bash
datum project add -f mapillary_vistas_panoptic ./dataset
```

Extra options for adding a source in the Mapillary Vistas format:

- `--use-original-config`: Use original `config_*.json` file for your version of
  Mapillary Vistas dataset. This options can helps to import dataset, in case
  when you don't have `config_*.json` file, but your dataset is using original
  categories of Mapillary Vistas dataset. The version of dataset will be detect
  by the name of annotation directory in your dataset (v1.2 or v2.0).
- `--keep-original-category-ids`: Add dummy label categories so that
 category indexes in the imported data source correspond to the category IDs
 in the original annotation file.

Example of using extra options:
```bash
datum project add -f mapillary_vistas ./dataset -- --use-original-config
```
Mapillary Vistas dataset has two versions: v1.2, v2.0.
They differ in the number of classes, the name of the classes, supported types
of annotations, and the names of the directory with annotations.
So, the directory with dataset should have one of these structures:

```
dataset
├── config_v1.2.json
├── config_v2.0.json
├── <training> # config file with description of classes (id, color, name)
│   ├── images
│   │   ├── <image_name1>.jpg
│   │   ├── <image_name2>.jpg
│   │   ├── ...
│   ├── v1.2
│   |   ├── instances # directory with instance masks
│   |   │   ├── <image_name1>.png
│   |   │   ├── <image_name2>.png
│   |   │   ├── ...
│   |   ├── labels # directory with class masks
│   |   │   ├── <image_name1>.png
│   |   │   ├── <image_name2>.png
│   |   │   ├── ...
│   |   └── panoptic # directory with panoptic masks and panoptic config file
│   |       ├── panoptic_2018.json # description of classes and annotations
│   |       ├── <image_name1>.png
│   |       ├── <image_name2>.png
│   |       ├── ...
│   └── v2.0
│       ├── instances # directory with instance masks
│       │   ├── <image_name1>.png
│       │   ├── <image_name2>.png
│       │   ├── ...
│       ├── labels # directory with class masks
│       │   ├── <image_name1>.png
│       │   ├── <image_name2>.png
│       │   ├── ...
│       ├── panoptic # directory with panoptic masks and panoptic config file
│       │   ├── panoptic_2020.json # description of classes and annotations
│       │   ├── <image_name1>.png
│       │   ├── <image_name2>.png
│       │   ├── ...
│       └── polygons # directory with description of polygons
│           ├── <image_name1>.json
│           ├── <image_name2>.json
│           ├── ...
├── <validation>
|   ├── ...
├── ...
```

To add custom classes, you can use [`dataset_meta.json`](/docs/data-formats/formats/index.rst#dataset-meta-info-file).

See examples of annotation files in
[test assets](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mapillary_vistas_dataset).
