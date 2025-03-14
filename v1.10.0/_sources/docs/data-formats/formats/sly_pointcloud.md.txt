# Supervisely Point Cloud

## Format specification

Specification for the Point Cloud data format is available
[here](https://docs.supervise.ly/data-organization/00_ann_format_navi).

You can also find examples of working with the dataset
[here](https://drive.google.com/file/d/1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl/view).

Supported annotation types:
- `cuboid_3d`

Supported annotation attributes:
- `track_id` (read/write, integer), responsible for `object` field
- `createdAt` (write, string),
- `updatedAt` (write, string),
- `labelerLogin` (write, string), responsible for the corresponding fields
  in the annotation file.
- arbitrary attributes

Supported image attributes:
- `description` (read/write, string),
- `createdAt` (write, string),
- `updatedAt` (write, string),
- `labelerLogin` (write, string), responsible for the corresponding fields
  in the annotation file.
- `frame` (read/write, integer). Indicates frame number of the image.
- arbitrary attributes

## Import Supervisely Point Cloud dataset

An example dataset in Supervisely Point Cloud format is available for download:

<https://drive.google.com/u/0/uc?id=1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl&export=download>

Point Cloud dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── ds0/
    │   ├── ann/
    │   │   ├── <pcdname1.pcd.json>
    │   │   ├── <pcdname2.pcd.json>
    │   │   └── ...
    │   ├── pointcloud/
    │   │   ├── <pcdname1.pcd>
    │   │   ├── <pcdname1.pcd>
    │   │   └── ...
    │   ├── related_images/
    │   │   ├── <pcdname1_pcd>/
    │   │   |  ├── <image_name.ext.json>
    │   │   |  ├── <image_name.ext.json>
    │   │   └── ...
    ├── key_id_map.json
    └── meta.json
```

Use the command below to import a Supervisely Point Cloud dataset:

``` bash
datum project create
datum project import -f sly_pointcloud <path/to/dataset>
```

To make sure that the selected dataset has been added to the project,
you can run `datum project info`, which will display the project and dataset
information.

## Export to other formats

Datumaro can convert Supervisely Point Cloud dataset into any other
format [Datumaro supports](/docs/data-formats/formats/index.rst).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. 3D point clouds can be saved in `KITTI Raw` format,
but not in `COCO keypoints`.

There are several ways to convert a Supervisely Point Cloud dataset
to other dataset formats:

``` bash
datum project create
datum project import -f sly_pointcloud <path/to/sly_pcd/>
datum project export -f kitti_raw -o <output/dir>
```
or
``` bash
datum convert -if sly_pointcloud -i <path/to/sly_pcd/> -f kitti_raw
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'sly_pointcloud')
dataset.export('save_dir', 'kitti_raw', save_media=True)
```

## Export to Supervisely Point Cloud

There are several ways to convert a dataset to Supervisely Point Cloud format:

``` bash
# export dataset into Supervisely Point Cloud format from existing project
datum project export -p <path/to/project> -f sly_pointcloud -o <output/dir> \
    -- --save-media
```
``` bash
# converting to Supervisely Point Cloud format from other format
datum convert -if kitti_raw -i <path/to/dataset> \
    -f sly_pointcloud -o <output/dir> -- --save-media
```

Extra options for exporting in Supervisely Point Cloud format:
- `--save-media` allow to export dataset with saving media files. This will
  include point clouds and related images (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--reindex` assigns new indices to frames and annotations.
- `--allow-undeclared-attrs` allows writing arbitrary annotation attributes.
  By default, only attributes specified in the input dataset metainfo
  will be written.

## Examples

### Example 1. Import dataset, compute statistics

```bash
datum project create -o project
datum project import -p project -f sly_pointcloud ../sly_dataset/
datum stats -p project
```

### Example 2. Convert Supervisely Point Clouds to KITTI Raw

``` bash
datum convert -if sly_pointcloud -i ../sly_pcd/ \
    -f kitti_raw -o my_kitti/ -- --save-media --reindex --allow-attrs
```

### Example 3. Create a custom dataset

``` python
import datumaro as dm

dataset = dm.Dataset.from_iterable([
    dm.DatasetItem(id='frame_1',
        annotations=[
            dm.Cuboid3d(id=206, label=0,
                position=[320.86, 979.18, 1.04],
                attributes={'occluded': False, 'track_id': 1, 'x': 1}),

            dm.Cuboid3d(id=207, label=1,
                position=[318.19, 974.65, 1.29],
                attributes={'occluded': True, 'track_id': 2}),
        ],
        pcd='path/to/pcd1.pcd',
        attributes={'frame': 0, 'description': 'zzz'}
    ),

    dm.DatasetItem(id='frm2',
        annotations=[
            dm.Cuboid3d(id=208, label=1,
                position=[23.04, 8.75, -0.78],
                attributes={'occluded': False, 'track_id': 2})
        ],
        pcd='path/to/pcd2.pcd', related_images=['image2.png'],
        attributes={'frame': 1}
    ),
], categories=['cat', 'dog'])

dataset.export('my_dataset/', format='sly_pointcloud', save_media=True,
    allow_undeclared_attrs=True)
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_sly_pointcloud_format.py)
