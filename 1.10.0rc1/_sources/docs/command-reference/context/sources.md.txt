# Source (context)

These commands are specific for Data Sources. Read more about them [here](/docs/user-manual/how_to_use_datumaro.md#datasets-and-data-sources).

## Import Dataset

Datasets can be added to a Datumaro project with the `import` command,
which adds a dataset link into the project and downloads (or copies)
the dataset. If you need to add a dataset already copied into the project,
use the [`add`](#add-dataset) command.

Dataset format readers can provide some additional import options. To pass
such options, use the `--` separator after the main command arguments.
The usage information can be printed with `datum import -f <format> -- --help`.

The list of currently available formats is listed in the command help output.

A dataset is imported by its URL. Currently, only local filesystem
paths are supported. The URL can be a file or a directory path
to a dataset. When the dataset is read, it is read as a whole.
However, many formats can have multiple subsets like `train`, `val`, `test`
etc. If you want to limit reading only to a specific subset, use
the `-r/--path` parameter. It can also be useful when subset files have
non-standard placement or names.

When a dataset is imported, the following things are done:
- URL is saved in the project config
- data in copied into the project

Each data source has a name assigned, which can be used in other commands. To
set a specific name, use the `-n/--name` parameter.

The dataset is added into the working tree of the project. A new commit
is _not_ done automatically.

Usage:

``` bash
datum project import [-h] [-n NAME] -f FORMAT [-r PATH] [--no-check]
  [-p PROJECT_DIR] url [-- EXTRA_FORMAT_ARGS]
```

Parameters:
- `<url>` (string) - A file of directory path to the dataset.
- `-f, --format` (string) - Dataset format
- `-r, --path` (string) - A path relative to the source URL the data source.
  Useful to specify a path to a subset, subtask, or a specific file in URL.
- `--no-check` - Don't try to read the source after importing
- `-n`, `--name` (string) - Name of the new source (default: generate
  automatically)
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra format args>` - Additional arguments for the format reader
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: create a project from images and annotations in different formats,
export as TFrecord for TF Detection API for model training

``` bash
# 'default' is the name of the subset below
datum project create
datum project import -f coco_instances -r annotations/instances_default.json path/to/coco
datum project import -f cvat <path/to/cvat/default.xml>
datum project import -f voc_detection -r custom_subset_dir/default.txt <path/to/voc>
datum project import -f datumaro <path/to/datumaro/default.json>
datum project import -f image_dir <path/to/images/dir>
datum project export -f tf_detection_api -- --save-media
```

## Add Dataset

Existing datasets can be added to a Datumaro project with the `add` command.
The command adds a project-local directory as a data source in the project.
Unlike the [`import`](#import-dataset)
command, it does not copy datasets and only works with local directories.
The source name is defined by the directory name.

Dataset format readers can provide some additional import options. To pass
such options, use the `--` separator after the main command arguments.
The usage information can be printed with `datum add -f <format> -- --help`.

The list of currently available formats is listed in the command help output.

A dataset is imported as a directory. When the dataset is read, it is read
as a whole. However, many formats can have multiple subsets like `train`,
`val`, `test` etc. If you want to limit reading only to a specific subset,
use the `-r/--path` parameter. It can also be useful when subset files have
non-standard placement or names.

The dataset is added into the working tree of the project. A new commit
is _not_ done automatically.

Usage:

``` bash
datum project add [-h] -f FORMAT [-r PATH] [--no-check]
  [-p PROJECT_DIR] path [-- EXTRA_FORMAT_ARGS]
```

Parameters:
- `<url>` (string) - A file of directory path to the dataset.
- `-f, --format` (string) - Dataset format
- `-r, --path` (string) - A path relative to the source URL the data source.
  Useful to specify a path to a subset, subtask, or a specific file in URL.
- `--no-check` - Don't try to read the source after importing
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra format args>` - Additional arguments for the format reader
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: create a project from images and annotations in different formats,
export in YOLO for model training

``` bash
datum project create
datum project add -f coco -r annotations/instances_train.json dataset1/
datum project add -f cvat dataset2/train.xml
datum project export -f yolo -- --save-media
```

Example: add an existing dataset into a project, avoid data copying

To add a dataset, we need to have it inside the project directory:

```bash
proj/
├─ .datumaro/
├─ .dvc/
├─ my_coco/
│  └─ images/
│     ├─ image1.jpg
│     └─ ...
│  └─ annotations/
│     └─ coco_annotation.json
├─ .dvcignore
└─ .gitignore
```

``` bash
datum project create -o proj/
mv ~/my_coco/ proj/my_coco/ # move the dataset into the project directory
datum project add -p proj/ -f coco proj/my_coco/
```

## Remove Datasets

To remove a data source from a project, use the `remove` command.

Usage:

``` bash
datum remove [-h] [--force] [--keep-data] [-p PROJECT_DIR] name [name ...]
```

Parameters:
- `<name>` (string) - The name of the source to be removed (repeatable)
- `-f, --force` - Do not fail and stop on errors during removal
- `--keep-data` - Do not remove source data from the working directory, remove
  only project metainfo.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example:

``` bash
datum project create
datum project import -f voc -n src1 <path/to/dataset/>
datum project remove src1
```
