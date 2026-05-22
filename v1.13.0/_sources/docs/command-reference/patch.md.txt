# Patch

## Patch Datasets

Updates items of the first dataset with items from the second one.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. The
`-f/--format` option can be used to specify the output format. When
updating in-place, use the `--overwrite` parameter along with the
`--save-media` export option (in-place updates fail by default
to prevent data loss).

The datasets are not required to have the same labels. The labels from
the "patch" dataset are projected onto the labels of the patched dataset,
so only the annotations with the matching labels are used, i.e.
all the annotations having unknown labels are ignored. Currently,
this command doesn't allow to update the label information in the
patched dataset.

The command supports passing extra exporting options for the output
dataset. The extra options should be passed after the main arguments
and after the `--` separator. Particularly, this is useful to include
images in the output dataset with `--save-media`.

This command can be applied to arbitrary datasets.

## Usage
```console
datum patch [-h] [-o DST_DIR] [-f FORMAT] [--overwrite]
               target patch
               [-- EXPORT_ARGS]
```

\<dataset_path\> - A [dataset path](../explanation/concept.rst#dataset-path-concepts), optionally with format specification (e.g., `path/to/dataset:coco`).

Parameters:
- `target` (string) - Target dataset path (path to dataset directory, optionally with format specification)
- `patch` (string) - Patch dataset path (path to dataset directory, optionally with format specification)
- `-o, --output-dir` (string) - Output directory (default: save in-place)
- `-f, --format` (string) - Output format (default: target dataset format)
- `--overwrite` - Overwrite existing files in the save directory, if it is not empty
- `-h, --help` - Print the help message and exit
- `extra_args` - Additional arguments for exporting (pass '-- -h' for help). Must be specified after the main command arguments and after the '--' separator

## Examples
- Update a VOC-like dataset with COCO-like annotations
  ```console
  datum patch --overwrite dataset1/:voc dataset2/:coco -- --save-media
  ```

- Generate a patched dataset
  ```console
  datum patch -o patched_dataset/ dataset1/ dataset2/
  ```

- Generate a patched dataset in a different format
  ```console
  datum patch -o patched_dataset/ -f yolo_ultralytics dataset1/ dataset2/
  ```
