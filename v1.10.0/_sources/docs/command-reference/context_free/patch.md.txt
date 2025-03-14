# Patch

## Patch Datasets

Updates items of the first dataset with items from the second one.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. When
updating in-place, use the `--overwrite` parameter along with the
`--save-media` export option (in-place updates fail by default
to prevent data loss).

```{eval-rst}
Unlike the regular project :ref:`data source joining <dataset_merging>`,
the datasets are not required to have the same labels. The labels from
the "patch" dataset are projected onto the labels of the patched dataset,
so only the annotations with the matching labels are used, i.e.
all the annotations having unknown labels are ignored. Currently,
this command doesn't allow to update the label information in the
patched dataset.
```

The command supports passing extra exporting options for the output
dataset. The extra options should be passed after the main arguments
and after the `--` separator. Particularly, this is useful to include
images in the output dataset with `--save-media`.

This command can be applied to the current project targets or
arbitrary datasets outside a project. Note that if the target dataset
is read-only (e.g. if it is a project, stage or a cache entry),
the output directory must be provided.

Usage:
```console
datum patch [-h] [-o DST_DIR] [--overwrite] [-p PROJECT_DIR]
            target patch
            [-- EXPORT_ARGS]
```

\<revpath\> - either [a dataset path or a revision path](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).

The current project (`-p/--project`) is also used as a context for
plugins, so it can be useful for dataset paths having custom formats.
When not specified, the current project's working tree is used.

Parameters:
- `<target dataset>` (string) - Target [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts)
- `<patch dataset>` (string) - Patch [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts)
- `-o, --output-dir` (string) - Output directory. By default, saves in-place
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <export args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Examples:
- Update a VOC-like dataset with COCO-like annotations
  ```console
  datum patch --overwrite <path/to/dataset1/>:voc <path/to/dataset2/>:coco -- --save-media
  ```

- Generate a patched dataset, based on a project
  ```console
  datum patch -o <path/to/patched/project1/> <path/to/project1/> <path/to/project2/>
  ```

- Update the "source1" source in the current project with a dataset
  ```console
  datum patch --overwrite source1 <path/to/dataset/>:coco
  ```

- Generate a patched source from a previous revision and a dataset
  ```console
  datum patch -o new_src2/ HEAD~2:source-2 <path/to/dataset>:yolo
  ```

- Update a dataset in a custom format, described in a project plugin
  ```console
  datum patch -p <path/to/project/> --overwrite <path/to/dataset1/>:my_format <path/to/dataset2/>:coco
  ```
