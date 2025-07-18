# Export

## Export Datasets

This command exports a project or a source as a dataset in some format.

Check [supported formats](../../data-formats/formats/index.rst)  for more info
about format specifications, supported options and other details.
The list of formats can be extended by custom plugins, check
[extending tips](../../user-manual/extending) for information on this topic.

Available formats are listed in the command help output.

Dataset format writers support additional export options. To pass
such options, use the `--` separator after the main command arguments.
The usage information can be printed with `datum import -f <format> -- --help`.

Common export options:
- Most formats (where applicable) support the `--save-media` option, which
  allows to export dataset images along with annotations. The option is
  disabled be default.
- If `--save-media` is used, the `image-ext` option can be passed to
  specify the output image file extension (`.jpg`, `.png` etc.). By default,
  tries to Datumaro keep the original image extension. This option
  allows to convert all the images from one format into another.

This command allows to use the `-f/--filter` parameter to select dataset
elements needed for exporting. Read the [`filter`](../context_free/filter)
command description for more info about this functionality.

The command can only be applied to a project build target, a stage
or the combined `project` target, in which case all the targets will
be affected.

Usage:

``` bash
datum project export [-h] [-e FILTER] [--filter-mode FILTER_MODE] [-o DST_DIR]
  [--overwrite] [-p PROJECT_DIR] -f FORMAT [target] [-- EXTRA_FORMAT_ARGS]
```

Parameters:
- `<target>` (string) - A project build target to be exported.
  By default, all project targets are affected.
- `-f, --format` (string) - Output format.
- `-e, --filter` (string) - XML XPath filter expression for dataset items
- `--filter-mode` (string) - The filtering mode. Default is the `i` mode.
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used.
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra format args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: save a project as a VOC-like dataset, include images, convert
images to `PNG` from other formats.

``` bash
datum project export \
  -p test_project \
  -o test_project-export \
  -f voc \
  -- --save-media --image-ext='.png'
```
