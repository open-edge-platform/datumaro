# Create

## Create project

The command creates an empty project. A project is required for the most of
Datumaro functionality.

By default, the project is created in the current directory. To specify
another output directory, pass the `-o/--output-dir` parameter. If output
already directory contains a Datumaro project, an error is raised, unless
`--overwrite` is used.

Usage:

``` bash
datum project create [-h] [-o DST_DIR] [--overwrite]
```

Parameters:
- `-o, --output-dir` (string) - Allows to specify an output directory.
  The current directory is used by default.
- `--overwrite` - Allows to overwrite existing project files in the output
  directory. Any other files are not touched.
- `-h, --help` - Print the help message and exit.

Examples:

Example: create an empty project in the `my_dataset` directory

``` bash
datum project create -o my_dataset/
```

Example: create a new empty project in the current directory, remove the
existing one

``` bash
datum project create
...
datum project create --overwrite
```
