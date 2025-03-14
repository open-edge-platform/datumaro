# Commit

This command allows to fix the current state of a project and
create a new revision from the working tree.

By default, this command checks sources in the working tree for
changes. If there are unknown changes found, an error will be raised,
unless `--allow-foreign` is used. If such changes are committed,
the source will only be available for reproduction from the project
cache, because Datumaro will not know how to repeat them.

The command will add the sources into the project cache. If you only
need to record revision metadata, you can use the `--no-cache` parameter.
This can be useful if you want to save disk space and/or have a backup copy
of datasets used in the project.

If there are no changes found, the command will stop. To allow empty
commits, use `--allow-empty`.

Usage:

``` bash
datum project commit [-h] -m MESSAGE [--allow-empty] [--allow-foreign]
  [--no-cache] [-p PROJECT_DIR]
```

Parameters:
- `--allow-empty` - Allow commits with no changes
- `--allow-foreign` - Allow commits with changes made not by Datumaro
- `--no-cache` - Don't put committed datasets into cache, save only metadata
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example:

``` bash
datum project create
datum project import -f coco <path/to/coco/>
datum project commit -m "Added COCO"
```
