# Status

This command prints the summary of the source changes between
the working tree of a project and its HEAD revision.

Prints lines in the following format:
`<status> <source name>`

The list of possible `status` values:
- `modified` - the source data exists and it is changed
- `foreign_modified` - the source data exists and it is changed,
  but Datumaro does not know about the way the differences were made.
  If changes are committed, they will only be available for reproduction
  from the project cache.
- `added` - the source was added in the working tree
- `removed` - the source was removed from the working tree. This status won't
  be reported if just the source _data_ is removed in the working tree.
  In such situation the status will be `missing`.
- `missing` - the source data is removed from the working directory.
  The source still can be restored from the project cache or reproduced.

Usage:

``` bash
datum project status [-h] [-p PROJECT_DIR]
```

Parameters:
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example output:

``` bash
added source-1
modified source-2
foreign_modified source-3
removed source-4
missing source-5
```
