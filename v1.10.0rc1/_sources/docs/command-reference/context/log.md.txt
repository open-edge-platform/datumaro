# Log

This command prints the history of the current project revision.

Prints lines in the following format:
`<short commit hash> <commit message>`

Usage:

``` bash
datum project log [-h] [-n MAX_COUNT] [-p PROJECT_DIR]
```

Parameters:
- `-n, --max-count` (number, default: 10) - The maximum number of
  previous revisions in the output
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example output:

``` bash
affbh33 Added COCO dataset
eeffa35 Added VOC dataset
```
