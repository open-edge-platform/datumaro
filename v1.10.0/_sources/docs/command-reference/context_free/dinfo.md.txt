# Info

## Print dataset info

This command outputs high level dataset information such as sample count,
categories and subsets.

Usage:

```console
datum dinfo [-h] [--all] [-p PROJECT_DIR] [revpath]
```

Parameters:
- `<target>` (string) - Target [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
  By default, prints info about the joined `project` dataset.
- `--all` - Print all the information: do not fold long lists of labels etc.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Examples:
- Print dataset info for the current project's working tree
  ```console
  datum dinfo
  ```

- Print dataset info for a COCO-like dataset
  ```console
  datum dinfo <path/to/dataset/>:coco
  ```

- Print dataset info for a source from a past revision
  ```console
  datum info HEAD~2:source-2
  ```

Sample output:

```
length: 5000
categories: label
  label:
    count: 80
    labels: person, bicycle, car, motorcycle (and 76 more)
subsets: minival2014
  'minival2014':
    length: 5000
    categories: label
      label:
        count: 80
        labels: person, bicycle, car, motorcycle (and 76 more)
```
