# Info

## Print dataset info

This command outputs high level dataset information such as sample count,
categories and subsets.

## Usage

```console
datum dinfo [-h] [--all] dataset_path
```

Parameters:
- `dataset_path` (string) - Target dataset path with optional format specification (path:format)
- `--all` - Print all information
- `-h, --help` - Print the help message and exit

## Examples
- Print dataset info for a path and a format name
  ```console
  datum dinfo path/to/dataset:voc
  ```

- Print dataset info for a COCO-like dataset
  ```console
  datum dinfo path/to/dataset:coco
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
