# How to use Datumaro

## Python Module Examples

As a standalone tool or a Python module:

``` bash
datum --help

python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

As a Python library:

``` python
import datumaro as dm
...
dataset = dm.Dataset.import_from(path, format)
...
```

## Command-line Examples

Example: create a project, add dataset, modify, restore an old version

``` bash
datum project create
datum project import <path/to/dataset> -f coco -n source1
datum project commit -m "Added a dataset"
datum transform -t shapes_to_boxes
datum filter -e '/item/annotation[label="cat" or label="dog"]' -m i+a
datum project commit -m "Transformed"
datum project checkout HEAD~1 -- source1 # restore a previous revision
datum project status # prints "modified source1"
datum project checkout source1 # restore the last revision
datum project export -f voc -- --save-media
```
