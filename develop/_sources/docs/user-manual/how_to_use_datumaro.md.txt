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

Example: convert dataset, transform and filter data

``` bash
# Convert a COCO dataset to VOC format
datum convert -i <path/to/dataset> -if coco -f voc -o <output_dir> -- --save-media

# Transform dataset (e.g., convert shapes to bounding boxes)
datum transform -i <path/to/dataset> -t shapes_to_boxes -o <output_dir>

# Filter dataset items (keep only items with cat or dog annotations)
datum filter -e '/item/annotation[label="cat" or label="dog"]' -m i+a <path/to/dataset> -o <output_dir>

# Merge multiple datasets
datum merge <path/to/dataset1> <path/to/dataset2> -f voc -o <output_dir> -- --save-media
```
