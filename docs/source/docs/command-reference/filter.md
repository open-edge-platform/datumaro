# Filter

## Filter datasets

This command allows to extract a sub-dataset from a dataset. The new dataset
includes only items satisfying some condition. The XML [XPath](https://devhints.io/xpath)
is used as a query format.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. When
updating in-place, use the `--overwrite` parameter (in-place
updates fail by default to prevent data loss).

There are several filtering modes available (the `-m/--mode` parameter).
Supported modes:
- `i`, `items`
- `a`, `annotations`
- `i+a`, `a+i`, `items+annotations`, `annotations+items`

When filtering annotations, use the `items+annotations`
mode to point that annotation-less dataset items should be
removed, otherwise they will be kept in the resulting dataset.
To select an annotation, write an XPath that returns `annotation`
elements (see examples).

Item representations can be printed with the `--dry-run` parameter:

``` xml
<item>
  <id>290768</id>
  <subset>minival2014</subset>
  <image>
    <width>612</width>
    <height>612</height>
    <depth>3</depth>
  </image>
  <annotation>
    <id>80154</id>
    <type>bbox</type>
    <label_id>39</label_id>
    <x>264.59</x>
    <y>150.25</y>
    <w>11.19</w>
    <h>42.31</h>
    <area>473.87</area>
  </annotation>
  <annotation>
    <id>669839</id>
    <type>bbox</type>
    <label_id>41</label_id>
    <x>163.58</x>
    <y>191.75</y>
    <w>76.98</w>
    <h>73.63</h>
    <area>5668.77</area>
  </annotation>
  ...
</item>
```

Usage:

```console
datumaro filter [-h] [-e FILTER] [-m MODE] [--dry-run] [-o DST_DIR] [--overwrite] target
```

Parameters:
- `target` (string) - Target dataset path with optional format (e.g., 'dataset/' or 'dataset/:voc')
- `-e, --filter` (string) - XML XPath filter expression for dataset items
- `-m, --mode` (string) - Filter mode (options: items, annotations, items+annotations; default: items)
- `--dry-run` - Print XML representations to be filtered and exit
- `-o, --output-dir` (string) - Output directory. If not specified, the results will be saved inplace
- `--overwrite` - Overwrite existing files in the save directory
- `-h, --help` - Print the help message and exit

Examples:
- Extract a dataset with images with `width` < `height`
  ```console
  datumaro filter -e '/item[image/width < image/height]' dataset/
  ```

- Extract a dataset with images of the `train` subset
  ```console
  datumaro filter -e '/item[subset="train"]' dataset/
  ```

- Extract a dataset with only large annotations of the `cat` class and any non-`persons`
  ```console
  datumaro filter --mode annotations \
    -e '/item/annotation[(label="cat" and area > 99.5) or label!="person"]' dataset/
  ```

- Extract a dataset with non-occluded annotations, remove empty images
  ```console
  datumaro filter -m i+a -e '/item/annotation[occluded="False"]' dataset/ -o output_dir
  ```

- Extract a dataset composed solely of items containing annotations
  ```console
  datumaro filter -e '/item[annotation]' dataset/
  ```
  The `item[annotation]` checks if there is a child named `annotation` within the `item` node.
