# Compare

## Compare datasets

This command compares two datasets and saves the results in the specified directory.

Datasets can be compared using different methods:
- [`table`](#table) - Generate a compare table mainly based on dataset statistics
- [`equality`](#equality) - Annotations are compared to be equal
- [`distance`](#distance) - A distance metric is used

## Usage
```console
datum compare [-h] [-o DST_DIR] [-m METHOD] [--overwrite] [--iou-thresh IOU_THRESH]
                [-f FORMAT] [-iia IGNORE_ITEM_ATTR] [-ia IGNORE_ATTR] [-if IGNORE_FIELD]
                [--match-images] [--all]
                <dataset1> <dataset2>
```

Compares two datasets by specifying their paths.

\<dataset\> - [a dataset path](../explanation/concept.rst#dataset-path-concepts) with optional format specification.

Parameters:
- `<dataset1>` (string) - The first dataset path to be compared
- `<dataset2>` (string) - The second dataset path to be compared
- `-m, --method` (string) - Comparison method, one of table, equality, distance (default: table)
- `-o, --output-dir` (string) - Directory to save comparison results (default: generate automatically)
- `--overwrite` - Overwrite existing files in the save directory
- `-f, --format` (string) - Output format, one of simple, tensorboard (default: simple)
- `-h, --help` - Print the help message and exit

- Distance comparison options:
  - `--iou-thresh` (number) - IoU match threshold for shapes (default: 0.5)

- Equality comparison options:
  - `-iia, --ignore-item-attr` (string) - Ignore item attribute (repeatable)
  - `-ia, --ignore-attr` (string) - Ignore annotation attribute (repeatable)
  - `-if, --ignore-field` (string) - Ignore annotation field (repeatable, default: ['id', 'group'])
  - `--match-images` - Match dataset items by image pixels instead of ids
  - `--all` - Include matches in the output
    Default is `id` and `group`
  - `--match-images` - Match dataset items by image pixels instead of ids
  - `--all` - Include matches in the output. By default, only differences are
    printed.

### Supported methods
#### `table`
This method allows comparing datasets based on dataset statistics and provides the results in a tabular format. The result report is saved in the formats of `table_compare.json` and `table_compare.txt`, each containing information for "High-level comparison," "Mid-level comparison," and "Low-level comparison."

Firstly, the "High-level comparison" provides information regarding the format, classes, images, and annotations for each dataset. For example:
```bash
+--------------------------+---------+---------------------+
| Field                    | First   | Second              |
+==========================+=========+=====================+
| Format                   | coco    | voc                 |
+--------------------------+---------+---------------------+
| Number of classes        | 2       | 4                   |
+--------------------------+---------+---------------------+
| Common classes           | a, b    | a, b                |
+--------------------------+---------+---------------------+
| Classes                  | a, b    | a, b, background, c |
+--------------------------+---------+---------------------+
| Images count             | 1       | 1                   |
+--------------------------+---------+---------------------+
| Unique images count      | 1       | 1                   |
+--------------------------+---------+---------------------+
| Repeated images count    | 0       | 0                   |
+--------------------------+---------+---------------------+
| Annotations count        | 1       | 2                   |
+--------------------------+---------+---------------------+
| Unannotated images count | 0       | 0                   |
+--------------------------+---------+---------------------+
```

Secondly, the "Mid-level comparison" displays image means, standard deviations, and label distributions for each subset in the datasets. For example:
```bash
+--------------------------+--------------------------+--------------------------+
| Field                    | First                    | Second                   |
+==========================+==========================+==========================+
| train - Image Mean (RGB) | 1.00,   1.00,   1.00     | 1.00,   1.00,   1.00     |
+--------------------------+--------------------------+--------------------------+
| train - Image Std (RGB)  | 0.00,   0.00,   0.00     | 0.00,   0.00,   0.00     |
+--------------------------+--------------------------+--------------------------+
| Label - a                | imgs: 1, percent: 1.0000 |                          |
+--------------------------+--------------------------+--------------------------+
| Label - b                |                          | imgs: 1, percent: 0.5000 |
+--------------------------+--------------------------+--------------------------+
| Label - background       |                          |                          |
+--------------------------+--------------------------+--------------------------+
| Label - c                |                          | imgs: 1, percent: 0.5000 |
+--------------------------+--------------------------+--------------------------+
```

The results are stored in the formats of `table_compare.json` and `table_compare.txt`.

- Compare two datasets for table, specify formats
  ```console
  datum compare <path/to/dataset1/>:voc <path/to/dataset2/>:coco
  ```

#### `equality`
This method shows how identical items and annotations are between datasets. It indicates the number of unmatched items in each dataset, as well as the quantity of conflicting items and the counts of matching and mismatching annotations. For example:
```bash
Found:
The first dataset has 10 unmatched items
The second dataset has 100 unmatched items
1 item conflicts
10 matching annotations
0 mismatching annotations
```
The detailed information is stored in `equality_compare.json`. If you'd like to review the specific details, please refer to this file.

Annotations are compared to be equal
- Compare two datasets for equality, exclude annotation groups
  and the `is_crowd` attribute from comparison
  ```console
  datum compare <dataset1> <dataset2> -m equality -if group -ia is_crowd
  ```

#### `distance`
This method demonstrates the consistency of annotations between dataset items. It presents the count of matched annotations between two items in a tabular format, comparing the numbers of label, bbox, polygon, and mask annotations. Additionally, it generates a confusion matrix for each annotation type, which is saved in the form of `<annotation_type>_confusion.png`. It also highlights cases where mismatching labels exist. For example:
```bash
Datasets have mismatching labels:
  #0: a != background
  #1: b != a
  #2:  < b
  #3:  < c
```

- Compare two datasets by distance, match boxes if IoU > 0.7,
  save results to TensorBoard
  ```console
  datum compare <dataset1> <dataset2> -m distance -f tensorboard --iou-thresh 0.7 -o compare/
  ```
