# Compare

## Compare datasets

This command compares two datasets and saves the results in the specified directory. The current project is considered to be "ground truth".

Datasets can be compared using different methods:
- [`table`](#table) - Generate a compare table mainly based on dataset statistics
- [`equality`](#equality) - Annotations are compared to be equal
- [`distance`](#distance) - A distance metric is used

This command has multiple forms:
```console
1) datum compare <revpath>
2) datum compare <revpath> <revpath>
```

1 - Compares the current project's main target (`project`) in the working tree with the specified dataset.
2 - Compares two specified datasets.

\<revpath\> - [a dataset path or a revision path](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).

Usage:
```console
datum compare [-h] [-o DST_DIR] [-m METHOD] [--overwrite] [-p PROJECT_DIR]
           [--iou-thresh IOU_THRESH] [-f FORMAT]
           [-iia IGNORE_ITEM_ATTR] [-ia IGNORE_ATTR] [-if IGNORE_FIELD]
           [--match-images] [--all]
           first_target [second_target]
```

Parameters:
- `<target>` (string) - Target [dataset revpaths](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts)
- `-m, --method` (string) - Comparison method.
- `-o, --output-dir` (string) - Output directory. By default, a new directory
  is created in the current directory.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

- Distance comparison options:
  - `--iou-thresh` (number) - The IoU threshold for spatial annotations
    (default is 0.5).
  - `-f, --format` (string) - Output format, one of `simple`
    (text files and images) and `tensorboard` (a TB log directory)

- Equality comparison options:
  - `-iia, --ignore-item-attr` (string) - Ignore an item attribute (repeatable)
  - `-ia, --ignore-attr` (string) - Ignore an annotation attribute (repeatable)
  - `-if, --ignore-field` (string) - Ignore an annotation field (repeatable)
    Default is `id` and `group`
  - `--match-images` - Match dataset items by image pixels instead of ids
  - `--all` - Include matches in the output. By default, only differences are
    printed.

### Support methods
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

Lastly, the "Low-level comparison" uses ShiftAnalyzer to show Covariate shift and Label shift between the two datasets. For example:
```bash
+-----------------+---------+
| Field           |   Value |
+=================+=========+
| Covariate shift |       0 |
+-----------------+---------+
| Label shift     |     nan |
+-----------------+---------+
```
The results are stored in the formats of `table_compare.json` and `table_compare.txt`.

- Compare the current working tree with a dataset in COCO data format to create the tabular report
  ```console
  datum compare <path/to/dataset2/>:coco
  ```

- Compare two projects for table
  ```console
  datum compare <path/to/other/project/> -if group -ia is_crowd
  ```

- Compare two datasets for table, specify formats
  ```console
  datum compare <path/to/dataset1/>:voc <path/to/dataset2/>:coco
  ```

- Compare a source from a previous revision and a dataset for table
  ```console
  datum compare HEAD~2:source-2 <path/to/dataset2/>:yolo
  ```

- Compare a dataset with model inference
  ```console
  datum project create <...>
  datum project import <...>
  datum model add mymodel <...>
  datum transform <...> -o inference
  datum compare inference -o compare
  ```

#### `equality`
This method shows how identical items and annotations are between datasets. It indicates the number of unmatched items in each project (dataset), as well as the quantity of conflicting items and the counts of matching and mismatching annotations. For example:
```bash
Found:
The first project has 10 unmatched items
The second project has 100 unmatched items
1 item conflicts
10 matching annotations
0 mismatching annotations
```
The detailed information is stored in `equality_compare.json`. If you'd like to review the specific details, please refer to this file.

Annotations are compared to be equal
- Compare two projects for equality, exclude annotation groups
  and the `is_crowd` attribute from comparison
  ```console
  datum compare <path/to/other/project/> -m equality -if group -ia is_crowd
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

- Compare two projects by distance, match boxes if IoU > 0.7,
  save results to TensorBoard
  ```console
  datum compare <path/to/other/project/> -m distance -f tensorboard --iou-thresh 0.7 -o compare/
  ```
