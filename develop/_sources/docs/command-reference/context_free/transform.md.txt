# Transform

## Transform Dataset

Often datasets need to be modified during preparation for model training and
experimenting. In trivial cases it can be done manually - e.g. image renaming
or label renaming. However, in more complex cases even simple modifications
can require too much efforts, distracting the user from the real work.
Datumaro provides the `datum transform` command to help in such cases.

This command allows to modify dataset images or annotations all at once.

> This command is designed for batch dataset processing, so if you only
> need to modify few elements of a dataset, you might want to use
> other approaches for better performance. A possible solution can be
> a simple script, which uses [Datumaro API](../../explanation/architecture).

The command can be applied to a dataset or a project build target,
a stage or the combined `project` target, in which case all the project
targets will be affected. A build tree stage will be recorded
if `--stage` is enabled, and the resulting dataset(-s) will be
saved if `--apply` is enabled.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. When
updating in-place, use the `--overwrite` parameter (in-place
updates fail by default to prevent data loss), unless a project
target is modified.

The current project (`-p/--project`) is also used as a context for
plugins, so it can be useful for dataset paths having custom formats.
When not specified, the current project's working tree is used.

Usage:

```console
datum transform [-h] -t TRANSFORM [-o DST_DIR] [--overwrite]
  [-p PROJECT_DIR] [--stage STAGE] [--apply APPLY] [target] [-- EXTRA_ARGS]
```

Parameters:
- `<target>` (string) - Target
  [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
  By default, transforms all targets of the current project.
- `-t, --transform` (string) - Transform method name
- `--stage` (bool) - Include this action as a project build step.
  If true, this operation will be saved in the project
  build tree, allowing to reproduce the resulting dataset later.
  Applicable only to main project targets (i.e. data sources
  and the `project` target, but not intermediate stages). Enabled by default.
- `--apply` (bool) - Run this command immediately. If disabled, only the
  build tree stage will be written. Enabled by default.
- `-o, --output-dir` (string) - Output directory. Can be omitted for
  main project targets (i.e. data sources and the `project` target, but not
  intermediate stages) and dataset targets. If not specified, the results
  will be saved inplace.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `<extra args>` - The list of extra transformation parameters. Should be
  passed after the `--` separator after the main command arguments. See
  transform descriptions for info about extra parameters. Use the `--help`
  option to print parameter info.

Examples:

- Split a VOC-like dataset randomly
  ```console
  datum transform -t random_split --overwrite path/to/dataset:voc
  ```

- Rename images in a project data source by a regex from `frame_XXX` to `XXX`

  **NOTE:** Please use double quotes (`"`) for regex representation. Check [Reason to use double quotes](https://stackoverflow.com/questions/51080215/differences-between-single-and-double-quotes-in-cmd).

  ```console
  datum project create <...>
  datum project import <...> -n source-1
  datum transform -t rename source-1 -- -e "|^frame_||"
  ```

### Built-in transforms

Basic dataset item manipulations:
- [`rename`](#rename) - Renames dataset items by regular expression
- [`id_from_image_name`](#id_from_image_name) - Renames dataset
  items to their image filenames
- [`reindex`](#reindex) - Renames dataset items with numbers
- [`sort`](#sort) - Sort dataset items
- [`ndr`](#ndr) - Removes duplicated images from dataset
- [`relevancy_sampler`](#relevancy_sampler) - Leaves only the most
  important images
  (requires model inference results)
- [`random_sampler`](#random_sampler) - Leaves no more than k items
  from the dataset randomly
- [`label_random_sampler`](#random_label_sampler) - Leaves at least
  k images with annotations per class
- [`resize`](#resize) - Resizes images and annotations in the dataset
- [`remove_images`](#remove_images) - Removes specific images
- [`remove_annotations`](#remove_annotations) - Removes annotations
- [`remove_attributes`](#remove_attributes) - Removes attributes
- [`astype_annotations`](#astype_annotations) - Transforms annotation types
- [`pseudo_labeling`](#pseudo_labeling) - Generates pseudo labels for unlabeled data
- [`correct`](#correct) - Corrects annotation types
- [`clean`](#clean) - Removes noisy data for tabular dataset

Subset manipulations:
- [`random_split`](#random_split) - Splits dataset into subsets
  randomly
- [`split`](#split) - Splits dataset into subsets for classification,
  detection, segmentation or re-identification
- [`map_subsets`](#map_subsets) - Renames and removes subsets

Annotation manipulations:
- [`remap_labels`](#remap_labels) - Renames, adds or removes
  labels in dataset
- [`project_labels`](#project_labels) - Sets dataset labels to
  the requested sequence
- [`shapes_to_boxes`](#shapes_to_boxes) - Replaces spatial
  annotations with bounding boxes
- [`boxes_to_masks`](#boxes_to_masks) - Converts bounding boxes
  to instance masks
- [`polygons_to_masks`](#polygons_to_masks) - Converts polygons
  to instance masks
- [`masks_to_polygons`](#masks_to_polygons) - Converts instance
  masks to polygons
- [`anns_to_labels`](#anns_to_labels) - Replaces annotations having
  labels with label annotations
- [`merge_instance_segments`](#merge_instance_segments) - Merges
  grouped spatial annotations into a mask
- [`crop_covered_segments`](#crop_covered_segments) - Removes
  occluded segments of covered masks
- [`bbox_value_decrement`](#bbox_value_decrement) - Subtracts
  1 from bbox coordinates

#### `rename`

Renames items in the dataset. Supports regular expressions.
The first character in the expression is a delimiter for
the pattern and replacement parts. Replacement part can also
contain `str.format` replacement fields with the `item`
(of type `DatasetItem`) object available.

Usage:
```console
rename [-h] [-e REGEX]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-e`, `--regex` (string) - Regex for renaming in the form
  `<sep><search><sep><replacement><sep>`

Examples:
- Replace 'pattern' with 'replacement'
  ```console
  datum transform -t rename -- -e "|pattern|replacement|"
  ```

- Remove the `frame_` prefix from item ids
  ```console
  datum transform -t rename -- -e "|^frame_|"
  ```

- Collect images from subdirectories into the base image directory using regex
  ```console
  datum transform -t rename -- -e "|^((.+[/\\])*)?(.+)$|\2|"
  ```

- Add subset prefix to images
  ```console
  datum transform -t rename -- -e "|(.*)|{item.subset}_\1|"
  ```

#### `id_from_image_name`

Renames items in the dataset based on the image file name, excluding the extension.
When 'ensure_unique' is enabled, a random suffix is appended to ensure each identifier is unique
in cases where the image name is not distinct. By default, the random suffix is three characters long,
but this can be adjusted with the 'suffix_length' parameter.

Usage:
```console
id_from_image_name [-h] [-u] [-l SUFFIX_LENGTH]
```

Optional arguments:
- `-h`, `--help` (flag) - show this help message and exit
- `-u`, `--ensure_unique` (flag) - Appends a random suffix to ensure each identifier is unique if the image name is duplicated
- `-l`, `--suffix_length` (int) - Alters the length of the random suffix if the `ensure_unique` is enabled(default: 3)

Examples:
- Renames items without duplication check
  ```console
  datum transform -t id_from_image_name
  ```

- Renames items with duplication check
  ```console
  datum transform -t id_from_image_name -- --ensure_unique
  ```

- Renames items with duplication check and alters the suffix length(default: 3)
  ```console
  datum transform -t id_from_image_name -- --ensure_unique --suffix_length 2
  ```

#### `reindex`

Replaces dataset item IDs with sequential indices.

Usage:
```console
reindex [-h] [-s START]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--start` (int) - Start value for item ids (default: 1)

#### `sort`

Sorts dataset items.

Usage:
```console
reindex [-h] [-s START]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--key` (string/callable) - key function to sort (default: sorted by `item.id`)

Examples:
- Sort by id converted into integer
  ```console
  datum transform -t ndr -- --key "lambda item: int(item.id)"
  ```

#### `ndr`

Removes near-duplicated images in subset.

Remove duplicated images from a dataset. Keep at most `-k/--num_cut`
resulting images.

Available oversampling policies (the `-e` parameter):
- `random` - sample from removed data randomly
- `similarity` - sample from removed data with ascending similarity score

Available undersampling policies (the `-u` parameter):
- `uniform` - sample data with uniform distribution
- `inverse` - sample data with reciprocal of the number of number of
  items with the same similarity

Usage:
```console
ndr [-h] [-w WORKING_SUBSET] [-d DUPLICATED_SUBSET] [-a {gradient}]
    [-k NUM_CUT] [-e {random,similarity}] [-u {uniform,inverse}] [-s SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-w`, `--working_subset` (str) - Name of the subset to operate
  (default: `None`)
- `-d`, `--duplicated_subset` (str) - Name of the subset for the removed
  data after NDR runs (default: duplicated)
- `-a`, `--algorithm` (one of: `gradient`) - Name of the algorithm to
  use (default: `gradient`)
- `-k`, `--num_cut` (int) - Maximum output dataset size
- `-e`, `--over_sample` (one of: `random`, `similarity`) - The policy to use
  when `num_cut` is bigger than result length (default: `random`)
- `-u`, `--under_sample` (one of: `uniform`, `inverse`) - The policy to use
  when `num_cut` is smaller than result length (default: `uniform`)
- `-s`, `--seed` (int) - Random seed

Examples:
- Apply NDR, return no more than 100 images
  ```console
  datum transform -t ndr -- \
    --working_subset train
    --algorithm gradient
    --num_cut 100
    --over_sample random
    --under_sample uniform
  ```

#### `relevancy_sampler`

Sampler that analyzes model inference results on the dataset
and picks the most relevant samples for training.

Creates a dataset from the `-k/--count` hardest items for a model.
The whole dataset or a single subset will be split into the `sampled`
and `unsampled` subsets based on the model confidence. The dataset
**must** contain model confidence values in the `scores` attributes
of annotations.

There are five methods of sampling (the `-m/--method` option):
- `topk` - Return the k items with the highest uncertainty data
- `lowk` - Return the k items with the lowest uncertainty data
- `randk` - Return random k items
- `mixk` - Return a half using topk, and the other half using lowk method
- `randtopk` - Select 3*k items randomly, and return the topk among them

Notes:
- Each image's inference result must contain the probability for
  all classes (in the `scores` attribute).
- Requesting a sample larger than the number of all images will return
  all images.

Usage:
```console
relevancy_sampler [-h] -k COUNT [-a {entropy}] [-i INPUT_SUBSET]
                  [-o SAMPLED_SUBSET] [-u UNSAMPLED_SUBSET]
                  [-m {topk,lowk,randk,mixk,randtopk}] [-d OUTPUT_FILE]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Number of items to sample
- `-a`, `--algorithm` (one of: `entropy`) - Sampling
  algorithm (default: `entropy`)
- `-i`, `--input_subset` (str) - Subset name to select sample
  from (default: `None`)
- `-o`, `--sampled_subset` (str) - Subset name to put sampled data
  to (default: `sample`)
- `-u`, `--unsampled_subset` (str) - Subset name to put the
  rest data to (default: `unsampled`)
- `-m`, `--sampling_method` (one of: `topk`, `lowk`, `randk`, `mixk`,
  `randtopk`) - Sampling method (default: `topk`)
- `-d`, `--output_file` (path) - A `.csv` file path to dump sampling results

Examples:
- Select the most relevant data subset of 20 images
  based on model certainty, put the result into `sample` subset
  and put all the rest into `unsampled` subset, use `train` subset
  as input. The dataset **must** contain model confidence values in the `scores`
  attributes of annotations.
  ```console
  datum transform -t relevancy_sampler -- \
    --algorithm entropy \
    --subset_name train \
    --sample_name sample \
    --unsampled_name unsampled \
    --sampling_method topk -k 20
  ```

#### `random_sampler`

Sampler that keeps no more than required number of items in the dataset.

Notes:
- Items are selected uniformly (tries to keep original item distribution
  by subsets)
- Requesting a sample larger than the number of all images will return
  all images

Usage:
```console
random_sampler [-h] -k COUNT [-s SUBSET] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Maximum number of items to sample
- `-s`, `--subset` (str) - Limit changes to this subset
  (default: affect all dataset)
- `--seed` (int) - Initial value for random number generator

Examples:
- Select subset of 20 images randomly
  ```console
  datum transform -t random_sampler -- -k 20
  ```

- Select subset of 20 images, modify only `train` subset
  ```console
  datum transform -t random_sampler -- -k 20 -s train
  ```

#### `random_label_sampler`

Sampler that keeps at least the required number of annotations of
each class in the dataset for each subset separately.

Consider using the "stats" command to get class distribution in the dataset.

Notes:
- Items can contain annotations of several selected classes
  (e.g. 3 bounding boxes per image). The number of annotations in the
  resulting dataset varies between `max(class counts)` and `sum(class counts)`
- If the input dataset does not has enough class annotations, the result
  will contain only what is available
- Items are selected uniformly
- For reasons above, the resulting class distribution in the dataset may
  not be the same as requested
- The resulting dataset will only keep annotations for classes with
  specified `count` > 0

Usage:
```console
label_random_sampler [-h] -k COUNT [-l LABEL_COUNTS] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Minimum number of annotations of each class
- `-l`, `--label` (str; repeatable) - Minimum number of annotations of
  a specific class. Overrides the `-k/--count` setting for the class.
  The format is `<label_name>:<count>`
- `--seed` (int) - Initial value for random number generator

Examples:
- Select a dataset with at least 10 images of each class
  ```console
  datum transform -t label_random_sampler -- -k 10
  ```

- Select a dataset with at least 20 `cat` images, 5 `dog`, 0 `car` and 10 of each
  unmentioned class
  ```console
  datum transform -t label_random_sampler -- \
    -l cat:20 \ # keep 20 images with cats
    -l dog:5 \ # keep 5 images with dogs
    -l car:0 \ # remove car annotations
    -k 10 # for remaining classes
  ```

#### `resize`

Resizes images and annotations in the dataset to the specified size.
Supports upscaling, downscaling and mixed variants.

Usage:
```console
resize [-h] [-dw WIDTH] [-dh HEIGHT]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-dw`, `--width` (int) - Destination image width
- `-dh`, `--height` (int) - Destination image height

Examples:
- Resize all images to 256x256 size
  ```
  datum transform -t resize -- -dw 256 -dh 256
  ```

#### `remove_images`

Removes specific dataset items by their ids.

Usage:
```console
remove_images [-h] [--id IDs]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--id` (str) - Item id to remove. Id is '<name>:<subset>' pair (repeatable)

Examples:
- Remove specific images from the dataset
  ```console
  datum transform -t remove_images -- --id 'image1:train' --id 'image2:test'
  ```

#### `remove_annotations`

Allows to remove annotations on specific dataset items.

Can be useful to clean the dataset from broken or unnecessary annotations.

Usage:
```console
remove_annotations [-h] [--id IDs]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--id` (str) - Item id to clean from annotations. Id is '<name>:<subset>' pair.
  If not specified, removes all annotations (repeatable)

Examples:
- Remove annotations from specific items in the dataset
  ```console
  datum transform -t remove_annotations -- --id 'image1:train' --id 'image2:test'
  ```

#### `remove_attributes`

Allows to remove item and annotation attributes in a dataset.

Can be useful to clean the dataset from broken or unnecessary attributes.

Usage:
```console
remove_attributes [-h] [--id IDs] [--attr ATTRIBUTE_NAME]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--id` (str) - Image id to clean from annotations. Id is '<name>:<subset>' pair.
  If not specified, affects all items and annotations (repeatable)
- `-a`, `--attr` (flag) - Attribute name to be removed. If not specified,
  removes all attributes (repeatable)

Examples:
- Remove the `is_crowd` attribute from dataset
  ```console
  datum transform -t remove_attributes -- \
    --attr 'is_crowd'
  ```

- Remove the `occluded` attribute from annotations of
  the `2010_001705` item in the `train` subset
  ```console
  datum transform -t remove_attributes -- \
    --id '2010_001705:train' --attr 'occluded'
  ```

#### `astype_annotations`

Enables the conversion of annotation types for the categories and individual items within a dataset.
This transform only supports tabular datasets. If you want to change annotation types in datasets of other types, please use a different transform.

Based on default setting it transforms the annotation types, changing them to 'Label' if they are categorical,
and to 'Caption' if they are of type string, float, or integer.
If you specifically set mapping, change annotation types based on the mapping.

Usage:
```console
astype_annotations [-h] [--mapping MAPPING]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--mapping` (str) - Annotations type in the form of: '<src>:<dst>' (repeatable)

Examples:
- Convert type of `title` and `rating` annotation
  ```console
  datum transform -t astype_annotations -- \
    --mapping 'title:text,rating:label'
  ```

#### `random_split`

Joins all subsets into one and splits the result into few parts.
It is expected that item ids are unique and subset ratios sum up to 1.

Usage:
```console
random_split [-h] [-s SPLITS] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--subset` (str, repeatable) - Subsets in the form: '<subset>:<ratio>'
  (repeatable, default: {`train`: 0.67, `test`: 0.33})
- `--seed` (int) - Random seed

Examples:
- Split a dataset randomly to `train` and `test` subsets, ratio is 2:1
  ```console
  datum transform -t random_split -- --subset train:.67 --subset test:.33
  ```

#### `split`

Splits a dataset for model training, using task information:

- classification splits
Splits dataset into subsets (train/val/test) in class-wise manner.
Splits dataset images in the specified ratio, keeping the initial
class distribution.

- detection & segmentation splits
Each image can have multiple object annotations - bbox, mask, polygon.
Since an image shouldn't be included in multiple subsets at the same time,
and image annotations shouldn't be split, in general, dataset annotations are
unlikely to be split exactly in the specified ratio.
This split tries to split dataset images as close as possible to the specified
ratio, keeping the initial class distribution.

- reidentification splits
In this task, the test set should consist of images of unseen people or
objects during the training phase.
This function splits a dataset in the following way:
1. Splits the dataset into `train + val` and `test` sets
  based on person or object ID.
2. Splits `test` set into `test-gallery` and `test-query` sets
  in class-wise manner.
3. Splits the `train + val` set into `train` and `val` sets
  in the same way.
The final subsets would be `train`, `val`, `test-gallery` and `test-query`.

Notes:
- Each image is expected to have only one `Annotation`. Unlabeled or
  multi-labeled images will be split into subsets randomly.
- If Labels also have attributes, also splits by attribute values.
- If there is not enough images in some class or attributes group,
  the split ratio can't be guaranteed.

In reidentification task,
- Object ID can be described by Label, or by attribute (`--attr` parameter)
- The splits of the test set are controlled by `--query` parameter
  Gallery ratio would be `1.0 - query`.

Usage:
```console
split [-h] [-t {classification,detection,segmentation,reid}]
      [-s SPLITS] [--query QUERY] [--attr ATTR_FOR_ID] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-t`, `--task` (one of: `classification`, `detection`, `segmentation`,
  `reid`) - Dataset task (default: `classification`)
- `-s`, `--subset` (str; repeatable) - Subsets in the form: '<subset>:<ratio>'
  (default: {`train`: 0.5, `val`: 0.2, `test`: 0.3})
- `--query` (float) - Query ratio in the test set (default: 0.5)
- `--attr` (str) - Attribute name representing the ID (default: use label)
- `--seed`(int) - Random seed

Examples:
- Split by ratio
  ```
  datum transform -t split -- -t classification \
    --subset train:.5 --subset val:.2 --subset test:.3

  datum transform -t split -- -t detection \
    --subset train:.5 --subset val:.2 --subset test:.3

  datum transform -t split -- -t segmentation \
    --subset train:.5 --subset val:.2 --subset test:.3

  datum transform -t split -- -t reid \
    --subset train:.5 --subset val:.2 --subset test:.3 --query .5
  ```

- Use `person_id` attribute for splitting
  ```console
  datum transform -t split -- -t detection --attr person_id
  ```

#### `map_subsets`

Renames subsets in the dataset.

Usage:
```console
map_subsets [-h] [-s MAPPING]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--subset` (str; repeatable) - Subset mapping of the form: `src:dst`

#### `remap_labels`

Changes labels in the dataset.

A label can be:
- renamed (and joined with existing) -
  when `--label <old_name>:<new_name>` is specified
- deleted - when `--label <name>:` is specified, or default action is `delete`
  and the label is not mentioned in the list. When a label
  is deleted, all the associated annotations are removed
- kept unchanged - when `--label <name>:<name>` is specified,
  or default action is `keep` and the label is not mentioned in the list
Annotations with no label are managed by the default action policy.

Usage:
```console
remap_labels [-h] [-l MAPPING] [--default {keep,delete}]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-l`, `--label` (str; repeatable) - Label in the form of: `<src>:<dst>`
- `--default` (one of: `keep`, `delete`) - Action for unspecified labels
  (default: `keep`)

Examples:
- Remove the `person` label (and corresponding annotations)
  ```console
  datum transform -t remap_labels -- -l person: --default keep
  ```

- Rename `person` to `pedestrian` and `human` to `pedestrian`, join annotations
  that had different classes under the same class id for `pedestrian`,
  don't touch other classes
  ```console
  datum transform -t remap_labels -- \
    -l person:pedestrian -l human:pedestrian --default keep
  ```

- Rename `person` to `car` and `cat` to `dog`, keep `bus`, remove others
  ```console
  datum transform -t remap_labels -- \
    -l person:car -l bus:bus -l cat:dog --default delete
  ```

#### `project_labels`

Changes the order of labels in the dataset from the existing
to the desired one, removes unknown labels and adds new labels.
Updates or removes the corresponding annotations.

Labels are matched by names (case dependent). Parent labels are
only kept if they are present in the resulting set of labels.
If new labels are added, and the dataset has mask colors defined,
new labels will obtain generated colors.

Useful for merging similar datasets, whose labels need to be aligned.

Usage:
```console
project_labels [-h] [-l DST_LABELS]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-l`, `--label` (str; repeatable) - Label name (ordered)

Examples:
- Set dataset labels to \[`person`, `cat`, `dog`\], remove others, add missing.
  Original labels (for example): `cat`, `dog`, `elephant`, `human`.
  New labels: `person` (added), `cat` (kept), `dog` (kept).
  ```console
  datum transform -t project_labels -- -l person -l cat -l dog
  ```

#### `shapes_to_boxes`

Converts spatial annotations (masks, polygons, polylines, points)
to enclosing bounding boxes.

Usage:
```console
shapes_to_boxes [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

Examples:
- Convert spatial annotations between each other
  ```console
  datum transform -t boxes_to_masks
  datum transform -t masks_to_polygons
  datum transform -t polygons_to_masks
  datum transform -t shapes_to_boxes
  ```

#### `boxes_to_masks`

Converts bounding boxes to masks.

Usage:
```console
boxes_to_masks [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `polygons_to_masks`

Converts polygons to masks.

Usage:
```console
polygons_to_masks [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `masks_to_polygons`

Converts masks to polygons.

Usage:
```console
masks_to_polygons [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `anns_to_labels`

Collects all labels from annotations (of all types) and transforms
them into a set of annotations of type `Label`

Usage:
```console
anns_to_labels [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `merge_instance_segments`

Replaces instance masks and, optionally, polygons with a single mask.
A group of annotations with the same group id is considered an "instance".
The largest annotation in the group is considered the group "head", so the
resulting mask takes properties from that annotation.

Usage:
```console
merge_instance_segments [-h] [--include-polygons]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--include-polygons` (flag) - Include polygons

#### `crop_covered_segments`

Sorts polygons and masks ("segments") according to `z_order`,
crops covered areas of underlying segments. If a segment is split
into several independent parts by the segments above, produces
the corresponding number of separate annotations joined into a group.

Usage:
```console
crop_covered_segments [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `bbox_value_decrement`

Subtracts one from the coordinates of bounding boxes

Usage:
```console
bbox_values_decrement [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

#### `pseudo_labeling`

Assigns pseudo-labels to items in a dataset based on their similarity to predefined labels. This class is useful for semi-supervised learning when dealing with missing or uncertain labels.

The process includes:

- Similarity Computation: Uses hashing techniques to compute the similarity between items and predefined labels.
- Pseudo-Label Assignment: Assigns the most similar label as a pseudo-label to each item.

Attributes:

- `extractor` (IDataset) - Provides access to dataset items and their annotations.
- `labels` (Optional[List[str]]) - List of predefined labels for pseudo-labeling. Defaults to all available labels if not provided.
- `explorer` (Optional[Explorer]) - Computes hash keys for items and labels. If not provided, a new Explorer is created.

Usage:
```console
pseudo_labeling [-h] [--labels LABELS]

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--labels` (str) - Comma-separated list of label names for pseudo-labeling

Examples:
- Assign pseudo-labels based on predefined labels
  ```console
  datum transform -t pseudo_labeling -- --labels 'label1,label2'
  ```

#### `correct`

Correct the dataset from a validation report

Usage:
```console
correct [-h] [-r REPORT_PATH]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-r`, `--reports` (str) - A validation report from a 'validate' CLI (default=validation_reports.json)

#### `clean`

Refines and preprocesses media items in a dataset, focusing on string, numeric, and categorical data. This transform is designed to clean and improve the quality of the data, making it more suitable for analysis and modeling.

The cleaning process includes:

- String Data: Removes unnecessary characters using NLP techniques.
- Numeric Data: Identifies and handles outliers and missing values.
- Categorical Data: Cleans and refines categorical information.

Usage:
```console
clean [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

Examples:
- Clean and preprocess dataset items
  ```console
  datum transform -t clean
  ```
