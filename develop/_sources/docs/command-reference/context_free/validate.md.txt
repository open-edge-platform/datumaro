# Validate

## Validate Dataset

This command inspects annotations with respect to the task type
and stores the results in JSON file.

The task types supported are `classification`, `detection`, `segmentation` and
`tabular` (the `-t/--task-type` parameter).

The validation result contains
- `annotation statistics` based on the task type
- `validation reports`, such as
  - items not having annotations
  - items having undefined annotations
  - imbalanced distribution in class/attributes
  - too small or large values
- `summary`

Usage:

```console
datum validate [-h] -t TASK [-s SUBSET_NAME] [-p PROJECT_DIR]
               [target] [-- EXTRA_ARGS]
```

Parameters:
- `<target>` (string) - Target
  [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
  By default, validates the current project.
- `-t, --task-type` (string) - Task type for validation
- `-s, --subset` (string) - Dataset subset to be validated
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `<extra args>` - The list of extra validation parameters. Should be passed
  after the `--` separator after the main command arguments:
  - `-fs, --few-samples-thr` (number) - The threshold for giving a warning
    for minimum number of samples per class
  - `-ir, --imbalance-ratio-thr` (number) - The threshold for giving
    imbalance data warning
  - `-m, --far-from-mean-thr` (number) - The threshold for giving
    a warning that data is far from mean
  - `-dr, --dominance-ratio-thr` (number) - The threshold for giving
    a warning bounding box imbalance
  - `-k, --topk-bins` (number) - The ratio of bins with the highest
    number of data to total bins in the histogram

Examples:
- Validate a project's subset as a classification dataset
  ```console
  datum validate -t classification -s subset
  ```

- Give warning when imbalance ratio of data with classification task
  over 40
  ```console
  datum validate -p <path/to/project/> -t classification -- -ir 40
  ```

### List of validation items (anomaly types)

| Anomaly Type | Description | Task Type |
| ------------ | ----------- | --------- |
| MissingLabelCategories | Metadata (ex. LabelCategories) should be defined | common |
| MissingAnnotation | No annotation found for an Item | common |
| MissingAttribute  | An attribute key is missing for an Item | common |
| MultiLabelAnnotations | Item needs a single label | classification |
| UndefinedLabel     | A label not defined in the metadata is found for an item | common |
| UndefinedAttribute | An attribute not defined in the metadata is found for an item | common |
| LabelDefinedButNotFound     | A label is defined, but not found actually | common |
| AttributeDefinedButNotFound | An attribute is defined, but not found actually | common |
| OnlyOneLabel          | The dataset consists of only label | common |
| OnlyOneAttributeValue | The dataset consists of only attribute value | common |
| FewSamplesInLabel     | The number of samples in a label might be too low | common |
| FewSamplesInAttribute | The number of samples in an attribute might be too low | common |
| ImbalancedLabels    | There is an imbalance in the label distribution | common |
| ImbalancedAttribute | There is an imbalance in the attribute distribution | common |
| ImbalancedDistInLabel     | Values (ex. bbox width) are not evenly distributed for a label | detection, segmentation |
| ImbalancedDistInAttribute | Values (ex. bbox width) are not evenly distributed for an attribute | detection, segmentation |
| NegativeLength | The width or height of bounding box is negative | detection |
| InvalidValue | There's invalid (ex. inf, nan) value for bounding box info. | detection |
| FarFromLabelMean | An annotation has an too small or large value than average for a label | detection, segmentation |
| FarFromAttrMean  | An annotation has an too small or large value than average for an attribute | detection, segmentation |
| BrokenAnnotation  | Some annotations are not defined for an item | tabular |
| EmptyLabel  | A value of the label column is not defined for an item | tabular |
| EmptyCaption  | A value of the caption column is not defined for an item | tabular |
| FewSamplesInCaption | The number of samples in a caption might be too low | tabular |
| RedundanciesInCaption | Redundancies of an caption for an item | tabular |
| ImbalancedCaptions  | There is an imbalance in the caption distribution | tabular |
| ImbalancedDistInCaption  | Values are not evenly distributed for a caption only if caption is number | tabular |
| FarFromCaptionMean | An annotation has an too small or large value than average for a caption only if caption is number | tabular |
| OutlierInCaption | An annotation has an outlier value based on Interquartile Range Method only if caption is number | tabular |

Validation Result Format:

<details>

```console
{
    'statistics': {
        ## common statistics
        'label_distribution': {
            'defined_labels': <dict>,   # <label:str>: <count:int>
            'undefined_labels': <dict>
            # <label:str>: {
            #     'count': <int>,
            #     'items_with_undefined_label': [<item_key>, ]
            # }
        },
        'attribute_distribution': {
            'defined_attributes': <dict>,
            # <label:str>: {
            #     <attribute:str>: {
            #         'distribution': {<attr_value:str>: <count:int>, },
            #         'items_missing_attribute': [<item_key>, ]
            #     }
            # }
            'undefined_attributes': <dict>
            # <label:str>: {
            #     <attribute:str>: {
            #         'distribution': {<attr_value:str>: <count:int>, },
            #         'items_with_undefined_attr': [<item_key>, ]
            #     }
            # }
        },
        'total_ann_count': <int>,
        'items_missing_annotation': <list>, # [<item_key>, ]

        ## statistics for classification task
        'items_with_multiple_labels': <list>, # [<item_key>, ]

        ## statistics for detection task
        'items_with_invalid_value': <dict>,
        # '<item_key>': {<ann_id:int>: [ <property:str>, ], }
        # - properties: 'x', 'y', 'width', 'height',
        #               'area(wxh)', 'ratio(w/h)', 'short', 'long'
        # - 'short' is min(w,h) and 'long' is max(w,h).
        'items_with_negative_length': <dict>,
        # '<item_key>': { <ann_id:int>: { <'width'|'height'>: <value>, }, }
        'bbox_distribution_in_label': <dict>, # <label:str>: <bbox_template>
        'bbox_distribution_in_attribute': <dict>,
        # <label:str>: {<attribute:str>: { <attr_value>: <bbox_template>, }, }
        'bbox_distribution_in_dataset_item': <dict>,
        # '<item_key>': <bbox count:int>

        ## statistics for segmentation task
        'items_with_invalid_value': <dict>,
        # '<item_key>': {<ann_id:int>: [ <property:str>, ], }
        # - properties: 'area', 'width', 'height'
        'mask_distribution_in_label': <dict>, # <label:str>: <mask_template>
        'mask_distribution_in_attribute': <dict>,
        # <label:str>: {
        #     <attribute:str>: { <attr_value>: <mask_template>, }
        # }
        'mask_distribution_in_dataset_item': <dict>,
        # '<item_key>': <mask/polygon count: int>

        ## statistics for tabular task
        'items_broken_annotation': <list>, # [<item_key>, ]
        'label_distribution': {
            'defined_labels': <dict>,   # <label:str>: <count:int>
            'empty_labels': <dict>
            # <label:str>: {
            #     'count': <int>,
            #     'items_with_empty_label': [<item_key>, ]
            # }
        },
        'caption_distribution': {
            'defined_captions': <dict>,   # <label:str>: <count:int>
            'empty_captions': <dict>
            # <label:str>: {
            #     'count': <int>,
            #     'items_with_empty_label': [<item_key>, ]
            # }
            'redundancies': <dict>
            # <label:str>: {
            #     'stopword': <dict>,
            #         'count': <int>,
            #         'items_with_redundancies': [<item_key>, ]
            #     'url': <dict>,
            #         'count': <int>,
            #         'items_with_redundancies': [<item_key>, ]
            #     }
            # }
        },

    },
    'validation_reports': <list>, # [ <validation_error_format>, ]
    # validation_error_format = {
    #     'anomaly_type': <str>,
    #     'description': <str>,
    #     'severity': <str>, # 'warning' or 'error'
    #     'item_id': <str>,  # optional, when it is related to a DatasetItem
    #     'subset': <str>,   # optional, when it is related to a DatasetItem
    # }
    'summary': {
        'errors': <count: int>,
        'warnings': <count: int>
    }
}

```

`item_key` is defined as,
``` python
item_key = (<DatasetItem.id:str>, <DatasetItem.subset:str>)
```

`bbox_template` and `mask_template` are defined as,

``` python
bbox_template = {
    'width': <numerical_stat_template>,
    'height': <numerical_stat_template>,
    'area(wxh)': <numerical_stat_template>,
    'ratio(w/h)': <numerical_stat_template>,
    'short': <numerical_stat_template>, # short = min(w, h)
    'long': <numerical_stat_template>   # long = max(w, h)
}
mask_template = {
    'area': <numerical_stat_template>,
    'width': <numerical_stat_template>,
    'height': <numerical_stat_template>
}
```

`numerical_stat_template` is defined as,

``` python
numerical_stat_template = {
    'items_far_from_mean': <dict>,
    # {'<item_key>': {<ann_id:int>: <value:float>, }, }
    'mean': <float>,
    'stddev': <float>,
    'min': <float>,
    'max': <float>,
    'median': <float>,
    'histogram': {
        'bins': <list>,   # [<float>, ]
        'counts': <list>, # [<int>, ]
    }
}
```

</details>
