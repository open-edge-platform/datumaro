# Stats

## Get Project Statistics

This command computes various project statistics, such as:
- image mean and std. dev. (RGB)
- class and attribute balance
- mask pixel balance
- segment area distribution

Usage:

```console
datum stats [-h] [-s SUBSET] [--image-stats IMAGE_STATS] [--ann-stats ANN_STATS] [-p PROJECT_DIR] [target]
```

Parameters:
- `<target>` (string) - Target
  [source revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
  By default, computes statistics of the merged dataset.
- `-s, --subset` (string) - Compute stats only for a specific subset
- `--image-stats` (bool) - Compute image mean and std (RGB) (default: `True`)
- `--ann-stats` (bool) - Compute annotation statistics (default: `True`)
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example:
- Compute project statistics
  ```console
  datum stats -p <path/to/project/>
  ```

Sample output:

<details>

``` json
{
    "annotations": {
        "labels": {
            "attributes": {
                "gender": {
                    "count": 358,
                    "distribution": {
                        "female": [
                            149,
                            0.41620111731843573
                        ],
                        "male": [
                            209,
                            0.5837988826815642
                        ]
                    },
                    "values count": 2,
                    "values present": [
                        "female",
                        "male"
                    ]
                },
                "view": {
                    "count": 340,
                    "distribution": {
                        "__undefined__": [
                            4,
                            0.011764705882352941
                        ],
                        "front": [
                            54,
                            0.1588235294117647
                        ],
                        "left": [
                            14,
                            0.041176470588235294
                        ],
                        "rear": [
                            235,
                            0.6911764705882353
                        ],
                        "right": [
                            33,
                            0.09705882352941177
                        ]
                    },
                    "values count": 5,
                    "values present": [
                        "__undefined__",
                        "front",
                        "left",
                        "rear",
                        "right"
                    ]
                }
            },
            "count": 2038,
            "distribution": {
                "car": [
                    340,
                    0.16683022571148184
                ],
                "cyclist": [
                    194,
                    0.09519136408243375
                ],
                "head": [
                    354,
                    0.17369970559371933
                ],
                "ignore": [
                    100,
                    0.04906771344455348
                ],
                "left_hand": [
                    238,
                    0.11678115799803729
                ],
                "person": [
                    358,
                    0.17566241413150147
                ],
                "right_hand": [
                    77,
                    0.037782139352306184
                ],
                "road_arrows": [
                    326,
                    0.15996074582924436
                ],
                "traffic_sign": [
                    51,
                    0.025024533856722278
                ]
            }
        },
        "segments": {
            "area distribution": [
                {
                    "count": 1318,
                    "max": 11425.1,
                    "min": 0.0,
                    "percent": 0.9627465303140978
                },
                {
                    "count": 1,
                    "max": 22850.2,
                    "min": 11425.1,
                    "percent": 0.0007304601899196494
                },
                {
                    "count": 0,
                    "max": 34275.3,
                    "min": 22850.2,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 45700.4,
                    "min": 34275.3,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 57125.5,
                    "min": 45700.4,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 68550.6,
                    "min": 57125.5,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 79975.7,
                    "min": 68550.6,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 91400.8,
                    "min": 79975.7,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 102825.90000000001,
                    "min": 91400.8,
                    "percent": 0.0
                },
                {
                    "count": 50,
                    "max": 114251.0,
                    "min": 102825.90000000001,
                    "percent": 0.036523009495982466
                }
            ],
            "avg. area": 5411.624543462382,
            "pixel distribution": {
                "car": [
                    13655,
                    0.0018431496518735067
                ],
                "cyclist": [
                    939005,
                    0.12674674030446592
                ],
                "head": [
                    0,
                    0.0
                ],
                "ignore": [
                    5501200,
                    0.7425510702956085
                ],
                "left_hand": [
                    0,
                    0.0
                ],
                "person": [
                    954654,
                    0.12885903974805205
                ],
                "right_hand": [
                    0,
                    0.0
                ],
                "road_arrows": [
                    0,
                    0.0
                ],
                "traffic_sign": [
                    0,
                    0.0
                ]
            }
        }
    },
    "annotations by type": {
        "bbox": {
            "count": 548
        },
        "caption": {
            "count": 0
        },
        "label": {
            "count": 0
        },
        "mask": {
            "count": 0
        },
        "points": {
            "count": 669
        },
        "polygon": {
            "count": 821
        },
        "polyline": {
            "count": 0
        }
    },
    "annotations count": 2038,
    "unannotated images": [
        "img00051",
        "img00052",
        "img00053",
        "img00054",
        "img00055",
    ],
    "unannotated images count": 5,

    "dataset": {
        "images count": 100,
        "unique images count": 97,
        "repeated images count": 3,
        "repeated images": [
            [["img00057", "default"], ["img00058", "default"]],
            [["img00059", "default"], ["img00060", "default"]],
            [["img00061", "default"], ["img00062", "default"]],
        ],
    },
    "subsets": {
        "default": {
            "images count": 100,
            "image mean (RGB)": [
                107.06903686941979,
                79.12831698580979,
                52.95829558185416
            ],
            "image std (RGB)": [
                49.40237673503467,
                43.29600731496902,
                35.47373007603151
            ],

        }
    },
}
```

</details>
