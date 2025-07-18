# Project (context)

## Migrate project

Updates the project from an old version to the current one and saves the
resulting project in the output directory. Projects cannot be updated
inplace.

The command tries to map the old source configuration to the new one.
This can fail in some cases, so the command will exit with an error,
unless `-f/--force` is specified. With this flag, the command will
skip these errors an continue its work.

Usage:

``` bash
datum project migrate [-h] -o DST_DIR [-f] [-p PROJECT_DIR] [--overwrite]
```

Parameters:
- `-o, --output-dir` (string) - Output directory for the updated project
- `-f, --force` - Ignore source import errors (default: False)
- `--overwrite` - Overwrite existing files in the save directory.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Examples:
- Migrate a project from v1 to v2, save the new project in other dir:
`datum project migrate -o <output/dir>`

## Print project info

Prints project configuration info such as available plugins, registered models,
imported sources and build tree.

Usage:

``` bash
datum project info [-h] [-p PROJECT_DIR] [revision]
```

Parameters:
- `<revision>` (string) - Target project revision. By default,
  uses the working tree.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Examples:
- Print project info for the current working tree:
`datum project info`

- Print project info for the previous revision:
`datum project info HEAD~1`

Sample output:

<details>

```
Project:
  location: /test_proj

Plugins:
  extractors: ade20k2017, ade20k2020, camvid, cifar, cityscapes, coco, coco_captions, coco_image_info, coco_instances, coco_labels, coco_panoptic, coco_person_keypoints, coco_stuff, cvat, datumaro, icdar_text_localization, icdar_text_segmentation, icdar_word_recognition, image_dir, image_zip, imagenet, imagenet_txt, kitti, kitti_detection, kitti_raw, kitti_segmentation, label_me, lfw, market1501, mnist, mnist_csv, mot_seq, mots, mots_png, open_images, sly_pointcloud, tf_detection_api, vgg_face2, voc, voc_action, voc_classification, voc_detection, voc_layout, voc_segmentation, wider_face, yolo, yolo_ultralytics

  converters: camvid, mot_seq_gt, coco_captions, coco, coco_image_info, coco_instances, coco_labels, coco_panoptic, coco_person_keypoints, coco_stuff, kitti, kitti_detection, kitti_segmentation, icdar_text_localization, icdar_text_segmentation, icdar_word_recognition, lfw, datumaro, open_images, image_zip, cifar, yolo, voc_action, voc_classification, voc, voc_detection, voc_layout, voc_segmentation, tf_detection_api, label_me, mnist, cityscapes, mnist_csv, kitti_raw, wider_face, vgg_face2, sly_pointcloud, mots_png, image_dir, imagenet_txt, market1501, imagenet, cvat, yolo_ultralytics

  launchers:

Models:

Sources:
  'source-2':
    format: voc
    url: /datasets/pascal/VOC2012
    location: /test_proj/source-2/
    options: {}
    hash: 3eb282cdd7339d05b75bd932a1fd3201
    stages:
      'root':
        type: source
        hash: 3eb282cdd7339d05b75bd932a1fd3201
  'source-3':
    format: imagenet
    url: /datasets/imagenet/ILSVRC2012_img_val/train
    location: /test_proj/source-3/
    options: {}
    hash: e47804a3ec1a54c9b145e5f1007ec72f
    stages:
      'root':
        type: source
        hash: e47804a3ec1a54c9b145e5f1007ec72f
```

</details>
