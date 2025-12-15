Examples
########

- Convert PASCAL VOC dataset to COCO format, keep only images with ``cat`` class
  presented:

.. code-block::

    # Download VOC dataset:
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    datum convert --input-format voc --input-path <path/to/voc> \
                  --output-format coco \
                  --filter '/item[annotation/label="cat"]' \
                  -- --reindex 1 # avoid annotation id conflicts

- Convert only non-``occluded`` annotations from a
  `CVAT <https://github.com/opencv/cvat>`_ project to TFrecord:

.. code-block::

    # export Datumaro dataset in CVAT UI, extract somewhere
    datum filter -e '/item/annotation[occluded="False"]' --mode items+anno <path/to/cvat> -o <output_dir>
    datum convert -i <output_dir> -f tf_detection_api -o <final_output> -- --save-media

- Annotate MS COCO dataset, extract image subset, re-annotate it in
  `CVAT <https://github.com/opencv/cvat>`_, update old dataset:

.. code-block::

    # Download COCO dataset http://cocodataset.org/#download
    # Put images to coco/images/ and annotations to coco/annotations/
    datum filter -e '/image[images_I_dont_like]' <path/to/coco> -o filtered_coco
    datum convert -i filtered_coco -f cvat -o cvat_export

- Annotate instance polygons in
  `CVAT <https://github.com/opencv/cvat>`_, export as masks in COCO:

.. code-block::

    datum convert --input-format cvat --input-path <path/to/cvat.xml> \
                  --output-format coco -- --segmentation-mode masks


- Change colors in PASCAL VOC-like ``.png`` masks:

.. code-block::

    # Create a color map file with desired colors:
    #
    # label : color_rgb : parts : actions
    # cat:0,0,255::
    # dog:255,0,0::
    #
    # Save as mycolormap.txt

    datum convert -i <path/to/voc/dataset> -if voc -f voc_segmentation -o <output_dir> -- --label-map mycolormap.txt
    # add "--apply-colormap=0" to save grayscale (indexed) masks
    # check "--help" option for more info
    # use "datum --loglevel debug" for extra conversion info

- Create a custom COCO-like dataset:

.. code-block::

    import numpy as np
    import datumaro as dm

    dataset = dm.Dataset.from_iterable([
      dm.DatasetItem(id='image1', subset='train',
        media=dm.Image.from_numpy(data=np.ones((5, 5, 3))),
        annotations=[
          dm.Bbox(1, 2, 3, 4, label=0),
        ]
      ),
      # ...
    ], categories=['cat', 'dog'])
    dataset.export('test_dataset/', 'coco')
