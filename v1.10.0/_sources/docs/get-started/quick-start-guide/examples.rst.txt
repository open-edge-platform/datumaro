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

    # export Datumaro dataset in CVAT UI, extract somewhere, go to the project dir
    datum filter -e '/item/annotation[occluded="False"]' --mode items+anno
    datum project export --format tf_detection_api -- --save-media

- Annotate MS COCO dataset, extract image subset, re-annotate it in
  `CVAT <https://github.com/opencv/cvat>`_, update old dataset:

.. code-block::

    # Download COCO dataset http://cocodataset.org/#download
    # Put images to coco/images/ and annotations to coco/annotations/
    mkdir my_project && cd my_project
    datum project create
    datum project import --format coco <path/to/coco>
    datum project export --filter '/image[images_I_dont_like]' --format cvat

- Annotate instance polygons in
  `CVAT <https://github.com/opencv/cvat>`_, export as masks in COCO:

.. code-block::

    datum convert --input-format cvat --input-path <path/to/cvat.xml> \
                  --output-format coco -- --segmentation-mode masks

- Apply an OpenVINO detection model to some COCO-like dataset,
  then compare annotations with ground truth and visualize in TensorBoard:

.. code-block::

    mkdir my_project && cd my_project
    datum project create
    datum project import --format coco <path/to/coco>
    # create model results interpretation script
    datum model add -n mymodel openvino \
      --weights model.bin --description model.xml \
      --interpretation-script parse_results.py
    datum model run --model -n mymodel --output-dir mymodel_inference/
    datum compare mymodel_inference/ --format tensorboard --output-dir compare

- Change colors in PASCAL VOC-like ``.png`` masks:

.. code-block::

    mkdir my_project && cd my_project
    datum project create
    datum project import --format voc <path/to/voc/dataset>

    # Create a color map file with desired colors:
    #
    # label : color_rgb : parts : actions
    # cat:0,0,255::
    # dog:255,0,0::
    #
    # Save as mycolormap.txt

    datum project export --format voc_segmentation -- --label-map mycolormap.txt
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
