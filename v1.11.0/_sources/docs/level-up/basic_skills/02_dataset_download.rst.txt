=========================
Level 2: Dataset download
=========================

Datumaro provides a way to download public datasets using `TensorFlow Datasets <https://www.tensorflow.org/datasets>`_ download API.
Using this feature, you can download some datasets in the `catalog <https://www.tensorflow.org/datasets/catalog/overview>`_.


Prepare installation
====================
To use Datumaro ``download`` feature, you should install Datumaro with ``[tf,tfds]`` extras:

  .. code-block:: bash

    pip install datumaro[tf,tfds]

.. note:: You cannot use Datumaro download feature if you installed Datumaro with the default option, e.g., ``pip install datumaro``. Please check it!

Which datasets are available?
=============================

.. tab-set::

  .. tab-item:: CLI

    You can see the list of available ``DATASET_ID`` using the following command.

    .. code-block:: bash

      datum download describe [--report-format {text,json}] [--report-file REPORT_FILE]

How can we download datasets?
=============================

.. tab-set::

  .. tab-item:: CLI

    You can actually download the dataset using the following command.
    You have to input ``-i DATASET_ID`` according to the id of dataset you want to download.
    Additionally, you can specify the output format (``-f OUTPUT_FORMAT``) and path (``-o DST_DIR``).

    .. code-block:: bash

      datum download get [-h] -i DATASET_ID [-f OUTPUT_FORMAT] [-o DST_DIR] [--overwrite] [-s SUBSET] ...

    .. note:: By default, ``download`` does not export the media files (e.g. images).
      We recommand you to run this command with ``--save-media`` option to export the media files as well,
      for example, ``datum download get -i tfds:mnist -- --save-media``.

In the :ref:`next level <Level 3: Data Import and Export>`, we will look into how to import and export the dataset using Datumaro!
