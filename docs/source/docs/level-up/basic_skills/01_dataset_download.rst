=========================
Level 1: Dataset download
=========================

Datumaro supports downloading public datasets from multiple sources: `TensorFlow Datasets <https://www.tensorflow.org/datasets>`_ and `Kaggle Datasets <https://www.kaggle.com/datasets>`_


Prepare installation
====================
To use Datumaro ``download`` feature, you should install Datumaro with ``[tf,tfds]`` extras for TensorFlow Datasets or ``[kaggle]`` for Kaggle Datasets:

.. tab-set::

  .. tab-item:: TensorFlow

    .. code-block:: bash

      pip install datumaro[tf,tfds]

  .. tab-item:: Kaggle

    .. code-block:: bash

      pip install datumaro[kaggle]

Which datasets are available?
=============================

You can browse the list of available TensorFlow Datasets `here <https://www.tensorflow.org/datasets/catalog/overview>`__ or using the command below.
For Kaggle Datasets, you can check `here <https://www.kaggle.com/datasets>`__.

.. tab-set::

  .. tab-item:: TensorFlow

    You can see the list of available ``DATASET_ID`` using the following command.

    .. code-block:: bash

      datum download tfds describe [--report-format {text,json}] [--report-file REPORT_FILE]

How can we download datasets?
=============================

.. tab-set::

  .. tab-item:: TensorFlow

    You can actually download the dataset using the following command.
    You have to input ``-i DATASET_ID`` according to the id of dataset you want to download.
    Additionally, you can specify the output format (``-f OUTPUT_FORMAT``) and path (``-o DST_DIR``).

    .. code-block:: bash

      datum download tfds get [-h] -i DATASET_ID [-f OUTPUT_FORMAT] [-o DST_DIR] [--overwrite] [-s SUBSET] ...

    .. note:: By default, ``download`` does not export the media files (e.g. images).
      We recommand you to run this command with ``--save-media`` option to export the media files as well,
      for example, ``datum download tfds get -i tfds:mnist -- --save-media``.

  .. tab-item:: Kaggle

    You can actually download the dataset using the following command.
    You have to input ``-i DATASET_ID`` according to the id of dataset you want to download.
    Additionally, you can specify the output format (``-f OUTPUT_FORMAT``) and path (``-o DST_DIR``).

    .. code-block:: bash

      datum download kaggle get [-h] -i DATASET_ID [-f OUTPUT_FORMAT] [-o DST_DIR] [--overwrite] [-s SUBSET] ...

    .. note:: By default, ``download`` does not export the media files (e.g. images).
      We recommand you to run this command with ``--save-media`` option to export the media files as well,
      for example, ``datum download kaggle get -i tfds:mnist -- --save-media``.

In the :ref:`next level <Level 2: Data Import and Export>`, we will look into how to import and export the dataset using Datumaro!
