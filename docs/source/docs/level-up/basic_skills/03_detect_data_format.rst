===================================================
Level 4: Detect Data Format from an Unknown Dataset
===================================================

Datumaro has a built-in function that allows users to detect the format of a dataset.
This feature is useful in situations where the original format of the data is unknown or unclear.
By utilizing this function, users can easily determine the format of the data and then proceed with the appropriate data handling processes.

Detect data format
==================

.. tab-set::

  .. tab-item:: CLI

    .. code-block:: bash

      datum detect <path/to/data>

    The printed format can be utilized as ``format`` argument when importing a dataset as following the
    :ref:`previous level <Level 3: Data Import and Export>`.

  .. tab-item:: Python

    .. code-block:: python

      from datumaro.components.environment import Environment

      data_path = '/path/to/data'

      env = Environment()

      detected_formats = env.detect_dataset(data_path)

    (Optional) With the detected format, we can import the dataset as below.

    .. code-block:: python

      from datumaro.components.dataset import Dataset

      dataset = Dataset.import_from(data_path, detected_formats[0])
