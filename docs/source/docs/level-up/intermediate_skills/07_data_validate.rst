===========================
Level 7: Dataset Validation
===========================


When creating a dataset, it is natural for imbalances to occur between categories, and sometimes
there may be very few data points for the minority class. In addition, inconsistent annotations may
be produced by annotators or over time. When training a model with such data, more attention should
be paid, and sometimes it may be necessary to filter or correct the data in advance. Datumaro provides
data validation functionality for this purpose.

More detailed descriptions about validation errors and warnings are given by :ref:`here <Validate>`.
The Python example for the usage of validator is described in this :doc:`notebook <../../jupyter_notebook_examples/notebooks/11_validate>`.


.. tab-set::

  .. tab-item:: Python

    .. code-block:: python

        from datumaro.components.environment import Environment
        from datumaro.components.dataset import Dataset

        data_path = '/path/to/data'

        env = Environment()

        detected_formats = env.detect_dataset(data_path)

        dataset = Dataset.import_from(data_path, detected_formats[0])

        from datumaro.plugins.validators import DetectionValidator

        validator = DetectionValidator() # Or ClassificationValidator or SegementationValidator

        reports = validator.validate(dataset)

  .. tab-item:: CLI

    You can validate a dataset directly using the context-free CLI:

    .. code-block:: bash

      datum validate --task-type <classification/detection/segmentation> --input-format coco_instances --input-path <path/to/data>

    The validation report will be saved as ``validation-report.json``.
