===========================
Level 8: Dataset Validation
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

  .. tab-item:: ProjectCLI

    With the project-based CLI, we first require to ``create`` a project by

    .. code-block:: bash

      datum project create -o <path/to/project>

    We now ``import`` MS-COCO validation data into the project through

    .. code-block:: bash

      datum project import --format coco_instances -p <path/to/project> <path/to/data>

    (Optional) When we import a data, the change is automatically commited in the project.
    This can be shown through ``log`` as

    .. code-block:: bash

      datum project log -p <path/to/project>

    (Optional) We can check the imported dataset information such as subsets, number of data, or
    categories through ``info``.

    .. code-block:: bash

      datum dinfo -p <path/to/project>

    Finally, we ``validate`` the data within the project as

    .. code-block:: bash

      datum validate --task-type <classification/detection/segmentation> --subset <subset_name> -p <path/to/project>

    We now have the validation report named by ``validation-report-<subset_name>.json``.
