===============================
Level 3: Data Import and Export
===============================

Datumaro is a tool that supports public data formats across a wide range of tasks such as
classification, detection, segmentation, pose estimation, or visual tracking.
To facilitate this, Datumaro provides assistance with data import and export via both Python API and CLI.
This makes it easier for users to work with various data formats using Datumaro.

Prepare dataset
===============

For the segmentation task, we here introduce the Cityscapes, which collects road scenes from 50
different cities and contains 5K fine-grained pixel-level annotations and 20K coarse annotations.
More detailed description is given by :ref:`here <Cityscapes>`.
The Cityscapes dataset is available for free `download <https://www.cityscapes-dataset.com/downloads/>`_.

Convert data format
===================

Users sometimes need to compare, merge, or manage various kinds of public datasets in a unified
system. To achieve this, Datumaro not only has ``import`` and ``export`` funcionalities, but also
provides ``convert``, which shortens the import and export into a single command line.
Let's convert the Cityscapes data into the MS-COCO format, which is described in :ref:`here <COCO>`.


.. tab-set::

  .. tab-item:: CLI

    Without creation of a project, we can achieve this with a single line command ``convert`` in Datumaro

    .. code-block:: bash

        datum convert -if cityscapes -i <path/to/cityscapes> -f coco_panoptic -o <path/to/output>

  .. tab-item:: Python

    With Python API, we can import the data through ``Dataset`` as below.

    .. code-block:: python

        from datumaro.components.dataset import Dataset

        data_path = '/path/to/cityscapes'
        data_format = 'cityscapes'

        dataset = Dataset.import_from(data_path, data_format)

    We then export the import dataset as

    .. code-block:: python

        output_path = '/path/to/output'

        dataset.export(output_path, format='coco_panoptic')

  .. tab-item:: ProjectCLI

    With the project-based CLI, we first require to ``create`` a project by

    .. code-block:: bash

        datum project create -o <path/to/project>

    We now ``import`` Cityscapes data into the project through

    .. code-block:: bash

        datum project import --format cityscapes -p <path/to/project> <path/to/cityscapes>

    (Optional) When we import a data, the change is automatically commited in the project.
    This can be shown through ``log`` as

    .. code-block:: bash

        datum project log -p <path/to/project>

    (Optional) We can check the imported dataset information such as subsets, number of data, or
    categories through ``info``.

    .. code-block:: bash

        datum project info -p <path/to/project>

    Finally, we ``export`` the data within the project with MS-COCO format as

    .. code-block:: bash

        datum project export --format coco -p <path/to/project> -o <path/to/save> -- --save-media

Even if you are not sure about the format of the dataset, there's no need to worry. You can easily detect the format in the next level, which is described in the :ref:`next level <Level 4: Detect Data Format from an Unknown Dataset>`!
