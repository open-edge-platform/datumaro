=========================================
Level 7: Merge Two Heterogeneous Datasets
=========================================


In the latest deep learning trends, training foundation models with larger datasets has become
increasingly popular. To achieve this, it is crucial to collect and prepare massive datasets for deep
learning model development. Collecting and labeling large datasets can be challenging, so
consolidating scattered datasets into a unified one is important. For instance, `Florence <https://arxiv.org/pdf/2111.11432.pdf>`_
created the FLOD-9M massive dataset by combining MS-COCO, LVIS, OpenImages, and Object365 datasets
to use for training.

In this tutorial, we provide the simple example for merging two datasets and the detailed description
for merge operation is given by :ref:`here <Merge>`.
The more advanced Python example with the label mapping between datasets is given
:doc:`here <../../jupyter_notebook_examples/notebooks/01_merge_multiple_datasets_for_classification>`.

Prepare datasets
================

We here download two aerial datasets named by Eurosat and UC Merced as a simple ImageNet format by

.. code-block:: bash

  datum download get -i tfds:eurosat -f imagenet --output-dir <path/to/eurosat> -- --save-media

  datum download get -i tfds:uc_merced -f imagenet --output-dir <path/to/uc_merced> -- --save-media

Merge datasets
==============

.. tab-set::

  .. tab-item:: CLI

    Without the project declaration, we can simply merge multiple datasets by

    .. code-block:: bash

      datum merge --merge_policy union --format imagenet --output-dir <path/to/output> <path/to/eurosat> <path/to/uc_merced> -- --save-media

    We now have the merge data with the merge report named by ``merge_report.json`` inside the output directory.

  .. tab-item:: Python

    .. code-block:: python

        from datumaro.components.dataset import Dataset

        eurosat_path = '/path/to/eurosat'
        eurosat = Dataset.import_from(eurosat_path, 'imagenet')

        uc_merced_path = '/path/to/uc_merced'
        uc_merced = Dataset.import_from(uc_merced_path, 'imagenet')

        from datumaro.components.hl_ops import HLOps

        merged = HLOps.merge(eurosat, uc_merced, merge_policy='union')

  .. tab-item:: ProjectCLI

    With the project-based CLI, we first create two project and import datasets into each project

    .. code-block:: bash

      datum project create --output-dir <path/to/project1>
      datum project import --format imagenet --project <path/to/project1> <path/to/eurosat>

      datum project create --output-dir <path/to/project2>
      datum project import --format imagenet --project <path/to/project2> <path/to/uc_merced>

    We merge two projects through

    .. code-block:: bash

      datum merge --merge-policy union --format imagenet --output-dir <path/to/output> <path/to/project1> <path/to/project2> -- --save-media

    Similar to merge without projects, we have the merge report named by ``merge_report.json`` inside the output directory.
    Finally, we import the merged data (``<path/to/output>``) into a project.
    In this tutorial, we create another project and import this into the project.

    .. code-block:: bash

      datum project create --output-dir <path/to/project3>
      datum project import --format imagenet --project <path/to/project3> <path/to/output>
