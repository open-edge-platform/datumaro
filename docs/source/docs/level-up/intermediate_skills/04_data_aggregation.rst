================================
Level 5: Data Subset Aggregation
================================


When working with public data, the dataset is sometimes provided with pre-divided training,
validation, and test subsets. However, in some cases, these subsets may not follow an identical
distribution, making it difficult to perform proper model comparison or selection. In this tutorial,
we will show an example of dataset aggregation and reorganization to address this issue.

Prepare datasets
================

As we did in :ref:`level 3 <Level 3: Data Import and Export>`, we use the Cityscapes dataset.
The Cityscapes dataset is divided into train, validation, and test subsets with the number of 2975,
500, and 1525 samples, respectively.

Again, more detailed description is given by :ref:`here <Cityscapes>`.
The Cityscapes dataset is available for free `download <https://www.cityscapes-dataset.com/downloads/>`_.

==============

.. tab-set::

  .. tab-item:: Python

    .. code-block:: python

        from datumaro.components.dataset import Dataset

        data_path = '/path/to/cityscapes'
        dataset = Dataset.import_from(data_path, 'cityscapes')

        from datumaro.components.hl_ops import HLOps

        aggregated = HLOps.aggregate(dataset, from_subsets=["train", "val", "test"], to_subset="default")

    (Optional) Through :ref:`splitter <Transform>`, we can reorganize the aggregated dataset with respect to the number of annotations in each subset.

    .. code-block:: python

      import datumaro.plugins.splitter as splitter

      splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
      task = splitter.SplitTask.segmentation.name

      resplitted = aggregated.transform("split", task=task, splits=splits)
