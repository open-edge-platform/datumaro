=====================================================
Level 14: Dataset Pruning
=====================================================


Datumaro support prune feature to extract representative subset of dataset. The pruned dataset allows us to examine the trade-off between
accuracy and convergence time when training on a reduced data sample. By selecting a subset of instances that captures the essential patterns
and characteristics of the data, we aim to evaluate the impact of dataset size on model performance.

More detailed descriptions about pruning are given by :doc:`Prune <../../command-reference/context_free/prune>`
The Python example for the usage of pruning is described in :doc:`here <../../jupyter_notebook_examples/notebooks/17_data_pruning>`.


.. tab-set::

    .. tab-item:: Python

        With Python API, we can prune dataset as below

        .. code-block:: python

            from datumaro.components.dataset import Dataset
            from datumaro.components.environment import Environment
            from datumaro.componenets.prune import prune

            data_path = '/path/to/data'

            env = Environment()
            detected_formats = env.detect_dataset(data_path)

            dataset = Dataset.import_from(data_path, detected_formats[0])

            prune = Prune(dataset, cluster_method='<how/to/prune/dataset>')

            result = prune.get_pruned(ratio='<how/much/to/prune/dataset>')

        We can choose the desired method as ``<how/to/prune/dataset>`` among the provided ones. The default value is ``random``.
        Additionally, we can specify how much of the dataset we want to retain by providing a float value between 0 and 1 for the ``<how/much/to/prune/dataset>`` parameter. The default value is 0.5.

    .. tab-item:: CLI

        Without the project declaration, we can simply ``prune`` dataset by

        .. code-block:: bash

            datum prune <target> -m METHOD -r RATIO -h HASH_TYPE

        We could use ``--overwrite`` instead of setting ``-o/--output-dir``.
        We can choose the desired method as ``METHOD`` among the provided ones. The default value is ``random``.
        Additionally, we can specify how much of the dataset we want to retain by providing a float value between 0 and 1 for the ``RATIO`` parameter. The default value is 0.5.


    .. tab-item:: ProjectCLI

        With the project-based CLI, we first require to ``create`` a project by

        .. code-block:: bash

            datum project create --output-dir <path/to/project>

        We now ``import`` data into project through

        .. code-block:: bash

            datum project import --project <path/to/project> <path/to/data>

        We can ``prune`` dataset

        .. code-block:: bash

            datum prune -m METHOD -r RATIO -h HASH_TYPE -p <path/to/project>

        We can choose the desired method as ``METHOD`` among the provided ones. The default value is ``random``.
        Additionally, we can specify how much of the dataset we want to retain by providing a float value between 0 and 1 for the ``RATIO`` parameter. The default value is 0.5.
