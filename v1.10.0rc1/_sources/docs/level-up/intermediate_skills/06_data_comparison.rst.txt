========================================================
Level 6: Data Comparison with Two Heterogeneous Datasets
========================================================

Comparison is a fundamental tool that enables users to identify and understand the discrepancies and variations that exist between datasets.
It allows for a comprehensive assessment of variations in data distribution, format, and annotation standards present across different sources.
By pinpointing the differences in data distribution, format, and annotation standards across multiple sources, the comparison paves the way for a streamlined and effective dataset consolidation process.
In essence, it serves as the cornerstone for achieving a cohesive and comprehensive large-scale dataset, a critical requirement for training deep learning models.

In this tutorial, we provide a simple example for comparing two datasets, and the detailed description of the comparison operation is given in the :doc:`Compare <../../command-reference/context_free/compare>` section.

Comparing Datasets
==================

.. tab-set::

    .. tab-item:: CLI

        Without the project declaration, you can simply compare multiple datasets using the following command:

        .. code-block:: bash

            datum compare <path/to/dataset1> <path/to/dataset2> -o result

        In this case, the ``table`` method is used to generate a comparison table. You will have the comparison report named ``table_compare.json`` and ``table_compare.txt`` inside the output directory.

        To compare if annotations are equal, use:

        .. code-block:: bash

            datum compare <path/to/dataset1> <path/to/dataset2> -m equality -o result

        You will have the comparison report named ``equality_compare.json`` inside the output directory.

        To compare a dataset from another project with a distance metric, use:

        .. code-block:: bash

            datum compare <path/to/other/project/> -m distance -o result

        You will have the comparison report named ``<annotation_type>_confusion.png`` inside the output directory. If there is a label difference, then a ``label_confusion`` result will be created. This supports ``label``, ``bbox``, ``polygon``, and ``mask`` annotation types.

    .. tab-item:: PythonCLI

        With the project-based CLI, you can compare the current project's main target (project) in the working tree with the specified dataset using the following command:

        .. code-block:: bash

            datum compare <path/to/specified/dataset>

        You can also simply compare multiple datasets by using:

        .. code-block:: bash

            datum compare <path/to/dataset1> <path/to/dataset2>
