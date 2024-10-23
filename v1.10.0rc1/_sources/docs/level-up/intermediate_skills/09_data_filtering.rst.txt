===========================
Level 9: Dataset Filtering
===========================

With the increasing availability of public data, the need for data filtering has become more apparent. Raw data often
contains irrelevant or unnecessary information, making it difficult to extract the desired insights or use it effectively
for decision-making purposes. Data filtering involves the process of identifying and selecting relevant data points while
excluding or removing the irrelevant ones to improve the quality and usability of the data. This process is essential for
ensuring that data can be used effectively and efficiently to drive insights and inform decisions. As the volume and complexity
of data continue to grow, data filtering will become an increasingly important aspect of data management and analysis.
By filtering the dataset in this way, we can create a subset of data that is tailored to our specific needs, making it easier
to extract meaningful insights or use it effectively for decision-making purposes.

In this tutorial, we provide the simple example of filtering dataset items and annotations.
To set the filtering condition, we can use

1) [**ProjectCLI**, **CLI**, **Python**] `XPATH <https://devhints.io/xpath>`_ query,
2) [**Python**] User-provided Python function query.

The detailed description for the `XPATH <https://devhints.io/xpath>`_ query is given by :doc:`this page <../../command-reference/context_free/filter>`.
The more advanced Python example is given :doc:`this notebook <../../jupyter_notebook_examples/notebooks/04_filter>`.

.. tab-set::

    .. tab-item:: ProjectCLI

        With the project-based CLI, we first create project and import datasets into the project

        .. code-block:: bash

            datum project create --output-dir <path/to/project>
            datum project import --format datumaro --project <path/to/project> <path/to/data>

        We filter dataset through

        .. code-block:: bash

            datum filter -e <how/to/filter/dataset> --project <path/to/project>

        We can set ``<how/to/filter/dataset>`` as your own filter like ``'/item/annotation[label="cat" and area > 85]'``.
        This example command will filter only items through the bbox annotations which have `cat` label and bbox area (`w * h`) more than 85.

    .. tab-item:: CLI

        Without the project declaration, we can simply filter dataset by

        .. code-block:: bash

            datum filter <target> -e <how/to/filter/dataset> --output-dir <path/to/output>

        We could use ``--overwrite`` instead of setting ``--output-dir``.
        And we can set ``<how/to/filter/dataset>`` as our own filter like ``'/item[subset="test"]'``
        to filter only items whose `subset` is `test`.

    .. tab-item:: Python

        With Python API, we can filter items as below

        .. code-block:: python

            from datumaro.components.dataset import Dataset

            dataset_path = '/path/to/data'
            dataset = Dataset.import_from(dataset_path, format='datumaro')

            filtered_result = Dataset.filter(dataset, 'how/to/filter/dataset')

        We can set ``<how/to/filter/dataset>`` as your own filter like ``'/item/annotation[occluded="True"]'``.
        This example command will filter only items through the annotation attribute which has `occluded`.

        In addition, you can filter dataset items with your custom Python fuction as well.
        For example, an example of filtering dataset items with images larger than 1024 pixels:

        .. code-block:: python

            from datumaro.components.dataset import Dataset
            from datumaro.components.media import Image

            def filter_func(item: DatasetItem) -> bool:
                h, w = item.media_as(Image).size
                return h > 1024 or w > 1024

            dataset_path = '/path/to/data'
            dataset = Dataset.import_from(dataset_path, format='datumaro')

            filtered_result = Dataset.filter(dataset, filter_func)

        On the other hand, it is possible to filter dataset annotations with the user-provided Python function.
        This is an example of removing bounding boxes sized greater than 50% of the image size:

        .. code-block:: python

            from datumaro.components.dataset import Dataset
            from datumaro.components.media import Image
            from datumaro.components.annotation import Annotation, Bbox

            def filter_func(item: DatasetItem, ann: Annotation) -> bool:
                # If the annotation is not a Bbox, do not filter
                if not isinstance(ann, Bbox):
                    return False

                h, w = item.media_as(Image).size
                image_size = h * w
                bbox_size = ann.h * ann.w

                # Accept Bboxes smaller than 50% of the image size
                return bbox_size < 0.5 * image_size

            dataset_path = '/path/to/data'
            dataset = Dataset.import_from(dataset_path, format='datumaro')

            filtered_result = Dataset.filter(dataset, filter_func, filter_annotations=True)
