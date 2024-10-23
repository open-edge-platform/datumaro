=====================================================
Level 10: Dataset Explorartion from a Query Image/Text
=====================================================


Datumaro support exploration feature to find out similar data for query among dataset. With query, the exploration result includes top-k similar data among dataset.
Through this feature, you could figure out dataset property. You could check the visualization result of exploration using `Visualizer`.

More detailed descriptions about explorer are given by :doc:`Explore <../../command-reference/context_free/explorer>`
The Python example for the usage of explorer is described in :doc:`here <../../jupyter_notebook_examples/notebooks/07_data_explorer>`.


.. tab-set::

    .. tab-item:: Python

        With Python API, we can explore similar items as below

        .. code-block:: python

            from datumaro.components.dataset import Dataset
            from datumaro.components.environment import Environment
            from datumaro.components.algorithms.hash_key_inference.explorer import Explorer

            data_path = '/path/to/data'

            env = Environment()
            detected_formats = env.detect_dataset(data_path)

            dataset = Dataset.import_from(data_path, detected_formats[0])

            explorer = Explorer(dataset)
            query = '/path/to/image/file'
            topk = 20
            topk_result = explorer.explore_topk(query, topk)
            dataset.export(dir, save_hashkey_meta=True)

        Through set ``save_hashkey_meta = True``, we could save ``hash_key`` of items, which is base of explorer. This allows we to re-explore this dataset without redundant hash calculations.

    .. tab-item:: CLI

        Without the project declaration, we can simply ``explore`` dataset like below.

        You can set the query using one of the following options: ``QUERY_PATH``, ``QUERY_ID``, or ``QUERY_STR``

        .. code-block:: bash

            datum explore <target> --query-img-path QUERY_PATH -topk TOPK_NUM

        ``QUERY_PATH`` could be image file path or list of them

        ``TOPK_NUM`` is an integer that you want to find the number of similar results for query

        Exploration result would be printed by log and result files would be copied into ``explore_result`` folder.

        .. code-block:: bash

            datum explore <target> --query-item-id QUERY_ID -topk TOPK_NUM

        ``QUERY_ID`` could be datasetitem id or list of them

        .. code-block:: bash

            datum explore <target> --query-str QUERY_STR -topk TOPK_NUM

        ``QUERY_STR`` could be text description or list of them

        .. code-block:: bash

            datum explore <target> --query-str QUERY_STR -topk TOPK_NUM -s -o DST_DIR

        To save the result, specify the output directory as ``DST_DIR``

    .. tab-item:: ProjectCLI

        With the project-based CLI, we first require to ``create`` a project by

        .. code-block:: bash

            datum project create --output-dir <path/to/project>

        We now ``import`` data in to project through

        .. code-block:: bash

            datum project import --project <path/to/project> <path/to/data>

        We can ``explore`` similar items for the query.

        You can set the query using one of the following options: ``QUERY_PATH``, ``QUERY_ID``, or ``QUERY_STR``

        .. code-block:: bash

            datum explore --query-img-path QUERY_PATH -topk TOPK_NUM -p <path/to/project>

        ``QUERY_PATH`` could be image file path or list of them

        ``TOPK_NUM`` is an integer that you want to find the number of similar results for query

        Exploration result would be printed by log and result files would be copied into ``explore_result`` folder.

        .. code-block:: bash

            datum explore <target> --query-item-id QUERY_ID -topk TOPK_NUM -p <path/to/project>

        ``QUERY_ID`` could be datasetitem id or list of them

        .. code-block:: bash

            datum explore <target> --query-str QUERY_STR -topk TOPK_NUM -p <path/to/project>

        ``QUERY_STR`` could be text description or list of them
