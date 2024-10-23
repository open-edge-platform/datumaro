============================
Level 12: Project Versioning
============================

Project versioning is a concept unique to Datumaro. Datumaro project includes a data source and revision tree,
where each revision represents the state of the data sources at a particular point in time.
With Datumaro's revision tree, you can control the versioning of datasets.
This means that the Datumaro project records the state of a dataset after it has been changed by an operation,
allowing you to go back to a previous state or create a new branch to have multiple versions of a dataset for different purposes from one source.
For more details, please refer to :ref:`Project versioning concepts` and :ref:`Project data model`.

Prerequisite
============

In this step, we create a project and import `UCMerced <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`_ dataset to the project.

.. tab-set::

  .. tab-item:: ProjectCLI

    At first, we create a project and import the dataset.

    .. code-block:: bash

        # Create a workspace and download UCMerced dataset
        mkdir -p ~/ws_datum/project
        cd ~/ws_datum
        datum download get -i tfds:uc_merced -o ./uc_merced -- --save-media

        # Create a project and import the dataset
        cd project
        datum project create
        datum project import -n my-dataset -f imagenet_txt ../uc_merced

        # Commit change
        datum project commit -m "Add my-dataset"
        datum dinfo

    You can see that the project successfully imports the dataset.

    .. code-block:: console

        length: 2100
        categories: label
        label:
            count: 21
            labels: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse (and 11 more)
        subsets: train
        'default':
            length: 2100
            categories: label
            label:
                count: 21
                labels: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse (and 11 more)

    You can check the revision history in the project by using the following commands.

    .. code-block:: bash

        datum project log

    It will shows the revision history as follows.

    .. code-block:: console

        c02ec1959c48d65d9558bad15108fe546ed2e4de Add my-dataset
        680d0437b86a99cc1e3b402bd47b87d3700b8387 Initial commit

Commit project revisions and checkout to the revision
=====================================================

In this step, we will apply :ref:`Transform` to the project, resulting in the creation of a new revision by the operation.
We can then commit this revision to the revision tree, allowing us to checkout to the revision anytime we need.
Additionally, if you require a new operation on the dataset for a new target, you can go back and create a new branch from the old revision.

.. tab-set::

  .. tab-item:: ProjectCLI

    At first, we split the dataset into "train" and "test" subsets and commit the change into the revision tree.

    .. code-block:: bash

        datum project transform -t random_split
        datum project commit -m "Split train-test"
        datum dinfo --all

    You can see that your dataset is successfully splitted into "train" and "subset".

    .. code-block:: console

        length: 2100
        categories: label
        label:
            count: 21
            labels: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, tenniscourt
        subsets: test, train
        'test':
            length: 693
            categories: label
            label:
                count: 21
                labels: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, tenniscourt
        'train':
            length: 1407
            categories: label
            label:
                count: 21
                labels: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, tenniscourt

    The revision history is also changed by your commit.

    .. code-block:: bash

        datum project log

    .. code-block:: console

        a0fdbbc6da25e5104ff927a321216affbb31fb75 Split train-test
        c02ec1959c48d65d9558bad15108fe546ed2e4de Add my-dataset
        680d0437b86a99cc1e3b402bd47b87d3700b8387 Initial commit

    This time, we can go back to the old revision to create another branch in the dataset.

    .. code-block:: bash

        # Checkout to the time when we added "my-dataset"
        datum project checkout c02ec1959c48d65d9558bad15108fe546ed2e4de
        datum project transform -t remap_labels -- -l airplane:airport -l runway:airport
        datum project commit -m "Remap labels airplane,runway -> airport"

    .. code-block:: bash

        datum dinfo --all

    Now, we have a different label categories ("airport" is added) but our dataset only has "train" subset since we go back to the old revision.

    .. code-block:: console

        length: 2100
        categories: label
        label:
            count: 20
            labels: agricultural, airport, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, sparseresidential, storagetanks, tenniscourt
        subsets: train
        'train':
            length: 2100
            categories: label
            label:
                count: 20
                labels: agricultural, airport, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, sparseresidential, storagetanks, tenniscourt

    You can see that the revision history is also updated.

    .. code-block:: bash

        datum project log

    .. code-block:: console

        e8e9c55b20992e48adf85f77753b52aba600abac Remap labels airplane,runway -> airport
        c02ec1959c48d65d9558bad15108fe546ed2e4de Add my-dataset
        680d0437b86a99cc1e3b402bd47b87d3700b8387 Initial commit

    Lastly, you can move to the other revision by the checkout command anytime you want.

    .. code-block:: bash

        # Checkout to the train-test split revision
        datum project checkout a0fdbbc6da25e5104ff927a321216affbb31fb75
        datum project log

    .. code-block:: console

        a0fdbbc6da25e5104ff927a321216affbb31fb75 Split train-test
        c02ec1959c48d65d9558bad15108fe546ed2e4de Add my-dataset
        680d0437b86a99cc1e3b402bd47b87d3700b8387 Initial commit
