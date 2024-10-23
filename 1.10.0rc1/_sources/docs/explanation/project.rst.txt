Command-Line Workflow
#####################

Project
-------

In Datumaro, most command-line commands operate on projects, but there are
also few commands operating on datasets directly. There are 2 basic ways
to use Datumaro from the command-line:

- Use the `convert <../command-reference/context_free/convert.md>`_,
  `compare <../command-reference/context_free/compare.md>`_, `merge <../command-reference/context_free/merge.md>`_
  commands directly on existing datasets.

- Create a Datumaro project and operate on it:

  - Create an empty project with `create <../command-reference/context/create.md>`_.

  - Import existing datasets with `import <../command-reference/context/sources.md#import-dataset>`_.

  - Modify the project with `transform <../command-reference/context_free/transform.md>`_ and `filter <../command-reference/context_free/filter.md>`_.

  - Create new revisions of the project with `commit <../command-reference/context/commit.md>`_,
    navigate over them using `checkout <../command-reference/context/checkout.md>`_, compare with
    `compare <../command-reference/context_free/compare.md>`_, compute statistics with
    `stats <../command-reference/context_free/stats.md>`_.

  - Export the resulting dataset with `export <../command-reference/context/export.md>`_.

  - Check project config with `project info <../command-reference/context/projects.md#print-project-info>`_.

Basically, a project is a combination of datasets, models and environment.

A project can contain an arbitrary number of datasets (see :ref:`Datasets and Data Sources`).
A project acts as a manager for them and allows to manipulate them
separately or as a whole, in which case it combines dataset items
from all the sources into one composite dataset. You can manage separate
datasets in a project by commands in the `datum source <../command-reference/context/sources.md>`_
command line context.

Note that **modifying operations** (``transform``, ``filter``, ``patch``)
**are applied in-place** to the datasets by default.

If you want to interact with models, you need to add them to the project
first using the `model add <../command-reference/context/models.md#register-model>`_ command.

A typical way to obtain Datumaro projects is to export tasks in
`CVAT <https://github.com/opencv/cvat>`_ UI.


Project data model
------------------

.. image:: ../../../images/project_model.svg
    :name: project model

Datumaro tries to combine a "Git for datasets" and a build system like
make or CMake for datasets in a single solution. Currently, ``Project``
represents a Version Control System for datasets, which is based on Git and DVC
projects. Each project ``Revision`` describes a build tree of a dataset
with all the related metadata. A build tree consists of a number of data
sources and transformation stages. Each data source has its own set of build
steps (stages). Datumaro supposes copying of datasets and working in-place by
default. Modifying operations are recorded in the project, so any of the
dataset revisions can be reproduced when needed. Multiple dataset versions can
be stored in different branches with the common data shared.

Let's consider an example of a build tree:

.. image:: ../../../images/build_tree.svg
    :name: build tree

There are 2 data sources in the example project. The resulting dataset
is obtained by simple merging (joining) the results of the input datasets.
"Source 1" and "Source 2" are the names of data sources in the project. Each
source has several stages with their own names. The first stage (called "root")
represents the original contents of a data source - the data at the
user-provided URL. The following stages represent operations, which needs to
be done with the data source to prepare the resulting dataset.

Roughly, such build tree can be created by the following commands (arguments
are omitted for simplicity):

.. code-block:: bash

  datum project create

  # describe the first source
  datum project import <...> -n source1
  datum filter <...> source1
  datum transform <...> source1
  datum transform <...> source1

  # describe the second source
  datum project import <...> -n source2
  datum model add <...>
  datum transform <...> source2
  datum transform <...> source2

Now, the resulting dataset can be built with:

.. code-block:: bash

  datum project export <...>


Project layout
--------------

.. code-block:: bash

  project/
  ├── .dvc/
  ├── .dvcignore
  ├── .git/
  ├── .gitignore
  ├── .datumaro/
  │   ├── cache/ # object cache
  │   │   └── <2 leading symbols of obj hash>/
  │   │       └── <remaining symbols of obj hash>/
  │   │           └── <object data>
  │   │
  │   ├── models/ # project-specific models
  │   │
  │   ├── plugins/ # project-specific plugins
  │   │   ├── plugin1/ # composite plugin, a directory
  │   │   |   ├── __init__.py
  │   │   |   └── file2.py
  │   │   ├── plugin2.py # simple plugin, a file
  │   │   └── ...
  │   │
  │   ├── tmp/ # temp files
  │   └── tree/ # working tree metadata
  │       ├── config.yml
  │       └── sources/
  │           ├── <source name 1>.dvc
  │           ├── <source name 2>.dvc
  │           └── ...
  │
  ├── <source name 1>/ # working directory for the source 1
  │   └── <source data>
  └── <source name 2>/ # working directory for the source 2
      └── <source data>


Datasets and Data Sources
-------------------------

A project can contain an arbitrary number of Data Sources. Each Data Source
describes a dataset in a specific format. A project acts as a manager for
the data sources and allows to manipulate them separately or as a whole, in
which case it combines dataset items from all the sources into one composite
dataset. You can manage separate sources in a project by commands in
the `datum source <../command-reference/context/sources>`_ command line context.

Datasets come in a wide variety of formats. Each dataset
format defines its own data structure and rules on how to
interpret the data. For example, the following data structure
is used in COCO format:

.. code-block:: bash

  /dataset/
  - ../../../images/<id>.jpg
  - /annotations/

Datumaro supports complete datasets, having both image data and
annotations, or incomplete ones, having annotations only.
Incomplete datasets can be used to prepare images and annotations
independently of each other, or to analyze or modify just the lightweight
annotations without the need to download the whole dataset.

Check `supported formats <../data-formats/formats/index.rst>`_ for more info
about format specifications, supported import and export options and other
details. The list of formats can be extended by custom plugins,
check `extending tips <../user-manual/extending.md>`_ for information on this
topic.

Use cases
---------

Let's consider few examples describing what Datumaro does for you behind the
scene.

The first example explains how working trees, working directories and the
cache interact. Suppose, there is a dataset which we want to modify and
export in some other format. To do it with Datumaro, we need to create a
project and register the dataset as a data source:

.. code-block:: bash

  datum project create
  datum project import <...> -n source1

The dataset will be copied to the working directory inside the project. It
will be added to the project working tree.

After the dataset is added, we want to transform it and filter out some
irrelevant samples, so we run the following commands:

.. code-block:: bash

  datum transform <...> source1
  datum filter <...> source1

The commands modify the data source inside the working directory, inplace.
The operations done are recorded in the working tree.

Now, we want to make a new version of the dataset and make a snapshot in the
project cache. So we ``commit`` the working tree:

.. code-block:: bash

  datum project commit <...>

.. image:: ../../../images/behavior_diag1.svg
    :name: cache interaction diagram 1

At this time, the data source is copied into the project cache and a new
project revision is created. The dataset operation history is saved, so
the dataset can be reproduced even if it is removed from the cache and the
working directory. Note, however, that the original dataset hash was not
computed, so Datumaro won't be able to compare dataset hash on re-downloading.
If it is desired, consider making a ``commit`` with an unmodified data source.

After this, we do some other modifications to the dataset and make a new
commit. Note that the dataset is not cached, until a ``commit`` is done.

When the dataset is ready and all the required operations are done, we
can `export` it to the required format. We can export the resulting dataset,
or any previous stage.

.. code-block:: bash

  datum project export <...> source1
  datum project export <...> source1.stage3

Let's extend the example. Imagine we have a project with 2 data sources.
Roughly, it corresponds to the following set of commands:

.. code-block:: bash

  datum project create
  datum project import <...> -n source1
  datum project import <...> -n source2
  datum transform <...> source1 # used 3 times
  datum transform <...> source2 # used 5 times

Then, for some reasons, the project cache was cleaned from ``source1`` revisions.
We also don't have anything in the project working directories - suppose,
the user removed them to save disk space.

Let's see what happens, if we call the ``compare`` command with 2 different
revisions now.

.. image:: ../../../images/behavior_diag2.svg
    :name: cache interaction diagram 2

Datumaro needs to reproduce 2 dataset revisions requested so that they could
be read and compared. Let's see how the first dataset is reproduced
step-by-step:

- ``source1.stage2`` will be looked for in the project cache. It won't be found, since the
  cache was cleaned.

- Then, Datumaro will look for previous source revisions in the cache and won't find any.

- The project can be marked read-only, if we are not working with the "current" project
  (which is specified by the ``-p/--project`` command parameter). In the example, the command is
  ``datum compare rev1:... rev2:...``, which means there is a project in the current directory, so the
  project we are working with is not read-only. If a command target was specified as
  ``datum compare <project>@<rev>:<source>``, the project would be loaded as read-only. If a project is
  read-only, we can't do anything more to reproduce the dataset and can only exit with an error (3a).
  The reason for such behavior is that the dataset downloading can be quite expensive (in terms of
  time, disk space etc.). It is supposed, that such side-effects should be controlled manually.

- If the project is not read-only (3b), Datumaro will try to download the original dataset
  and reproduce the resulting dataset. The data hash will be computed and hashes will be compared (if
  the data source had hash computed on addition). On success, the data will be put into the cache.

- The downloaded dataset will be read and the remaining operations from the source history will be
  re-applied.

- The resulting dataset might be cached in some cases.

- The resulting dataset is returned.


The ``source2`` will be looked for the same way. In our case, it will be found
in the cache and returned. Once both datasets are restored and read, they
are compared.

Consider other situation. Let's try to ``export`` the ``source1``. Suppose
we have a clear project cache and the ``source1`` has a copy in the working
directory.

.. image:: ../../../images/behavior_diag3.svg
    :name: cache interaction diagram 3

Again, Datumaro needs to reproduce a dataset revision (stage) requested.

- It looks for the dataset in the working directory and finds some data. If there is no source
  working directory, Datumaro will try to reproduce the source using the approach described above (1b).

- The data hash is computed and compared with the one saved in the history. If the hashes match,
  the dataset is read and returned (4). Note: we can't use the cached hash stored in the working tree
  info - it can be outdated, so we need to compute it again.

- Otherwise, Datumaro tries to detect the stage by the data hash. If the current stage is not
  cached, the tree is the working tree and the working directory is not empty, the working copy is
  hashed and matched against the source stage list. If there is a matching stage, it will be read and
  the missing stages will be added. The result might be cached in some cases. If there is no matching
  stage in the source history, the situation can be contradictory. Currently, an error is raised (3b).

- The resulting dataset is returned.


After the requested dataset is obtained, it is exported in the requested format.

To sum up, Datumaro tries to restore a dataset from the project cache or
reproduce it from sources. It can be done as long as the source operations
are recorded and any step data is available. Note that cache objects share
common files, so if there are only annotation differences between datasets,
or data sources contain the same images, there will only be a single copy
of the related media files. This helps to keep storage use reasonable and
avoid unnecessary data copies.
