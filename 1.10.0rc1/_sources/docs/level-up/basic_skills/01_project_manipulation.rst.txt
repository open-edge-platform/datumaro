=============================
Level 1: Project Manipulation
=============================

Projects are designed to facilitate the use of Datumaro CLI commands for complex tasks.
If you need more information about the project's concept and meaning, please refer to :ref:`Projects`.
If you plan to perform various dataset management tasks using the Datumaro Python API instead of CLI commands, you may be able to skip this page.
However, even when using the Python API, you will need to refer back to the project if you require dataset versioning.

Create a project
================

.. tab-set::

  .. tab-item:: ProjectCLI

    The following command creates a project in the current working directory.

    .. code-block:: bash

      cd <path/where/you/want>
      datum project create

Import a dataset to the project
===============================

.. tab-set::

  .. tab-item:: ProjectCLI

    After creating the project, you can import datasets to the project.
    In this example, we import :ref:`Cityscapes` dataset to the project with naming it ``my-dataset``.

    .. code-block:: bash

      datum project import -n my-dataset --format cityscapes -p <path/to/project> <path/to/cityscapes>

Remove a dataset from the project
=================================

.. tab-set::

  .. tab-item:: ProjectCLI

    Conversely, it is also possible to delete a dataset that has been added to the project.
    This command removes the :ref:`Cityscapes` dataset named ``my-dataset`` from the previous step.

    .. code-block:: bash

      datum project remove my-dataset

Print dataset information
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    We can also check the dataset information added in the project.
    This is an example of printing information about the dataset named ``my-dataset``.

    .. code-block:: bash

      datum dinfo my-dataset

Add model to project
====================

.. tab-set::

  .. tab-item:: ProjectCLI

    You can add an AI model into a project. The model requires an inference launcher for its model format.
    Currently, we only support `OpenVINO™ <https://github.com/openvinotoolkit/openvino>`_ launcher.
    Here is an example to add an `OpenVINO™ <https://github.com/openvinotoolkit/openvino>`_ model to the project.

    .. code-block:: bash

      datum model add -n my-model -l openvino -- -d <path/to/model.xml> -w <path/to/model.bin> -i <path/to/interpreter.py>

    .. note::
      In addition to entering the path to the model weights (``-w WEIGHTS``) and metafiles (``-d DESCRIPTION``),
      you must enter the interpreter file path (``-i INTERPRETER``) written in Python to interpret that model output as well.

      .. collapse:: An example of the interpreter (``ssd_mobilenet_coco_detection_interp.py``)

        .. literalinclude:: ../../../../../datumaro/plugins/openvino_plugin/samples/ssd_mobilenet_coco_detection_interp.py
          :language: python

Remove model from project
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    We can remove ``my-model`` model from the project as follows.

    .. code-block:: bash

      datum model remove my-model

Print project information
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    We can print an overall information of the project.

    .. code-block:: bash

      datum project info

.. note::
  Many CLI commands, including those we introduce above, have ``-p PROJECT_DIR``, ``--project PROJECT_DIR`` CLI arguments.
  This argument allows you to specify the path of the target project where the CLI operation will be executed.
  This is useful if you don't want to change your current working directory.
