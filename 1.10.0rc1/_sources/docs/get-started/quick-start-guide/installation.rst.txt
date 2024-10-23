Installation
############

Dependencies
------------

- Python (3.9+)
- Optional: OpenVINO, TensorFlow, PyTorch, MxNet, Caffe, Accuracy Checker, Git

Installation steps
------------------

Optionally, set up a virtual environment:

.. code-block::

    python -m pip install virtualenv
    python -m virtualenv venv
    . venv/bin/activate

Install:

1. From PyPI (**recommended**)

    .. code-block::

        pip install datumaro[default]


    Alternatively, for zsh users:

    .. code-block::

        pip install 'datumaro[default]'


2. From the GitHub repository (**not recommended, for advanced users**)

    Installation from the repository source is not recommended.
    This is because it requires that C++ and Rust build systems are prepared in your local environment before installation.
    Datumaro includes C++ and Rust implementations to accelerate some workloads to overcome Python's innate slowness.

    .. code-block::

        # Prerequisite (For Unix-like systems)
        # Install C++ build system
        sudo apt-get install build-essential
        # Install Rust build system
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

        # Install from the GitHub repository
        pip install 'datumaro[default] @ git+https://github.com/openvinotoolkit/datumaro'


Plugins
^^^^^^^

Datumaro has many plugins, which are responsible for dataset formats,
model launchers and other optional components. If a plugin has dependencies,
they can require additional installation. You can find the list of all the
plugin dependencies in the [plugins](/docs/user-manual/extending) section.

``Optional dependencies``
These components are only required for plugins and not installed by default:

- OpenVINO
- Accuracy Checker
- TensorFlow
- PyTorch
- MxNet
- Caffe

Customizing installation
^^^^^^^^^^^^^^^^^^^^^^^^

- Datumaro has the following installation options:

  - ``pip install datumaro`` - for core library functionality
  - ``pip install datumaro[default]`` - for normal CLI experience

  In restricted installation environments, where some dependencies are
  not available, or if you need only the core library functionality,
  you can install Datumaro without extra plugins.

  The CLI variant (``datumaro[default]``) requires Git to be installed and
  available to work with Datumaro projects and dataset versioning features.
  You can find installation instructions for your platform `here <https://git-scm.com/downloads>`_.

  In some cases, installing just the core library may be not enough,
  because there can be limited options of installing graphical libraries
  in the system (various Docker environments, servers etc). You can select
  between using ``opencv-python`` and ``opencv-python-headless`` by setting the
  ``DATUMARO_HEADLESS`` environment variable to ``0`` or ``1`` before installing
  the package. It requires installation from sources (using ``--no-binary``):
  This option can't be covered by extras due to Python packaging system limitations.

.. code-block::

    DATUMARO_HEADLESS=1 pip install datumaro --no-binary=datumaro

- When installing directly from the repository, you can change the
  installation branch with ``...@<branch_name>``. Also use ``--force-reinstall``
  parameter in this case. It can be useful for testing of unreleased
  versions from GitHub pull requests.
