Supported Plugins
#################

List of plugins available through the CLI

transform module
================

Applies a batch operation to a dataset and produces a new dataset.

Learn more about :mod:`transform <datumaro.plugins.transforms>`

To get help, run:

    .. code-block::

        datum transform -h

To get help by class and modules, run:

- :mod:`Rename <datumaro.plugins.transforms.Rename>`
    Renames items in the dataset

    .. code-block::

        datum transform -t rename -- -h

- :mod:`RemapLabels <datumaro.plugins.transforms.RemapLabels>`
    Changes labels in the dataset.

    .. code-block::

        datum transform -t remap_labels -- -h

- :mod:`ProjectLabels <datumaro.plugins.transforms.ProjectLabels>`
    Changes the order of labels in the dataset from the existing
    to the desired one, removes unknown labels and adds new labels.
    Updates or removes the corresponding annotations.

    .. code-block::

        datum transform -t project_labels -- -h

- :mod:`ResizeTransform <datumaro.plugins.transforms.ResizeTransform>`
    Resizes images and annotations in the dataset to the specified size.

    .. code-block::

        datum transform -t resize -- -h

- :mod:`RemoveItems <datumaro.plugins.transforms.RemoveItems>`
    Allows to remove specific dataset items from dataset by their ids.

    .. code-block::

        datum transform -t remove_items -- -h

- :mod:`RemoveAnnotations <datumaro.plugins.transforms.RemoveAnnotations>`
    Allows to remove annotations on specific dataset items.

    .. code-block::

        datum transform -t remove_annotations -- -h

- :mod:`RemoveAttributes <datumaro.plugins.transforms.RemoveAttributes>`
    Allows to remove item and annotation attributes in a dataset.

    .. code-block::

        datum transform -t remove_attributes -- -h

- :mod:`NDR <datumaro.plugins.ndr>`
    Removes near-duplicated images in subset.
    Remove duplicated images from a dataset.

    .. code-block::

        datum transform -t ndr -- -h

- :mod:`Splitter <datumaro.plugins.splitter>`

    .. code-block::

        datum transform -t split -- -h

- :ref:`sampler package`

    - :mod:`RandomSampler <datumaro.plugins.sampler.random_sampler.RandomSampler>`
        Sampler that keeps no more than required number of items
        in the dataset.

        .. code-block::

            datum transform -t random_sampler -- -h

    - :mod:`LabelRandomSampler <datumaro.plugins.sampler.random_sampler.LabelRandomSampler>`
        Sampler that keeps at least the required number of annotations
        of each class in the dataset for each subset separately.

        .. code-block::

            datum transform -t label_random_sampler -- -h
