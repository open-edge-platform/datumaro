===========================
Level 11: Data Generation
===========================


Pre-training of deep learning models for vision tasks can increase model accuracy.
Training model with the synthetic dataset is one of famous pre-training approach
since the manual annotations is quite expensive work.

Base on the [FractalDB]_,
Datumaro provides a fractal image dataset (FractalDB) generator that can be utilized to pre-train the vision models.
Learning visual features of FractalDB is known to increase the performance of Vision Transformer (ViT) models.
Note that a fractal patterns in FractalDB is calculated mathmatically using the interated function system (IFS) with random parameters.
We thus don't need to concern about any privacy issues.


.. tab-set::

  .. tab-item:: CLI

    We can ``generate`` the synthetic images by the following CLI command:

    .. code-block:: bash

      datum generate -o <path/to/data> --count GEN_IMG_COUNT --shape GEN_IMG_SHAPE

    ``GEN_IMG_COUNT`` is an integer that indicates the number of images to be generated. (e.g. ``--count 300``)
    ``GEN_IMG_SHAPE`` is the shape (width height) of generated images (e.g. ``--shape 240 180``)

  .. tab-item:: Python

    With Pthon API, we can generate the synthetic images as below.

    .. code-block:: python

        from datumaro.plugins.synthetic_data import FractalImageGenerator

        FractalImageGenerator(output_dir=<path/to/data>, count=GEN_IMG_COUNT, shape=GEN_IMG_SHAPE).generate_dataset()

    ``GEN_IMG_COUNT`` is an integer that indicates the number of images to be generated. (e.g. ``count=300``)
    ``GEN_IMG_SHAPE`` is a tuple representing the shape of generated images as (width, height) (e.g. ``shape=(240, 180)``)

Congratulations! You completed reading all Datumaro level-up documents for the intermediate skills.

.. [FractalDB] Nakashima, Kodai, et al. "Can vision transformers learn without natural images?." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 2. 2022.
