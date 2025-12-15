Usage
#####

There are several options available:

Standalone tool
---------------

Datuaro as a standalone tool allows to do various dataset operations from
the command line interface:

.. code-block::

    datum --help
    python -m datumaro --help

Python module
-------------

Datumaro can be used in custom scripts as a Python module. Used this way, it
allows to use its features from an existing codebase, enabling dataset
reading, exporting and iteration capabilities, simplifying integration of custom
formats and providing high performance operations:

.. code-block::

    import datumaro as dm

    dataset = dm.Dataset.import_from('path/', 'voc')

    # keep only annotated images
    dataset.select(lambda item: len(item.annotations) != 0)

    # change dataset labels and corresponding annotations
    dataset.transform('remap_labels',
        mapping={
          'cat': 'dog', # rename cat to dog
          'truck': 'car', # rename truck to car
          'person': '', # remove this label
        },
        default='delete') # remove everything else

    # iterate over the dataset elements
    for item in dataset:
        print(item.id, item.annotations)

    # export the resulting dataset in COCO format
    dataset.export('dst/dir', 'coco', save_media=True)
