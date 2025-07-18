Architecture
############
.. _developer_manual:

Basics
------

The central part of the library is the :mod:`Dataset <datumaro.components.dataset.Dataset>` class, which represents
a dataset and allows to iterate over its elements.
:mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` , an element of a dataset, represents a single
dataset entry with annotations - an image, video sequence, audio track etc.
It can contain only annotated data or meta information, only annotations, or
all of this.

Basic library usage and data flow:

.. code-block::

   Extractors -> Dataset -> Converter
                    |
                Filtration
             Transformations
                Statistics
                 Merging
                Inference
             Quality Checking
                Comparison
                   ...


#. Data is read (or produced) by one or many :mod:`Extractor <datumaro.components.extractor.Extractor>` s and
   `merged <#dataset-merging>`_ into a :mod:`Dataset <datumaro.components.dataset.Dataset>`
#. The dataset is processed in some way
#. The dataset is saved with a :mod:`Converter <datumaro.components.converter.Converter>`

Datumaro has a number of dataset and annotation features:


* iteration over dataset elements
* filtering of datasets and annotations by a custom criteria
* working with subsets (e.g. `train` , `val` , `test` )
* computing of dataset statistics
* comparison and merging of datasets
* various annotation operations

.. code-block:: python

   from datumaro import Bbox, Polygon, Dataset, DatasetItem

   # Import and export a dataset
   dataset = Dataset.import_from('src/dir', 'voc')
   dataset.export('dst/dir', 'coco')

   # Create a dataset, convert polygons to masks, save in PASCAL VOC format
   dataset = Dataset.from_iterable([
     DatasetItem(id='image1', annotations=[
       Bbox(x=1, y=2, w=3, h=4, label=1),
       Polygon([1, 2, 3, 2, 4, 4], label=2, attributes={'occluded': True}),
     ]),
   ], categories=['cat', 'dog', 'person'])
   dataset.transform('polygons_to_masks')
   dataset.export('dst/dir', 'voc')

The Dataset class
^^^^^^^^^^^^^^^^^

The :mod:`Dataset <datumaro.components.dataset.Dataset>` class from the :mod:`datumaro.components.dataset` module represents
a dataset, consisting of multiple :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` s. Annotations are
represented by members of the :mod:`datumaro.components.extractor` module,
such as :mod:`Label <datumaro.components.annotation.Label>` , :mod:`Mask <datumaro.components.annotation.Mask>`
or :mod:`Polygon <datumaro.components.annotation.Polygon>`. A dataset can contain items from one or
multiple subsets (e.g. `train` , `test` , `val` etc.), the list of dataset
subsets is available in :mod:`dataset.subsets() <datumaro.components.dataset.Dataset.subsets>`.

A :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` is an element of a dataset.
Its :mod:`id <datumaro.components.extractor.DatasetItem.id>` is the name of the
corresponding image, video frame, or other media being annotated.
An item can have some :mod:`attributes <datumaro.components.extractor.DatasetItem.attributes>` , associated media info and :mod:`annotations <datumaro.components.extractor.DatasetItem.annotations>`.

Datasets typically have annotations, and these annotations can
require additional information to be interpreted correctly. For instance, it
can be class names, class hierarchy, keypoint connections,
class colors for masks, class attributes. Such information is stored in
:mod:`dataset.categories() <datumaro.components.dataset.Dataset.categories>`,
which is a mapping from :mod:`AnnotationType <datumaro.components.annotation.AnnotationType>`
to a corresponding `...Categories` class. Each annotation type
can have its :mod:`Categories <datumaro.components.annotation.Categories>`. Typically, there will be at least :mod:`LabelCategories <datumaro.components.annotation.LabelCategories>` ;
if there are instance masks, the dataset will contain :mod:`MaskCategories <datumaro.components.annotation.MaskCategories>` etc.
The "main" type of categories is :mod:`LabelCategories <datumaro.components.annotation.LabelCategories>` - annotations and other
categories use label indices from this object.

The main operation for a dataset is iteration over its elements
( :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` s). An item corresponds to a single image, a video sequence,
etc. There are also many other operations available, such as filtration
( :mod:`dataset.select() <datumaro.components.dataset.Dataset.select>` ), transformation (:mod:`dataset.transform()<datumaro.components.dataset.Dataset.transform>`),
exporting ( :mod:`dataset.export() <datumaro.components.dataset.Dataset.export>` ) and others. A :mod:`Dataset <datumaro.components.dataset.Dataset>` is an `Iterable` and
`Extractor` by itself.

A :mod:`Dataset <datumaro.components.dataset.Dataset>` can be created from scratch by its class constructor.
Categories can be set immediately or later with the
:mod:`define_categories() <datumaro.components.dataset.Dataset.define_categories>` method, but only once. You can create a dataset filled
with initial :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` s with :mod:`Dataset.from_iterable() <datumaro.components.dataset.Dataset.from_iterable>`.
If you need to create a dataset from one or many other extractors
(or datasets), it can be done with :mod:`Dataset.from_extractors() <datumaro.components.dataset.Dataset.from_extractors>`.

If a dataset is created from multiple extractors with
:mod:`Dataset.from_extractors() <datumaro.components.dataset.Dataset.from_extractors>` , the source datasets will be `joined <#dataset-merging>`_ ,
so their categories must match. If datasets have mismatching categories,
use the more complex :mod:`IntersectMerge <datumaro.components.operations.IntersectMerge>` class from :mod:`datumaro.components.operations` ,
which will merge all the labels and remap the shifted indices in annotations.

A :mod:`Dataset <datumaro.components.dataset.Dataset>` can be loaded from an existing dataset on disk with
:mod:`Dataset.import_from() <datumaro.components.dataset.Dataset.import_from>` (for arbitrary formats) and
:mod:`Dataset.load() <datumaro.components.dataset.Dataset.load>` (for the Datumaro data format).

By default, :mod:`Dataset <datumaro.components.dataset.Dataset>` works lazily, which means all the operations requiring
iteration over inputs will be deferred as much as possible. If you don't want
such behavior, use the :mod:`init_cache() <datumaro.components.dataset.Dataset.init_cache>` method or wrap the code in
:mod:`eager_mode <datumaro.components.dataset.eager_mode>` (from :mod:`datumaro.components.dataset` ), which will load all
the annotations into memory. The media won't be loaded unless the data
is required, because it can quickly waste all the available memory.
You can check if the dataset is cached with the :mod:`is_cache_initialized <datumaro.components.dataset.Dataset.is_cache_initialized>`
attribute.

Once created, a dataset can be modified in batch mode with transforms or
directly with the :mod:`put() <datumaro.components.dataset.Dataset.put>` and :mod:`remove() <datumaro.components.dataset.Dataset.remove>` methods. :mod:`Dataset <datumaro.components.dataset.Dataset>` instances
record information about changes done, which can be obtained by :mod:`get_patch() <datumaro.components.dataset.Dataset.get_patch>`.
The patch information is used automatically on saving and exporting to
reduce the amount of disk writes. Changes can be flushed with
:mod:`flush_changes() <datumaro.components.dataset.Dataset.flush_changes>`.

.. code-block:: python

   from datumaro import Bbox, Label, Polygon, Dataset, DatasetItem

   # create a dataset directly from items
   dataset1 = Dataset.from_iterable([
       DatasetItem(id='image1', annotations=[
           Bbox(x=1, y=2, w=3, h=4, label=1),
           Polygon([1, 2, 3, 2, 4, 4], label=2),
       ]),
   ], categories=['cat', 'dog', 'person', 'truck'])

   dataset2 = Dataset(categories=dataset1.categories())
   dataset2.put(DatasetItem(id='image2', annotations=[
       Label(label=3),
       Bbox(x=2, y=0, w=3, h=1, label=2)
   ]))

   # create a dataset from other datasets
   dataset = Dataset.from_extractors(dataset1, dataset2)

   # keep only annotated images
   dataset.select(lambda item: len(item.annotations) != 0)

   # change dataset labels
   dataset.transform('remap_labels',
       {
           'cat': 'dog', # rename cat to dog
           'truck': 'car', # rename truck to car
           'person': '', # remove this label
       },
       default='delete')

   # iterate over elements
   for item in dataset:
       print(item.id, item.annotations)

   # iterate over subsets as Datasets
   for subset_name, subset in dataset.subsets().items():
       for item in subset:
           print(item.id, item.annotations)

.. _dataset_merging:

Dataset merging
~~~~~~~~~~~~~~~

There are 2 methods of merging datasets in Datumaro:


* simple merging ("joining")
* complex merging

The simple merging ("joining")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach finds the corresponding :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` s in inputs,
finds equal annotations and leaves only the unique set of annotations.
This approach requires all the inputs to have categories with the same
labels (or no labels) in the same order.

This algorithm is applied automatically in :mod:`Dataset.from_extractors() <datumaro.components.dataset.Dataset.from_extractors>`
and when the build targets are merged in the :mod:`Project.Tree.make_dataset() <datumaro.components.project.Tree.make_dataset>`.

The complex merging
~~~~~~~~~~~~~~~~~~~

If datasets have mismatching categories, they can't be
merged by the simple approach, because it can lead to errors in the
resulting annotations. For complex cases Datumaro provides a more
sophisticated algorithm, which finds matching annotations by computing
distances between them. Labels and attributes are deduced by voting,
spatial annotations use the corresponding metrics like
Intersection-over-Union (IoU), OKS, PDJ and others.

The categories of the input datasets are compared, the matching ones
complement missing information in each other, the mismatching ones are
appended after next. Label indices in annotations are shifted to the
new values.

The complex algorithm is available in the :mod:`IntersectMerge <datumaro.components.operations.IntersectMerge>` class
from :mod:`datumaro.components.operations`. It must be used explicitly.
This class also allows to check the inputs and the output dataset
for errors and problems.

Projects
^^^^^^^^

Projects are intended for complex use of Datumaro. They provide means of
persistence, versioning, high-level operations for datasets and also
allow to extend Datumaro via :ref:`Plugins`. A project provides
access to build trees and revisions, data sources, models, configuration,
plugins and cache. Projects can have multiple data sources, which are
`joined <#dataset-merging>`_ on dataset creation. Project configuration is available
in :mod:`project.config <datumaro.components.project.Project.config>`. To add a data source into a :mod:`Project <datumaro.components.project.Project>` , use
the :mod:`import_source() <datumaro.components.project.Project.import_source>` method. The build tree of the current working
directory can be converted to a :mod:`Dataset <datumaro.components.dataset.Dataset>` with
:mod:`project.working_tree.make_dataset() <datumaro.components.project.Project.working_tree>`.

The :mod:`Environment <datumaro.components.environment>` class is responsible for accessing built-in and
project-specific plugins. For a :mod:`Project <datumaro.components.project.Project>` object, there is an instance of
related :mod:`Environment <datumaro.components.environment>` in :mod:`project.env <datumaro.components.project.Project.env>`.

Check the :ref:`Data Model section of the User Manual <Project data model>`:
for more info about Project behavior and high-level details.

Library contents
----------------

Dataset Formats
^^^^^^^^^^^^^^^

The framework provides functions to read and write datasets in specific formats.
It is supported by :mod:`Extractor <datumaro.components.extractor>` s, :mod:`Importer <datumaro.plugins.coco_format.importer>` s, and :mod:`Converter <datumaro.components.converter.Converter>` s.

Dataset reading is supported by :mod:`Extractor <datumaro.components.extractor>` s and :mod:`Importer <datumaro.plugins.coco_format.importer>` s:

* An :mod:`Extractor <datumaro.components.extractor>` produces a list of :mod:`DatasetItem <datumaro.components.extractor.DatasetItem>` s corresponding to the
  dataset. Annotations are available in the :mod:`DatasetItem.annotations <datumaro.components.extractor.DatasetItem.annotations>` list.
  The :mod:`SourceExtractor <datumaro.components.extractor.SourceExtractor>` class is designed for loading simple, single-subset
  datasets. It should be used by default. The :mod:`Extractor <datumaro.components.extractor>` base class should
  be used when :mod:`SourceExtractor <datumaro.components.extractor.SourceExtractor>` 's functionality is not enough.
* An :mod:`Importer <datumaro.plugins.coco_format.importer>` detects dataset files and generates dataset loading parameters
  for the corresponding :mod:`Extractor <datumaro.components.extractor>` s. :mod:`Importer <datumaro.plugins.coco_format.importer>` s are optional, they
  only extend the Extractor functionality and make them more flexible and
  simple. They are mostly used to locate dataset subsets, but they also can
  do some data compatibility checks and have other required logic.

It is possible to add custom :mod:`Extractor <datumaro.components.extractor>` s and :mod:`Importer <datumaro.plugins.coco_format.importer>` s. To do this, you need
to put an :mod:`Extractor <datumaro.components.extractor>` and :mod:`Importer <datumaro.plugins.coco_format.importer>` implementations to a plugin directory.

Dataset writing is supported by :mod:`Converter <datumaro.components.converter.Converter>` s.
A :mod:`Converter <datumaro.components.converter.Converter>` produces a dataset of a specific format from dataset items.
It is possible to add custom :mod:`Converter <datumaro.components.converter.Converter>` s. To do this, you need to put a
:mod:`Converter <datumaro.components.converter.Converter>` implementation script to a plugin directory.

Dataset Conversions ("Transforms")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :mod:`Transform <datumaro.components.extractor.Transform>` is a function for altering a dataset and producing a new one.
It can update dataset items, annotations, classes, and other properties.
A list of available transforms for dataset conversions can be extended by
adding a :mod:`Transform <datumaro.components.extractor.Transform>` implementation script into a plugin directory.

Model launchers
^^^^^^^^^^^^^^^

A list of available launchers for model execution can be extended by
adding a :mod:`Launcher <datumaro.components.launcher.Launcher>` implementation script into a plugin directory.

Plugins
-------

Datumaro comes with a number of built-in formats and other tools,
but it also can be extended by plugins. Plugins are optional components,
which dependencies are not installed by default.
In Datumaro there are several types of plugins, which include:


* :mod:`Extractor <datumaro.components.extractor>` - produces dataset items from data source
* :mod:`Importer <datumaro.plugins.coco_format.importer>` - recognizes dataset type and creates project
* :mod:`Converter <datumaro.components.converter.Converter>` - exports dataset to a specific format
* :mod:`transformation <datumaro.plugins.transforms>` - modifies dataset items or other properties
* :mod:`launcher <datumaro.components.launcher>` - executes models

A plugin is a regular Python module. It must be present in a plugin directory:


* ``<project_dir>/.datumaro/plugins`` for project-specific plugins
* ``<datumaro_dir>/plugins`` for global plugins

A plugin can be used either via the :mod:`Environment <datumaro.components.environment>` class instance,
or by regular module importing:

.. code-block:: python

   import datumaro as dm
   from datumaro.plugins.yolo_format.converter import YoloConverter

   # Import a dataset
   dataset = dm.Dataset.import_from(src_dir, 'voc')

   # Load an existing project, save the dataset in some project-specific format
   project = dm.project.Project('project/')
   project.env.converters['custom_format'].convert(dataset, save_dir=dst_dir)

   # Save the dataset in some built-in format
   dm.Environment().converters['yolo'].convert(dataset, save_dir=dst_dir)
   YoloConverter.convert(dataset, save_dir=dst_dir)

:ref:`Using datumaro as a python module <datumaro>`

Writing a plugin
^^^^^^^^^^^^^^^^

A plugin is a Python module with any name, which exports some symbols. Symbols,
starting with ``_`` are not exported by default. To export a symbol,
inherit it from one of the special classes:

.. code-block:: python

   from datumaro import Importer, Extractor, Transform, Launcher, Converter

The `exports` list of the module can be used to override default behavior:

.. code-block:: python

   class MyComponent1: ...
   class MyComponent2: ...
   exports = [MyComponent2] # exports only MyComponent2

There is also an additional class to modify plugin appearance in command line:

.. code-block:: python

   from datumaro import Converter

   class MyPlugin(Converter):
       """
       Optional documentation text, which will appear in command-line help
       """

       NAME = 'optional_custom_plugin_name'

       def build_cmdline_parser(self, **kwargs):
           parser = super().build_cmdline_parser(**kwargs)
           # set up argparse.ArgumentParser instance
           # the parsed args are supposed to be used as invocation options
           return parser

Plugin example
~~~~~~~~~~~~~~

.. code-block::

   datumaro/plugins/
   - my_plugin1/file1.py
   - my_plugin1/file2.py
   - my_plugin2.py

``my_plugin1/file2.py`` contents:

.. code-block:: python

   from datumaro import Transform
   from .file1 import something, useful

   class MyTransform(Transform):
       NAME = "custom_name" # could be generated automatically

       """
       Some description. The text will be displayed in the command line output.
       """

       @classmethod
       def build_cmdline_parser(cls, **kwargs):
           parser = super().build_cmdline_parser(**kwargs)
           parser.add_argument('-q', help="Very useful parameter")
           return parser

       def __init__(self, extractor, q):
           super().__init__(extractor)
           self.q = q

       def transform_item(self, item):
           return item

``my_plugin2.py`` contents:

.. code-block:: python

   from datumaro import Converter, Extractor

   class MyFormat: ...
   class _MyFormatConverter(Converter): ...
   class MyFormatExtractor(Extractor): ...

   exports = [MyFormat] # explicit exports declaration
   # MyFormatExtractor and _MyFormatConverter won't be exported

Command-line
------------

Basically, the interface is divided on contexts and single commands.
Contexts are semantically grouped commands, related to a single topic or target.
Single commands are handy shorter alternatives for the most used commands
and also special commands, which are hard to be put into any specific context.
`Docker <https://www.docker.com/>`_ is an example of similar approach.

.. graphviz:: command_line.dot

:ref:`List of plugins available through the CLI <Supported Plugins>`

Model-View-ViewModel (MVVM) UI pattern is used.

.. raw:: html
   :file: ../../../images/command_diag2.svg
