# Extending

There are few ways to extend and customize Datumaro behavior, which is
supported by plugins. Check [our contribution guide](https://github.com/openvinotoolkit/datumaro/blob/develop/contributing.md)
for details on plugin implementation. In general, a plugin is a Python module.
It must be put into a plugin directory:
- `<project_dir>/.datumaro/plugins` for project-specific plugins
- `<datumaro_dir>/plugins` for global plugins

## Built-in plugins

Datumaro provides several builtin plugins. Plugins can have dependencies,
which need to be installed separately.

### TensorFlow

The plugin provides support of TensorFlow Detection API format, which includes
boxes and masks.

**Dependencies**

The plugin depends on TensorFlow, which can be installed with `pip`:

``` bash
pip install tensorflow
```
or
``` bash
pip install tensorflow-gpu
```
or
``` bash
pip install datumaro[tf]
```
or
``` bash
pip install datumaro[tf-gpu]
```

### Accuracy Checker

This plugin allows to use [Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker)
to launch deep learning models from various frameworks
(Caffe, MxNet, PyTorch, OpenVINO, ...) through Accuracy Checker's API.

**Dependencies**

The plugin depends on Accuracy Checker, which can be installed with `pip`:

``` bash
pip install 'git+https://github.com/openvinotoolkit/open_model_zoo.git#subdirectory=tools/accuracy_checker'
```

To execute models with deep learning frameworks, they need to be installed too.

### OpenVINO™

This plugin provides support for model inference with [OpenVINO™](https://01.org/openvinotoolkit).

**Dependencies**

The plugin depends on the OpenVINO™ Toolkit, which can be installed by
following [these instructions](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_runtime.html)

## Dataset Formats

Dataset reading is supported by Extractors and Importers.
An Extractor produces a list of dataset items corresponding
to the dataset. An Importer creates a project from the data source location.
It is possible to add custom Extractors and Importers. To do this, you need
to put an Extractor and Importer implementation scripts to a plugin directory.

Dataset writing is supported by Converters.
A Converter produces a dataset of a specific format from dataset items.
It is possible to add custom Converters. To do this, you need to put a Converter
implementation script to a plugin directory.

## Dataset Conversions ("Transforms")

A Transform is a function for altering a dataset and producing a new one.
It can update dataset items, annotations, classes, and other properties.
A list of available transforms for dataset conversions can be extended by
adding a Transform implementation script into a plugin directory.

## Model launchers

A list of available launchers for model execution can be extended by adding
a Launcher implementation script into a plugin directory.
