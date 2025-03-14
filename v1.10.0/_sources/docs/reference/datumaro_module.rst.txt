Datumaro Module
===============
.. _datumaro_module:

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: _autosummary

    datumaro

The recommended way to use Datumaro API is:

.. code-block:: python

    import datumaro as dm

Once you can use the components:

* :mod:`dm.components.annotation <datumaro.components.annotation>`
    * :class:`dm.Annotation <datumaro.components.annotation>`
    * :class:`dm.AnnotationType <datumaro.components.annotation>`
    * :class:`dm.Bbox <datumaro.components.annotation>`
    * :class:`dm.Caption <datumaro.components.annotation>`
    * :class:`dm.Categories <datumaro.components.annotation>`
    * :class:`dm.CompiledMask <datumaro.components.annotation>`
    * :class:`dm.Cuboid3d <datumaro.components.annotation>`
    * :class:`dm.Ellipse <datumaro.components.annotation>`
    * :class:`dm.Label <datumaro.components.annotation>`
    * :class:`dm.LabelCategories <datumaro.components.annotation>`
    * :class:`dm.Mask <datumaro.components.annotation>`
    * :class:`dm.MaskCategories <datumaro.components.annotation>`
    * :class:`dm.Points <datumaro.components.annotation>`
    * :class:`dm.PointsCategories <datumaro.components.annotation>`
    * :class:`dm.PolyLine <datumaro.components.annotation>`
    * :class:`dm.Polygon <datumaro.components.annotation>`
    * :class:`dm.RleMask <datumaro.components.annotation>`
* :mod:`dm.components.media <datumaro.components.media>`
    * :class:`dm.ByteImage <datumaro.components.media>`
    * :class:`dm.Image <datumaro.components.media>`
    * :class:`dm.MediaElement <datumaro.components.media>`
    * :class:`dm.Video <datumaro.components.media>`
    * :class:`dm.VideoFrame <datumaro.components.media>`
* :mod:`dm.components.cli_plugin <datumaro.components.cli_plugin>`
    * :class:`dm.CliPlugin <datumaro.components.cli_plugin>`
* :mod:`dm.components.dataset <datumaro.components.dataset>`
    * :class:`dm.Dataset <datumaro.components.dataset>`
    * :class:`dm.DatasetPatch <datumaro.components.dataset>`
    * :class:`dm.DatasetSubset <datumaro.components.dataset>`
    * :class:`dm.ItemStatus <datumaro.components.dataset>`
* :mod:`dm.components.dataset_base <datumaro.components.dataset_base>`
    * :class:`dm.DatasetBase <datumaro.components.dataset_base>`
    * :class:`dm.DatasetItem <datumaro.components.dataset_base>`
    * :class:`dm.IDataset <datumaro.components.dataset_base>`
    * :class:`dm.SubsetBase <datumaro.components.dataset_base>`
* :mod:`dm.components.environment <datumaro.components.environment>`
    * :class:`dm.Environment <datumaro.components.environment>`
    * :class:`dm.PluginRegistry <datumaro.components.environment>`
* :mod:`dm.components.exporter <datumaro.components.exporter>`
    * :class:`dm.ExportErrorPolicy <datumaro.components.exporter>`
    * :class:`dm.Exporter <datumaro.components.exporter>`
    * :class:`dm.FailingExportErrorPolicy <datumaro.components.exporter>`
* :mod:`dm.components.importer <datumaro.components.importer>`
    * :class:`dm.FailingImportErrorPolicy <datumaro.components.importer>`
    * :class:`dm.ImportErrorPolicy <datumaro.components.importer>`
    * :class:`dm.Importer <datumaro.components.importer>`
* :mod:`dm.components.hl_ops <datumaro.components.hl_ops>`
    * :class:`dm.HLOps <datumaro.components.hl_ops>`
* :mod:`dm.components.transformer <datumaro.components.transformer>`
    * :class:`dm.ItemTransform <datumaro.components.transformer>`
    * :class:`dm.Transform <datumaro.components.transformer>`
* :mod:`dm.components.launcher <datumaro.components.launcher>`
    * :class:`dm.Launcher <datumaro.components.launcher>`
    * :class:`dm.ModelTransform <datumaro.components.launcher>`
* :mod:`dm.components.algorithms.noisy_label_detection.loss_dynamics_analyzer <datumaro.components.algorithms.noisy_label_detection.loss_dynamics_analyzer>`
    * :class:`dm.LossDynamicsAnalyzer <datumaro.components.algorithms.noisy_label_detection.loss_dynamics_analyzer>`
* :mod:`dm.components.media_manager <datumaro.components.media_manager>`
    * :class:`dm.MediaManager <datumaro.components.media_manager>`
* :mod:`dm.components.progress_reporting <datumaro.components.progress_reporting>`
    * :class:`dm.NullProgressReporter <datumaro.components.progress_reporting>`
    * :class:`dm.ProgressReporter <datumaro.components.progress_reporting>`
    * :class:`dm.SimpleProgressReporter <datumaro.components.progress_reporting>`
    * :class:`dm.TQDMProgressReporter <datumaro.components.progress_reporting>`
* :mod:`dm.components.algorithms.rise <datumaro.components.algorithms.rise>`
    * :class:`dm.RISE <datumaro.components.algorithms.rise>`
* :mod:`dm.components.validator <datumaro.components.validator>`
    * :class:`dm.Validator <datumaro.components.validator>`
* :mod:`dm.components.visualizer <datumaro.components.visualizer>`
    * :class:`dm.Visualizer <datumaro.components.visualizer>`
