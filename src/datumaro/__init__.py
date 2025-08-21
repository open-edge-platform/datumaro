# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import warnings

from . import errors as errors
from . import ops as ops
from .components.annotation import (
    NO_GROUP,
    Annotation,
    AnnotationType,
    Bbox,
    BinaryMaskImage,
    Caption,
    Categories,
    Colormap,
    CompiledMask,
    CompiledMaskImage,
    Cuboid3d,
    Ellipse,
    IndexMaskImage,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RgbColor,
    RleMask,
)
from .components.cli_plugin import CliPlugin
from .components.dataset import Dataset, DatasetPatch, DatasetSubset, StreamDataset, eager_mode
from .components.dataset_base import (
    DEFAULT_SUBSET_NAME,
    CategoriesInfo,
    DatasetBase,
    DatasetItem,
    IDataset,
    SubsetBase,
)
from .components.dataset_item_storage import ItemStatus
from .components.environment import Environment
from .components.exporter import Exporter, ExportErrorPolicy, FailingExportErrorPolicy
from .components.hl_ops import HLOps
from .components.importer import FailingImportErrorPolicy, Importer, ImportErrorPolicy
from .components.media import Image, MediaElement, Video, VideoFrame
from .components.progress_reporting import (
    NullProgressReporter,
    ProgressReporter,
    SimpleProgressReporter,
    TQDMProgressReporter,
)
from .components.registry import PluginRegistry
from .components.transformer import ItemTransform, Transform
from .components.validator import Validator
from .components.visualizer import Visualizer
from .version import __version__
