# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TypeVar, Union

import numpy as np

from datumaro.components.annotation import Annotation
from datumaro.components.dataset_base import DatasetItem
from datumaro.util.deprecation import deprecated

__all__ = ["IModelInterpreter", "PrepInfo", "ModelPred", "LauncherInputType"]

PrepInfo = TypeVar("PrepInfo")
ModelPred = Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
LauncherInputType = Union[np.ndarray, Dict[str, np.ndarray]]


@deprecated(deprecated_version="1.11", removed_version="1.12")
class IModelInterpreter(ABC):
    @abstractmethod
    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        """Preprocessing an dataset item input.

        Parameters:
            img: DatasetItem input

        Returns:
            It returns a tuple of preprocessed input and preprocessing information.
            The preprocessing information will be used in the postprocessing step.
            One use case for this would be an affine transformation of the output bounding box
            obtained by object detection models. Input images for those models are usually resized
            to fit the model input dimensions. As a result, the postprocessing step should refine
            the model output bounding box to match the original input image size.
        """
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        """Postprocess a model prediction.

        Parameters:
            pred: Model prediction
            info: Preprocessing information coming from the preprocessing step

        Returns:
            A list of annotations which is created from the model predictions
        """
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def get_categories(self):
        """It should be implemented if the model generate a new categories"""
        raise NotImplementedError("Function should be implemented.")
