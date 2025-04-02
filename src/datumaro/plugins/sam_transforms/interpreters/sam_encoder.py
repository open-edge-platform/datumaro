# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import cv2
import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, FeatureVector
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.sam_transforms.utils import rescale_img_keeping_aspect_ratio


class SAMEncoderInterpreter(IModelInterpreter):
    h_model = 1024
    w_model = 1024

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        output = rescale_img_keeping_aspect_ratio(img, self.h_model, self.w_model, padding=False)

        # From BGR to RGB
        output.image = cv2.cvtColor(output.image, cv2.COLOR_BGR2RGB)
        # From HWC to CHW
        output.image = output.image.transpose(2, 0, 1)
        # To FP32
        output.image = output.image.astype(np.float32)

        return output.image, output.scale

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        embed = pred.get("image_embeddings")

        assert embed is not None, 'Model output should have "image_embeddings".'

        return [FeatureVector(vector=embed)]

    def get_categories(self):
        return None
