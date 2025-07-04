# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import hashlib
import logging as log
import os
import os.path as osp
from importlib.resources import open_text
from multiprocessing import get_context
from random import Random
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import requests

from datumaro.components.generator import DatasetGenerator
from datumaro.util.definitions import get_datumaro_cache_dir
from datumaro.util.deprecation import deprecated
from datumaro.util.image import save_image
from datumaro.util.scope import on_error_do, on_exit_do, scope_add, scoped

from .utils import IFSFunction, augment, colorize, suppress_computation_warnings


@deprecated(deprecated_version="1.11", removed_version="1.12")
class FractalImageGenerator(DatasetGenerator):
    """
    ImageGenerator generates 3-channel synthetic images with provided shape.
    Uses the algorithm from the article: https://arxiv.org/abs/2103.13023
    """

    _MODEL_PROTO_FILENAME = "colorization_deploy_v2.prototxt"
    _MODEL_WEIGHTS_FILENAME = "colorization_release_v2.caffemodel"
    _HULL_PTS_FILE_NAME = "pts_in_hull.npy"
    _COLORS_FILE = "background_colors.txt"

    def __init__(
        self,
        output_dir: str,
        count: int,
        shape: Tuple[int, int],
        model_path: str = get_datumaro_cache_dir(),
    ) -> None:
        assert 0 < count, "Image count cannot be lesser than 1"
        self._count = count

        self._output_dir = output_dir
        self._model_dir = model_path

        self._cpu_count = min(os.cpu_count(), self._count)

        assert len(shape) == 2
        self._height, self._width = shape

        self._weights = self._create_weights(IFSFunction.NUM_PARAMS)
        self._threshold = 0.2
        self._iterations = 200000
        self._num_of_points = 100000

        self._initialize_params()

    def generate_dataset(self) -> None:
        log.info(
            "Generation of '%d' 3-channel images with height = '%d' and width = '%d'",
            self._count,
            self._height,
            self._width,
        )

        self._download_colorization_model(self._model_dir)

        mp_ctx = get_context("spawn")  # On Mac 10.15 and Python 3.7 fork leads to hangs
        with mp_ctx.Pool(processes=self._cpu_count) as pool:
            try:
                params = pool.map(
                    self._generate_category,
                    [Random(i) for i in range(self._categories)],  # nosec B311
                )
            finally:
                pool.close()
                pool.join()

        instances_weights = np.repeat(self._weights, self._instances, axis=0)
        weight_per_img = np.tile(instances_weights, (self._categories, 1))
        params = np.array(params, dtype=object)
        repeated_params = np.repeat(params, self._weights.shape[0] * self._instances, axis=0)
        repeated_params = repeated_params[: self._count]
        weight_per_img = weight_per_img[: self._count]
        assert weight_per_img.shape[0] == len(repeated_params) == self._count

        splits = min(self._cpu_count, self._count)
        params_per_proc = np.array_split(repeated_params, splits)
        weights_per_proc = np.array_split(weight_per_img, splits)

        generation_params = []
        offset = 0
        for param, w in zip(params_per_proc, weights_per_proc):
            indices = list(range(offset, offset + len(param)))
            offset += len(param)
            generation_params.append((param, w, indices))

        with mp_ctx.Pool(processes=self._cpu_count) as pool:
            try:
                pool.starmap(self._generate_image_batch, generation_params)
            finally:
                pool.close()
                pool.join()

    @scoped
    def _generate_image_batch(
        self, params: np.ndarray, weights: np.ndarray, indices: List[int]
    ) -> None:
        scope_add(suppress_computation_warnings())

        proto = osp.join(self._model_dir, self._MODEL_PROTO_FILENAME)
        model = osp.join(self._model_dir, self._MODEL_WEIGHTS_FILENAME)
        npy = osp.join(self._model_dir, self._HULL_PTS_FILE_NAME)
        pts_in_hull = np.load(npy).transpose().reshape(2, 313, 1, 1).astype(np.float32)

        with open_text(__package__, self._COLORS_FILE) as f:
            background_colors = np.loadtxt(f)

        net = cv.dnn.readNetFromCaffe(proto, model)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

        for i, param, w in zip(indices, params, weights):
            image = self._generate_image(
                Random(i),  # nosec B311
                param,
                self._iterations,
                self._height,
                self._width,
                draw_point=False,
                weight=w,
            )
            color_image = colorize(image, net)
            aug_image = augment(Random(i), color_image, background_colors)  # nosec B311
            save_image(
                osp.join(self._output_dir, "{:06d}.png".format(i)), aug_image, create_dir=True
            )

    def _generate_image(
        self,
        rng: Random,
        params: np.ndarray,
        iterations: int,
        height: int,
        width: int,
        draw_point: bool = True,
        weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ifs_function = IFSFunction(rng, prev_x=0.0, prev_y=0.0)
        for param in params:
            ifs_function.add_param(
                param[: ifs_function.NUM_PARAMS], param[ifs_function.NUM_PARAMS], weight
            )
        ifs_function.calculate(iterations)
        img = ifs_function.draw(height, width, draw_point)
        return img

    @scoped
    def _generate_category(self, rng: Random, base_h: int = 512, base_w: int = 512) -> np.ndarray:
        scope_add(suppress_computation_warnings())

        pixels = -1
        i = 0
        while pixels < self._threshold and i < self._iterations:
            param_size = rng.randint(2, 7)
            params = np.zeros((param_size, IFSFunction.NUM_PARAMS + 1), dtype=np.float32)

            sum_proba = 1e-5
            for p_idx in range(param_size):
                a, b, c, d, e, f = [rng.uniform(-1.0, 1.0) for _ in range(IFSFunction.NUM_PARAMS)]
                prob = abs(a * d - b * c)
                sum_proba += prob
                params[p_idx] = a, b, c, d, e, f, prob
            params[:, IFSFunction.NUM_PARAMS] /= sum_proba

            fractal_img = self._generate_image(rng, params, self._num_of_points, base_h, base_w)
            pixels = np.count_nonzero(fractal_img) / (base_h * base_w)
            i += 1
        return params

    def _initialize_params(self) -> None:
        if self._count < self._weights.shape[0]:
            self._weights = self._weights[: self._count, :]

        instances_categories = np.ceil(self._count / self._weights.shape[0])
        self._instances = np.ceil(np.sqrt(instances_categories)).astype(int)
        self._categories = np.ceil(instances_categories / self._instances).astype(int)

    @staticmethod
    def _create_weights(num_params):
        # weights from https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf
        BASE_WEIGHTS = np.ones((num_params,))
        WEIGHT_INTERVAL = 0.4
        INTERVAL_MULTIPLIERS = (-2, -1, 1, 2)
        weight_vectors = [BASE_WEIGHTS]

        for weight_index in range(num_params):
            for multiplier in INTERVAL_MULTIPLIERS:
                modified_weights = BASE_WEIGHTS.copy()
                modified_weights[weight_index] += multiplier * WEIGHT_INTERVAL
                weight_vectors.append(modified_weights)
        weights = np.array(weight_vectors)
        return weights

    @classmethod
    def _download_colorization_model(cls, save_dir: str) -> None:
        prototxt_file_name = cls._MODEL_PROTO_FILENAME
        caffemodel_file_name = cls._MODEL_WEIGHTS_FILENAME
        hull_file_name = cls._HULL_PTS_FILE_NAME

        proto_path = osp.join(save_dir, prototxt_file_name)
        model_path = osp.join(save_dir, caffemodel_file_name)
        hull_path = osp.join(save_dir, hull_file_name)
        if not (
            osp.exists(proto_path) and osp.exists(model_path) and osp.exists(hull_path)
        ) and not os.access(save_dir, os.W_OK):
            raise ValueError(
                "Please provide a path to a colorization model directory or "
                "a path to a writable directory to download the model"
            )

        for url, filename, size, sha512_checksum in [
            (
                f"https://raw.githubusercontent.com/richzhang/colorization/a1642d6ac6fc80fe08885edba34c166da09465f6/colorization/models/{prototxt_file_name}",
                prototxt_file_name,
                9945,
                "e3dd9188771202bd296623510bcf527b41c130fc9bae584e61dcdf66917b8c4d147b7b838fec0685568f7f287235c34e8b8e9c0482b555774795be89f0442820",
            ),
            (
                f"http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/{caffemodel_file_name}",
                caffemodel_file_name,
                128946764,
                "3d773dd83cfcf8e846e3a9722a4d302a3b7a0f95a0a7ae1a3d3ef5fe62eecd617f4f30eefb1d8d6123be4a8f29f7c6e64f07b36193f45710b549f3e4796570f1",
            ),
            (
                f"https://raw.githubusercontent.com/richzhang/colorization/a1642d6ac6fc80fe08885edba34c166da09465f6/colorization/resources/{hull_file_name}",
                hull_file_name,
                5088,
                "bf59a8a4e74b18948e4aeaa430f71eb8603bd9dbbce207ea086dd0fb976a34672beaeea6f1233a21687da710e0f8d36e86133a8532265dfda52994a7d6f0dbf5",
            ),
        ]:
            save_path = osp.join(save_dir, filename)
            if osp.exists(save_path):
                continue

            log.info("Downloading the '%s' file to '%s'", filename, save_dir)
            try:
                cls._download_file(
                    url, save_path, expected_size=size, expected_checksum=sha512_checksum
                )
            except Exception as e:
                raise Exception(f"Failed to download the '{filename}' file: {str(e)}") from e

    @staticmethod
    @scoped
    def _download_file(
        url: str, output_path: str, *, timeout: int = 60, expected_size: int, expected_checksum: str
    ) -> None:
        BLOCK_SIZE = 2**20

        assert not osp.exists(output_path)

        tmp_path = output_path + ".tmp"
        if osp.exists(tmp_path):
            raise Exception(f"Can't write temporary file '{tmp_path}' - file exists")

        response = requests.get(url, timeout=timeout, stream=True)
        on_exit_do(response.close)

        response.raise_for_status()

        checksum_counter = hashlib.sha512()
        actual_size = 0

        with open(tmp_path, "wb") as fd:
            on_error_do(os.unlink, tmp_path)

            for chunk in response.iter_content(chunk_size=BLOCK_SIZE):
                actual_size += len(chunk)
                if actual_size > expected_size:
                    # There is also the context-length header, but it can be corrupted or invalid
                    # for different reasons
                    raise Exception(
                        f"The downloaded file has unexpected size, expected {expected_size}."
                    )

                checksum_counter.update(chunk)

                fd.write(chunk)

        actual_checksum = checksum_counter.hexdigest()
        if actual_checksum.lower() != expected_checksum.lower():
            raise Exception("The downloaded file has unexpected checksum")

        os.rename(tmp_path, output_path)
