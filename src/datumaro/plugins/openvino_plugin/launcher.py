# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=exec-used


import logging as log
import os.path as osp
import shutil
import urllib
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import numpy as np
from openvino import Core
from tqdm import tqdm

from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.errors import DatumaroError
from datumaro.util.definitions import get_datumaro_cache_dir
from datumaro.util.samples import get_samples_path


class _OpenvinoImporter(CliPlugin):
    @staticmethod
    def _parse_output_layers(s):
        return [s.strip() for s in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-d", "--description", required=True, help="Path to the model description file (.xml)"
        )
        parser.add_argument(
            "-w", "--weights", required=True, help="Path to the model weights file (.bin)"
        )
        parser.add_argument(
            "-i",
            "--interpreter",
            required=True,
            help="Path to the network output interprter script (.py)",
        )
        parser.add_argument("--device", default="CPU", help="Target device (default: %(default)s)")
        parser.add_argument(
            "--output-layers",
            type=cls._parse_output_layers,
            help="A comma-separated list of extra output layers",
        )
        return parser

    @staticmethod
    def copy_model(model_dir, model):
        shutil.copy(model["description"], osp.join(model_dir, osp.basename(model["description"])))
        model["description"] = osp.basename(model["description"])

        shutil.copy(model["weights"], osp.join(model_dir, osp.basename(model["weights"])))
        model["weights"] = osp.basename(model["weights"])

        shutil.copy(model["interpreter"], osp.join(model_dir, osp.basename(model["interpreter"])))
        model["interpreter"] = osp.basename(model["interpreter"])


@dataclass
class OpenvinoModelInfo:
    interpreter: Optional[str]
    description: Optional[str]
    weights: Optional[str]
    model_dir: Optional[str]

    def validate(self):
        """Validate integrity of the member variables"""

        def _validate(key: str):
            path = getattr(self, key)
            if not osp.isfile(path):
                path = osp.join(self.model_dir, path)
            if not osp.isfile(path):
                raise DatumaroError(f'Failed to open model {key} file "{path}"')
            setattr(self, key, path)

        for field in fields(self):
            if field.name != "model_dir":
                _validate(field.name)


@dataclass
class BuiltinOpenvinoModelInfo(OpenvinoModelInfo):
    downloadable_models = {
        "clip_text_ViT-B_32",
        "clip_visual_ViT-B_32",
        "clip_visual_vit_l_14_336px_int8",
        "clip_text_vit_l_14_336px_int8",
        "googlenet-v4-tf",
    }

    @classmethod
    def create_from_model_name(cls, model_name: str) -> "BuiltinOpenvinoModelInfo":
        openvino_plugin_samples_dir = get_samples_path()
        interpreter = osp.join(openvino_plugin_samples_dir, model_name + "_interp.py")
        interpreter = interpreter if osp.exists(interpreter) else interpreter

        model_dir = get_datumaro_cache_dir()

        # Please visit open-model-zoo repository for OpenVINO public models if you are interested in
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md
        url_folder = "https://storage.openvinotoolkit.org/repositories/datumaro/models/"

        description = osp.join(model_dir, model_name + ".xml")
        if not osp.exists(description):
            description = (
                cls._download_file(osp.join(url_folder, model_name + ".xml"), description)
                if model_name in cls.downloadable_models
                else None
            )

        weights = osp.join(model_dir, model_name + ".bin")
        if not osp.exists(weights):
            weights = (
                cls._download_file(osp.join(url_folder, model_name + ".bin"), weights)
                if model_name in cls.downloadable_models
                else None
            )

        return cls(
            interpreter=interpreter,
            description=description,
            weights=weights,
            model_dir=model_dir,
        )

    @staticmethod
    def _download_file(url: str, file_root: str) -> str:
        log.info('Downloading: "{}" to {}\n'.format(url, file_root))
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as source, open(file_root, "wb") as output:  # nosec B310
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        return file_root

    def override(self, other: OpenvinoModelInfo) -> None:
        """Override builtin model variables to other"""

        def _apply(key: str) -> None:
            other_item = getattr(other, key)
            self_item = getattr(self, key)
            if other_item is None and self_item:
                log.info(f"Override description with the builtin model {key}: {self.description}.")
                setattr(other, key, self_item)

        for field in fields(self):
            _apply(field.name)


class OpenvinoLauncher(LauncherWithModelInterpreter):
    cli_plugin = _OpenvinoImporter

    def __init__(
        self,
        description: Optional[str] = None,
        weights: Optional[str] = None,
        interpreter: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        output_layers: List[str] = [],
        device: Optional[str] = None,
        compile_model_config: Optional[Dict] = None,
    ):
        model_info = OpenvinoModelInfo(
            interpreter=interpreter,
            description=description,
            weights=weights,
            model_dir=model_dir,
        )
        if model_name:
            builtin_model_info = BuiltinOpenvinoModelInfo.create_from_model_name(model_name)
            builtin_model_info.override(model_info)

        model_info.validate()

        super().__init__(model_interpreter_path=model_info.interpreter)

        self.model_info = model_info

        self._device = device or "CPU"
        self._compile_model_config = compile_model_config

        self._core = Core()
        self._network = self._core.read_model(model_info.description, model_info.weights)

        if output_layers:
            log.info(f"Add additional output layers {output_layers} to the model outputs.")
            self._network.add_outputs(output_layers)

        self._check_model_support(self._network, self._device)
        self._load_executable_net()

    @property
    def inputs(self):
        return self._network.inputs

    @property
    def outputs(self):
        return self._network.outputs

    def _check_model_support(self, net, device):
        not_supported_layers = set(
            name for name, dev in self._core.query_model(net, device).items() if not dev
        )
        if len(not_supported_layers) != 0:
            log.error(
                "The following layers are not supported "
                "by the plugin for device '%s': %s." % (device, ", ".join(not_supported_layers))
            )
            raise NotImplementedError("Some layers are not supported on the device")

    def _load_executable_net(self, batch_size: int = 1):
        network = self._network

        iter_inputs = iter(network.inputs)
        self._input_blob = next(iter_inputs)

        is_dynamic_layout = False
        try:
            self._input_layout = self._input_blob.shape
        except ValueError:
            # In case of that the input has dynamic shape
            self._input_layout = self._input_blob.partial_shape
            is_dynamic_layout = True

        if is_dynamic_layout:
            self._input_layout[0] = batch_size
            network.reshape({self._input_blob: self._input_layout})
        else:
            model_batch_size = self._input_layout[0]
            if batch_size != model_batch_size:
                log.warning(
                    "Input layout of the model is static, so that we cannot change "
                    f"the model batch size ({model_batch_size}) to batch size ({batch_size})! "
                    "Set the batch size to {model_batch_size}."
                )
                batch_size = model_batch_size

        self._batch_size = batch_size

        self._net = self._core.compile_model(
            model=network,
            device_name=self._device,
            config=self._compile_model_config,
        )
        self._request = self._net.create_infer_request()

    def infer(self, inputs: LauncherInputType) -> List[ModelPred]:
        batch_size = len(inputs)
        if self._batch_size < batch_size:
            self._load_executable_net(batch_size)

        inputs = (
            {self._input_blob.get_any_name(): inputs} if isinstance(inputs, np.ndarray) else inputs
        )
        results = self._request.infer(inputs=inputs)

        outputs_group_by_item = [
            {key.any_name: output for key, output in zip(results.keys(), outputs)}
            for outputs in zip(*results.values())
        ]

        return outputs_group_by_item
