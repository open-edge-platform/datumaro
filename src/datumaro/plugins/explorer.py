# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Sequence

import numpy as np
from tokenizers import Tokenizer

from datumaro.components.annotation import Annotation, HashKey
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher
from datumaro.util.deprecation import deprecated


@deprecated(deprecated_version="1.11", removed_version="1.12")
class ExplorerLauncher(OpenvinoLauncher):
    def __init__(
        self,
        description=None,
        weights=None,
        interpreter=None,
        model_dir=None,
        model_name=None,
        output_layers=None,
        device=None,
    ):
        super().__init__(
            description, weights, interpreter, model_dir, model_name, output_layers, device
        )

        self._device = device or "cpu"
        self._output_blobs = next(iter(self._net.outputs))
        self._tokenizer = None

    def _tokenize(self, texts: str, context_length: int = 77, truncate: bool = True):
        if not self._tokenizer:
            checkpoint = "openai/clip-vit-base-patch32"
            self._tokenizer = Tokenizer.from_pretrained(checkpoint)
        tokens = self._tokenizer.encode(texts).ids
        result = np.zeros((1, context_length))

        if len(tokens) > context_length:
            if truncate:
                eot_token = tokens.ids[-1]
                tokens = tokens[:context_length]
                tokens[-1] = eot_token

        for i, token in enumerate(tokens):
            result[:, i] = token
        return result

    def _compute_hash(self, features):
        features = np.sign(features)
        hash_key = np.clip(features, 0, None)
        hash_key = hash_key.astype(np.uint8)
        hash_key = np.packbits(hash_key, axis=-1)
        return hash_key

    def infer_text(self, text: str, use_prompt: bool = True) -> HashKey:
        prompt_text = f"a photo of a {text}" if use_prompt else text
        inputs = self._tokenize(prompt_text)
        preds = self.infer(inputs)
        anns = self.postprocess(preds[0], None)
        return anns[0]

    def infer_item(self, item: DatasetItem) -> HashKey:
        anns = self.launch([item])[0]
        return anns[0]

    def launch(self, batch: Sequence[DatasetItem]) -> List[List[Annotation]]:
        outputs = super().launch(batch)
        return outputs

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
