# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import os.path as osp
from contextlib import contextmanager
from functools import wraps
from glob import iglob
from typing import Callable, Dict, List, Optional, Type, TypeVar

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.contexts.importer import (
    FailingImportErrorPolicy,
    ImportContext,
    ImportErrorPolicy,
    NullImportContext,
    _ImportFail,
)
from datumaro.components.errors import DatasetImportError, DatasetNotFoundError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.merge.extractor_merger import ExtractorMerger
from datumaro.util.definitions import SUBSET_NAME_WHITELIST

T = TypeVar("T")

__all__ = [
    "FailingImportErrorPolicy",
    "ImportContext",
    "ImportErrorPolicy",
    "Importer",
    "NullImportContext",
    "_ImportFail",
    "with_subset_dirs",
]


class Importer(CliPlugin):
    DETECT_CONFIDENCE = FormatDetectionConfidence.LOW

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        if not cls.find_sources_with_params(context.root_path):
            context.fail("specific requirement information unavailable")

        return cls.DETECT_CONFIDENCE

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def find_sources(cls, path: str) -> List[Dict]:
        raise NotImplementedError

    @classmethod
    def find_sources_with_params(cls, path: str, **extra_params) -> List[Dict]:
        return cls.find_sources(path)

    def __call__(self, path, stream: bool = False, **extra_params):
        if not path or not osp.exists(path):
            raise DatasetNotFoundError(path, self.NAME)

        found_sources = self.find_sources_with_params(osp.normpath(path), **extra_params)
        if not found_sources:
            raise DatasetNotFoundError(path, self.NAME)

        sources = []
        for desc in found_sources:
            params = dict(extra_params)
            params.update(desc.get("options", {}))

            if stream and self.can_stream:
                params.update({"stream": True})
            elif stream and not self.can_stream:
                raise DatasetImportError(f"{self.__class__.__name__} cannot stream, but stream=True.")

            desc["options"] = params
            sources.append(desc)

        return sources

    @classmethod
    def _find_sources_recursive(
        cls,
        path: str,
        ext: Optional[str],
        extractor_name: str,
        filename: str = "*",
        dirname: str = "",
        file_filter: Optional[Callable[[str], bool]] = None,
        max_depth: int = 3,
        recursive: bool = False,
    ):
        """
        Finds sources in the specified location, using the matching pattern
        to filter file names and directories.
        Supposed to be used, and to be the only call in subclasses.

        Parameters:
            path: a directory or file path, where sources need to be found.
            ext: file extension to match. To match directories,
                set this parameter to None or ''. Comparison is case-independent,
                a starting dot is not required.
            extractor_name: the name of the associated Extractor type
            filename: a glob pattern for file names
            dirname: a glob pattern for filename prefixes
            file_filter: a callable (abspath: str) -> bool, to filter paths found
            max_depth: the maximum depth for recursive search.
            recursive: If recursive is true, the pattern '**' will match any files and
                zero or more directories and subdirectories.

        Returns: a list of source configurations
            (i.e. Extractor type names and c-tor parameters)
        """

        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            ext = ext.lower()

        if (path.lower().endswith(ext) and osp.isfile(path)) or (
            not ext
            and dirname
            and osp.isdir(path)
            and os.sep + osp.normpath(dirname.lower()) + os.sep in osp.abspath(path.lower()) + os.sep
        ):
            sources = [{"url": path, "format": extractor_name}]
        else:
            sources = []
            for d in range(max_depth + 1):
                sources.extend(
                    {"url": p, "format": extractor_name}
                    for p in iglob(osp.join(path, *("*" * d), dirname, filename + ext), recursive=recursive)
                    if (callable(file_filter) and file_filter(p)) or (not callable(file_filter))
                )
                if sources:
                    break
        return sources

    @property
    def can_stream(self) -> bool:
        """Flag to indicate whether the importer can stream the dataset item or not."""
        return False

    def get_extractor_merger(self) -> Optional[Type[ExtractorMerger]]:
        """Extractor merger dedicated for the data format

        Datumaro import process spawns multiple `DatasetBase` for the detected sources.
        We can find a bunch of the detected sources from the given directory path.
        It is usually each detected source is corresponded to the subset of dataset in many data formats.

        Parameters:
            stream: There can exist a branch according to `stream` flag

        Returns:
            If None, use `Dataset.from_extractors()` to merge the extractors,
            Otherwise, use the return type to merge the extractors.
        """
        return None


def with_subset_dirs(input_cls: Importer):
    @wraps(input_cls, updated=())
    class WrappedImporter(input_cls):
        NAME = input_cls.NAME

        @classmethod
        def detect(
            cls,
            context: FormatDetectionContext,
        ) -> Optional[FormatDetectionConfidence]:
            @contextmanager
            def _change_context_root_path(context: FormatDetectionContext, path: str):
                tmp = context.root_path
                context._root_path = path
                yield
                context._root_path = tmp

            confs = []
            path = context.root_path

            if not osp.isdir(path):
                context.fail(
                    f"{input_cls.NAME} should require an input as a directory path. "
                    f"However, {path} is not a directory path."
                )

            for sub_dir in os.listdir(path):
                if sub_dir.lower() not in SUBSET_NAME_WHITELIST:
                    continue

                sub_path = osp.join(path, sub_dir)
                if osp.isdir(sub_path):
                    with _change_context_root_path(context, sub_path):
                        conf = input_cls.detect(context)
                    if conf is not None:
                        confs += [conf]

            if len(confs) == 0:
                context.fail(f"{input_cls.NAME} cannot find its subdirectory structure.")

            return max(confs)

        def __call__(self, path, **extra_params):
            sources = []
            for sub_dir in os.listdir(path):
                sub_path = osp.join(path, sub_dir)
                if osp.isdir(sub_path):
                    source = input_cls.__call__(self, sub_path, **extra_params)

                    if len(source) != 1:
                        raise DatasetImportError(f"@with_subset_dirs only allows one source format from {sub_path}.")

                    if "subset" in source[0]:
                        raise DatasetImportError(
                            f"@with_subset_dirs does not allows a subset key in source: {source[0]}."
                        )

                    source[0]["options"]["subset"] = sub_dir
                    sources += source

            return sources

        def __reduce__(self):
            return (input_cls.__class__, ())

    return WrappedImporter
