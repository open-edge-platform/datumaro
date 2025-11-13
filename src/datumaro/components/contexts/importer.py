# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT


from typing import NoReturn, Tuple

import attr
from attr import define, field

from datumaro.components.errors import AnnotationImportError, DatumaroError, ItemImportError
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter


class _ImportFail(DatumaroError):
    pass


class ImportErrorPolicy:
    def report_item_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item.
        If this function returns, the extractor must skip the item.
        """

        if not isinstance(error, _ImportFail):
            ie = ItemImportError(item_id)
            ie.__cause__ = error
            return self._handle_item_error(ie)
        raise error

    def report_annotation_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item annotation.
        If this function returns, the extractor must skip the annotation.
        """

        if not isinstance(error, _ImportFail):
            ie = AnnotationImportError(item_id)
            ie.__cause__ = error
            return self._handle_annotation_error(ie)
        raise error

    def _handle_item_error(self, error: ItemImportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def _handle_annotation_error(self, error: AnnotationImportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def fail(self, error: Exception) -> NoReturn:
        raise _ImportFail from error


class FailingImportErrorPolicy(ImportErrorPolicy):
    pass


@define(eq=False)
class ImportContext:
    progress_reporter: ProgressReporter = field(
        default=None, converter=attr.converters.default_if_none(factory=NullProgressReporter)
    )
    error_policy: ImportErrorPolicy = field(
        default=None, converter=attr.converters.default_if_none(factory=FailingImportErrorPolicy)
    )


class NullImportContext(ImportContext):
    pass
