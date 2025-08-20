# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
import shutil
from tempfile import mkdtemp
from typing import NoReturn, Optional, Tuple, TypeVar, Union

import attr
from attrs import define, field

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.errors import (
    AnnotationExportError,
    DatasetExportError,
    DatumaroError,
    ItemExportError,
)
from datumaro.components.media import Image, PointCloud, Video, VideoFrame
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter
from datumaro.util.meta_file_util import save_meta_file
from datumaro.util.scope import on_error_do, scoped

T = TypeVar("T")


class _ExportFail(DatumaroError):
    pass


class ExportErrorPolicy:
    def report_item_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item.
        If this function returns, the converter must skip the item.
        """

        if not isinstance(error, _ExportFail):
            ie = ItemExportError(item_id)
            ie.__cause__ = error
            return self._handle_item_error(ie)
        else:
            raise error

    def report_annotation_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item annotation.
        If this function returns, the converter must skip the annotation.
        """

        if not isinstance(error, _ExportFail):
            ie = AnnotationExportError(item_id)
            ie.__cause__ = error
            return self._handle_annotation_error(ie)
        else:
            raise error

    def _handle_item_error(self, error: ItemExportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def _handle_annotation_error(self, error: AnnotationExportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def fail(self, error: Exception) -> NoReturn:
        raise _ExportFail from error


class FailingExportErrorPolicy(ExportErrorPolicy):
    pass


@define(eq=False)
class ExportContext:
    progress_reporter: ProgressReporter = field(
        default=None, converter=attr.converters.default_if_none(factory=NullProgressReporter)
    )
    error_policy: ExportErrorPolicy = field(
        default=None, converter=attr.converters.default_if_none(factory=FailingExportErrorPolicy)
    )


class NullExportContext(ExportContext):
    pass


class Exporter(CliPlugin):
    DEFAULT_IMAGE_EXT = None

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--save-media",
            action="store_true",
            help="Save media (default: False)",
        )
        parser.add_argument(
            "--image-ext",
            default=None,
            help="Image extension (default: keep or use format default%s)"
            % (" " + cls.DEFAULT_IMAGE_EXT if cls.DEFAULT_IMAGE_EXT else ""),
        )
        parser.add_argument(
            "--save-dataset-meta",
            action="store_true",
            help="Save dataset meta file (default: %(default)s)",
        )

        return parser

    @classmethod
    def convert(cls, extractor, save_dir, **options):
        converter = cls(extractor, save_dir, **options)
        return converter.apply()

    @classmethod
    @scoped
    def patch(cls, dataset, patch, save_dir, **options):
        # This solution is not any better in performance than just
        # writing a dataset, but in case of patching (i.e. writing
        # to the previous location), it allows to avoid many problems
        # with removing and replacing existing files. Surely, this
        # approach also has problems with removal of the given directory.
        # Problems can occur if we can't remove the directory,
        # or want to reuse the given directory. It can happen if it
        # is mounted or (sym-)linked.
        # Probably, a better solution could be to wipe directory
        # contents and write new data there. Note that directly doing this
        # also doesn't work, because images may be needed for writing.

        if not osp.isdir(save_dir):
            return cls.convert(dataset, save_dir, **options)

        tmpdir = mkdtemp(dir=osp.dirname(save_dir), prefix=osp.basename(save_dir), suffix=".tmp")
        on_error_do(shutil.rmtree, tmpdir, ignore_errors=True)
        shutil.copymode(save_dir, tmpdir)

        retval = cls.convert(dataset, tmpdir, **options)

        shutil.rmtree(save_dir)
        os.replace(tmpdir, save_dir)

        return retval

    def apply(self):
        """Execute the data-format conversion"""
        return self._apply_impl()

    def _apply_impl(self):
        raise NotImplementedError("Should be implemented in a subclass")

    def __init__(
        self,
        extractor: IDataset,
        save_dir: str,
        *,
        save_media: Optional[bool] = None,
        image_ext: Optional[str] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        stream: bool = False,
        ctx: Optional[ExportContext] = None,
    ):
        default_image_ext = default_image_ext or self.DEFAULT_IMAGE_EXT
        assert default_image_ext
        self._default_image_ext = default_image_ext

        self._save_media = save_media
        self._image_ext = image_ext

        self._extractor = extractor
        self._save_dir = save_dir

        self._save_dataset_meta = save_dataset_meta

        # TODO: refactor this variable.
        # Can be used by a subclass to store the current patch info
        from datumaro.components.dataset import DatasetPatch

        if isinstance(extractor, DatasetPatch.DatasetPatchWrapper):
            self._patch = extractor.patch
        else:
            self._patch = None

        if stream and not self.can_stream:
            raise DatasetExportError(
                f"{self.__class__.__name__} cannot export a dataset in a stream manner"
            )
        self._stream = stream

        self._ctx: ExportContext = ctx or NullExportContext()

    def _find_image_ext(self, item: Union[DatasetItem, Image]):
        src_ext = None

        if isinstance(item, DatasetItem) and isinstance(item.media, Image):
            src_ext = item.media.ext
        elif isinstance(item, Image):
            src_ext = item.ext

        return self._image_ext or src_ext or self._default_image_ext

    def _make_item_filename(self, item, *, name=None, subdir=None):
        name = name or item.id
        subdir = subdir or ""
        return osp.join(subdir, name)

    def _make_image_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + self._find_image_ext(item)

    def _make_pcd_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + ".pcd"

    def _save_image(
        self,
        item,
        path=None,
        *,
        name=None,
        subdir=None,
        basedir=None,
    ):
        assert not (
            (subdir or name or basedir) and path
        ), "Can't use both subdir or name or basedir and path arguments"

        if not isinstance(item.media, Image) or not item.media.has_data:
            log.warning("Item '%s' has no image", item.id)
            return

        basedir = basedir or self._save_dir
        path = path or osp.join(basedir, self._make_image_filename(item, name=name, subdir=subdir))
        path = osp.abspath(path)

        item.media.save(path)

    def _save_point_cloud(self, item=None, path=None, *, name=None, subdir=None, basedir=None):
        assert not (
            (subdir or name or basedir) and path
        ), "Can't use both subdir or name or basedir and path arguments"

        if not item.media or not isinstance(item.media, PointCloud):
            log.warning("Item '%s' has no pcd", item.id)
            return

        basedir = basedir or self._save_dir
        path = path or osp.join(basedir, self._make_pcd_filename(item, name=name, subdir=subdir))
        path = osp.abspath(path)

        os.makedirs(osp.dirname(path), exist_ok=True)
        item.media.save(path)

    def _save_meta_file(self, path):
        save_meta_file(path, self._extractor.categories())

    @property
    def can_stream(self) -> bool:
        """Flag to indicate whether the exporter can export the dataset in a stream manner or not."""
        return False


# TODO: Currently, ExportContextComponent is introduced only for Datumaro and DatumaroBinary format
# for multi-processing. We need to propagate this to everywhere in Datumaro 1.2.0


class ExportContextComponent:
    def __init__(
        self,
        save_dir: str,
        save_media: bool,
        images_dir: str,
        pcd_dir: str,
        video_dir: str,
        image_ext: Optional[str] = None,
        default_image_ext: Optional[str] = None,
        source_path: Optional[str] = None,
    ):
        self._save_dir = save_dir
        self._save_media = save_media
        self._images_dir = images_dir
        self._pcd_dir = pcd_dir
        self._video_dir = video_dir
        self._image_ext = image_ext
        self._default_image_ext = default_image_ext
        self._source_path = source_path

    def find_image_ext(self, item: Union[DatasetItem, Image]):
        src_ext = None

        if isinstance(item, DatasetItem) and isinstance(item.media, Image):
            src_ext = item.media.ext
        elif isinstance(item, Image):
            src_ext = item.ext

        return self._image_ext or src_ext or self._default_image_ext

    def _make_item_filename(self, item, *, name=None, subdir=None):
        name = name or item.id
        subdir = subdir or ""
        return osp.join(subdir, name)

    def make_image_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + self.find_image_ext(item)

    def make_pcd_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + ".pcd"

    def make_pcd_extra_image_filename(self, item, idx, image, *, name=None, subdir=None):
        return self._make_item_filename(
            item, name=name if name else f"{item.id}/extra_image_{idx}", subdir=subdir
        ) + self.find_image_ext(image)

    def make_video_filename(self, item, *, name=None):
        STR_WRONG_MEDIA_TYPE = "Video item's media type should be Video or VideoFrame"
        assert isinstance(item, DatasetItem), STR_WRONG_MEDIA_TYPE

        if isinstance(item.media, VideoFrame):
            video_file_name = osp.basename(item.media.video.path)
        elif isinstance(item.media, Video):
            video_file_name = osp.basename(item.media.path)
        else:
            assert False, STR_WRONG_MEDIA_TYPE

        return video_file_name

    def save_image(
        self,
        item: DatasetItem,
        *,
        encryption: bool = False,
        basedir: Optional[str] = None,
        subdir: Optional[str] = None,
        fname: Optional[str] = None,
    ):
        if not isinstance(item.media, Image) or not item.media.has_data:
            log.warning("Item '%s' has no image", item.id)
            return

        basedir = self._images_dir if basedir is None else basedir
        basedir = osp.join(basedir, subdir) if subdir is not None else basedir
        fname = self.make_image_filename(item) if fname is None else fname
        path = osp.join(basedir, fname)
        path = osp.abspath(path)

        os.makedirs(osp.dirname(path), exist_ok=True)
        item.media.save(path)

    def save_point_cloud(
        self,
        item: DatasetItem,
        *,
        basedir: Optional[str] = None,
        subdir: Optional[str] = None,
        fname: Optional[str] = None,
    ):
        if not item.media or not isinstance(item.media, PointCloud):
            log.warning("Item '%s' has no pcd", item.id)
            return

        basedir = self._pcd_dir if basedir is None else basedir
        basedir = osp.join(basedir, subdir) if subdir is not None else basedir
        fname = self.make_pcd_filename(item) if fname is None else fname
        path = osp.join(basedir, fname)
        path = osp.abspath(path)

        os.makedirs(osp.dirname(path), exist_ok=True)

        def helper(i, image):
            basedir = self._images_dir
            basedir = osp.join(basedir, subdir) if subdir is not None else basedir
            return {"fp": osp.join(basedir, self.make_pcd_extra_image_filename(item, i, image))}

        item.media.save(path, helper)

    def save_video(
        self,
        item: DatasetItem,
        *,
        basedir: Optional[str] = None,
        subdir: Optional[str] = None,
        fname: Optional[str] = None,
    ):
        if not item.media or not isinstance(item.media, (Video, VideoFrame)):
            log.warning("Item '%s' has no video", item.id)
            return
        basedir = self._video_dir if basedir is None else basedir
        basedir = osp.join(basedir, subdir) if subdir is not None else basedir
        fname = self.make_video_filename(item) if fname is None else fname

        path = osp.join(basedir, fname)
        path = osp.abspath(path)

        # To prevent the video from being overwritten
        # (A video can have same path but different start/end frames)
        if not osp.exists(path):
            os.makedirs(osp.dirname(path), exist_ok=True)
            if isinstance(item.media, VideoFrame):
                item.media.video.save(path)
            else:  # Video
                item.media.save(path)

    @property
    def images_dir(self) -> str:
        return self._images_dir

    @property
    def pcd_dir(self) -> str:
        return self._pcd_dir

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @property
    def save_media(self) -> bool:
        return self._save_media

    @property
    def source_path(self) -> str:
        return self._source_path if self._source_path else ""
