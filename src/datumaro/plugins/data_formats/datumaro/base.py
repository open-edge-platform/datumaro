# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import re
from typing import Dict, List, Optional, Set, Type

from datumaro.components.annotation import (
    NO_OBJECT_ID,
    AnnotationType,
    Bbox,
    Caption,
    Cuboid2D,
    Cuboid3d,
    Ellipse,
    GroupType,
    Label,
    LabelCategories,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
)
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError, MediaTypeError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud, Video, VideoFrame
from datumaro.plugins.data_formats.datumaro.page_mapper import DatumPageMapper
from datumaro.util import parse_json_file
from datumaro.version import __version__

from .format import DATUMARO_FORMAT_VERSION, DatumaroPath

__all__ = ["DatumaroBase"]


class JsonReader:
    def __init__(
        self,
        path: str,
        subset: str,
        rootpath: str,
        images_dir: str,
        pcd_dir: str,
        video_dir: str,
        ctx: ImportContext,
    ) -> None:
        self._subset = subset
        self._rootpath = rootpath
        self._images_dir = images_dir
        self._pcd_dir = pcd_dir
        self._video_dir = video_dir
        self._videos = {}
        self._ctx = ctx
        self.ann_types = {}

        self._reader = self._init_reader(path)
        self.media_type = self._load_media_type(self._reader)
        self.infos = self._load_infos(self._reader)
        self.categories = self._load_categories(self._reader)
        self.items = self._load_items(self._reader)

    def _init_reader(self, path: str):
        return parse_json_file(path)

    @staticmethod
    def _load_media_type(parsed) -> Type[MediaElement]:
        media_type = parsed.get("media_type", MediaType.IMAGE)
        return MediaType(media_type).media

    @staticmethod
    def _load_ann_types(parsed) -> Set[AnnotationType]:
        return parsed.get("ann_types", set())

    @staticmethod
    def _load_infos(parsed) -> Dict:
        return parsed.get("infos", {})

    @staticmethod
    def _load_categories(parsed) -> Dict:
        categories = {}

        parsed_label_cat = parsed["categories"].get(AnnotationType.label.name)
        if parsed_label_cat:
            label_categories = LabelCategories(attributes=parsed_label_cat.get("attributes", []))
            for item in parsed_label_cat["labels"]:
                label_categories.add(
                    item["name"],
                    parent=item["parent"],
                    attributes=item.get("attributes", []),
                )

            for item in parsed_label_cat.get("label_groups", []):
                label_categories.add_label_group(
                    item["name"],
                    labels=item["labels"],
                    group_type=GroupType.from_str(item["group_type"]),
                )

            categories[AnnotationType.label] = label_categories

        parsed_mask_cat = parsed["categories"].get(AnnotationType.mask.name)
        if parsed_mask_cat:
            colormap = {}
            for item in parsed_mask_cat["colormap"]:
                colormap[int(item["label_id"])] = (item["r"], item["g"], item["b"])

            mask_categories = MaskCategories(colormap=colormap)
            categories[AnnotationType.mask] = mask_categories

        parsed_points_cat = parsed["categories"].get(AnnotationType.points.name)
        if parsed_points_cat:
            point_categories = PointsCategories()
            for item in parsed_points_cat["items"]:
                point_categories.add(
                    int(item["label_id"]),
                    item["labels"],
                    joints=item["joints"],
                    positions=item.get("positions"),
                )

            categories[AnnotationType.points] = point_categories

        return categories

    def _load_items(self, parsed) -> List:
        item_descs: List = parsed["items"]
        pbar = self._ctx.progress_reporter

        def _gen():
            # Reverse the list to pop from the front of it
            item_descs.reverse()
            while item_descs:
                yield item_descs.pop()

        items = []
        ann_types = set()
        actual_media_types = set()
        for item_desc in pbar.iter(
            _gen(), desc=f"Importing '{self._subset}'", total=len(item_descs)
        ):
            item = self._parse_item(item_desc)
            if item is not None:
                items.append(item)
                for ann in item.annotations:
                    ann_types.add(ann.type)
                if item.media:
                    actual_media_types.add(item.media.type)

        self.ann_types = ann_types

        if len(actual_media_types) == 1:
            actual_media_type = actual_media_types.pop()
        elif len(actual_media_types) > 1:
            actual_media_type = MediaType.MEDIA_ELEMENT
        else:
            actual_media_type = None

        if actual_media_type and not issubclass(actual_media_type.media, self.media_type):
            raise MediaTypeError(
                f"Unexpected media type of a dataset '{self.media_type}'. "
                f"Expected media type is '{actual_media_type.media}."
            )

        return items

    def _parse_item(self, item_desc: Dict) -> Optional[DatasetItem]:
        STR_MULTIPLE_MEDIA = "DatasetItem cannot contain multiple media types"
        try:
            item_id = item_desc["id"]

            media = None
            image_info = item_desc.get("image")
            if image_info:
                image_filename = image_info.get("path") or item_id + DatumaroPath.IMAGE_EXT
                image_path = osp.join(self._images_dir, self._subset, image_filename)
                if not osp.isfile(image_path):
                    # backward compatibility
                    old_image_path = osp.join(self._images_dir, image_filename)
                    if osp.isfile(old_image_path):
                        image_path = old_image_path

                media = Image.from_file(path=image_path, size=image_info.get("size"))

            pcd_info = item_desc.get("point_cloud")
            if media and pcd_info:
                raise MediaTypeError(STR_MULTIPLE_MEDIA)
            if pcd_info and (pcd_path := pcd_info.get("path")):
                point_cloud = osp.join(self._pcd_dir, self._subset, pcd_path)

                related_images = None
                ri_info = item_desc.get("related_images")
                if ri_info:
                    related_images = [
                        Image.from_file(
                            size=ri.get("size"),
                            path=osp.join(self._images_dir, self._subset, ri.get("path")),
                        )
                        for ri in ri_info
                    ]

                media = PointCloud.from_file(path=point_cloud, extra_images=related_images)

            video_frame_info = item_desc.get("video_frame")
            if media and video_frame_info:
                raise MediaTypeError(STR_MULTIPLE_MEDIA)
            if video_frame_info:
                video_path = osp.join(
                    self._video_dir, self._subset, video_frame_info.get("video_path")
                )
                if video_path not in self._videos:
                    self._videos[video_path] = Video(video_path)
                video = self._videos[video_path]

                frame_index = video_frame_info.get("frame_index")

                media = VideoFrame(video, frame_index)

            video_info = item_desc.get("video")
            if media and video_info:
                raise MediaTypeError(STR_MULTIPLE_MEDIA)
            if video_info:
                video_path = osp.join(self._video_dir, self._subset, video_info.get("path"))
                if video_path not in self._videos:
                    self._videos[video_path] = Video(video_path)
                step = video_info.get("step", 1)
                start_frame = video_info.get("start_frame", 0)
                end_frame = video_info.get("end_frame", None)
                media = Video(
                    path=video_path, step=step, start_frame=start_frame, end_frame=end_frame
                )

            media_desc = item_desc.get("media")
            if not media and media_desc and media_desc.get("path"):
                media = MediaElement(path=media_desc.get("path"))

        except Exception as e:
            self._ctx.error_policy.report_item_error(
                e, item_id=(item_desc.get("id", None), self._subset)
            )
            return None

        annotations = self._load_annotations(item_desc)

        return DatasetItem(
            id=item_id,
            subset=self._subset,
            annotations=annotations,
            media=media,
            attributes=item_desc.get("attr"),
        )

    def _load_annotations(self, item: Dict):
        loaded = []

        for ann in item.get("annotations", []):
            try:
                ann_id = ann.get("id")
                ann_type = AnnotationType[ann["type"]]
                attributes = ann.get("attributes")
                group = ann.get("group")
                object_id = ann.get("object_id", NO_OBJECT_ID)

                label_id = ann.get("label_id")
                z_order = ann.get("z_order")
                points = ann.get("points")

                if ann_type == AnnotationType.label:
                    loaded.append(
                        Label(
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                        )
                    )

                elif ann_type == AnnotationType.mask:
                    rle = ann["rle"]
                    rle["counts"] = rle["counts"].encode("ascii")
                    loaded.append(
                        RleMask(
                            rle=rle,
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.polyline:
                    loaded.append(
                        PolyLine(
                            points,
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.polygon:
                    loaded.append(
                        Polygon(
                            points,
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.bbox:
                    x, y, w, h = ann["bbox"]
                    loaded.append(
                        Bbox(
                            x,
                            y,
                            w,
                            h,
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.points:
                    loaded.append(
                        Points(
                            points,
                            label=label_id,
                            id=ann_id,
                            visibility=ann.get("visibility"),
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.caption:
                    caption = ann.get("caption")
                    loaded.append(Caption(caption, id=ann_id, attributes=attributes, group=group))

                elif ann_type == AnnotationType.cuboid_3d:
                    loaded.append(
                        Cuboid3d(
                            ann.get("position"),
                            ann.get("rotation"),
                            ann.get("scale"),
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                        )
                    )

                elif ann_type == AnnotationType.ellipse:
                    loaded.append(
                        Ellipse(
                            *points,
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )

                elif ann_type == AnnotationType.cuboid_2d:
                    loaded.append(
                        Cuboid2D(
                            list(map(tuple, points)),
                            label=label_id,
                            id=ann_id,
                            attributes=attributes,
                            group=group,
                            object_id=object_id,
                            z_order=z_order,
                        )
                    )
                else:
                    raise NotImplementedError()
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(
                    e, item_id=(ann.get("id", None), self._subset)
                )

        return loaded

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        yield from self.items


class StreamJsonReader(JsonReader):
    def __init__(
        self,
        path: str,
        subset: str,
        rootpath: str,
        images_dir: str,
        pcd_dir: str,
        video_dir: str,
        ctx: ImportContext,
    ) -> None:
        super().__init__(path, subset, rootpath, images_dir, pcd_dir, video_dir, ctx)
        self._length = None

    def __len__(self):
        return len(self._reader)

    def __iter__(self):
        ann_types = set()
        pbar = self._ctx.progress_reporter
        for item_desc in pbar.iter(
            self._reader,
            desc=f"Importing '{self._subset}'",
        ):
            item = self._parse_item(item_desc)
            yield item

            if item is not None:
                for ann in item.annotations:
                    ann_types.add(ann.type)
        self.ann_types = ann_types

    def _init_reader(self, path: str) -> DatumPageMapper:
        return DatumPageMapper(path)

    @staticmethod
    def _load_media_type(page_mapper: DatumPageMapper) -> Type[MediaElement]:
        media_type = page_mapper.media_type

        if media_type is None:
            return MediaType.IMAGE.media

        return media_type.media

    @staticmethod
    def _load_ann_types(page_mapper: DatumPageMapper) -> Set[AnnotationType]:
        ann_types = page_mapper.ann_types

        if ann_types is None:
            return set()

        return ann_types

    @staticmethod
    def _load_infos(page_mapper: DatumPageMapper) -> Dict:
        return page_mapper.infos

    @staticmethod
    def _load_categories(page_mapper: DatumPageMapper) -> Dict:
        return JsonReader._load_categories({"categories": page_mapper.categories})

    def _load_items(self, parsed) -> List:
        return []


class DatumaroBase(SubsetBase):
    LEGACY_VERSION = "legacy"
    CURRENT_DATUMARO_FORMAT_VERSION = DATUMARO_FORMAT_VERSION

    # If Datumaro format version goes up, it will be
    # ALLOWED_VERSIONS = {LEGACY_VERSION, 1.0, ..., CURRENT_DATUMARO_FORMAT_VERSION}
    ALLOWED_VERSIONS = {LEGACY_VERSION, CURRENT_DATUMARO_FORMAT_VERSION}

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        stream: bool = False,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isfile(path), path
        subset = osp.splitext(osp.basename(path))[0] if subset is None else subset
        self._stream = stream
        super().__init__(subset=subset, ctx=ctx)
        self._init_path(path)

        dm_version = self._get_dm_format_version(path)

        # when backward compatibility happen, we should implement version specific readers
        if dm_version not in self.ALLOWED_VERSIONS:
            raise DatasetImportError(
                f"Datumaro format version of the given dataset is {dm_version}, "
                f"but not supported by this Datumaro version: {__version__}. "
                f"The allowed datumaro format versions are {self.ALLOWED_VERSIONS}. "
                "Please install the latest Datumaro."
            )

        self._load_impl(path)

    def _init_path(self, path: str):
        """Initialize path variables"""
        rootpath = ""
        if path.endswith(osp.join(DatumaroPath.ANNOTATIONS_DIR, osp.basename(path))):
            rootpath = path.rsplit(DatumaroPath.ANNOTATIONS_DIR, maxsplit=1)[0]
        self._rootpath = rootpath

        images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, DatumaroPath.IMAGES_DIR)
        self._images_dir = images_dir

        pcd_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.PCD_DIR)):
            pcd_dir = osp.join(rootpath, DatumaroPath.PCD_DIR)
        self._pcd_dir = pcd_dir

        video_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.VIDEO_DIR)):
            video_dir = osp.join(rootpath, DatumaroPath.VIDEO_DIR)
        self._video_dir = video_dir

    @property
    def is_stream(self) -> bool:
        return self._stream

    def infos(self):
        return self._reader.infos

    def categories(self):
        return self._reader.categories

    def media_type(self):
        return self._reader.media_type

    def ann_types(self):
        return self._reader.ann_types

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro format."""
        self._reader = (
            JsonReader(
                path,
                self._subset,
                self._rootpath,
                self._images_dir,
                self._pcd_dir,
                self._video_dir,
                self._ctx,
            )
            if not self._stream
            else StreamJsonReader(
                path,
                self._subset,
                self._rootpath,
                self._images_dir,
                self._pcd_dir,
                self._video_dir,
                self._ctx,
            )
        )
        return self._reader

    def _get_dm_format_version(self, path) -> str:
        """
        Get Datumaro format at exporting the dataset

        Note that the legacy Datumaro doesn't store the version into exported dataset.
        Thus it returns DatumaroBase.LEGACY_VERSION
        """
        # We can assume that the version information will be within the first 1 KB of the file.
        # This is because we are the producer of Datumaro format.
        search_size = 1024  # 1 KB

        pattern = '"dm_format_version"\s*:\s*"(\w+)"'

        with open(path, "r", encoding="utf-8") as fp:
            out = fp.read(search_size)
            found = re.search(pattern, out)

            if not found:
                return self.LEGACY_VERSION

        version_str = found.group(1)

        if re.match("\d+\.d+", version_str) is None:
            raise DatasetImportError(f"Invalid version string: {version_str} ")

        return version_str

    def __len__(self) -> int:
        return len(self._reader)

    def __iter__(self) -> DatasetItem:
        yield from self._reader
