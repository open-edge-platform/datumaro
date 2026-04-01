from __future__ import annotations

import io
from abc import ABC, abstractmethod
from functools import partial as _partial
from typing import Any

import numpy as np
from PIL import Image as PILImage

from datumaro import Dataset as LegacyDataset
from datumaro import DatasetItem, Image, MediaElement
from datumaro.components.media import FromDataMixin, FromFileMixin, ImageFromBytes, Video, VideoFrame
from datumaro.experimental import AttributeInfo, Sample, Schema
from datumaro.experimental.fields import (
    ImageInfo,
    ImagePathField,
    MediaPathField,
    VideoFramePathField,
    image_bytes_field,
    image_callable_field,
    image_info_field,
    image_path_field,
    media_path_field,
    video_frame_path_field,
)
from datumaro.experimental.media import LazyImage, LazyVideoFrame


class ForwardMediaConverter(ABC):
    """Base class for forward media type converters."""

    @classmethod
    @abstractmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return list of media types this converter can handle."""

    @classmethod
    @abstractmethod
    def create(
        cls, dataset: LegacyDataset, semantic: str = "default", name_prefix: str = ""
    ) -> ForwardMediaConverter | None:
        """Create converter instance if dataset is supported, None otherwise.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """

    @abstractmethod
    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this media type."""

    @abstractmethod
    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert media from a DatasetItem to new dataset format."""


def _image_callable_impl(bytes_source: Any, is_callable: bool = False):
    """Convert image bytes (or bytes provider) to a numpy array.

    Implemented at module scope so that partials of this function are pickleable
    and thus safe to use with multi-processing data loaders.
    """
    # Get the bytes data (either directly or from callable)
    bytes_data = bytes_source() if is_callable else bytes_source
    if not isinstance(bytes_data, bytes):
        raise TypeError(f"Expected bytes data, got {type(bytes_data)}")
    # Convert bytes to image array using PIL
    with PILImage.open(io.BytesIO(bytes_data)) as pil_image:
        processed_image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return np.array(processed_image, dtype=np.uint8)


class ForwardImageMediaConverter(ForwardMediaConverter):
    """Forward converter for Image media type supporting both file paths and byte data."""

    def __init__(
        self,
        media_mixin: type,
        has_image_info: bool,
        semantic: str = "default",
        name_prefix: str = "",
        has_callable_data: bool = False,
    ):
        """Initialize converter with format preference and image info availability."""
        self.media_mixin = media_mixin
        self.has_image_info = has_image_info
        self.has_callable_data = has_callable_data
        self.semantic = semantic
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return list of media types this converter can handle."""
        return [Image]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: str = "default", name_prefix: str = ""
    ) -> ForwardImageMediaConverter | None:
        """Create converter instance, detecting whether to use paths or bytes.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
        """
        found_media_type: type | None = None
        has_image_info = True  # Assume all images have size until proven otherwise
        has_callable_data = False  # Track if any FromDataMixin has callable _data

        for item in dataset:
            if isinstance(item.media, Image):
                media_type = type(item.media)
                if found_media_type is not None and media_type != found_media_type:
                    raise ValueError(
                        f"The dataset has a mix of different image media types: "
                        f"{found_media_type} and {media_type}. This is not supported by the converter."
                    )

                found_media_type = media_type

                # Check if this image has size info
                if not item.media.has_size:
                    has_image_info = False

                # Check if this is FromDataMixin with callable _data
                if isinstance(item.media, FromDataMixin) and callable(item.media._data):
                    has_callable_data = True

        if found_media_type is None:
            return None

        if issubclass(found_media_type, FromDataMixin):
            media_mixin = FromDataMixin
        elif issubclass(found_media_type, FromFileMixin):
            media_mixin = FromFileMixin
        else:
            raise ValueError(f"Unknown media mixin for {found_media_type}.")

        return cls(
            media_mixin=media_mixin,
            has_image_info=has_image_info,
            semantic=semantic,
            name_prefix=name_prefix,
            has_callable_data=has_callable_data,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes: dict[str, AttributeInfo] = {}

        if self.media_mixin == FromDataMixin:
            if self.has_callable_data:
                attributes[self.name_prefix + "image_callable"] = AttributeInfo(
                    type=callable, field=image_callable_field(semantic=self.semantic)
                )
            else:
                attributes[self.name_prefix + "image_bytes"] = AttributeInfo(
                    type=bytes, field=image_bytes_field(semantic=self.semantic)
                )
        elif self.media_mixin == FromFileMixin:
            attributes[self.name_prefix + "image_path"] = AttributeInfo(
                type=str, field=image_path_field(semantic=self.semantic)
            )
        else:
            raise RuntimeError(f"Media mixin not implemented: {self.media_mixin}")

        # Add image info field if all images have size
        if self.has_image_info:
            attributes[self.name_prefix + "image_info"] = AttributeInfo(
                type=ImageInfo, field=image_info_field(semantic=self.semantic)
            )

        return attributes

    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if isinstance(item.media, (Image, ImageFromBytes)):
            if self.media_mixin == FromDataMixin:
                if self.has_callable_data:
                    # Use a top-level callable to ensure picklability across workers
                    is_callable = callable(item.media._data)
                    bytes_source = item.media._data
                    result[self.name_prefix + "image_callable"] = _partial(
                        _image_callable_impl, bytes_source, is_callable
                    )
                else:
                    result[self.name_prefix + "image_bytes"] = item.media._data
            elif self.media_mixin == FromFileMixin:
                result[self.name_prefix + "image_path"] = item.media.path
            else:
                raise RuntimeError(f"Media mixin not implemented: {self.media_mixin}")

            # Add image info if available
            if self.has_image_info and item.media.has_size:
                height, width = item.media.size  # size returns (H, W)
                result[self.name_prefix + "image_info"] = ImageInfo(width=width, height=height)

        return result


class ForwardVideoMediaConverter(ForwardMediaConverter):
    """Forward converter for VideoFrame media type.

    Converts legacy VideoFrame media to the new dataset format using
    video_frame_path_field, storing the video file path and frame index.
    """

    def __init__(self, semantic: str = "default", name_prefix: str = ""):
        """Initialize converter.

        Args:
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        self.semantic = semantic
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return list of media types this converter can handle."""
        return [VideoFrame]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: str = "default", name_prefix: str = ""
    ) -> ForwardVideoMediaConverter | None:
        """Create converter instance if dataset contains VideoFrame media.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        has_video_frame = False

        for item in dataset:
            if isinstance(item.media, VideoFrame):
                has_video_frame = True
                break

        if not has_video_frame:
            return None

        return cls(semantic=semantic, name_prefix=name_prefix)

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for video frame media.

        Generates a video_frame_path field that stores both the video file
        path and the frame index.
        """
        attributes: dict[str, AttributeInfo] = {}
        attributes[self.name_prefix + "video_frame"] = AttributeInfo(
            type=LazyVideoFrame, field=video_frame_path_field(semantic=self.semantic)
        )
        return attributes

    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert VideoFrame media to video_frame_path field values.

        Extracts the video file path and frame index from the legacy VideoFrame.
        """
        result: dict[str, Any] = {}

        if isinstance(item.media, VideoFrame):
            video_path = item.media.video.path
            frame_index = item.media.index
            result[self.name_prefix + "video_frame"] = LazyVideoFrame(
                video_path=video_path,
                frame_index=frame_index,
            )

        return result


class ForwardMixedMediaConverter(ForwardMediaConverter):
    """Forward converter for datasets containing VideoFrame media (with or without images).

    When a legacy dataset has video frames (either alone or mixed with images),
    this converter uses a unified media_path_field (MediaPathField) to store
    either type.  Images are converted to LazyImage and video frames to
    LazyVideoFrame.

    For datasets containing *only* plain images (no VideoFrame items),
    :class:`ForwardImageMediaConverter` should be used instead.

    This converter is only used for file-path-based images; byte/numpy images
    in mixed datasets are not supported (they should not normally occur in
    practice since video frames are always file-based).
    """

    def __init__(self, semantic: str = "default", name_prefix: str = ""):
        """Initialize converter.

        Args:
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        self.semantic = semantic
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return an empty list — this converter is not registered by media type.

        It is selected explicitly by get_forward_media_converter when any
        VideoFrame items are detected in the dataset.
        """
        return []

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: str = "default", name_prefix: str = ""
    ) -> ForwardMixedMediaConverter | None:
        """Create converter if dataset contains any non-pure-image media.

        This converter is used whenever the dataset has video frames (with or
        without images).  If the dataset contains *only* plain images (no
        VideoFrame items at all), this converter returns ``None`` so that the
        more specific :class:`ForwardImageMediaConverter` can be used instead.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        has_video_frame = False

        for item in dataset:
            if isinstance(item.media, VideoFrame):
                has_video_frame = True
                break

        if has_video_frame:
            return cls(semantic=semantic, name_prefix=name_prefix)

        return None

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes using a unified media_path field."""
        return {
            self.name_prefix + "media": AttributeInfo(
                type=LazyImage | LazyVideoFrame,
                field=media_path_field(semantic=self.semantic),
            )
        }

    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert Image or VideoFrame media to a unified media_path value."""
        result: dict[str, Any] = {}
        key = self.name_prefix + "media"

        if isinstance(item.media, VideoFrame):
            result[key] = LazyVideoFrame(
                video_path=item.media.video.path,
                frame_index=item.media.index,
            )
        elif isinstance(item.media, Image) and isinstance(item.media, FromFileMixin):
            result[key] = LazyImage(path=item.media.path)

        return result


class BackwardMediaConverter(ABC):
    """Base class for backward media type converters."""

    @classmethod
    @abstractmethod
    def create_from_schema(cls, schema: Schema) -> BackwardMediaConverter | None:
        """Create converter instance if schema is supported, None otherwise."""

    @abstractmethod
    def get_media_type(self) -> type[MediaElement[Any]]:
        """Get the legacy media type this converter produces."""

    @abstractmethod
    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert sample media to legacy MediaElement."""


class BackwardImageMediaConverter(BackwardMediaConverter):
    """Backward converter for Image media type."""

    def __init__(self, image_path_attr: str):
        """Initialize with the name of the image path attribute."""
        self.image_path_attr = image_path_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardImageMediaConverter | None:
        """Create converter instance if schema contains image_path field."""
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, ImagePathField):
                return cls(image_path_attr=attr_name)
        return None

    def get_media_type(self) -> type[MediaElement[Any]]:
        return Image

    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert image_path back to Image MediaElement."""
        image_path = getattr(sample, self.image_path_attr)
        return Image.from_file(path=image_path)  # pyright: ignore[reportUnknownMemberType]


class BackwardVideoMediaConverter(BackwardMediaConverter):
    """Backward converter for VideoFrame media type.

    Converts new dataset video frame references back to legacy VideoFrame objects.
    Caches Video objects so that multiple frames from the same video share the
    same Video instance.
    """

    def __init__(self, video_frame_attr: str):
        """Initialize with the name of the video frame attribute.

        Args:
            video_frame_attr: Name of the schema attribute containing LazyVideoFrame data
        """
        self.video_frame_attr = video_frame_attr
        self._video_cache: dict[str, Video] = {}

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardVideoMediaConverter | None:
        """Create converter instance if schema contains video_frame_path field."""
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, VideoFramePathField):
                return cls(video_frame_attr=attr_name)
        return None

    def get_media_type(self) -> type[MediaElement[Any]]:
        return VideoFrame

    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert LazyVideoFrame back to legacy VideoFrame.

        Caches Video objects so that frames from the same video share a single
        Video instance, matching the legacy format's structure.
        """
        lazy_frame: LazyVideoFrame = getattr(sample, self.video_frame_attr)
        video_path = str(lazy_frame.video_path)

        if video_path not in self._video_cache:
            self._video_cache[video_path] = Video(path=video_path)

        video = self._video_cache[video_path]
        return VideoFrame(video=video, index=lazy_frame.frame_index)


class BackwardMixedMediaConverter(BackwardMediaConverter):
    """Backward converter for datasets using a unified MediaPathField.

    Handles schemas that use a single media_path_field to store both image
    and video frame references. Each sample is inspected at conversion time
    to determine whether it contains a LazyImage or LazyVideoFrame, and the
    appropriate legacy media type is returned.

    Because a mixed dataset can contain both Image and VideoFrame items, this
    converter reports MediaElement as the legacy media type (the common base).
    Video objects are cached to ensure frames from the same video share one
    Video instance.
    """

    def __init__(self, media_attr: str):
        """Initialize with the name of the unified media attribute.

        Args:
            media_attr: Name of the schema attribute containing the MediaPathField data
        """
        self.media_attr = media_attr
        self._video_cache: dict[str, Video] = {}

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardMixedMediaConverter | None:
        """Create converter instance if schema contains a MediaPathField."""
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, MediaPathField):
                return cls(media_attr=attr_name)
        return None

    def get_media_type(self) -> type[MediaElement[Any]]:
        # Mixed datasets can hold both Image and VideoFrame items.
        # Return MediaElement (the common base) so the legacy dataset
        # does not restrict the media type.
        return MediaElement

    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert a LazyImage or LazyVideoFrame back to the appropriate legacy media.

        Returns Image for LazyImage values and VideoFrame for LazyVideoFrame values.
        """
        value = getattr(sample, self.media_attr)

        if isinstance(value, LazyVideoFrame):
            video_path = str(value.video_path)
            if video_path not in self._video_cache:
                self._video_cache[video_path] = Video(path=video_path)
            video = self._video_cache[video_path]
            return VideoFrame(video=video, index=value.frame_index)

        if isinstance(value, LazyImage):
            return Image.from_file(path=str(value.path))  # pyright: ignore[reportUnknownMemberType]

        raise TypeError(f"Expected LazyImage or LazyVideoFrame, got {type(value)}")
