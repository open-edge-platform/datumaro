# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os
import os.path as osp
from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError, InvalidAnnotationError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class ImagenetTxtPath:
    LABELS_FILE = "synsets.txt"
    IMAGE_DIR = "images"


class _LabelsSource(Enum):
    file = auto()
    generate = auto()


def _parse_annotation_line(line: str) -> Tuple[str, str, Sequence[int]]:
    item = line.split('"')
    if len(item) > 1:
        if len(item) == 3:
            item_id = item[1]
            item = item[2].split()
            image = item_id + item[0]
            label_ids = [int(id) for id in item[1:]]
        else:
            raise InvalidAnnotationError("Line %s: unexpected number of quotes in filename" % line)
    else:
        item = line.split()
        item_id = osp.splitext(item[0])[0]
        image = item[0]
        label_ids = [int(id) for id in item[1:]]

    return item_id, image, label_ids


class ImagenetTxtBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
        labels: Union[Iterable[str], str] = _LabelsSource.file.name,
        labels_file: str = ImagenetTxtPath.LABELS_FILE,
        image_dir: Optional[str] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find dataset file", path)

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        root_dir = osp.dirname(path)
        if not image_dir:
            image_dir = ImagenetTxtPath.IMAGE_DIR
        self.image_dir = osp.join(root_dir, image_dir)

        self._generate_labels = False

        if isinstance(labels, str):
            labels_source = _LabelsSource[labels]

            if labels_source == _LabelsSource.generate:
                labels = ()
                self._generate_labels = True
            elif labels_source == _LabelsSource.file:
                if has_meta_file(root_dir):
                    labels = parse_meta_file(root_dir).keys()
                else:
                    labels = self._parse_labels(osp.join(root_dir, labels_file))
            else:
                assert False, "Unhandled labels source %s" % labels_source
        else:
            assert all(isinstance(e, str) for e in labels)

        self._categories = self._load_categories(labels)

        self._items = list(self._load_items(path).values())

    @staticmethod
    def _parse_labels(path):
        with open(path, encoding="utf-8") as labels_file:
            return [s.strip() for s in labels_file]

    def _load_categories(self, labels):
        return {AnnotationType.label: LabelCategories.from_iterable(labels)}

    def _load_items(self, path):
        items = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                item_id, image, label_ids = _parse_annotation_line(line)

                anno = []
                label_categories = self._categories[AnnotationType.label]

                for label in label_ids:
                    if label < 0:
                        raise DatasetImportError(f"Image '{item_id}': invalid label id '{label}'")

                    if len(label_categories) <= label:
                        if self._generate_labels:
                            while len(label_categories) <= label:
                                label_categories.add(f"class-{len(label_categories)}")
                        else:
                            raise DatasetImportError(f"Image '{item_id}': unknown label id '{label}'")

                    anno.append(Label(label))
                    self._ann_types.add(AnnotationType.label)

                items[item_id] = DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=Image.from_file(path=osp.join(self.image_dir, image)),
                    annotations=anno,
                )
        return items


class ImagenetTxtImporter(Importer, CliPlugin):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_path = context.require_file(f"*{cls._ANNO_EXT}", exclude_fnames=ImagenetTxtPath.LABELS_FILE)

        with context.probe_text_file(
            annot_path,
            "must be an ImageNet-like annotation file",
        ) as f:
            for line in f:
                _, _, label_ids = _parse_annotation_line(line)
                if label_ids:
                    break
            else:
                # If there are no labels in the entire file, it's probably
                # not actually an ImageNet file.
                raise Exception

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--labels",
            choices=_LabelsSource.__members__,
            default=_LabelsSource.file.name,
            help="Where to get label descriptions from (use "
            "'file' to load from the file specified by --labels-file; "
            "'generate' to create generic ones)",
        )
        parser.add_argument(
            "--labels-file",
            default=ImagenetTxtPath.LABELS_FILE,
            help="Path to the file with label descriptions (synsets.txt)",
        )
        return parser

    @classmethod
    def find_sources_with_params(cls, path, **extra_params):
        if "labels" not in extra_params or extra_params["labels"] == _LabelsSource.file.name:
            labels_file_name = osp.basename(extra_params.get("labels_file") or ImagenetTxtPath.LABELS_FILE)

            def file_filter(p):
                return osp.basename(p) != labels_file_name

        else:
            file_filter = None

        return cls._find_sources_recursive(path, ".txt", "imagenet_txt", file_filter=file_filter)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class ImagenetTxtExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        subset_dir = self._save_dir
        os.makedirs(subset_dir, exist_ok=True)

        extractor = self._extractor
        for subset_name, subset in self._extractor.subsets().items():
            annotation_file = osp.join(subset_dir, "%s.txt" % subset_name)

            labels = {}
            for item in subset:
                item_id = item.id
                if len(item_id.split()) > 1:
                    item_id = '"' + item_id + '"'
                item_id += self._find_image_ext(item)
                labels[item_id] = set(p.label for p in item.annotations if p.type == AnnotationType.label)

                if self._save_media and item.media:
                    self._save_image(item, subdir=ImagenetTxtPath.IMAGE_DIR)

            annotation = ""
            for item_id, item_labels in labels.items():
                annotation += "%s %s\n" % (item_id, " ".join(str(l) for l in item_labels))

            with open(annotation_file, "w", encoding="utf-8") as f:
                f.write(annotation)

        if self._save_dataset_meta:
            self._save_meta_file(subset_dir)
        else:
            labels_file = osp.join(subset_dir, ImagenetTxtPath.LABELS_FILE)
            with open(labels_file, "w", encoding="utf-8") as f:
                f.writelines(l.name + "\n" for l in extractor.categories().get(AnnotationType.label, LabelCategories()))
