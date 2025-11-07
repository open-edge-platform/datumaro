import os.path as osp
from unittest.case import TestCase, skipIf

import numpy as np

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.extractor_tfds import AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE
from datumaro.components.media import Image, MediaElement
from datumaro.util.image import decode_image, encode_image
from tests.utils.test_utils import compare_datasets, mock_tfds_data

if TFDS_EXTRACTOR_AVAILABLE:
    import tensorflow_datasets as tfds


@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class TfdsDatasetsTest(TestCase):
    def test_metadata(self):
        env = Environment()

        for dataset in AVAILABLE_TFDS_DATASETS.values():
            assert isinstance(dataset.metadata.human_name, str)
            assert dataset.metadata.human_name != ""

            assert dataset.metadata.default_output_format in env.exporters

            assert issubclass(dataset.metadata.media_type, MediaElement)

            # The home URL is optional, but currently every dataset has one.
            assert isinstance(dataset.metadata.home_url, str)
            assert dataset.metadata.home_url != ""

    def test_remote_metadata(self):
        with mock_tfds_data():
            dataset = AVAILABLE_TFDS_DATASETS["mnist"]

            remote_meta = dataset.query_remote_metadata()

            # verify that the remote metadata contains a copy of the local metadata
            for attribute in dataset.metadata.__attrs_attrs__:
                assert getattr(dataset.metadata, attribute.name) == getattr(remote_meta, attribute.name)

            tfds_info = tfds.builder("mnist").info

            assert remote_meta.description == tfds_info.description
            assert remote_meta.download_size == tfds_info.download_size
            assert remote_meta.num_classes == len(tfds_info.features["label"].names)
            assert remote_meta.version == tfds_info.version

            assert len(remote_meta.subsets) == len(tfds_info.splits)

            for split_name, split in tfds_info.splits.items():
                assert remote_meta.subsets[split_name].num_items == sum(split.shard_lengths)


@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class TfdsExtractorTest(TestCase):
    def test_data_access(self):
        with mock_tfds_data(subsets=("train", "val")):
            extractor = AVAILABLE_TFDS_DATASETS["mnist"].make_extractor()
            self.assertEqual(len(extractor), 2)

            expected_train_subset = Dataset(extractor).filter("/item[subset = 'train']")

            compare_datasets(self, expected_train_subset, Dataset(extractor.get_subset("train")))

            self.assertRaises(KeyError, extractor.get_subset, "test")

            subsets = extractor.subsets()
            self.assertEqual(["train", "val"], sorted(subsets))
            compare_datasets(self, expected_train_subset, Dataset(subsets["train"]))

            self.assertIsNotNone(extractor.get("0"))
            self.assertIsNotNone(extractor.get("0", subset="train"))
            self.assertIsNotNone(extractor.get("0", subset="val"))
            self.assertIsNone(extractor.get("x"))
            self.assertIsNone(extractor.get("0", subset="test"))

    def test_can_extract_mnist(self):
        with mock_tfds_data():
            tfds_ds, tfds_info = tfds.load("mnist", split="train", with_info=True)
            tfds_example = next(iter(tfds_ds))

            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="0",
                        subset="train",
                        media=Image.from_numpy(data=tfds_example["image"].numpy().squeeze(axis=2)),
                        annotations=[Label(tfds_example["label"].numpy())],
                    ),
                ],
                categories=tfds_info.features["label"].names,
            )

            extractor = AVAILABLE_TFDS_DATASETS["mnist"].make_extractor()
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    def _test_can_extract_cifar(self, name):
        with mock_tfds_data():
            tfds_ds, tfds_info = tfds.load(name, split="train", with_info=True)
            tfds_example = next(iter(tfds_ds))

            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=tfds_example["id"].numpy().decode("UTF-8"),
                        subset="train",
                        media=Image.from_numpy(data=tfds_example["image"].numpy()[..., ::-1]),
                        annotations=[Label(tfds_example["label"].numpy())],
                    ),
                ],
                categories=tfds_info.features["label"].names,
            )

            extractor = AVAILABLE_TFDS_DATASETS[name].make_extractor()
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    def test_can_extract_cifar10(self):
        self._test_can_extract_cifar("cifar10")

    def test_can_extract_cifar100(self):
        self._test_can_extract_cifar("cifar100")

    def test_can_extract_coco(self):
        tfds_example = {
            "image": encode_image(np.ones((20, 10)), ".png"),
            "image/filename": "test.png",
            "image/id": 123,
            "objects": {
                "bbox": [[0.1, 0.2, 0.3, 0.4]],
                "label": [5],
                "is_crowd": [True],
            },
        }

        with mock_tfds_data(example=tfds_example):
            tfds_info = tfds.builder("coco/2014").info

            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="test",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((20, 10))),
                        annotations=[
                            Bbox(2, 2, 2, 4, label=5, attributes={"is_crowd": True}),
                        ],
                        attributes={"id": 123},
                    ),
                ],
                categories=tfds_info.features["objects"].feature["label"].names,
            )

            extractor = AVAILABLE_TFDS_DATASETS["coco/2014"].make_extractor()
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    def test_can_extract_imagenet_v2(self):
        with mock_tfds_data():
            tfds_ds, tfds_info = tfds.load(
                "imagenet_v2",
                split="train",
                with_info=True,
                # We can't let TFDS decode the image for us, because:
                # a) imagenet_v2 produces JPEG-encoded images;
                # b) TFDS decodes them via TensorFlow;
                # c) Datumaro decodes them via OpenCV.
                # So for the decoded results to match, we have to decode
                # them via OpenCV as well.
                decoders={"image": tfds.decode.SkipDecoding()},
            )
            tfds_example = next(iter(tfds_ds))

            example_file_name = tfds_example["file_name"].numpy().decode("UTF-8")

            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=osp.splitext(example_file_name)[0],
                        subset="train",
                        media=Image.from_numpy(
                            data=decode_image(tfds_example["image"].numpy()),
                        ),
                        annotations=[Label(tfds_example["label"].numpy())],
                    ),
                ],
                categories=tfds_info.features["label"].names,
            )

            extractor = AVAILABLE_TFDS_DATASETS["imagenet_v2"].make_extractor()
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    def test_can_extract_voc(self):
        # TFDS is unable to generate fake examples for object detection
        # datasets. See <https://github.com/tensorflow/datasets/issues/3633>.
        tfds_example = {
            "image/filename": "test.png",
            "image": encode_image(np.ones((20, 10)), ".png"),
            "objects": {
                "bbox": [[0.1, 0.2, 0.3, 0.4]],
                "label": [5],
                "is_difficult": [True],
                "is_truncated": [False],
                "pose": [0],
            },
        }

        with mock_tfds_data(example=tfds_example):
            tfds_info = tfds.builder("voc/2012").info

            pose_names = tfds_info.features["objects"].feature["pose"].names

            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="test",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((20, 10))),
                        annotations=[
                            Bbox(
                                2,
                                2,
                                2,
                                4,
                                label=5,
                                attributes={
                                    "difficult": True,
                                    "truncated": False,
                                    "pose": pose_names[0].title(),
                                },
                            ),
                        ],
                    ),
                ],
                categories=tfds_info.features["objects"].feature["label"].names,
            )

            extractor = AVAILABLE_TFDS_DATASETS["voc/2012"].make_extractor()
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)
