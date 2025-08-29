from unittest import TestCase

import numpy as np

import datumaro.plugins.ndr as ndr
from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaShapeError
from datumaro.components.media import Image


class NDRTest(TestCase):
    def _generate_dataset(self, config, num_duplicate, dataset="classification"):
        subsets = ["train", "val", "test"]
        if dataset == "classification":
            dummy_images = [
                np.random.randint(0, 255, size=(224, 224, 3)) for _ in range(num_duplicate)
            ]
        if dataset == "invalid_channel":
            dummy_images = [
                np.random.randint(0, 255, size=(224, 224, 2)) for _ in range(num_duplicate)
            ]
        if dataset == "invalid_dimension":
            dummy_images = [
                np.random.randint(0, 255, size=(224, 224, 3, 3)) for _ in range(num_duplicate)
            ]
        iterable = []
        label_cat = LabelCategories()
        idx = 0
        for label_id, label in enumerate(config.keys()):
            label_cat.add(label, attributes=None)
            num_item = config[label]
            for subset in subsets:
                for _ in range(num_item):
                    idx += 1
                    iterable.append(
                        DatasetItem(
                            idx,
                            subset=subset,
                            annotations=[Label(label_id)],
                            media=Image.from_numpy(data=dummy_images[idx % num_duplicate]),
                        )
                    )
        categories = {AnnotationType.label: label_cat}
        dataset = Dataset.from_iterable(iterable, categories)
        return dataset

    def test_ndr_with_error(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        with self.assertRaisesRegex(ValueError, "Invalid working_subset name"):
            source = self._generate_dataset(config, 3)
            subset = "no_such_subset"
            result = ndr.NDR(source, working_subset=subset)
            len(result)

        with self.assertRaisesRegex(ValueError, "working_subset == duplicated_subset"):
            source = self._generate_dataset(config, 3)
            result = ndr.NDR(source, working_subset="train", duplicated_subset="train")
            len(result)

        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            source = self._generate_dataset(config, 3)
            algorithm = "no_such_algo"
            result = ndr.NDR(source, working_subset="train", algorithm=algorithm)
            len(result)

        with self.assertRaisesRegex(
            ValueError, "The number of images is smaller than the cut you want"
        ):
            source = self._generate_dataset(config, 3)
            result = ndr.NDR(source, working_subset="train", num_cut=10000)
            len(result)

        with self.assertRaisesRegex(ValueError, "Unknown oversampling method"):
            source = self._generate_dataset(config, 10)
            sampling = "no_such_sampling"
            result = ndr.NDR(
                source, working_subset="train", num_cut=100, seed=12145, over_sample=sampling
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "Unknown undersampling method"):
            source = self._generate_dataset(config, 10)
            sampling = "no_such_sampling"
            result = ndr.NDR(
                source, working_subset="train", num_cut=1, seed=12145, under_sample=sampling
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "unexpected number of channels"):
            source = self._generate_dataset(config, 10, "invalid_channel")
            result = ndr.NDR(source, working_subset="train")
            len(result)

        with self.assertRaises(MediaShapeError):
            source = self._generate_dataset(config, 10, "invalid_dimension")
            result = ndr.NDR(source, working_subset="train")
            len(result)

    def test_ndr_without_cut(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(source, working_subset="train", seed=12145)

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_can_use_undersample_uniform(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", num_cut=1, under_sample="uniform", seed=12145
        )

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_can_use_undersample_inverse(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", num_cut=1, under_sample="inverse", seed=12145
        )

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_can_use_oversample_random(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", num_cut=10, over_sample="random", seed=12145
        )

        self.assertEqual(10, len(result.get_subset("train")))
        self.assertEqual(290, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_can_use_oversample_similarity(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", num_cut=10, over_sample="similarity", seed=12145
        )

        self.assertEqual(10, len(result.get_subset("train")))
        self.assertEqual(290, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_gradient_fails_on_invalid_parameters(self):
        source = self._generate_dataset({"label1": 5}, 10)

        with self.assertRaisesRegex(ValueError, "Invalid block_shape"):
            result = ndr.NDR(
                source,
                working_subset="train",
                over_sample="random",
                block_shape=(3, 6, 6),
                algorithm="gradient",
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "block_shape should be positive"):
            result = ndr.NDR(
                source,
                working_subset="train",
                over_sample="random",
                block_shape=(-1, 0),
                algorithm="gradient",
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "sim_threshold should be large than 0"):
            result = ndr.NDR(
                source,
                working_subset="train",
                over_sample="random",
                sim_threshold=0,
                block_shape=(8, 8),
                algorithm="gradient",
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "hash_dim should be smaller than feature shape"):
            result = ndr.NDR(
                source,
                working_subset="train",
                over_sample="random",
                hash_dim=1024,
                block_shape=(8, 8),
                algorithm="gradient",
            )
            len(result)

        with self.assertRaisesRegex(ValueError, "hash_dim should be positive"):
            result = ndr.NDR(
                source,
                working_subset="train",
                over_sample="random",
                hash_dim=-5,
                block_shape=(8, 8),
                algorithm="gradient",
            )
            len(result)

    def test_ndr_gradient_can_use_block(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)
        result = ndr.NDR(
            source, working_subset="train", over_sample="random", block_shape=(8, 8), seed=12145
        )

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_gradient_can_use_hash_dim(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", over_sample="random", hash_dim=16, seed=12145
        )

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_gradient_can_use_sim_thresh(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result = ndr.NDR(
            source, working_subset="train", over_sample="random", sim_threshold=0.7, seed=12145
        )

        self.assertEqual(1, len(result.get_subset("train")))
        self.assertEqual(299, len(result.get_subset("duplicated")))
        self.assertEqual(300, len(result.get_subset("val")))
        self.assertEqual(300, len(result.get_subset("test")))
        # Check source
        self.assertEqual(300, len(source.get_subset("train")))
        self.assertEqual(300, len(source.get_subset("val")))
        self.assertEqual(300, len(source.get_subset("test")))

    def test_ndr_seed(self):
        config = {"label1": 100, "label2": 100, "label3": 100}
        # train : 300, val : 300, test : 300
        np.random.seed(1234)
        source = self._generate_dataset(config, 10)

        result1 = ndr.NDR(source, working_subset="train", seed=12345)
        result2 = ndr.NDR(source, working_subset="train", seed=12345)
        result3 = ndr.NDR(source, working_subset="train", seed=12)

        self.assertEqual(tuple(result1.get_subset("train")), tuple(result2.get_subset("train")))
        self.assertNotEqual(tuple(result1.get_subset("train")), tuple(result3.get_subset("train")))
        self.assertNotEqual(tuple(result2.get_subset("train")), tuple(result3.get_subset("train")))
