from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import product
from typing import Dict
from unittest import TestCase, skipIf

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.sampler.random_sampler import LabelRandomSampler, RandomSampler
from tests.utils.test_utils import compare_datasets, compare_datasets_strict

try:
    import pandas as pd

    from datumaro.plugins.sampler.algorithm.entropy import SampleEntropy as entropy
    from datumaro.plugins.sampler.relevancy_sampler import RelevancySampler

    has_libs = True
except ImportError:
    has_libs = False


@skipIf(not has_libs, "pandas library is not available")
class TestRelevancySampler(TestCase):
    @staticmethod
    def _get_probs(out_range=False):
        probs = []
        inference_file = "tests/assets/sampler/inference.csv"
        with open(inference_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            col = 0
            for row in csv_reader:
                if col == 0:
                    col += 1
                    continue
                if out_range:
                    probs.append(list(map(lambda x: -float(x), row[1:4])))
                else:
                    probs.append(list(map(float, row[1:4])))
        return probs

    def _generate_classification_dataset(
        self, config, subset=None, empty_scores=False, out_range=False, no_attr=False, no_img=False
    ):
        probs = self._get_probs(out_range)
        if subset is None:
            self.subset = ["train", "val", "test"]
        else:
            self.subset = subset

        iterable = []
        label_cat = LabelCategories()
        idx = 0
        for label_id, label in enumerate(config.keys()):
            num_item = config[label]
            label_cat.add(label, attributes=None)
            for _ in range(num_item):
                scores = probs[idx]
                idx += 1
                if empty_scores:
                    scores = []
                attr = {"scores": scores}
                if no_attr:
                    attr = {}
                img = Image.from_file(path=f"test/dataset/{idx}.jpg", size=(90, 90))
                if no_img:
                    img = None
                iterable.append(
                    DatasetItem(
                        idx,
                        subset=self.subset[idx % len(self.subset)],
                        annotations=[
                            Label(
                                label_id,
                                attributes=attr,
                            )
                        ],
                        media=img,
                    )
                )
        categories = {AnnotationType.label: label_cat}
        dataset = Dataset.from_iterable(iterable, categories)
        return dataset

    def test_sampler_get_sample_classification(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config, ["train"])
        num_pre_train_subset = len(source.get_subset("train"))

        num_sample = 5

        with self.subTest("Top-K method"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            topk_expected_result = [1, 4, 9, 10, 26]
            topk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(topk_result), topk_expected_result)

        with self.subTest("Low-K method"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="lowk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            lowk_expected_result = [2, 6, 14, 21, 23]
            lowk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(lowk_result), lowk_expected_result)

        with self.subTest("Rand-K method"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="randk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

        with self.subTest("Mix-K method"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="mixk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            mixk_expected_result = [2, 4, 10, 23, 26]
            mixk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="mixk",
                count=6,
                output_file=None,
            )
            self.assertEqual(6, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            mixk_expected_result = [2, 4, 6, 10, 23, 26]
            mixk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

        with self.subTest("Randtop-K method"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="randtopk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

    def test_sampler_gives_error(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }
        num_sample = 5

        source = self._generate_classification_dataset(config)

        with self.subTest("Not found"):
            with self.assertRaisesRegex(Exception, "Unknown subset"):
                subset = "hello"
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset=subset,
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Unknown element"):
                algorithm = "hello"
                result = RelevancySampler(
                    source,
                    algorithm=algorithm,
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Unknown element"):
                sampling_method = "hello"
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method=sampling_method,
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Invalid Value"):
            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = 0
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = -1
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = "string"
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "extension"):
                output_file = "string.xml"
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
                    output_file=output_file,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid Data, ImageID not found in data"):
                sub = source.get_subset("train")

                data_df = defaultdict(list)
                infer_df = defaultdict(list)

                for data in sub:
                    width, height = data.media.size
                    data_df["Width"].append(width)
                    data_df["Height"].append(height)
                    data_df["ImagePath"].append(data.media.path)

                    for annotation in data.annotations:
                        probs = annotation.attributes["scores"]
                        infer_df["ImageID"].append(data.id)

                        for prob_idx, prob in enumerate(probs):
                            infer_df[f"ClassProbability{prob_idx + 1}"].append(prob)

                data_df = pd.DataFrame(data_df)
                infer_df = pd.DataFrame(infer_df)

                entropy(data_df, infer_df)

            with self.assertRaisesRegex(Exception, "Invalid Data, ImageID not found in inference"):
                sub = source.get_subset("train")

                data_df = defaultdict(list)
                infer_df = defaultdict(list)

                for data in sub:
                    width, height = data.media.size
                    data_df["ImageID"].append(data.id)
                    data_df["Width"].append(width)
                    data_df["Height"].append(height)
                    data_df["ImagePath"].append(data.media.path)

                    for annotation in data.annotations:
                        probs = annotation.attributes["scores"]

                        for prob_idx, prob in enumerate(probs):
                            infer_df[f"ClassProbability{prob_idx + 1}"].append(prob)

                data_df = pd.DataFrame(data_df)
                infer_df = pd.DataFrame(infer_df)

                entropy(data_df, infer_df)

    def test_sampler_get_invalid_data(self):
        with self.subTest("empty dataset"):
            config = {
                "label1": 0,
                "label2": 0,
                "label3": 0,
            }

            source = self._generate_classification_dataset(config)
            with self.assertRaisesRegex(Exception, "Unknown subset"):
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Dataset without Scores (Probability)"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, empty_scores=True)
            with self.assertRaisesRegex(Exception, "ClassProbability"):
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Out of range, probability (Less than 0 or more than 1)"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, empty_scores=False, out_range=True)
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No Scores Attribute Data"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, no_attr=True)
            with self.assertRaisesRegex(Exception, "does not have 'scores'"):
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No Image Data"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, no_img=True)
            with self.assertRaisesRegex(Exception, "does not have image info"):
                result = RelevancySampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

    def test_sampler_number_of_samples(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)
        num_pre_train_subset = len(source.get_subset("train"))

        with self.subTest("k > num of data with top-k"):
            num_sample = 500
            sampling_method = "topk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with low-k"):
            num_sample = 500
            sampling_method = "lowk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with rand-k"):
            num_sample = 500
            sampling_method = "randk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with mix-k"):
            num_sample = 500
            sampling_method = "mixk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with randtop-k"):
            num_sample = 500
            sampling_method = "randtopk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with top-k"):
            num_sample = 10
            sampling_method = "topk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with low-k"):
            num_sample = 10
            sampling_method = "lowk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with rand-k"):
            num_sample = 10
            sampling_method = "randk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with mix-k"):
            num_sample = 10
            sampling_method = "mixk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with randtop-k"):
            num_sample = 10
            sampling_method = "randtopk"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

            num_sample = 9

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(len(result.get_subset("sample")), 9)

    def test_sampler_accumulated_sampling(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 3
            sample_subset_name = "sample"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample * 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample * 3)

        with self.subTest("Same Subset, 2, 3, 4 sampling"):
            sample_subset_name = "sample"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 5)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 5)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 9)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 9)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 3
            sample_subset_name = "sample"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset=sample_subset_name,
                unsampled_subset="val",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset=sample_subset_name,
                unsampled_subset="test",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - num_sample)

        with self.subTest("Different Subset, 2, 3, 4 sampling"):
            sample_subset_name = "sample"

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset=sample_subset_name,
                unsampled_subset="val",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 5)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 3)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset=sample_subset_name,
                unsampled_subset="test",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 9)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 4)

    def test_sampler_unaccumulated_sampling(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 3

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample2",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample * 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample3",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample * 3)

        with self.subTest("Same Subset, 2, 3, 4 sampling"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample2",
                unsampled_subset="train",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 5)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample3",
                unsampled_subset="train",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("sample3")), 4)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 9)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 3

            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset="sample2",
                unsampled_subset="val",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - num_sample)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset="sample3",
                unsampled_subset="test",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - num_sample)

        with self.subTest("Different Subset, 2, 3, 4 sampling"):
            result = RelevancySampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset="sample2",
                unsampled_subset="val",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 3)

            result = RelevancySampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset="sample3",
                unsampled_subset="test",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("sample3")), 4)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 4)

    def test_sampler_parser(self):
        from argparse import ArgumentParser

        assert isinstance(RelevancySampler.build_cmdline_parser(), ArgumentParser)


class TestRandomSampler(TestCase):
    @staticmethod
    def _make_dataset(config: Dict[str, int]):
        return Dataset.from_iterable(
            [DatasetItem(i, subset=subset) for subset, subset_size in config.items() for i in range(subset_size)]
        )

    def test_can_sample_when_no_subsets(self):
        n = 10
        source = self._make_dataset({None: n})

        for k in [5, 10, 15]:
            with self.subTest(k=k):
                actual = RandomSampler(source, k)

                self.assertEqual(min(k, n), len(actual))

    def test_can_sample_when_has_subsets(self):
        n = 10
        source = self._make_dataset({"a": 7, "b": 3})

        for k in [5, 10, 15]:
            with self.subTest(k=k):
                actual = RandomSampler(source, k)

                self.assertEqual(min(k, n), len(actual))

    def test_can_sample_when_subset_selected(self):
        source = self._make_dataset({"a": 7, "b": 3})

        s = "a"
        for k in [5, 7, 15]:
            with self.subTest(k=k, s=s):
                actual = RandomSampler(source, k, subset=s)

                self.assertEqual(min(k, len(source.get_subset(s))), len(actual.get_subset(s)))
                compare_datasets_strict(self, source.get_subset("b"), actual.get_subset("b"))

    def test_can_reproduce_sequence(self):
        source = self._make_dataset({"a": 7, "b": 3})

        seed = 42
        actual1 = RandomSampler(source, 5, seed=seed)
        actual2 = RandomSampler(source, 5, seed=seed)

        compare_datasets_strict(self, actual1, actual2)

    def test_can_change_sequence(self):
        source = self._make_dataset({"a": 7, "b": 3})

        actual1 = RandomSampler(source, 5, seed=1)
        actual2 = RandomSampler(source, 5, seed=2)

        with self.assertRaises(AssertionError):
            compare_datasets_strict(self, actual1, actual2)


class TestLabelRandomSampler(TestCase):
    def test_can_sample_with_common_count(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(i, subset=s, annotations=[Label(l)])
                for i, (s, l, _) in enumerate(product(["a", "b"], [0, 1, 2], [0, 1, 2]))
            ],
            categories=["a", "b", "c"],
        )

        actual = LabelRandomSampler(source, count=2)

        counts_a = Counter(a.label for item in actual.get_subset("a") for a in item.annotations)
        counts_b = Counter(a.label for item in actual.get_subset("b") for a in item.annotations)
        self.assertEqual(set(counts_a.values()), {2})
        self.assertEqual(set(counts_b.values()), {2})

    def test_can_sample_with_selective_count(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(i, subset=s, annotations=[Label(l)])
                for i, (s, l, _) in enumerate(product(["a", "b"], [0, 1, 2], [0, 1, 2]))
            ],
            categories=["a", "b", "c"],
        )

        actual = LabelRandomSampler(source, count=2, label_counts={"a": 0, "b": 1})

        counts_a = Counter(a.label for item in actual.get_subset("a") for a in item.annotations)
        counts_b = Counter(a.label for item in actual.get_subset("b") for a in item.annotations)
        self.assertEqual(
            counts_a,
            {
                actual.categories()[AnnotationType.label].find("b")[0]: 1,
                actual.categories()[AnnotationType.label].find("c")[0]: 2,
            },
        )
        self.assertEqual(
            counts_b,
            {
                actual.categories()[AnnotationType.label].find("b")[0]: 1,
                actual.categories()[AnnotationType.label].find("c")[0]: 2,
            },
        )

    def test_can_change_output_labels(self):
        expected = Dataset.from_iterable([], categories=["a"])

        source = Dataset.from_iterable([], categories=["a", "b", "c"])
        actual = LabelRandomSampler(source, label_counts={"a": 1, "b": 0})

        compare_datasets(self, expected, actual)

    def test_can_reiterate_sequence(self):
        source = Dataset.from_iterable(
            [
                DatasetItem("1", subset="a", annotations=[Label(0), Label(1)]),
                DatasetItem("2", subset="a", annotations=[Label(1)]),
                DatasetItem("3", subset="a", annotations=[Label(2)]),
                DatasetItem("4", subset="a", annotations=[Label(1), Label(2)]),
                DatasetItem("5", subset="b", annotations=[Label(0)]),
                DatasetItem("6", subset="b", annotations=[Label(0), Label(2)]),
                DatasetItem("7", subset="b", annotations=[Label(1), Label(2)]),
                DatasetItem("8", subset="b", annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        transformed = LabelRandomSampler(source, count=2)

        actual1 = Dataset.from_extractors(transformed)
        actual1.init_cache()

        actual2 = Dataset.from_extractors(transformed)
        actual2.init_cache()

        compare_datasets_strict(self, actual1, actual2)

    def test_can_reproduce_sequence(self):
        source = Dataset.from_iterable(
            [
                DatasetItem("1", subset="a", annotations=[Label(0), Label(1)]),
                DatasetItem("2", subset="a", annotations=[Label(1)]),
                DatasetItem("3", subset="a", annotations=[Label(2)]),
                DatasetItem("4", subset="a", annotations=[Label(1), Label(2)]),
                DatasetItem("5", subset="b", annotations=[Label(0)]),
                DatasetItem("6", subset="b", annotations=[Label(0), Label(2)]),
                DatasetItem("7", subset="b", annotations=[Label(1), Label(2)]),
                DatasetItem("8", subset="b", annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        seed = 42
        actual1 = LabelRandomSampler(source, count=2, seed=seed)
        actual2 = LabelRandomSampler(source, count=2, seed=seed)

        compare_datasets_strict(self, actual1, actual2)

    def test_can_change_sequence(self):
        source = Dataset.from_iterable(
            [
                DatasetItem("1", subset="a", annotations=[Label(0), Label(1)]),
                DatasetItem("2", subset="a", annotations=[Label(1)]),
                DatasetItem("3", subset="a", annotations=[Label(2)]),
                DatasetItem("4", subset="a", annotations=[Label(1), Label(2)]),
                DatasetItem("5", subset="b", annotations=[Label(0)]),
                DatasetItem("6", subset="b", annotations=[Label(0), Label(2)]),
                DatasetItem("7", subset="b", annotations=[Label(1), Label(2)]),
                DatasetItem("8", subset="b", annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        actual1 = LabelRandomSampler(source, count=2, seed=1)
        actual2 = LabelRandomSampler(source, count=2, seed=2)

        with self.assertRaises(AssertionError):
            compare_datasets_strict(self, actual1, actual2)
