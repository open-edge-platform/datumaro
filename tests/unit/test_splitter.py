from unittest import TestCase

import numpy as np

import datumaro.plugins.splitter as splitter
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.operations import compute_ann_statistics


class SplitterTest(TestCase):
    @staticmethod
    def _get_subset(idx):
        subsets = ["", "a", "b", "", "", "a", "", "b", "", "a"]
        return subsets[idx % len(subsets)]

    def _generate_dataset(self, config):
        # counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        # attr1 = ['attr1', 'attr2']
        # attr2 = ['attr1', 'attr3']
        # config = { "label1": { "attrs": attr1, "counts": counts },
        #            "label2": { "attrs": attr2, "counts": counts }}
        iterable = []
        label_cat = LabelCategories()
        idx = 0
        for label_id, label in enumerate(config.keys()):
            anames = config[label]["attrs"]
            counts = config[label]["counts"]
            label_cat.add(label, attributes=anames)
            if isinstance(counts, dict):
                for attrs, count in counts.items():
                    attributes = dict()
                    if isinstance(attrs, tuple):
                        for aname, value in zip(anames, attrs):
                            attributes[aname] = value
                    else:
                        attributes[anames[0]] = attrs
                    for _ in range(count):
                        idx += 1
                        iterable.append(
                            DatasetItem(
                                idx,
                                subset=self._get_subset(idx),
                                annotations=[Label(label_id, attributes=attributes)],
                                media=Image.from_numpy(data=np.ones((1, 1, 3))),
                            )
                        )
            else:
                for _ in range(counts):
                    idx += 1
                    iterable.append(
                        DatasetItem(
                            idx,
                            subset=self._get_subset(idx),
                            annotations=[Label(label_id)],
                            media=Image.from_numpy(data=np.ones((1, 1, 3))),
                        )
                    )
        categories = {AnnotationType.label: label_cat}
        return Dataset.from_iterable(iterable, categories=categories)

    def test_split_for_classification_multi_class_no_attr(self):
        config = {
            "label1": {"attrs": None, "counts": 10},
            "label2": {"attrs": None, "counts": 20},
            "label3": {"attrs": None, "counts": 30},
        }
        source = self._generate_dataset(config)
        task = splitter.SplitTask.classification.name

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.Split(source, task, splits, seed=100)

        self.assertEqual(42, len(actual.get_subset("train")))
        self.assertEqual(18, len(actual.get_subset("test")))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset("train"))
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(7, dist_train["label1"][0])
        self.assertEqual(14, dist_train["label2"][0])
        self.assertEqual(21, dist_train["label3"][0])

        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset("test"))
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(3, dist_test["label1"][0])
        self.assertEqual(6, dist_test["label2"][0])
        self.assertEqual(9, dist_test["label3"][0])

    def test_split_for_classification_single_class_single_attr(self):
        counts = {0: 10, 1: 20, 2: 30}
        config = {"label": {"attrs": ["attr"], "counts": counts}}
        source = self._generate_dataset(config)
        task = splitter.SplitTask.classification.name

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.Split(source, task, splits, seed=100)

        self.assertEqual(42, len(actual.get_subset("train")))
        self.assertEqual(18, len(actual.get_subset("test")))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset("train"))
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(7, attr_train["attr"]["distribution"]["0"][0])
        self.assertEqual(14, attr_train["attr"]["distribution"]["1"][0])
        self.assertEqual(21, attr_train["attr"]["distribution"]["2"][0])

        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset("test"))
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(3, attr_test["attr"]["distribution"]["0"][0])
        self.assertEqual(6, attr_test["attr"]["distribution"]["1"][0])
        self.assertEqual(9, attr_test["attr"]["distribution"]["2"][0])

    def test_split_for_classification_single_class_multi_attr(self):
        counts = {
            (0, 0): 20,
            (0, 1): 20,
            (0, 2): 30,
            (1, 0): 20,
            (1, 1): 10,
            (1, 2): 20,
        }
        attrs = ["attr1", "attr2"]
        config = {"label": {"attrs": attrs, "counts": counts}}
        source = self._generate_dataset(config)
        task = splitter.SplitTask.classification.name

        with self.subTest("zero remainder"):
            splits = [("train", 0.7), ("test", 0.3)]
            actual = splitter.Split(source, task, splits, seed=100)

            self.assertEqual(84, len(actual.get_subset("train")))
            self.assertEqual(36, len(actual.get_subset("test")))

            # check stats for train
            stat_train = compute_ann_statistics(actual.get_subset("train"))
            attr_train = stat_train["annotations"]["labels"]["attributes"]
            self.assertEqual(49, attr_train["attr1"]["distribution"]["0"][0])
            self.assertEqual(35, attr_train["attr1"]["distribution"]["1"][0])
            self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])
            self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])
            self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])

            # check stats for test
            stat_test = compute_ann_statistics(actual.get_subset("test"))
            attr_test = stat_test["annotations"]["labels"]["attributes"]
            self.assertEqual(21, attr_test["attr1"]["distribution"]["0"][0])
            self.assertEqual(15, attr_test["attr1"]["distribution"]["1"][0])
            self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])
            self.assertEqual(9, attr_test["attr2"]["distribution"]["1"][0])
            self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])

        with self.subTest("non-zero remainder"):
            splits = [("train", 0.95), ("test", 0.05)]
            actual = splitter.Split(source, task, splits, seed=100)

            self.assertEqual(114, len(actual.get_subset("train")))
            self.assertEqual(6, len(actual.get_subset("test")))

    def test_split_for_classification_multi_label_with_attr(self):
        counts = {
            (0, 0): 20,
            (0, 1): 20,
            (0, 2): 30,
            (1, 0): 20,
            (1, 1): 10,
            (1, 2): 20,
        }
        attr1 = ["attr1", "attr2"]
        attr2 = ["attr1", "attr3"]
        config = {
            "label1": {"attrs": attr1, "counts": counts},
            "label2": {"attrs": attr2, "counts": counts},
        }
        source = self._generate_dataset(config)
        task = splitter.SplitTask.classification.name

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.Split(source, task, splits, seed=100)

        train = actual.get_subset("train")
        test = actual.get_subset("test")
        self.assertEqual(168, len(train))
        self.assertEqual(72, len(test))

        # check stats for train
        stat_train = compute_ann_statistics(train)
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(84, dist_train["label1"][0])
        self.assertEqual(84, dist_train["label2"][0])
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(49 * 2, attr_train["attr1"]["distribution"]["0"][0])
        self.assertEqual(35 * 2, attr_train["attr1"]["distribution"]["1"][0])
        self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])
        self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])
        self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])
        self.assertEqual(28, attr_train["attr3"]["distribution"]["0"][0])
        self.assertEqual(21, attr_train["attr3"]["distribution"]["1"][0])
        self.assertEqual(35, attr_train["attr3"]["distribution"]["2"][0])

        # check stats for test
        stat_test = compute_ann_statistics(test)
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(36, dist_test["label1"][0])
        self.assertEqual(36, dist_test["label2"][0])
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(21 * 2, attr_test["attr1"]["distribution"]["0"][0])
        self.assertEqual(15 * 2, attr_test["attr1"]["distribution"]["1"][0])
        self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])
        self.assertEqual(9, attr_test["attr2"]["distribution"]["1"][0])
        self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])
        self.assertEqual(12, attr_test["attr3"]["distribution"]["0"][0])
        self.assertEqual(9, attr_test["attr3"]["distribution"]["1"][0])
        self.assertEqual(15, attr_test["attr3"]["distribution"]["2"][0])

        with self.subTest("random seed test"):
            r1 = splitter.Split(source, task, splits, seed=1234)
            r2 = splitter.Split(source, task, splits, seed=1234)
            r3 = splitter.Split(source, task, splits, seed=4321)
            self.assertEqual(list(r1.get_subset("test")), list(r2.get_subset("test")))
            self.assertNotEqual(list(r1.get_subset("test")), list(r3.get_subset("test")))

    def test_split_for_classification_zero_ratio(self):
        config = {
            "label1": {"attrs": None, "counts": 5},
        }
        source = self._generate_dataset(config)
        splits = [("train", 0.1), ("val", 0.9), ("test", 0.0)]
        task = splitter.SplitTask.classification.name

        actual = splitter.Split(source, task, splits, seed=100)

        self.assertEqual(1, len(actual.get_subset("train")))
        self.assertEqual(4, len(actual.get_subset("val")))
        self.assertEqual(0, len(actual.get_subset("test")))

    def test_split_for_classification_unlabeled(self):
        with self.subTest("no label"):
            iterable = [DatasetItem(i, annotations=[]) for i in range(10)]
            source = Dataset.from_iterable(iterable, categories=["a", "b"])
            splits = [("train", 0.7), ("test", 0.3)]
            task = splitter.SplitTask.classification.name
            actual = splitter.Split(source, task, splits, seed=100)

            self.assertEqual(7, len(actual.get_subset("train")))
            self.assertEqual(3, len(actual.get_subset("test")))

        with self.subTest("multi label"):
            anns = [Label(0), Label(1)]
            iterable = [DatasetItem(i, annotations=anns) for i in range(10)]
            source = Dataset.from_iterable(iterable, categories=["a", "b"])
            splits = [("train", 0.7), ("test", 0.3)]
            task = splitter.SplitTask.classification.name
            actual = splitter.Split(source, task, splits, seed=100)

            self.assertEqual(7, len(actual.get_subset("train")))
            self.assertEqual(3, len(actual.get_subset("test")))

    def test_split_for_classification_gives_error(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b", "c"],
        )
        task = splitter.SplitTask.classification.name

        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("test", 1.5)]
                splitter.Split(source, task, splits)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                splitter.Split(source, task, splits)

        with self.subTest("duplicated subset name"), self.assertRaisesRegex(Exception, "duplicated"):
            splits = [("train", 0.5), ("train", 0.2), ("test", 0.3)]
            splitter.Split(source, task, splits)

    def test_split_for_reidentification(self):
        """
        Test ReidentificationSplit using Dataset with label (ImageNet style)
        """

        def _get_present(stat):
            values_present = []
            for label, dist in stat["distribution"].items():
                if dist[0] > 0:
                    values_present.append(label)
            return set(values_present)

        for with_attr in [True, False]:
            if with_attr:
                counts = {i: (i % 3 + 1) * 7 for i in range(10)}
                config = {"person": {"attrs": ["PID"], "counts": counts}}
                attr_for_id = "PID"
            else:
                counts = {}
                config = dict()
                for i in range(10):
                    label = "label%d" % i
                    count = (i % 3 + 1) * 7
                    counts[label] = count
                    config[label] = {"attrs": None, "counts": count}
                attr_for_id = None
            source = self._generate_dataset(config)
            splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
            task = splitter.SplitTask.reid.name
            query = 0.4 / 0.7
            actual = splitter.Split(source, task, splits, query, attr_for_id)

            stats = dict()
            for sname in ["train", "val", "test-query", "test-gallery"]:
                subset = actual.get_subset(sname)
                stat = compute_ann_statistics(subset)["annotations"]["labels"]
                if with_attr:
                    stat = stat["attributes"]["PID"]
                stats[sname] = stat

            # check size of subsets
            self.assertEqual(65, stats["train"]["count"])
            self.assertEqual(26, stats["val"]["count"])
            self.assertEqual(18, stats["test-gallery"]["count"])
            self.assertEqual(24, stats["test-query"]["count"])

            # check ID separation between test set and others
            train_ids = _get_present(stats["train"])
            test_ids = _get_present(stats["test-gallery"])
            for pid in train_ids:
                assert pid not in test_ids
            self.assertEqual(7, len(train_ids))
            self.assertEqual(3, len(test_ids))
            self.assertEqual(train_ids, _get_present(stats["val"]))
            self.assertEqual(test_ids, _get_present(stats["test-query"]))

            # check trainval set statistics
            trainval = stats["train"]["count"] + stats["val"]["count"]
            expected_train_count = int(trainval * 0.5 / 0.7)
            expected_val_count = int(trainval * 0.2 / 0.7)
            self.assertEqual(expected_train_count, stats["train"]["count"])
            self.assertEqual(expected_val_count, stats["val"]["count"])
            dist_train = stats["train"]["distribution"]
            dist_val = stats["val"]["distribution"]
            for pid in train_ids:
                total = counts[int(pid)] if with_attr else counts[pid]
                self.assertEqual(int(total * 0.5 / 0.7), dist_train[pid][0])
                self.assertEqual(int(total * 0.2 / 0.7), dist_val[pid][0])

            # check teset set statistics
            dist_gallery = stats["test-gallery"]["distribution"]
            dist_query = stats["test-query"]["distribution"]
            for pid in test_ids:
                total = counts[int(pid)] if with_attr else counts[pid]
                self.assertEqual(int(total * 0.3 / 0.7), dist_gallery[pid][0])
                self.assertEqual(int(total * 0.4 / 0.7), dist_query[pid][0])

    def test_split_for_reidentification_randomseed(self):
        """
        Test randomseed for reidentification
        """
        counts = {}
        config = dict()
        for i in range(10):
            label = "label%d" % i
            count = (i % 3 + 1) * 7
            counts[label] = count
            config[label] = {"attrs": None, "counts": count}
        source = self._generate_dataset(config)
        task = splitter.SplitTask.reid.name
        splits = [("train", 0.5), ("test", 0.5)]
        query = 0.4 / 0.7
        r1 = splitter.Split(source, task, splits, query, seed=1234)
        r2 = splitter.Split(source, task, splits, query, seed=1234)
        r3 = splitter.Split(source, task, splits, query, seed=4321)
        self.assertEqual(list(r1.get_subset("train")), list(r2.get_subset("train")))
        self.assertNotEqual(list(r1.get_subset("train")), list(r3.get_subset("train")))

    def test_split_for_reidentification_rebalance(self):
        """
        rebalance function shouldn't gives error when there's no exchange
        """
        config = dict()
        for i in range(100):
            label = "label%03d" % i
            config[label] = {"attrs": None, "counts": 7}
        source = self._generate_dataset(config)
        task = splitter.SplitTask.reid.name
        splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
        query = 0.4 / 0.7
        actual = splitter.Split(source, task, splits, query, seed=100)

        self.assertEqual(350, len(actual.get_subset("train")))
        self.assertEqual(140, len(actual.get_subset("val")))
        self.assertEqual(90, len(actual.get_subset("test-gallery")))
        self.assertEqual(120, len(actual.get_subset("test-query")))

    def test_split_for_reidentification_unlabeled(self):
        query = 0.5
        task = splitter.SplitTask.reid.name

        with self.subTest("no label"):
            iterable = [DatasetItem(i, annotations=[]) for i in range(10)]
            source = Dataset.from_iterable(iterable, categories=["a", "b"])
            splits = [("train", 0.6), ("test", 0.4)]
            actual = splitter.Split(source, task, splits, query, seed=100)
            self.assertEqual(10, len(actual.get_subset("not-supported")))

        with self.subTest("multi label"):
            anns = [Label(0), Label(1)]
            iterable = [DatasetItem(i, annotations=anns) for i in range(10)]
            source = Dataset.from_iterable(iterable, categories=["a", "b"])
            splits = [("train", 0.6), ("test", 0.4)]
            actual = splitter.Split(source, task, splits, query, seed=100)

            self.assertEqual(10, len(actual.get_subset("not-supported")))

    def test_split_for_reidentification_gives_error(self):
        query = 0.4 / 0.7  # valid query ratio
        task = splitter.SplitTask.reid.name

        counts = {i: (i % 3 + 1) * 7 for i in range(10)}
        config = {"person": {"attrs": ["PID"], "counts": counts}}
        source = self._generate_dataset(config)
        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("val", 0.2), ("test", 0.3)]
                splitter.Split(source, task, splits, query)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.6), ("val", 0.2), ("test", 0.3)]
                splitter.Split(source, task, splits, query)

            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
                actual = splitter.Split(source, task, splits, -query)

        with self.subTest("duplicated subset name"), self.assertRaisesRegex(Exception, "duplicated"):
            splits = [("train", 0.5), ("train", 0.2), ("test", 0.3)]
            splitter.Split(source, task, splits, query)

        with self.subTest("wrong subset name"), self.assertRaisesRegex(Exception, "Subset name"):
            splits = [("_train", 0.5), ("val", 0.2), ("test", 0.3)]
            splitter.Split(source, task, splits, query)

        with self.subTest("wrong attribute name for person id"):
            splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
            actual = splitter.Split(source, task, splits, query)

            with self.assertRaisesRegex(Exception, "Unknown subset"):
                actual.get_subset("test")

    def _generate_detection_segmentation_dataset(self, **kwargs):
        annotation_type = kwargs.get("annotation_type")
        with_attr = kwargs.get("with_attr", False)
        nimages = kwargs.get("nimages", 10)

        label_cat = LabelCategories()
        for i in range(6):
            label = "label%d" % (i + 1)
            if with_attr is True:
                attributes = {"attr0", "attr%d" % (i + 1)}
            else:
                attributes = {}
            label_cat.add(label, attributes=attributes)
        categories = {AnnotationType.label: label_cat}

        iterable = []
        attr_val = 0
        totals = np.zeros(3)
        objects = [(1, 5, 2), (3, 4, 1), (2, 3, 4), (1, 1, 1), (2, 4, 2)]
        for img_id in range(nimages):
            cnts = objects[img_id % len(objects)]
            totals += cnts
            annotations = []
            for label_id, count in enumerate(cnts):
                attributes = {}
                if with_attr:
                    attr_val += 1
                    attributes["attr0"] = attr_val % 3
                    attributes["attr%d" % (label_id + 1)] = attr_val % 2
                for ann_id in range(count):
                    annotation_type(
                        annotations,
                        label_id=label_id,
                        ann_id=ann_id,
                        attributes=attributes,
                    )
            item = DatasetItem(
                img_id,
                subset=self._get_subset(img_id),
                annotations=annotations,
                attributes={"id": img_id},
            )
            iterable.append(item)

        dataset = Dataset.from_iterable(iterable, categories)
        return dataset, totals

    @staticmethod
    def _get_append_bbox(dataset_type):
        def append_bbox_coco(annotations, **kwargs):
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_bbox_voc(annotations, **kwargs):
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"] + 1,
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )  # obj
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"] + 3,
                    group=kwargs["ann_id"],
                )
            )  # part
            annotations.append(Label(kwargs["label_id"] + 3, attributes=kwargs["attributes"]))

        def append_bbox_yolo(annotations, **kwargs):
            annotations.append(Bbox(1, 1, 2, 2, label=kwargs["label_id"]))
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_bbox_cvat(annotations, **kwargs):
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                    z_order=kwargs["ann_id"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_bbox_labelme(annotations, **kwargs):
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_bbox_mot(annotations, **kwargs):
            annotations.append(
                Bbox(
                    1,
                    1,
                    2,
                    2,
                    label=kwargs["label_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_bbox_widerface(annotations, **kwargs):
            annotations.append(Bbox(1, 1, 2, 2, attributes=kwargs["attributes"]))
            annotations.append(Label(0, attributes=kwargs["attributes"]))

        functions = {
            "coco": append_bbox_coco,
            "voc": append_bbox_voc,
            "yolo": append_bbox_yolo,
            "cvat": append_bbox_cvat,
            "labelme": append_bbox_labelme,
            "mot": append_bbox_mot,
            "widerface": append_bbox_widerface,
        }

        return functions.get(dataset_type, append_bbox_cvat)

    @staticmethod
    def _get_append_mask(dataset_type):
        def append_mask_coco(annotations, **kwargs):
            annotations.append(
                Mask(
                    np.array([[0, 0, 0, 1, 0]]),
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_mask_voc(annotations, **kwargs):
            annotations.append(
                Mask(
                    np.array([[0, 0, 0, 1, 0]]),
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"] + 1,
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )  # obj
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))
            annotations.append(
                Mask(
                    np.array([[0, 0, 0, 1, 0]]),
                    label=kwargs["label_id"] + 3,
                    group=kwargs["ann_id"],
                )
            )  # part
            annotations.append(Label(kwargs["label_id"] + 3, attributes=kwargs["attributes"]))

        def append_mask_labelme(annotations, **kwargs):
            annotations.append(
                Mask(
                    np.array([[0, 0, 0, 1, 0]]),
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_mask_mot(annotations, **kwargs):
            annotations.append(
                Mask(
                    np.array([[0, 0, 0, 1, 0]]),
                    label=kwargs["label_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        functions = {
            "coco": append_mask_coco,
            "voc": append_mask_voc,
            "labelme": append_mask_labelme,
            "mot": append_mask_mot,
        }

        return functions.get(dataset_type, append_mask_coco)

    @staticmethod
    def _get_append_polygon(dataset_type):
        def append_polygon_coco(annotations, **kwargs):
            annotations.append(
                Polygon(
                    [0, 0, 1, 0, 1, 2, 0, 2],
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_polygon_voc(annotations, **kwargs):
            annotations.append(
                Polygon(
                    [0, 0, 1, 0, 1, 2, 0, 2],
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"] + 1,
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )  # obj
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))
            annotations.append(
                Polygon(
                    [0, 0, 1, 0, 1, 2, 0, 2],
                    label=kwargs["label_id"] + 3,
                    group=kwargs["ann_id"],
                )
            )  # part
            annotations.append(Label(kwargs["label_id"] + 3, attributes=kwargs["attributes"]))

        def append_polygon_yolo(annotations, **kwargs):
            annotations.append(Bbox(1, 1, 2, 2, label=kwargs["label_id"]))
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_polygon_cvat(annotations, **kwargs):
            annotations.append(
                Polygon(
                    [0, 0, 1, 0, 1, 2, 0, 2],
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                    z_order=kwargs["ann_id"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        def append_polygon_labelme(annotations, **kwargs):
            annotations.append(
                Polygon(
                    [0, 0, 1, 0, 1, 2, 0, 2],
                    label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(Label(kwargs["label_id"], attributes=kwargs["attributes"]))

        functions = {
            "coco": append_polygon_coco,
            "voc": append_polygon_voc,
            "yolo": append_polygon_yolo,
            "cvat": append_polygon_cvat,
            "labelme": append_polygon_labelme,
        }

        return functions.get(dataset_type, append_polygon_coco)

    def test_split_for_detection(self):
        dtypes = ["coco", "voc", "yolo", "cvat", "labelme", "mot", "widerface"]
        task = splitter.SplitTask.detection.name
        params = []
        for dtype in dtypes:
            for with_attr in [False, True]:
                params.append((dtype, with_attr, 10, 5, 3, 2))
                params.append((dtype, with_attr, 10, 7, 0, 3))

        for dtype, with_attr, nimages, train, val, test in params:
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_bbox(dtype),
                with_attr=with_attr,
                nimages=nimages,
            )
            total = np.sum([train, val, test])
            splits = [
                ("train", train / total),
                ("val", val / total),
                ("test", test / total),
            ]
            with self.subTest(
                dtype=dtype,
                with_attr=with_attr,
                nimage=nimages,
                train=train,
                val=val,
                test=test,
                task=task,
            ):
                actual = splitter.Split(source, task, splits, seed=100)

                self.assertEqual(train, len(actual.get_subset("train")))
                self.assertEqual(val, len(actual.get_subset("val")))
                self.assertEqual(test, len(actual.get_subset("test")))

        # random seed test
        source, _ = self._generate_detection_segmentation_dataset(
            annotation_type=self._get_append_bbox("cvat"),
            with_attr=True,
            nimages=10,
        )

        splits = [("train", 0.5), ("test", 0.5)]
        r1 = splitter.Split(source, task, splits, seed=1234)
        r2 = splitter.Split(source, task, splits, seed=1234)
        r3 = splitter.Split(source, task, splits, seed=4321)
        self.assertEqual(list(r1.get_subset("test")), list(r2.get_subset("test")))
        self.assertNotEqual(list(r1.get_subset("test")), list(r3.get_subset("test")))

    def test_split_for_detection_with_unlabeled(self):
        source, _ = self._generate_detection_segmentation_dataset(
            annotation_type=self._get_append_bbox("cvat"),
            with_attr=True,
            nimages=10,
        )
        for i in range(10):
            source.put(DatasetItem(i + 10, annotations={}))

        splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
        task = splitter.SplitTask.detection.name
        actual = splitter.Split(source, task, splits, seed=100)
        self.assertEqual(10, len(actual.get_subset("train")))
        self.assertEqual(4, len(actual.get_subset("val")))
        self.assertEqual(6, len(actual.get_subset("test")))

    def test_split_for_detection_gives_error(self):
        source, _ = self._generate_detection_segmentation_dataset(
            annotation_type=self._get_append_bbox("cvat"),
            with_attr=True,
            nimages=5,
        )
        task = splitter.SplitTask.detection.name

        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("test", 1.5)]
                splitter.Split(source, task, splits)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                splitter.Split(source, task, splits)

        with self.subTest("duplicated subset name"), self.assertRaisesRegex(Exception, "duplicated"):
            splits = [("train", 0.5), ("train", 0.2), ("test", 0.3)]
            splitter.Split(source, task, splits)

    def test_no_subset_name_and_count_restriction(self):
        splits = [
            ("_train", 0.5),
            ("valid", 0.1),
            ("valid2", 0.1),
            ("test*", 0.2),
            ("test2", 0.1),
        ]

        with self.subTest("classification"):
            config = {"label1": {"attrs": None, "counts": 10}}
            task = splitter.SplitTask.classification.name
            source = self._generate_dataset(config)
            actual = splitter.Split(source, task, splits, seed=100)
            self.assertEqual(5, len(actual.get_subset("_train")))
            self.assertEqual(1, len(actual.get_subset("valid")))
            self.assertEqual(1, len(actual.get_subset("valid2")))
            self.assertEqual(2, len(actual.get_subset("test*")))
            self.assertEqual(1, len(actual.get_subset("test2")))

        with self.subTest("detection"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_bbox("cvat"),
                with_attr=True,
                nimages=10,
            )
            task = splitter.SplitTask.detection.name
            actual = splitter.Split(source, task, splits, seed=21)
            self.assertEqual(4, len(actual.get_subset("_train")))
            self.assertEqual(1, len(actual.get_subset("valid")))
            self.assertEqual(2, len(actual.get_subset("valid2")))
            self.assertEqual(2, len(actual.get_subset("test*")))
            self.assertEqual(1, len(actual.get_subset("test2")))

        with self.subTest("segmentation"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_mask("coco"),
                with_attr=True,
                nimages=10,
            )
            task = splitter.SplitTask.detection.name
            actual = splitter.Split(source, task, splits, seed=100)
            self.assertEqual(5, len(actual.get_subset("_train")))
            self.assertEqual(1, len(actual.get_subset("valid")))
            self.assertEqual(1, len(actual.get_subset("valid2")))
            self.assertEqual(2, len(actual.get_subset("test*")))
            self.assertEqual(1, len(actual.get_subset("test2")))

            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_polygon("coco"),
                with_attr=True,
                nimages=10,
            )
            actual = splitter.Split(source, task, splits, seed=100)
            self.assertEqual(5, len(actual.get_subset("_train")))
            self.assertEqual(1, len(actual.get_subset("valid")))
            self.assertEqual(1, len(actual.get_subset("valid2")))
            self.assertEqual(2, len(actual.get_subset("test*")))
            self.assertEqual(1, len(actual.get_subset("test2")))

    def test_split_for_segmentation(self):
        with self.subTest("mask annotation"):
            dtypes = ["coco", "voc", "labelme", "mot"]
            task = splitter.SplitTask.segmentation.name
            params = []
            for dtype in dtypes:
                for with_attr in [False, True]:
                    params.append((dtype, with_attr, 10, 5, 3, 2))
                    params.append((dtype, with_attr, 10, 7, 0, 3))

            for dtype, with_attr, nimages, train, val, test in params:
                source, _ = self._generate_detection_segmentation_dataset(
                    annotation_type=self._get_append_mask(dtype),
                    with_attr=with_attr,
                    nimages=nimages,
                )
                total = np.sum([train, val, test])
                splits = [
                    ("train", train / total),
                    ("val", val / total),
                    ("test", test / total),
                ]
                with self.subTest(
                    dtype=dtype,
                    with_attr=with_attr,
                    nimage=nimages,
                    train=train,
                    val=val,
                    test=test,
                    task=task,
                ):
                    actual = splitter.Split(source, task, splits, seed=100)

                    self.assertEqual(train, len(actual.get_subset("train")))
                    self.assertEqual(val, len(actual.get_subset("val")))
                    self.assertEqual(test, len(actual.get_subset("test")))

            # random seed test
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_mask("coco"),
                with_attr=True,
                nimages=10,
            )

            splits = [("train", 0.5), ("test", 0.5)]
            r1 = splitter.Split(source, task, splits, seed=1234)
            r2 = splitter.Split(source, task, splits, seed=1234)
            r3 = splitter.Split(source, task, splits, seed=4321)
            self.assertEqual(list(r1.get_subset("test")), list(r2.get_subset("test")))
            self.assertNotEqual(list(r1.get_subset("test")), list(r3.get_subset("test")))

        with self.subTest("polygon annotation"):
            dtypes = ["coco", "voc", "labelme", "yolo", "cvat"]
            task = splitter.SplitTask.segmentation.name
            params = []
            for dtype in dtypes:
                for with_attr in [False, True]:
                    params.append((dtype, with_attr, 10, 5, 3, 2))
                    params.append((dtype, with_attr, 10, 7, 0, 3))

            expected = []
            for dtype, with_attr, nimages, train, val, test in params:
                source, _ = self._generate_detection_segmentation_dataset(
                    annotation_type=self._get_append_polygon(dtype),
                    with_attr=with_attr,
                    nimages=nimages,
                )
                total = np.sum([train, val, test])
                splits = [
                    ("train", train / total),
                    ("val", val / total),
                    ("test", test / total),
                ]
                with self.subTest(
                    dtype=dtype,
                    with_attr=with_attr,
                    nimage=nimages,
                    train=train,
                    val=val,
                    test=test,
                    task=task,
                ):
                    actual = splitter.Split(source, task, splits, seed=21)

                    expected.append(
                        [
                            dtype,
                            with_attr,
                            len(actual.get_subset("train")),
                            len(actual.get_subset("val")),
                            len(actual.get_subset("test")),
                        ]
                    )

                    self.assertEqual(train, len(actual.get_subset("train")))
                    self.assertEqual(val, len(actual.get_subset("val")))
                    self.assertEqual(test, len(actual.get_subset("test")))

            # random seed test
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_polygon("coco"),
                with_attr=True,
                nimages=10,
            )

            splits = [("train", 0.5), ("test", 0.5)]
            r1 = splitter.Split(source, task, splits, seed=1234)
            r2 = splitter.Split(source, task, splits, seed=1234)
            r3 = splitter.Split(source, task, splits, seed=4321)
            self.assertEqual(list(r1.get_subset("test")), list(r2.get_subset("test")))
            self.assertNotEqual(list(r1.get_subset("test")), list(r3.get_subset("test")))

    def test_split_for_segmentation_with_unlabeled(self):
        with self.subTest("mask annotation"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_mask("coco"),
                with_attr=True,
                nimages=10,
            )
            for i in range(10):
                source.put(DatasetItem(i + 10, annotations={}))

            splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
            task = splitter.SplitTask.segmentation.name
            actual = splitter.Split(source, task, splits, seed=100)
            self.assertEqual(10, len(actual.get_subset("train")))
            self.assertEqual(4, len(actual.get_subset("val")))
            self.assertEqual(6, len(actual.get_subset("test")))

        with self.subTest("polygon annotation"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_polygon("coco"),
                with_attr=True,
                nimages=10,
            )
            for i in range(10):
                source.put(DatasetItem(i + 10, annotations={}))

            splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
            task = splitter.SplitTask.segmentation.name
            actual = splitter.Split(source, task, splits, seed=100)
            self.assertEqual(10, len(actual.get_subset("train")))
            self.assertEqual(4, len(actual.get_subset("val")))
            self.assertEqual(6, len(actual.get_subset("test")))

    def test_split_for_segmentation_gives_error(self):
        with self.subTest("mask annotation"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_mask("coco"),
                with_attr=True,
                nimages=5,
            )
            task = splitter.SplitTask.segmentation.name

            with self.subTest("wrong ratio"):
                with self.assertRaisesRegex(Exception, "in the range"):
                    splits = [("train", -0.5), ("test", 1.5)]
                    splitter.Split(source, task, splits)

                with self.assertRaisesRegex(Exception, "Sum of ratios"):
                    splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                    splitter.Split(source, task, splits)

            with self.subTest("duplicated subset name"), self.assertRaisesRegex(Exception, "duplicated"):
                splits = [("train", 0.5), ("train", 0.2), ("test", 0.3)]
                splitter.Split(source, task, splits)

        with self.subTest("polygon annotation"):
            source, _ = self._generate_detection_segmentation_dataset(
                annotation_type=self._get_append_polygon("coco"),
                with_attr=True,
                nimages=5,
            )
            task = splitter.SplitTask.segmentation.name

            with self.subTest("wrong ratio"):
                with self.assertRaisesRegex(Exception, "in the range"):
                    splits = [("train", -0.5), ("test", 1.5)]
                    splitter.Split(source, task, splits)

                with self.assertRaisesRegex(Exception, "Sum of ratios"):
                    splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                    splitter.Split(source, task, splits)

            with self.subTest("duplicated subset name"), self.assertRaisesRegex(Exception, "duplicated"):
                splits = [("train", 0.5), ("train", 0.2), ("test", 0.3)]
                splitter.Split(source, task, splits)
