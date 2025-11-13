# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from unittest import TestCase

from datumaro.components.format_detection import (
    DetectedFormat,
    FormatDetectionConfidence,
    FormatDetectionUnsupported,
    FormatRequirementsUnmet,
    RejectionReason,
    apply_format_detector,
    detect_dataset_format,
)
from tests.utils.test_utils import TestDir


class FormatDetectionTest(TestCase):
    def setUp(self) -> None:
        test_dir_context = TestDir()
        self._dataset_root = test_dir_context.__enter__()
        self.addCleanup(test_dir_context.__exit__, None, None, None)


class ApplyFormatDetectorTest(FormatDetectionTest):
    def test_default_result(self):
        result = apply_format_detector(self._dataset_root, lambda c: None)
        self.assertEqual(result, FormatDetectionConfidence.MEDIUM)

    def test_custom_confidence(self):
        result = apply_format_detector(self._dataset_root, lambda c: FormatDetectionConfidence.LOW)
        self.assertEqual(result, FormatDetectionConfidence.LOW)

    def test_root_path(self):
        provided_root = None

        def detect(context):
            nonlocal provided_root
            provided_root = context.root_path

        apply_format_detector(self._dataset_root, detect)
        self.assertEqual(provided_root, self._dataset_root)

    def test_fail(self):
        def detect(context):
            context.fail("abcde")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ("abcde",))

    def test_require_file_success(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w"):
            pass

        selected_file = None

        def detect(context):
            nonlocal selected_file
            selected_file = context.require_file("**/[fg]oo*.t?t")

        apply_format_detector(self._dataset_root, detect)

        self.assertEqual(selected_file, "foobar.txt")

    def test_require_file_failure(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w"):
            pass

        def detect(context):
            context.require_file("*/*")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn("*/*", result.exception.failed_alternatives[0])

    def test_require_file_exclude_fname_one(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w"):
            pass

        def detect(context):
            context.require_file("foobar.*", exclude_fnames="*.txt")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn("foobar.*", result.exception.failed_alternatives[0])
        self.assertIn("*.txt", result.exception.failed_alternatives[0])

    def test_require_file_exclude_fname_many(self):
        for ext in ("txt", "lst"):
            with open(osp.join(self._dataset_root, f"foobar.{ext}"), "w"):
                pass

        def detect(context):
            context.require_file("foobar.*", exclude_fnames=("*.txt", "*.lst"))

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn("foobar.*", result.exception.failed_alternatives[0])
        self.assertIn("*.txt", result.exception.failed_alternatives[0])
        self.assertIn("*.lst", result.exception.failed_alternatives[0])

    def test_require_files_success(self):
        for stem in "cba":
            with open(osp.join(self._dataset_root, f"{stem}.txt"), "w"):
                pass

        selected_files = []

        def detect(context):
            selected_files.extend(context.require_files("*.txt", exclude_fnames="b.txt"))

        apply_format_detector(self._dataset_root, detect)

        self.assertEqual(selected_files, ["a.txt", "c.txt"])

    def test_require_files_failure(self):
        def detect(context):
            context.require_files("*.txt")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn("*.txt", result.exception.failed_alternatives[0])

    def test_probe_text_file_success(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w") as f:
            print("123", file=f)

        def detect(context):
            with context.probe_text_file("foobar.txt", "abcde") as f:
                if next(f) != "123\n":
                    raise Exception

        apply_format_detector(self._dataset_root, detect)

    def test_probe_text_file_failure_bad_file(self):
        def detect(context):
            with context.probe_text_file("foobar.txt", "abcde"):
                pass

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ("foobar.txt: abcde",))

    def test_probe_text_file_failure_exception(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w"):
            pass

        def detect(context):
            with context.probe_text_file("foobar.txt", "abcde"):
                raise Exception

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ("foobar.txt: abcde",))

    def test_probe_text_file_nested_req(self):
        with open(osp.join(self._dataset_root, "foobar.txt"), "w"):
            pass

        def detect(context):
            with context.probe_text_file("foobar.txt", "abcde"):
                context.fail("abcde")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ("abcde",))

    def test_require_any_success(self):
        alternatives_executed = set()

        def detect(context):
            nonlocal alternatives_executed
            with context.require_any():
                with context.alternative():
                    alternatives_executed.add(1)
                    context.fail("bad alternative 1")
                with context.alternative():
                    alternatives_executed.add(2)
                    # good alternative 2
                with context.alternative():
                    alternatives_executed.add(3)
                    context.fail("bad alternative 3")

        apply_format_detector(self._dataset_root, detect)

        self.assertEqual(alternatives_executed, {1, 2, 3})

    def test_require_any_failure(self):
        def detect(context):
            with context.require_any():
                with context.alternative():
                    context.fail("bad alternative 1")
                with context.alternative():
                    context.fail("bad alternative 2")

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ("bad alternative 1", "bad alternative 2"))

    def test_raise_unsupported(self):
        def detect(context):
            context.raise_unsupported()

        with self.assertRaises(FormatDetectionUnsupported):
            apply_format_detector(self._dataset_root, detect)


class DetectDatasetFormat(FormatDetectionTest):
    def test_no_input_formats(self):
        detected_datasets = detect_dataset_format((), self._dataset_root)
        self.assertEqual(detected_datasets, [])

    def test_general_case(self):
        formats = [
            # aaa should be rejected after bbb is checked
            ("aaa", lambda context: FormatDetectionConfidence.LOW),
            ("bbb", lambda context: None),
            ("ccc", lambda context: context.fail("test unmet requirement")),
            # ddd should be rejected immediately
            ("ddd", lambda context: FormatDetectionConfidence.LOW),
            ("eee", lambda context: None),
            ("fff", lambda context: context.raise_unsupported()),
        ]

        rejected_formats = {}

        def rejection_callback(format, reason, message):
            rejected_formats[format] = (reason, message)

        detected_datasets = detect_dataset_format(formats, self._dataset_root, rejection_callback=rejection_callback)

        detected_dataset_names = [detected_dataset.name for detected_dataset in detected_datasets]

        self.assertEqual(set(detected_dataset_names), {"bbb", "eee"})

        self.assertEqual(rejected_formats.keys(), {"aaa", "ccc", "ddd", "fff"})

        for name in ("aaa", "ddd"):
            self.assertEqual(rejected_formats[name][0], RejectionReason.insufficient_confidence)

        self.assertEqual(rejected_formats["ccc"][0], RejectionReason.unmet_requirements)
        self.assertIn("test unmet requirement", rejected_formats["ccc"][1])

        self.assertEqual(rejected_formats["fff"][0], RejectionReason.detection_unsupported)

    def test_no_callback(self):
        formats = [
            ("bbb", lambda context: None),
            ("ccc", lambda context: context.fail("test unmet requirement")),
        ]

        detected_datasets = detect_dataset_format(formats, self._dataset_root)

        detected_dataset_names = [detected_dataset.name for detected_dataset in detected_datasets]

        self.assertEqual(detected_dataset_names, ["bbb"])


class DetectedFormatTest:
    def test_compare(self):
        # Comparison should be decided by the confidence
        decr_confs = [
            FormatDetectionConfidence.MEDIUM,
            FormatDetectionConfidence.LOW,
            FormatDetectionConfidence.EXTREME_LOW,
            FormatDetectionConfidence.NONE,
        ]
        incr_confs = decr_confs[::-1]

        detects = [DetectedFormat(conf, f"{[0] * idx}") for idx, conf in enumerate(decr_confs)]

        assert [detect.confidence for detect in detects] != incr_confs

        detects.sort()

        assert [detect.confidence for detect in detects] == incr_confs

    def test_eq(self):
        # Equivalance should be decided by the format name
        assert DetectedFormat(FormatDetectionConfidence.LOW, "format") == DetectedFormat(
            FormatDetectionConfidence.MEDIUM, "format"
        )
        assert DetectedFormat(FormatDetectionConfidence.LOW, "format1") != DetectedFormat(
            FormatDetectionConfidence.MEDIUM, "format2"
        )
