import contextlib
import io
import json
import os
import os.path as osp
import shutil
from typing import List
from unittest.case import TestCase

from datumaro.plugins.data_formats.ade20k2017 import Ade20k2017Importer
from datumaro.plugins.data_formats.ade20k2020 import Ade20k2020Importer
from datumaro.plugins.data_formats.camvid import CamvidImporter
from datumaro.plugins.data_formats.image_dir import ImageDirImporter
from datumaro.plugins.data_formats.lfw import LfwImporter
from datumaro.util.os_util import suppress_output

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run

ADE20K2017_DIR = get_test_asset_path("ade20k2017_dataset", "dataset")
ADE20K2020_DIR = get_test_asset_path("ade20k2020_dataset", "dataset")
LFW_DIR = get_test_asset_path("lfw_dataset")


class DetectFormatTest(TestCase):
    def _extract_detect_format_name(self, output_file: io.StringIO) -> List[str]:
        output = output_file.getvalue()
        if "Ambiguous dataset; detected the following formats:\n\n" in output:
            tokens = output.replace(
                "Ambiguous dataset; detected the following formats:\n\n", ""
            ).split("- ")[1:]
            return [token.replace("\n", "") for token in tokens]

        return [output.replace("Detected format: ", "").replace("\n", "")]

    def test_unambiguous(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self, "detect", ADE20K2017_DIR)

        output = self._extract_detect_format_name(output_file)

        self.assertEqual([Ade20k2017Importer.NAME], output)
        self.assertNotEqual([Ade20k2020Importer.NAME], output)

    def test_deep_nested_folders(self):
        with TestDir() as test_dir:
            output_file = io.StringIO()

            annotation_dir = osp.join(test_dir, "a", "b", "c", "annotations")
            os.makedirs(annotation_dir)
            shutil.copy(osp.join(LFW_DIR, "test", "annotations", "pairs.txt"), annotation_dir)

            with contextlib.redirect_stdout(output_file):
                run(self, "detect", test_dir, "--depth", "3")

            output = self._extract_detect_format_name(output_file)

            self.assertEqual([LfwImporter.NAME], output)

    def test_nested_folders(self):
        with TestDir() as test_dir:
            output_file = io.StringIO()

            annotation_dir = osp.join(test_dir, "a", "training/street")
            os.makedirs(annotation_dir)
            shutil.copy(osp.join(ADE20K2020_DIR, "training/street/1.json"), annotation_dir)

            with contextlib.redirect_stdout(output_file):
                run(self, "detect", test_dir)

            output = self._extract_detect_format_name(output_file)

            self.assertEqual([Ade20k2020Importer.NAME], output)

    def test_ambiguous(self):
        with TestDir() as test_dir:
            annotation_dir = osp.join(test_dir, "training/street")
            os.makedirs(annotation_dir)

            for asset in [
                osp.join(ADE20K2017_DIR, "training/street/1_atr.txt"),
                osp.join(ADE20K2020_DIR, "training/street/1.json"),
            ]:
                shutil.copy(asset, annotation_dir)

            output_file = io.StringIO()

            with contextlib.redirect_stdout(output_file):
                run(self, "detect", test_dir)

            output = self._extract_detect_format_name(output_file)

            self.assertEqual([Ade20k2017Importer.NAME, Ade20k2020Importer.NAME], output)

    # Ideally, there should be a test for the case where no formats match,
    # but currently that's impossible, because some low-confidence detectors
    # always match.

    def test_rejections(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self, "detect", "--show-rejections", ADE20K2017_DIR)

        output = output_file.getvalue()

        self.assertIn(Ade20k2017Importer.NAME, output)

        self.assertIn(CamvidImporter.NAME, output)

    def test_json_report(self):
        with suppress_output(), TestDir() as test_dir:
            report_path = osp.join(test_dir, "report.json")

            run(
                self,
                "detect",
                "--show-rejections",
                "--json-report",
                report_path,
                ADE20K2017_DIR,
            )

            with open(report_path, "rb") as report_file:
                report = json.load(report_file)

        self.assertIsInstance(report, dict)
        self.assertIn("detected_formats", report)
        self.assertEqual(["ade20k2017"], report["detected_formats"])

        self.assertIn("rejected_formats", report)

        self.assertIn("camvid", report["rejected_formats"])
        camvid_rejection = report["rejected_formats"]["camvid"]

        self.assertIn("reason", camvid_rejection)
        self.assertEqual(camvid_rejection["reason"], "unmet_requirements")
        self.assertIn("message", camvid_rejection)
        self.assertIsInstance(camvid_rejection["message"], str)
        self.assertTrue(CamvidImporter._ANNO_EXT in camvid_rejection["message"])
