# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging
import os
import os.path as osp
import platform
import shutil
import unittest
from contextlib import suppress
from typing import Iterator
from unittest import TestCase, mock

import pytest

from datumaro.util import is_method_redefined
from datumaro.util.definitions import get_datumaro_cache_dir
from datumaro.util.multi_procs_util import consumer_generator
from datumaro.util.os_util import ensure_within_base, join_within_base, walk
from datumaro.util.scope import Scope, on_error_do, on_exit_do, scoped
from tests.utils.test_utils import TestDir


class TestException(Exception):
    pass


class ScopeTest(TestCase):
    def test_calls_only_exit_callback_on_exit(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        with Scope() as scope:
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)

        error_cb.assert_not_called()
        exit_cb.assert_called_once()

    def test_calls_both_callbacks_on_error(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        with self.assertRaises(TestException), Scope() as scope:
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)
            raise TestException

        error_cb.assert_called_once()
        exit_cb.assert_called_once()

    def test_adds_cm(self):
        cm = mock.Mock()
        cm.__enter__ = mock.MagicMock(return_value=42)
        cm.__exit__ = mock.MagicMock()

        with Scope() as scope:
            retval = scope.add(cm)

        cm.__enter__.assert_called_once()
        cm.__exit__.assert_called_once()
        self.assertEqual(42, retval)

    def test_calls_cm_on_error(self):
        cm = mock.Mock()
        cm.__enter__ = mock.MagicMock()
        cm.__exit__ = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.add(cm)
            raise TestException

        cm.__enter__.assert_called_once()
        cm.__exit__.assert_called_once()

    def test_decorator_calls_on_error(self):
        cb = mock.MagicMock()

        @scoped("scope")
        def foo(scope=None):
            scope.on_error_do(cb)
            raise TestException

        with suppress(TestException):
            foo()

        cb.assert_called_once()

    def test_decorator_does_not_call_on_no_error(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        @scoped("scope")
        def foo(scope=None):
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)

        foo()

        error_cb.assert_not_called()
        exit_cb.assert_called_once()

    def test_decorator_supports_implicit_form(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        @scoped
        def foo():
            on_error_do(error_cb)
            on_exit_do(exit_cb)
            raise TestException

        with suppress(TestException):
            foo()

        error_cb.assert_called_once()
        exit_cb.assert_called_once()

    def test_can_fowrard_args(self):
        cb = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.on_error_do(cb, 5, ignore_errors=True, kwargs={"a2": 2})
            raise TestException

        cb.assert_called_once_with(5, a2=2)

    def test_decorator_can_return_on_success_in_implicit_form(self):
        @scoped
        def f():
            return 42

        retval = f()

        self.assertEqual(42, retval)

    def test_decorator_can_return_on_success_in_explicit_form(self):
        @scoped("scope")
        def f(scope=None):
            return 42

        retval = f()

        self.assertEqual(42, retval)


class TestOsUtils(TestCase):
    def test_can_walk_with_maxdepth(self):
        with TestDir() as rootdir:
            os.makedirs(osp.join(rootdir, "1", "2", "3", "4"))

            visited = set(d for d, _, _ in walk(rootdir, max_depth=2))
            self.assertEqual(
                {
                    osp.join(rootdir),
                    osp.join(rootdir, "1"),
                    osp.join(rootdir, "1", "2"),
                },
                visited,
            )


class JoinWithinBaseTest(TestCase):
    def setUp(self):
        self.base_dir_cm = TestDir()
        self.base_dir = self.base_dir_cm.__enter__()
        self.addCleanup(self.base_dir_cm.__exit__, None, None, None)

    def test_can_join_descendant_paths(self):
        self.assertEqual(
            osp.join(self.base_dir, "train", "img001.jpg"),
            join_within_base(self.base_dir, "train", "img001.jpg"),
        )
        self.assertEqual(osp.join(self.base_dir, "img001.jpg"), join_within_base(self.base_dir, "img001.jpg"))

    def test_rejects_dotdot_component(self):
        with self.assertRaises(ValueError):
            join_within_base(self.base_dir, "..", "escape.txt")

        with self.assertRaises(ValueError):
            join_within_base(self.base_dir, "a", "..", "..", "escape.txt")

    def test_rejects_absolute_part(self):
        with self.assertRaises(ValueError):
            join_within_base(self.base_dir, "/etc/passwd")

    @unittest.skipIf(platform.system() != "Windows", "drive letters are only meaningful on Windows")
    def test_rejects_different_drive_on_windows_style_path(self):
        with self.assertRaises(ValueError):
            join_within_base("C:\\base", "D:\\other\\file.txt")

    def test_rejects_prefix_collision_sibling(self):
        # "<base_dir>_sibling" starts with the same characters as base_dir but
        # is not actually contained within it: a naive string-prefix check
        # would wrongly accept this, since osp.join() replaces base_dir
        # entirely once it hits this absolute part.
        sibling = self.base_dir + "_sibling"
        with self.assertRaises(ValueError):
            join_within_base(self.base_dir, sibling, "file.txt")

    @unittest.skipIf(platform.system() == "Windows", "symlinks require elevated privileges on Windows")
    def test_rejects_symlink_escaping_base_dir(self):
        outside_dir = osp.join(self.base_dir, "..", "outside_" + osp.basename(self.base_dir))
        outside_dir = osp.abspath(outside_dir)
        os.makedirs(outside_dir)
        self.addCleanup(shutil.rmtree, outside_dir, ignore_errors=True)

        link_path = osp.join(self.base_dir, "escape_link")
        os.symlink(outside_dir, link_path)

        with self.assertRaises(ValueError):
            join_within_base(self.base_dir, "escape_link", "file.txt")

    @unittest.skipIf(platform.system() == "Windows", "symlinks require elevated privileges on Windows")
    def test_accepts_symlink_staying_within_base_dir(self):
        real_subdir = osp.join(self.base_dir, "real_subdir")
        os.makedirs(real_subdir)

        link_path = osp.join(self.base_dir, "linked_subdir")
        os.symlink(real_subdir, link_path)

        result = join_within_base(self.base_dir, "linked_subdir", "file.txt")
        self.assertEqual(osp.join(self.base_dir, "linked_subdir", "file.txt"), result)

    def test_rejects_win32_trimmed_dotdot_component(self):
        for unsafe_part in (".. ", "...", ".. ."):
            with self.assertRaises(ValueError):
                join_within_base(self.base_dir, unsafe_part, "file.txt")


class EnsureWithinBaseTest(TestCase):
    def setUp(self):
        self.base_dir_cm = TestDir()
        self.base_dir = self.base_dir_cm.__enter__()
        self.addCleanup(self.base_dir_cm.__exit__, None, None, None)

    def test_accepts_path_within_base(self):
        path = osp.join(self.base_dir, "train", "img001.jpg")
        self.assertEqual(path, ensure_within_base(path, self.base_dir))

    def test_rejects_path_outside_base(self):
        with self.assertRaises(ValueError):
            ensure_within_base(osp.join(self.base_dir, "..", "escape.txt"), self.base_dir)

    def test_rejects_absolute_path_outside_base(self):
        with self.assertRaises(ValueError):
            ensure_within_base("/etc/passwd", self.base_dir)

    def test_rejects_prefix_collision_sibling(self):
        sibling = self.base_dir + "_sibling"
        with self.assertRaises(ValueError):
            ensure_within_base(osp.join(sibling, "file.txt"), self.base_dir)


class TestMemberRedefined(TestCase):
    class Base:
        def method(self):
            pass

    def test_can_detect_no_changes_in_derived_class(self):
        class Derived(self.Base):
            pass

        self.assertFalse(is_method_redefined("method", self.Base, Derived))

    def test_can_detect_no_changes_in_derived_instance(self):
        class Derived(self.Base):
            pass

        self.assertFalse(is_method_redefined("method", self.Base, Derived()))

    def test_can_detect_changes_in_derived_class(self):
        class Derived(self.Base):
            def method(self):
                pass

        self.assertTrue(is_method_redefined("method", self.Base, Derived))

    def test_can_detect_changes_in_derived_instance(self):
        class Derived(self.Base):
            def method(self):
                pass

        self.assertTrue(is_method_redefined("method", self.Base, Derived()))

    def test_can_detect_changes_in_patched_instance(self):
        obj = self.Base()
        with mock.patch.object(obj, "method"):
            self.assertTrue(is_method_redefined("method", self.Base, obj))


class DefinitionsTest:
    @pytest.fixture
    def fxt_writable_path(self, test_dir: str) -> str:
        dst = os.path.join(test_dir, "writable")
        os.makedirs(dst)
        os.chmod(dst, 0o755)
        return dst

    @pytest.fixture
    def fxt_non_writable_path(self, test_dir: str) -> str:
        dst = os.path.join(test_dir, "non-writable")
        os.makedirs(dst)
        os.chmod(dst, 0o000)
        yield dst
        os.chmod(dst, 0o755)

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="os.chmod() cannot be used for Windows.",
    )
    def test_get_datumaro_cache_dir(
        self, fxt_writable_path: str, fxt_non_writable_path: str, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.ERROR):
            get_datumaro_cache_dir(fxt_writable_path)
            assert len(caplog.records) == 0
        with caplog.at_level(logging.ERROR):
            get_datumaro_cache_dir(fxt_non_writable_path)
            assert len(caplog.records) == 1


@pytest.mark.new
class MultiProcUtilTest:
    @pytest.fixture
    def fxt_producer_generator(self):
        class TestObject:
            def __init__(self, value: int) -> None:
                self.value = value

        def test_func() -> Iterator[TestObject]:
            for i in range(1000):
                yield TestObject(i)

        return test_func

    def test_succeed(self, fxt_producer_generator):
        with consumer_generator(producer_generator=fxt_producer_generator()) as f:
            for expect, actual in enumerate(f):
                assert expect == actual.value

    def test_raise_exception_in_main_thread(self, fxt_producer_generator, caplog: pytest.LogCaptureFixture):
        try:
            with consumer_generator(
                producer_generator=fxt_producer_generator(),
                enqueue_timeout=0.05,
                join_timeout=None,
            ) as f:
                for expect, actual in enumerate(f):
                    assert expect == actual.value
                    raise Exception
        except Exception:
            assert any(
                record.message == "Item to enqueue is left. However, the main process is terminated."
                for record in caplog.records
            )
