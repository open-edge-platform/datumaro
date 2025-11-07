# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging
import os
import os.path as osp
import platform
from contextlib import suppress
from typing import Iterator
from unittest import TestCase, mock

import pytest

from datumaro.util import is_method_redefined
from datumaro.util.definitions import get_datumaro_cache_dir
from datumaro.util.multi_procs_util import consumer_generator
from datumaro.util.os_util import walk
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
