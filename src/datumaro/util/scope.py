# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
from contextlib import ExitStack, contextmanager
from functools import partial, wraps
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple, TypeVar

from attrs import frozen

from datumaro.util import optional_arg_decorator

T = TypeVar("T")


class Scope:
    """
    A context manager that allows to register error and exit callbacks.
    """

    _thread_locals = threading.local()

    @frozen
    class _ExitHandler:
        callback: Callable[[], Any]
        ignore_errors: bool = True

        def __exit__(self, exc_type, exc_value, exc_traceback):
            try:
                self.callback()
            except Exception:
                if not self.ignore_errors:
                    raise

    @frozen
    class _ErrorHandler(_ExitHandler):
        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type:
                return super().__exit__(exc_type=exc_type, exc_value=exc_value, exc_traceback=exc_traceback)

    def __init__(self):
        self._stack = ExitStack()
        self.enabled = True

    def on_error_do(
        self,
        callback: Callable,
        *args,
        kwargs: Optional[Dict[str, Any]] = None,
        ignore_errors: bool = False,
    ):
        """
        Registers a function to be called on scope exit because of an error.

        If ignore_errors is True, the errors from this function call
        will be ignored.
        """

        self._register_callback(
            self._ErrorHandler,
            ignore_errors=ignore_errors,
            callback=callback,
            args=args,
            kwargs=kwargs,
        )

    def on_exit_do(
        self,
        callback: Callable,
        *args,
        kwargs: Optional[Dict[str, Any]] = None,
        ignore_errors: bool = False,
    ):
        """
        Registers a function to be called on scope exit.
        """

        self._register_callback(
            self._ExitHandler,
            ignore_errors=ignore_errors,
            callback=callback,
            args=args,
            kwargs=kwargs,
        )

    def _register_callback(
        self,
        handler_type,
        callback: Callable,
        args: Tuple[Any] = None,
        kwargs: Dict[str, Any] = None,
        ignore_errors: bool = False,
    ):
        if args or kwargs:
            callback = partial(callback, *args, **(kwargs or {}))

        self._stack.push(handler_type(callback, ignore_errors=ignore_errors))

    def add(self, cm: ContextManager[T]) -> T:
        """
        Enters a context manager and adds it to the exit stack.

        Returns: cm.__enter__() result
        """

        return self._stack.enter_context(cm)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self) -> Scope:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.enabled:
            return

        self._stack.__exit__(exc_type, exc_value, exc_traceback)
        self._stack.pop_all()  # prevent issues on repetitive calls

    @classmethod
    def current(cls) -> Scope:
        return cls._thread_locals.current

    @contextmanager
    def as_current(self):
        previous = getattr(self._thread_locals, "current", None)
        self._thread_locals.current = self
        try:
            yield
        finally:
            self._thread_locals.current = previous


@optional_arg_decorator
def scoped(func, arg_name=None):
    """
    A function decorator, which allows to do actions with the current scope,
    such as registering error and exit callbacks and context managers.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        with Scope() as scope:
            if arg_name is None:
                with scope.as_current():
                    ret_val = func(*args, **kwargs)
            else:
                kwargs[arg_name] = scope
                ret_val = func(*args, **kwargs)
            return ret_val

    return wrapped_func


# Shorthands for common cases
def on_error_do(callback, *args, ignore_errors=False, kwargs=None):
    return Scope.current().on_error_do(callback, *args, ignore_errors=ignore_errors, kwargs=kwargs)


def on_exit_do(callback, *args, ignore_errors=False, kwargs=None):
    return Scope.current().on_exit_do(callback, *args, ignore_errors=ignore_errors, kwargs=kwargs)


def scope_add(cm: ContextManager[T]) -> T:
    return Scope.current().add(cm)
