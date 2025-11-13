# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import time
from typing import Iterable, Optional, Tuple, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


class ProgressReporter:
    """
    Only one set of methods must be called:
        - start - report_status - finish
        - iter
        - split

    This class is supposed to manage the state of children progress bars
    and release of their resources, if necessary.
    """

    @property
    def period(self) -> float:
        """
        Returns reporting period.

        For example, 0.1 would mean every 10%.
        """
        raise NotImplementedError

    @property
    def interval(self) -> float:
        """
        Returns reporting time interval in second.
        """
        raise NotImplementedError

    def start(self, total: int, *, desc: Optional[str] = None):
        """Initializes the progress bar"""
        raise NotImplementedError

    def report_status(self, progress: int):
        """Updates the progress bar"""
        raise NotImplementedError

    def finish(self):
        """Finishes the progress bar"""
        # pylint: disable=unnecessary-pass

    def iter(self, iterable: Iterable[T], *, total: Optional[int] = None, desc: Optional[str] = None) -> Iterable[T]:
        """
        Traverses the iterable and reports progress simultaneously.

        Starts and finishes the progress bar automatically.

        Args:
            iterable: An iterable to be traversed
            total: The expected number of iterations. If not provided, will
              try to use iterable.__len__.
            desc: The status message

        Returns:
            An iterable over elements of the input sequence
        """

        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)

        self.start(total, desc=desc)

        if total:
            display_step = math.ceil(total * self.period)

        s = time.time()
        for i, elem in enumerate(iterable):
            if not total or (display_step and i % display_step == 0) or time.time() - s > self.interval:
                s = time.time()
                self.report_status(i)

            yield elem

        self.finish()

    def split(self, count: int) -> Tuple[ProgressReporter, ...]:
        """
        Splits the progress bar into few independent parts.
        In case of 0 must return an empty tuple.

        This class is supposed to manage the state of children progress bars
        and release of their resources, if necessary.
        """
        raise NotImplementedError


class NullProgressReporter(ProgressReporter):
    @property
    def period(self) -> float:
        return 0

    @property
    def interval(self) -> float:
        return float("inf")

    def start(self, total: int, *, desc: Optional[str] = None):
        pass

    def report_status(self, progress: int):
        pass

    def iter(self, iterable: Iterable[T], *, total: Optional[int] = None, desc: Optional[str] = None) -> Iterable[T]:
        yield from iterable

    def split(self, count: int) -> Tuple[ProgressReporter]:
        return (self,) * count


class SimpleProgressReporter(ProgressReporter):
    def __init__(self, period: float = 0.1, interval: float = float("inf")):
        self._period = period
        self._interval = interval

    @property
    def period(self) -> float:
        return self._period

    @property
    def interval(self) -> float:
        return self._interval

    def start(self, total: int, *, desc: Optional[str] = None):
        self._total = total
        self._desc = desc

    def report_status(self, progress: int):
        status = str(self._desc) if self._desc else ""
        status += f" {progress:0{len(str(self._total))}d}/{self._total} ({progress / self._total * 100:6.2f}%)"
        print(status)

    def finish(self):
        self.report_status(self._total)

    def split(self, count: int):
        return (SimpleProgressReporter(self._period, self._interval),) * count


class TQDMProgressReporter(ProgressReporter):
    def __init__(self, period: float = 0.1, interval: float = 0.1, **options):
        self._period = period
        self._interval = interval
        self._options = options

    @property
    def period(self) -> float:
        return self._period

    @property
    def interval(self) -> float:
        return self._interval

    def start(self, total: int, *, desc: Optional[str] = None):
        options = self._options.copy()
        if desc is not None:
            options["desc"] = desc
        self._total = total
        self._pbar = tqdm(total=total, **options)
        self._cur = 0

    def report_status(self, progress: int):
        self._pbar.update(progress - self._cur)
        self._cur = progress

    def finish(self):
        if self._total is None:
            self._total = self._cur  # Total can be None
        self._pbar.update(self._total - self._cur)
        self._pbar.close()

    def split(self, count: int):
        return (TQDMProgressReporter(self._period, self._interval, **self._options),) * count
