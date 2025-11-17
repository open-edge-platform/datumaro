# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from contextlib import contextmanager
from enum import IntEnum
from queue import Full, Queue
from threading import Condition, Thread
from typing import Any, Generator, Iterator, Optional, TypeVar

__all__ = ["consumer_generator"]


class ProducerMessage(IntEnum):
    START = 0
    END = 1


Item = TypeVar("Item")


@contextmanager
def consumer_generator(
    producer_generator: Iterator[Item],
    queue_size: int = 100,
    enqueue_timeout: float = 5.0,
    join_timeout: Optional[float] = 10.0,
) -> Generator[Iterator[Item], None, None]:
    """Context manager that creates a generator to consume items produced by another generator.

    This context manager sets up a producer thread that generates items from the `producer_generator`
    and enqueues them to be consumed by the consumer generator, which is also created by this function.

    Parameters:
        producer_generator: A generator that produces items.
        queue_size: The maximum size of the shared queue between the producer and consumer.
        enqueue_timeout: The maximum time to wait for enqueuing an item to the queue if it's full.
        join_timeout: The maximum time to wait for the producer thread to finish when exiting the context.
            If None, wait until the producer thread terminates.

    Returns:
        Iterator: A context for iterating over the generated items.
    """
    queue = Queue(maxsize=queue_size)
    lock = Condition()
    is_terminated = False

    def _enqueue(item: Any, queue: Queue):
        while True:
            try:
                queue.put(item, block=True, timeout=enqueue_timeout)
                return
            except Full:
                with lock:
                    if is_terminated:
                        raise RuntimeError("Item to enqueue is left. However, the main process is terminated.")

    def _target(queue: Queue) -> None:
        try:
            _enqueue(ProducerMessage.START, queue)

            for item in producer_generator:
                _enqueue(item, queue)

            _enqueue(ProducerMessage.END, queue)
        except RuntimeError as e:
            log.error(e)
            return

    producer = Thread(target=_target, args=(queue,))
    producer.start()

    def _generator() -> Iterator[Item]:
        while True:
            item = queue.get()

            if item == ProducerMessage.START:
                continue
            elif item == ProducerMessage.END:
                return

            yield item

    try:
        yield _generator()
    finally:
        with lock:
            is_terminated = True
        producer.join(timeout=join_timeout)
