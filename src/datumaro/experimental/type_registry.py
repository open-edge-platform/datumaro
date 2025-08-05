# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Type conversion registry for extensible tensor/array type support.

This module provides a runtime-extensible registry system for converting between
different tensor libraries (PyTorch, NumPy, JAX, TensorFlow, etc.) and Polars
DataFrames. New types can be registered at runtime without modifying core code.
"""

from typing import Any, Callable

import numpy as np
import polars as pl

# Type conversion registry - extensible at runtime
_to_numpy_converters: dict[type, Callable[[Any], np.ndarray[Any, Any]]] = {
    np.ndarray: lambda x: x,
}

_from_polars_converters: dict[type, Callable[[Any], Any]] = {
    np.ndarray: lambda x: np.array(x),
    int: lambda x: int(x),
    float: lambda x: float(x),
    str: lambda x: str(x),
}


def register_numpy_converter(
    source_type: type, converter_func: Callable[[Any], np.ndarray[Any, Any]]
) -> None:
    """Register a converter function to convert from source_type to numpy array.

    Args:
        source_type: The source type to convert from
        converter_func: Function that takes a value of source_type and returns np.ndarray

    Example:
        >>> import jax.numpy as jnp
        >>> register_numpy_converter(jnp.ndarray, lambda x: np.array(x))
    """
    _to_numpy_converters[source_type] = converter_func


def register_from_polars_converter(target_type: type, converter_func: Callable[[Any], Any]) -> None:
    """Register a converter function to convert from polars data to target_type.

    Args:
        target_type: The target type to convert to
        converter_func: Function that takes polars data and returns target_type

    Example:
        >>> import jax.numpy as jnp
        >>> register_from_polars_converter(jnp.ndarray, lambda x: jnp.array(x))
    """
    _from_polars_converters[target_type] = converter_func


def to_numpy(value: Any, dtype: Any = None) -> np.ndarray[Any, Any]:
    """Convert any registered type to numpy array with optional dtype conversion.

    Args:
        value: Value to convert to numpy array
        dtype: Optional Polars dtype to ensure numpy array has correct dtype

    Returns:
        numpy array representation of the value with correct dtype

    Raises:
        TypeError: If the value type is not registered for conversion

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> numpy_array = to_numpy(tensor)
        >>> isinstance(numpy_array, np.ndarray)
        True
    """
    value_type = type(value)  # type: ignore

    if value_type in _to_numpy_converters:
        numpy_value = _to_numpy_converters[value_type](value)

        # Apply dtype conversion if specified
        if dtype is not None:
            if dtype == pl.Float32:
                numpy_value = numpy_value.astype(np.float32)
            elif dtype == pl.Float64:
                numpy_value = numpy_value.astype(np.float64)
            elif dtype == pl.Int32:
                numpy_value = numpy_value.astype(np.int32)
            elif dtype == pl.Int64:
                numpy_value = numpy_value.astype(np.int64)

        return numpy_value

    raise TypeError(f"No converter registered for type {value_type}")


def from_polars_data(polars_data: Any, target_type: type) -> Any:
    """Convert polars data to target type.

    Args:
        polars_data: Data from polars DataFrame
        target_type: Target type to convert to

    Returns:
        Value converted to target_type

    Raises:
        TypeError: If target_type is not registered for conversion

    Example:
        >>> import torch
        >>> polars_data = [1, 2, 3]
        >>> tensor = from_polars_data(polars_data, torch.Tensor)
        >>> isinstance(tensor, torch.Tensor)
        True
    """
    if target_type in _from_polars_converters:
        return _from_polars_converters[target_type](polars_data)

    raise TypeError(f"No converter registered for type {target_type}")


# Register PyTorch converters if available
try:
    import torch  # pyright: ignore[reportMissingImports]

    register_numpy_converter(
        torch.Tensor, lambda x: x.detach().cpu().numpy()
    )  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    register_from_polars_converter(
        torch.Tensor, lambda x: torch.tensor(x)
    )  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType, reportUnknownArgumentType]
except ImportError:
    pass
