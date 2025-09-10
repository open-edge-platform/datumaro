# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Type conversion registry for extensible tensor/array type support.

This module provides a runtime-extensible registry system for converting between
different tensor libraries (PyTorch, NumPy, JAX, TensorFlow, etc.) and Polars
DataFrames. New types can be registered at runtime without modifying core code.
"""

import sys
import types
from typing import Any, Callable, Union

import numpy as np
import polars as pl


def polars_to_numpy_dtype(polars_dtype: pl.DataType) -> np.dtype[Any]:
    """Convert a Polars dtype to the corresponding NumPy dtype.

    Args:
        polars_dtype: Polars data type to convert

    Returns:
        Corresponding NumPy dtype

    Raises:
        TypeError: If no mapping exists for the given Polars dtype

    Example:
        >>> numpy_dtype = polars_to_numpy_dtype(pl.Float32)
        >>> numpy_dtype == np.float32
        True
    """
    # Basic numeric types
    if polars_dtype == pl.Float32:
        return np.dtype(np.float32)
    elif polars_dtype == pl.Float64:
        return np.dtype(np.float64)
    elif polars_dtype == pl.Int8:
        return np.dtype(np.int8)
    elif polars_dtype == pl.Int16:
        return np.dtype(np.int16)
    elif polars_dtype == pl.Int32:
        return np.dtype(np.int32)
    elif polars_dtype == pl.Int64:
        return np.dtype(np.int64)
    elif polars_dtype == pl.UInt8:
        return np.dtype(np.uint8)
    elif polars_dtype == pl.UInt16:
        return np.dtype(np.uint16)
    elif polars_dtype == pl.UInt32:
        return np.dtype(np.uint32)
    elif polars_dtype == pl.UInt64:
        return np.dtype(np.uint64)
    elif polars_dtype == pl.Boolean:
        return np.dtype(np.bool_)
    elif polars_dtype == pl.Binary:
        return np.dtype(np.bytes_)
    else:
        raise TypeError(f"No NumPy dtype mapping for Polars dtype: {polars_dtype}")


# Type conversion registry - extensible at runtime
_to_numpy_converters: dict[type, Callable[[Any], np.ndarray[Any, Any]]] = {
    np.ndarray: lambda x: x,
    bytes: lambda x: np.array(x),
}

_from_polars_converters: dict[type, Callable[[Any], Any]] = {
    np.ndarray: lambda x: np.array(x),
    int: lambda x: int(x),
    float: lambda x: float(x),
    str: lambda x: str(x),
    bytes: lambda x: bytes(x),
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
            if numpy_value.dtype == object:
                nested_func = np.vectorize(
                    lambda x: to_numpy(x, dtype), otypes=numpy_value.dtype.char
                )
                numpy_value = nested_func(numpy_value)
            else:
                target_numpy_dtype = polars_to_numpy_dtype(dtype)
                numpy_value = numpy_value.astype(target_numpy_dtype)

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
    # Handle direct type matches first
    if target_type in _from_polars_converters:
        return _from_polars_converters[target_type](polars_data)

    # Handle Union types (e.g., torch.Tensor | np.ndarray)
    # Check if target_type is a Union type (Python 3.10+ style or typing.Union)
    is_union = False
    union_args = None

    # Check for types.UnionType (Python 3.10+ syntax: A | B)
    if sys.version_info >= (3, 10) and isinstance(target_type, types.UnionType):
        is_union = True
        union_args = target_type.__args__

    # Check for typing.Union (older syntax: Union[A, B])
    try:
        from typing import get_args, get_origin

        if get_origin(target_type) is Union:
            is_union = True
            union_args = get_args(target_type)
    except Exception:
        pass

    if is_union and union_args:
        # Try each type in the union until one succeeds
        for union_type in union_args:
            if union_type in _from_polars_converters:
                try:
                    return _from_polars_converters[union_type](polars_data)
                except KeyError:
                    # If conversion fails, try the next type in the union
                    continue
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


# Register PIL Image converters if available
try:
    from PIL import Image

    register_numpy_converter(Image.Image, lambda x: np.array(x))
    register_from_polars_converter(Image.Image, lambda x: Image.fromarray(np.array(x)))
except ImportError:
    pass


def convert_image_type(image: Any, target_type: type) -> Any:
    """
    Convert an image between different types (numpy, PIL, torch).
    This function provides direct conversion between image types using
    the registered converters in the type registry.
    Args:
        image: Source image (numpy.ndarray, PIL.Image.Image, or torch.Tensor)
        target_type: Target type to convert to
    Returns:
        Image converted to the target type
    Raises:
        TypeError: If source or target type is not supported
    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> import torch
        >>>
        >>> # Convert numpy array to PIL Image
        >>> np_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        >>> pil_image = convert_image_type(np_image, Image.Image)
        >>>
        >>> # Convert PIL Image to torch tensor
        >>> torch_image = convert_image_type(pil_image, torch.Tensor)
    """
    current_type = type(image)

    # Define supported image types - only numpy, PIL Image, and torch Tensor
    supported_image_types = get_supported_image_types()

    # Validate that target_type is a supported image type
    if target_type not in supported_image_types:
        supported_names = [t.__name__ for t in supported_image_types]
        raise TypeError(
            f"Target type {target_type.__name__} not supported. Supported image types: {supported_names}"
        )

    # If already the target type, return as-is
    if current_type == target_type:
        return image

    # Convert via numpy as intermediate format
    try:
        # First convert to numpy if not already
        if current_type == np.ndarray:
            numpy_image = image
        else:
            numpy_image = to_numpy(image)

        # Then convert from numpy to target type
        if target_type == np.ndarray:
            return numpy_image
        else:
            # Convert numpy to target via polars-style conversion
            return _from_polars_converters[target_type](numpy_image)

    except Exception as e:
        raise TypeError(f"Cannot convert from {current_type} to {target_type}: {e}")


def get_supported_image_types() -> list[type]:
    """
    Get a list of all supported image types for conversion.
    Returns:
        List of supported image types
    """
    supported_types = [np.ndarray]  # numpy is always supported

    # Add conditionally available types
    try:
        from PIL import Image

        if Image.Image in _from_polars_converters:
            supported_types.append(Image.Image)
    except ImportError:
        pass

    # Check for torch
    try:
        import torch

        if torch.Tensor in _from_polars_converters:
            supported_types.append(torch.Tensor)
    except ImportError:
        pass

    return supported_types
