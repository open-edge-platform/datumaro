# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""Tests for the experimental type registry system."""

import sys
from typing import Optional, Union

import numpy as np
import pytest

from datumaro.experimental.type_registry import from_polars_data, register_from_polars_converter, to_numpy


def test_basic_type_conversion():
    """Test basic type conversion for registered types."""
    # Test int conversion
    result = from_polars_data(42, int)
    assert result == 42
    assert isinstance(result, int)

    # Test float conversion
    result = from_polars_data(3.14, float)
    assert result == 3.14
    assert isinstance(result, float)

    # Test str conversion
    result = from_polars_data("hello", str)
    assert result == "hello"
    assert isinstance(result, str)

    # Test numpy array conversion
    data = [1, 2, 3]
    result = from_polars_data(data, np.ndarray)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_union_type_conversion():
    """Test Union type conversion with both syntax styles."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    data = [1.0, 2.0, 3.0]

    # Test typing.Union syntax
    union_type = Union[torch.Tensor, np.ndarray]
    result = from_polars_data(data, union_type)
    assert isinstance(result, torch.Tensor)
    assert result.tolist() == [1.0, 2.0, 3.0]

    # Test modern syntax (Python 3.10+)
    if sys.version_info >= (3, 10):
        modern_union = torch.Tensor | np.ndarray
        result = from_polars_data(data, modern_union)
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]


def test_union_type_fallback_behavior():
    """Test that Union types fall back to subsequent types when first conversion fails."""

    # Create custom types for testing fallback
    class FailingConverter:
        def __init__(self, data):
            raise KeyError("This converter always fails")

    class WorkingConverter:
        def __init__(self, data):
            self.data = data

    # Register converters
    register_from_polars_converter(FailingConverter, lambda x: FailingConverter(x))
    register_from_polars_converter(WorkingConverter, lambda x: WorkingConverter(x))

    # Test that union falls back to the working converter
    union_type = Union[FailingConverter, WorkingConverter]
    data = [1, 2, 3]

    result = from_polars_data(data, union_type)
    assert isinstance(result, WorkingConverter)
    assert result.data == [1, 2, 3]


def test_union_type_errors():
    """Test Union type error handling when no converters work."""

    class UnregisteredType1:
        pass

    class UnregisteredType2:
        pass

    union_type = Union[UnregisteredType1, UnregisteredType2]
    data = [1, 2, 3]

    with pytest.raises(TypeError, match="No converter registered for type"):
        from_polars_data(data, union_type)


def test_union_type_ordering():
    """Test Union type behavior with multiple working converters - first wins."""

    # Register multiple working converters
    class TypeA:
        def __init__(self, data):
            self.data = f"A:{data}"

    class TypeB:
        def __init__(self, data):
            self.data = f"B:{data}"

    register_from_polars_converter(TypeA, lambda x: TypeA(x))
    register_from_polars_converter(TypeB, lambda x: TypeB(x))

    # Should pick the first available type in the union
    union_type = Union[TypeA, TypeB]
    data = [1, 2, 3]

    result = from_polars_data(data, union_type)
    assert isinstance(result, TypeA)
    assert result.data == "A:[1, 2, 3]"

    # Change order - should pick TypeB first now
    union_type_reordered = Union[TypeB, TypeA]
    result = from_polars_data(data, union_type_reordered)
    assert isinstance(result, TypeB)
    assert result.data == "B:[1, 2, 3]"


def test_torch_converter_functionality():
    """Test PyTorch tensor converter registration and functionality."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    # Test torch tensor to numpy conversion
    tensor_data = torch.tensor([1.0, 2.0, 3.0])
    numpy_result = to_numpy(tensor_data)
    assert isinstance(numpy_result, np.ndarray)
    np.testing.assert_array_almost_equal(numpy_result, np.array([1.0, 2.0, 3.0]))

    # Test polars data to torch conversion
    polars_data = [1.0, 2.0, 3.0]
    torch_result = from_polars_data(polars_data, torch.Tensor)
    assert isinstance(torch_result, torch.Tensor)
    assert torch_result.tolist() == [1.0, 2.0, 3.0]


def test_numpy_converter_functionality():
    """Test the numpy converter registration and usage."""
    # Test basic numpy conversion
    data = [[1, 2], [3, 4]]
    result = to_numpy(np.array(data))
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(data))


def test_optional_type_with_none():
    """Test Optional type conversion when polars_data is None."""
    # Test Optional[int] with None
    result = from_polars_data(None, Optional[int])
    assert result is None

    # Test Optional[str] with None
    result = from_polars_data(None, Optional[str])
    assert result is None

    # Test Optional[float] with None
    result = from_polars_data(None, Optional[float])
    assert result is None


def test_optional_type_with_data():
    """Test Optional type conversion when polars_data has actual data."""
    # Test Optional[int] with actual int data
    result = from_polars_data(42, Optional[int])
    assert result == 42
    assert isinstance(result, int)

    # Test Optional[str] with actual string data
    result = from_polars_data("hello", Optional[str])
    assert result == "hello"
    assert isinstance(result, str)

    # Test Optional[float] with actual float data
    result = from_polars_data(3.14, Optional[float])
    assert result == 3.14
    assert isinstance(result, float)


def test_union_with_none_explicit():
    """Test explicit Union[Type, None] conversion."""
    # Test Union[int, None] with None
    result = from_polars_data(None, Union[int, None])
    assert result is None

    # Test Union[int, None] with actual data
    result = from_polars_data(123, Union[int, None])
    assert result == 123
    assert isinstance(result, int)

    # Test Union[None, str] (reverse order) with None
    result = from_polars_data(None, Union[None, str])
    assert result is None

    # Test Union[None, str] with actual data
    result = from_polars_data("test", Union[None, str])
    assert result == "test"
    assert isinstance(result, str)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+ union syntax")
def test_modern_union_with_none():
    """Test Python 3.10+ Union syntax (A | None) conversion."""
    # Test int | None with None
    union_type = eval("int | None")
    result = from_polars_data(None, union_type)
    assert result is None

    # Test int | None with actual data
    result = from_polars_data(456, union_type)
    assert result == 456
    assert isinstance(result, int)

    # Test None | str (reverse order) with None
    union_type = eval("None | str")
    result = from_polars_data(None, union_type)
    assert result is None

    # Test None | str with actual data
    result = from_polars_data("modern", union_type)
    assert result == "modern"
    assert isinstance(result, str)


def test_optional_vs_regular_union():
    """Test that Optional types behave differently from regular Union types."""

    # Create custom types for testing
    class TypeA:
        def __init__(self, data):
            self.data = f"A:{data}"

    class TypeB:
        def __init__(self, data):
            self.data = f"B:{data}"

    register_from_polars_converter(TypeA, lambda x: TypeA(x))
    register_from_polars_converter(TypeB, lambda x: TypeB(x))

    # Regular Union[TypeA, TypeB] should convert to first available type
    data = "test"
    result = from_polars_data(data, Union[TypeA, TypeB])
    assert isinstance(result, TypeA)
    assert result.data == "A:test"

    # Optional Union[TypeA, None] should return None when data is None
    result = from_polars_data(None, Union[TypeA, None])
    assert result is None

    # Optional Union[TypeA, None] should convert to TypeA when data is not None
    result = from_polars_data(data, Union[TypeA, None])
    assert isinstance(result, TypeA)
    assert result.data == "A:test"


def test_optional_with_numpy_arrays():
    """Test Optional type conversion with numpy arrays."""
    # Test Optional[np.ndarray] with None
    result = from_polars_data(None, Optional[np.ndarray])
    assert result is None

    # Test Optional[np.ndarray] with actual data
    data = [1, 2, 3]
    result = from_polars_data(data, Optional[np.ndarray])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_nested_optional_error_handling():
    """Test error handling for unsupported nested Optional types."""
    # This should work fine - basic Optional
    result = from_polars_data(None, Optional[int])
    assert result is None

    # Test with unregistered type in Optional
    class UnregisteredType:
        pass

    # Should raise TypeError when trying to convert to unregistered type
    with pytest.raises(TypeError, match="No converter registered for type"):
        from_polars_data("test", Optional[UnregisteredType])


def test_list_converter_functionality():
    """Test the list to numpy converter registration and usage."""
    # Test list to numpy conversion
    data = [1, 2, 3, 4]
    result = to_numpy(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    # Test nested list conversion
    nested_data = [[1, 2], [3, 4]]
    result = to_numpy(nested_data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_points_converter_functionality():
    """Test the Points to numpy converter registration and usage."""
    from datumaro.components.annotation import Points

    # Create Points object
    points_data = [10.0, 20.0, 30.0, 40.0]
    visibility = [Points.Visibility.visible, Points.Visibility.hidden]
    points_obj = Points(points_data, visibility=visibility)

    # Test Points to numpy conversion
    result = to_numpy(points_obj)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[10.0, 20.0, 2.0], [30.0, 40.0, 1.0]]))
