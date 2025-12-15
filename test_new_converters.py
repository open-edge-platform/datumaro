#!/usr/bin/env python3
"""Quick test for the new converters."""

from datumaro.experimental.converters import ConverterRegistry

print("Registered converters:")
for c in ConverterRegistry.list_converters():
    print(f"  - {c.__name__}")

print("\nAll imports successful!")
