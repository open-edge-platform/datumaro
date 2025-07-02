# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import unittest
import warnings

from datumaro.util.deprecation import deprecated


class TestDeprecationDecorator(unittest.TestCase):
    def test_deprecated_class_warning(self):
        # Create a test class with the deprecated decorator
        @deprecated()
        class TestClass:
            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

        # Test that a warning is raised when instantiating the class
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = TestClass(42)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            expected_message = "The TestClass class will be deprecated in version 1.11 and will be removed in version 1.12."
            self.assertEqual(str(w[0].message), expected_message)
