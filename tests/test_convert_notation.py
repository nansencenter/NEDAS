import unittest
import numpy as np
import re
from NEDAS.config.parse_config import convert_notation

class TestConvertNotation(unittest.TestCase):
    def test_dict_conversion(self):
        data = {"key1": "1e-3", "key2": "inf", "key3": "true"}
        expected = {"key1": 0.001, "key2": np.inf, "key3": True}
        self.assertEqual(convert_notation(data), expected)

    def test_list_conversion(self):
        data = ["-1e5", "-inf", "false"]
        expected = [-100000.0, -np.inf, False]
        self.assertEqual(convert_notation(data), expected)

    def test_nested_structure(self):
        data = {"outer": ["1e2", {"inner": "no"}]}
        expected = {"outer": [100.0, {"inner": False}]}
        self.assertEqual(convert_notation(data), expected)

    def test_string_to_bool(self):
        self.assertTrue(convert_notation("yes"))
        self.assertTrue(convert_notation("T"))
        self.assertFalse(convert_notation("no"))
        self.assertFalse(convert_notation(".false."))

    def test_string_to_float(self):
        self.assertEqual(convert_notation("1e-3"), 0.001)
        self.assertEqual(convert_notation("-1E4"), -10000.0)

    def test_inf_handling(self):
        self.assertEqual(convert_notation("inf"), np.inf)
        self.assertEqual(convert_notation("-inf"), -np.inf)

    def test_no_conversion(self):
        self.assertEqual(convert_notation("random string"), "random string")
        self.assertEqual(convert_notation(123), 123)
        self.assertEqual(convert_notation(["text", 45]), ["text", 45])

if __name__ == "__main__":
    unittest.main()
