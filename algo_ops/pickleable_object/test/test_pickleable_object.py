import os
import unittest

from algo_ops.dependency.tester_util import iter_params
from algo_ops.pickleable_object.pickleable_object import PickleableObject


class PklTestClass(PickleableObject):
    def __init__(self, a: int = 2):
        self.a = a


class TestPickleableObject(unittest.TestCase):
    """
    Tests inheritance of PickleableObject framework with sample class.
    """

    def test_framework(self) -> None:
        test_obj = PklTestClass()
        pkl_path = "pkl1_test.pkl"
        self.assertEqual(test_obj.a, 2)
        self.assertTrue(not os.path.exists(pkl_path))
        test_obj.to_pickle(out_pkl_path=pkl_path)
        self.assertTrue(os.path.exists(pkl_path))
        loaded_obj = PklTestClass.load_from_pickle(pkl_path=pkl_path)
        self.assertTrue(test_obj is not loaded_obj)
        self.assertEqual(test_obj.a, loaded_obj.a)
        self.assertEqual(loaded_obj.a, 2)
        os.unlink(pkl_path)

    def test_framework_vary_parameter(self) -> None:
        test_obj = PklTestClass(a=3)
        pkl_path = "pkl2_test.pkl"
        self.assertEqual(test_obj.a, 3)
        self.assertTrue(not os.path.exists(pkl_path))
        test_obj.to_pickle(out_pkl_path=pkl_path)
        self.assertTrue(os.path.exists(pkl_path))
        loaded_obj = PklTestClass.load_from_pickle(pkl_path=pkl_path)
        self.assertTrue(test_obj is not loaded_obj)
        self.assertEqual(test_obj.a, loaded_obj.a)
        self.assertEqual(loaded_obj.a, 3)
        os.unlink(pkl_path)

    @iter_params(compression=[None, "gzip", "bz2", "lzma", "blosc"])
    def test_compression(self, compression: str) -> None:
        """
        Tests compression functions of PickleableObject.
        """
        test_obj = PklTestClass()
        pkl_path = "pkl3_test.pkl"
        self.assertEqual(test_obj.a, 2)
        self.assertTrue(not os.path.exists(pkl_path))
        test_obj.to_pickle(out_pkl_path=pkl_path, compression=compression)
        self.assertTrue(os.path.exists(pkl_path))
        loaded_obj = PklTestClass.load_from_pickle(
            pkl_path=pkl_path, compression=compression
        )
        self.assertTrue(test_obj is not loaded_obj)
        self.assertEqual(test_obj.a, loaded_obj.a)
        self.assertEqual(loaded_obj.a, 2)
        os.unlink(pkl_path)
