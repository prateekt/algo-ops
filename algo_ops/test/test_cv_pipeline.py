import os
import shutil
import unittest

import cv2
import numpy as np

from algo_ops.ops.cv import CVOp
from algo_ops.ops.text import TextOp
from algo_ops.pipeline.cv_pipeline import CVPipeline


class TestCVPipeline(unittest.TestCase):
    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_image = os.path.join(dir_path, "data", "joy_of_data.png")

    @staticmethod
    def _gray_scale(img: np.array) -> np.array:
        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def _invert_img(img: np.array) -> np.array:
        # invert so white text becomes black
        inv_img = cv2.bitwise_not(img)
        return inv_img

    def test_cv_op(self) -> None:
        """
        Test a single CV Op.
        """

        # clear results dirs
        if os.path.exists("test_profile"):
            shutil.rmtree("test_profile")
        if os.path.exists("cvop_results"):
            shutil.rmtree("cvop_results")
        self.assertTrue(not os.path.exists("test_profile"))
        self.assertTrue(not os.path.exists("cvop_results"))

        # init and test empty state
        op = CVOp(func=self._invert_img, profiling_figs_path="test_profile")
        self.assertTrue(isinstance(op, CVOp))
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(op.name, "_invert_img")
        self.assertEqual(op.exec_func, self._invert_img)
        self.assertEqual(len(op.execution_times), 0)
        self.assertEqual(op.profiling_figs_path, "test_profile")
        for method in [
            op.vis_input,
            op.vis,
            op.save_input,
            op.save_output,
            op.vis_profile,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test that Op executes as expected and matches manual call to function
        pipeline_output = op.exec(inp=self.test_image)
        original_inp = cv2.imread(filename=self.test_image)
        original_inp = cv2.cvtColor(original_inp, cv2.COLOR_BGR2RGB)
        manual_output = self._invert_img(original_inp)
        self.assertTrue(np.array_equal(pipeline_output, manual_output))
        self.assertTrue(np.array_equal(op.output, manual_output))
        self.assertTrue(np.array_equal(op.input, original_inp))

        # test saving input / outputs
        op.save_input(out_path="cvop_results")
        op.save_output(out_path="cvop_results")
        fig_files = ["_invert_img", "_invert_img_input"]
        self.assertTrue(os.path.exists("cvop_results"))
        for fig_file in fig_files:
            fig_path = os.path.join("cvop_results", fig_file + ".png")
            self.assertTrue(os.path.exists(fig_path))
        shutil.rmtree("cvop_results")

        # test profiling data
        self.assertEqual(len(op.execution_times), 1)
        op.vis_profile()
        self.assertEqual(len(os.listdir("test_profile")), 1)
        self.assertTrue(os.path.exists("test_profile"))
        self.assertTrue(os.path.exists(os.path.join("test_profile", "_invert_img.png")))
        shutil.rmtree("test_profile")

    def test_cv_pipeline(self) -> None:
        """
        End-to-end test example of CVPipeline.
        """

        # init pipeline and check empty state
        pipeline = CVPipeline.init_from_funcs(
            funcs=[self._gray_scale, self._invert_img],
            profiling_figs_path="profiling_figs",
        )
        self.assertTrue(isinstance(pipeline, CVPipeline))
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        with self.assertRaises(ValueError):
            pipeline.save_input()
        with self.assertRaises(ValueError):
            pipeline.save_output()
        with self.assertRaises(ValueError):
            pipeline.vis_input()

        # test that pipeline output matches stacking functions manually
        pipeline_output = pipeline.exec(inp=self.test_image)
        original_inp = cv2.imread(filename=self.test_image)
        original_inp = cv2.cvtColor(original_inp, cv2.COLOR_BGR2RGB)
        manual_output = self._invert_img(self._gray_scale(original_inp))
        self.assertTrue(np.array_equal(pipeline_output, manual_output))
        self.assertTrue(np.array_equal(pipeline.output, manual_output))
        self.assertTrue(np.array_equal(pipeline.input, self.test_image))

        # test the pipeline saving input cannot be used
        with self.assertRaises(ValueError):
            pipeline.save_input()

        # test saving outputs
        if os.path.exists("pipeline_outputs"):
            shutil.rmtree("pipeline_outputs")
        pipeline.save_output(out_path="pipeline_outputs")
        self.assertTrue(os.path.exists("pipeline_outputs"))
        self.assertEqual(len(os.listdir("pipeline_outputs")), 3)
        shutil.rmtree("pipeline_outputs")

        # test profiling
        if os.path.exists("profiling_figs"):
            shutil.rmtree("profiling_figs")
        pipeline.vis_profile()
        fig_files = [
            "['_gray_scale', '_invert_img']",
            "['_gray_scale', '_invert_img']_violin",
            "_gray_scale",
            "_invert_img",
        ]
        self.assertTrue(os.path.exists("profiling_figs"))
        for fig_file in fig_files:
            fig_path = os.path.join("profiling_figs", fig_file + ".png")
            self.assertTrue(os.path.exists(fig_path))
        shutil.rmtree("profiling_figs")

    def test_invalid_config(self) -> None:
        """
        Test invalid configs.
        """

        with self.assertRaises(AssertionError):
            CVPipeline.init_from_funcs(
                funcs=[self._gray_scale, self._invert_img],
                profiling_figs_path="profiling_figs",
                op_class=TextOp,
            )
        with self.assertRaises(AssertionError):
            CVPipeline.init_from_funcs(
                funcs=[self._gray_scale, self._invert_img],
                profiling_figs_path="profiling_figs",
                op_class=None,
            )
