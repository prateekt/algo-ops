import os
import shutil
import unittest

import cv2
import ezplotly.settings as plotting_settings
import numpy as np

from algo_ops.dependency.tester_util import clean_paths
from algo_ops.ops.cv import CVOp, ImageResult
from algo_ops.ops.text import TextOp
from algo_ops.pipeline.cv_pipeline import CVPipeline


class TestCVPipeline(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(
            dirs=("test_profile", "cvop_results", "profiling_figs", "pipeline_outputs"),
            files=("test.pkl",),
        )

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_image = os.path.join(dir_path, "data", "joy_of_data.png")
        plotting_settings.SUPPRESS_PLOTS = True
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

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

        # init and test empty state
        op = CVOp(func=self._invert_img)
        self.assertTrue(isinstance(op, CVOp))
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(op.name, "_invert_img")
        self.assertEqual(op.exec_func, self._invert_img)
        self.assertEqual(len(op.execution_times), 0)
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
        output = op.exec(inp=self.test_image)
        self.assertTrue(isinstance(op.input, ImageResult))
        self.assertTrue(isinstance(op.output, ImageResult))
        self.assertEqual(op.output, output)
        original_inp = cv2.imread(filename=self.test_image)
        original_inp = cv2.cvtColor(original_inp, cv2.COLOR_BGR2RGB)
        manual_output = self._invert_img(original_inp)
        self.assertTrue(np.array_equal(output.img, manual_output))
        self.assertTrue(np.array_equal(op.output.img, manual_output))
        self.assertTrue(np.array_equal(op.input.img, original_inp))

        # test vis (with plots suppressed)
        op.vis_input()
        op.vis()

        # test saving input / outputs
        op.save_input(out_path="cvop_results")
        op.save_output(out_path="cvop_results")
        fig_files = ["_invert_img", "_invert_img_input"]
        self.assertTrue(os.path.exists("cvop_results"))
        for fig_file in fig_files:
            fig_path = os.path.join("cvop_results", fig_file + ".png")
            self.assertTrue(os.path.exists(fig_path))

        # test profiling data
        self.assertEqual(len(op.execution_times), 1)
        op.vis_profile(profiling_figs_path="test_profile")
        self.assertEqual(len(os.listdir("test_profile")), 1)
        self.assertTrue(os.path.exists("test_profile"))
        self.assertTrue(os.path.exists(os.path.join("test_profile", "_invert_img.png")))

        # test pickling
        op.to_pickle(out_pkl_path="test.pkl")

    def test_cv_pipeline(self) -> None:
        """
        End-to-end test example of CVPipeline.
        """

        # init pipeline and check empty state
        pipeline = CVPipeline.init_from_funcs(
            funcs=[self._gray_scale, self._invert_img]
        )
        self.assertTrue(isinstance(pipeline, CVPipeline))
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        for method in (
            pipeline.vis_input,
            pipeline.vis,
            pipeline.save_input,
            pipeline.save_output,
            pipeline.vis_profile,
        ):
            with self.assertRaises(ValueError):
                method()

        # test that pipeline output matches stacking functions manually
        pipeline_output = pipeline.exec(inp=self.test_image)
        self.assertTrue(isinstance(pipeline.input, ImageResult))
        self.assertTrue(isinstance(pipeline.output, ImageResult))
        self.assertTrue(isinstance(pipeline_output, ImageResult))
        self.assertEqual(pipeline.output, pipeline_output)
        original_inp = cv2.imread(filename=self.test_image)
        original_inp = cv2.cvtColor(original_inp, cv2.COLOR_BGR2RGB)
        manual_output = self._invert_img(self._gray_scale(original_inp))
        self.assertTrue(np.array_equal(pipeline_output.img, manual_output))
        self.assertTrue(np.array_equal(pipeline.output.img, manual_output))
        self.assertTrue(np.array_equal(pipeline.input.file_path, self.test_image))

        # test vis (with plots suppressed)
        with self.assertRaises(ValueError):
            pipeline.vis_input()
        pipeline.vis()

        # test the pipeline saving input cannot be used
        with self.assertRaises(ValueError):
            pipeline.save_input()

        # test saving outputs
        if os.path.exists("pipeline_outputs"):
            shutil.rmtree("pipeline_outputs")
        pipeline.save_output(out_path="pipeline_outputs")
        self.assertTrue(os.path.exists("pipeline_outputs"))
        self.assertEqual(len(os.listdir("pipeline_outputs")), 3)

        # test profiling
        pipeline.vis_profile(profiling_figs_path="profiling_figs")
        self.assertTrue(os.path.exists("profiling_figs"))
        for fig_file in (
            "['_gray_scale', '_invert_img']",
            "['_gray_scale', '_invert_img']_violin",
            "_gray_scale",
            "_invert_img",
        ):
            fig_path = os.path.join("profiling_figs", fig_file + ".png")
            self.assertTrue(os.path.exists(fig_path))

        # test pickling
        pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_invalid_config(self) -> None:
        """
        Test invalid configs.
        """

        with self.assertRaises(AssertionError):
            CVPipeline.init_from_funcs(
                funcs=[self._gray_scale, self._invert_img],
                op_class=TextOp,
            )
        with self.assertRaises(AssertionError):
            CVPipeline.init_from_funcs(
                funcs=[self._gray_scale, self._invert_img],
                op_class=None,
            )
