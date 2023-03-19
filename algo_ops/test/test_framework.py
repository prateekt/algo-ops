import os
import unittest

import ezplotly.settings as plotting_settings

from algo_ops.dependency.tester_util import clean_paths
from algo_ops.ops.op import Op
from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
import algo_ops.ops.settings as settings


class TestStruct:
    def __init__(self, a: int):
        self.a = a + 1

    def __str__(self):
        return str(self.a)


class TestAlgoOpsFramework(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(
            dirs=(
                "test_save_input",
                "test_save_output",
                "profiling_figs",
                "saving_test",
                "bad_pkl",
                "test_profile",
            ),
            files=("test.pkl", "reverse.txt", "reverse_input.txt"),
        )

    def setUp(self) -> None:
        plotting_settings.SUPPRESS_PLOTS = True
        settings.DEBUG_MODE = False
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

    # test funcs
    @staticmethod
    def reverse(s: str) -> str:
        return s[::-1]

    @staticmethod
    def append_a(s: str) -> str:
        return s + "a"

    @staticmethod
    def append_b(s: str) -> str:
        return s + "b"

    @staticmethod
    def append_something(s: str, something: str) -> str:
        return s + something

    def test_text_op(self) -> None:
        """
        Test an atomic TextOp.
        """

        # create operation from reverse function
        op = TextOp(func=self.reverse)
        self.assertEqual(op.exec_func, self.reverse)
        self.assertEqual(op.name, "reverse")
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(list(op.execution_times), [])

        # test with empty buffers; not much such work.
        for method in (op.save_input, op.save_output, op.vis_profile):
            with self.assertRaises(ValueError):
                method()
        op.vis_input()
        op.vis()

        # test op execution
        output = op.exec(inp="ab")
        self.assertEqual(output, "ba")
        self.assertEqual(output, self.reverse(s="ab"))
        self.assertEqual(op.input, "ab")
        self.assertEqual(op.output, "ba")
        self.assertEqual(len(op.execution_times), 1)

        # test op pickle and recover state
        op.to_pickle(out_pkl_path="test.pkl")
        reloaded_op = TextOp.load_from_pickle(pkl_path="test.pkl")
        self.assertEqual(op.input, reloaded_op.input)
        self.assertEqual(op.output, reloaded_op.output)
        self.assertEqual(op.execution_times, reloaded_op.execution_times)

        # test op execution again
        output = op.exec(inp="a")
        self.assertEqual(output, "a")
        self.assertEqual(output, self.reverse(s="a"))
        self.assertEqual(op.input, "a")
        self.assertEqual(op.output, "a")
        self.assertEqual(len(op.execution_times), 2)

        # test op saving input and output
        op.save_input()
        op.save_output()
        self.assertTrue(os.path.exists("reverse.txt"))
        self.assertTrue(os.path.exists("reverse_input.txt"))
        with open("reverse.txt", "r") as fin:
            self.assertEqual(fin.read(), "a")
        with open("reverse_input.txt", "r") as fin:
            self.assertEqual(fin.read(), "a")

        # test op profiling
        op.vis_profile(profiling_figs_path="profiling_figs")
        self.assertTrue(os.path.exists(os.path.join("profiling_figs", "reverse.png")))

        # test op pickle and recover state
        op.to_pickle(out_pkl_path="test.pkl")
        reloaded_op = TextOp.load_from_pickle(pkl_path="test.pkl")
        self.assertEqual(op.input, reloaded_op.input)
        self.assertEqual(op.output, reloaded_op.output)
        self.assertEqual(op.execution_times, reloaded_op.execution_times)

    def test_pipeline_framework(self) -> None:
        """
        Tests a series of TextOps in the pipeline framework.
        """

        # construct pipeline and check that Ops exist and have empty IO buffers. Inputs and outputs can be printed,
        # but not saved when TextOp pipeline is empty.
        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_b, self.reverse, self.reverse],
            op_class=TextOp,
        )
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        expected_op_names = ["append_a", "append_b", "reverse", "reverse"]
        expected_funcs = [self.append_a, self.append_b, self.reverse, self.reverse]
        for i, op_name in enumerate(pipeline.ops.keys()):
            op = pipeline.ops[op_name]
            self.assertTrue(isinstance(op, Op))
            self.assertTrue(isinstance(op, TextOp))
            op_hash_name = str(op) + ":" + expected_op_names[i]
            self.assertEqual(op_name, op_hash_name)
            self.assertEqual(op.name, expected_op_names[i])
            self.assertEqual(op.input, None)
            self.assertEqual(op.output, None)
            self.assertEqual(op.exec_func, expected_funcs[i])
        for method in [pipeline.save_input, pipeline.save_output, pipeline.vis_profile]:
            with self.assertRaises(ValueError):
                method()

        # test running pipeline and check ops IO buffers post execution
        final_output = pipeline.exec(inp="g")
        self.assertEqual(final_output, "gab")
        op_names = list(pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "g")
            else:
                self.assertEqual(op.input, pipeline.ops[op_names[i - 1]].output)
            if i == len(op_names) - 1:
                self.assertEqual(op.output, final_output)
            else:
                self.assertEqual(op.output, pipeline.ops[op_names[i + 1]].input)

        # pickle pipeline state
        pipeline.to_pickle(out_pkl_path="test.pkl")

        # test running pipeline again and check ops IO buffers post execution
        final_output = pipeline.exec(inp="a")
        self.assertEqual(final_output, "aab")
        op_names = list(pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "a")
            else:
                self.assertEqual(op.input, pipeline.ops[op_names[i - 1]].output)
            if i == len(op_names) - 1:
                self.assertEqual(op.output, final_output)
            else:
                self.assertEqual(op.output, pipeline.ops[op_names[i + 1]].input)

        # test pipeline save input/output
        with self.assertRaises(ValueError):
            pipeline.save_input()
        pipeline.save_output(out_path="saving_test")
        self.assertTrue(os.path.exists("saving_test"))
        self.assertEqual(len(os.listdir("saving_test")), 5)

        # pipeline vis test
        pipeline.vis()
        pipeline.vis_profile(profiling_figs_path="profiling_figs")
        fig_files = [
            "['append_a', 'append_b', 'reverse', 'reverse']",
            "['append_a', 'append_b', 'reverse', 'reverse']_violin",
            "append_a",
            "append_b",
            "reverse",
        ]
        for fig_file in fig_files:
            fig_path = os.path.join("profiling_figs", fig_file + ".png")
            self.assertTrue(os.path.exists(fig_path))

        # test reloading pipeline after first and check reloaded pipeline
        reloaded_pipeline = Pipeline.load_from_pickle(pkl_path="test.pkl")
        assert isinstance(reloaded_pipeline, Pipeline)
        op_names = list(reloaded_pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = reloaded_pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "g")
            else:
                self.assertEqual(
                    op.input, reloaded_pipeline.ops[op_names[i - 1]].output
                )
            if i == len(op_names) - 1:
                self.assertEqual(op.output, "gab")
            else:
                self.assertEqual(
                    op.output, reloaded_pipeline.ops[op_names[i + 1]].input
                )

    def test_parameter_fixing(self) -> None:
        """
        Test fixing a parameter in a function.
        """

        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_something, self.reverse, self.reverse],
            op_class=TextOp,
        )
        pipeline.set_pipeline_params(
            func_name="append_something", params={"something": "b"}
        )
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        expected_op_names = ["append_a", "append_something", "reverse", "reverse"]
        for i, op_name in enumerate(pipeline.ops.keys()):
            op = pipeline.ops[op_name]
            self.assertTrue(isinstance(op, Op))
            self.assertTrue(isinstance(op, TextOp))
            op_hash_name = str(op) + ":" + expected_op_names[i]
            self.assertEqual(op_name, op_hash_name)
            self.assertEqual(op.name, expected_op_names[i])
            self.assertEqual(op.input, None)
            self.assertEqual(op.output, None)

    @staticmethod
    def _fake_gen_ans_and_compare(inp: str, pred: str) -> bool:
        correct_ans = inp + "abc"
        return pred == correct_ans

    def test_evaluator(self) -> None:
        """
        Test evaluator capability.
        """

        # init pipeline
        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_b, self.reverse, self.reverse],
            op_class=TextOp,
        )

        # evaluate on a set of inputs. The pipeline should get it all wrong
        # and produce debug pickle files that can be loaded.
        inputs = ["a", "b", "cc"]
        results = pipeline.evaluate(
            inputs=inputs,
            eval_func=self._fake_gen_ans_and_compare,
            incorrect_pkl_path="bad_pkl",
        )
        self.assertEqual(results, [("aab", False), ("bab", False), ("ccab", False)])
        self.assertTrue(os.path.exists("bad_pkl"))
        for inp in inputs:
            self.assertTrue(os.path.exists("bad_pkl/" + inp + ".pkl"))
            reloaded_pipeline = Pipeline.load_from_pickle("bad_pkl/" + inp + ".pkl")
            self.assertTrue(isinstance(reloaded_pipeline, Pipeline))
            self.assertEqual(reloaded_pipeline.input, inp)

        # test pickle
        pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_pipeline_of_pipelines(self) -> None:
        """
        Test that pipelines of pipelines work.
        """
        pipeline1 = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_something, self.reverse, self.reverse],
            op_class=TextOp,
        )
        pipeline1.set_pipeline_params(
            func_name="append_something", params={"something": "b"}
        )
        pipeline2 = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_b, self.reverse],
            op_class=TextOp,
        )
        total_pipeline = Pipeline(ops=[pipeline1, pipeline2])
        output = total_pipeline.exec(inp="z")
        self.assertEqual(pipeline1.input, "z")
        self.assertEqual(pipeline1.output, "zab")
        self.assertEqual(pipeline2.input, "zab")
        self.assertEqual(pipeline2.output, "babaz")
        self.assertEqual(output, "babaz")
        with self.assertRaises(ValueError):
            total_pipeline.vis_input()
        pipeline1.vis()
        pipeline2.vis()
        total_pipeline.vis()
        total_pipeline.vis_profile(profiling_figs_path="test_profile")
        self.assertEqual(len(os.listdir("test_profile")), 10)

        # test pickle
        pipeline1.to_pickle(out_pkl_path="test.pkl")
        pipeline2.to_pickle(out_pkl_path="test.pkl")
        total_pipeline.to_pickle(out_pkl_path="test.pkl")

    @staticmethod
    def f(a: int) -> TestStruct:
        return TestStruct(a=a)

    def test_text_op_struct(self) -> None:
        """
        Test TextOp with embedded TestStruct.
        """

        op = TextOp(func=self.f)
        op.exec(inp=2)

        # test save input
        op.save_input(out_path="test_save_input")
        test_file = os.path.join("test_save_input", "f_input.txt")
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as fin:
            line = fin.read()
        self.assertEqual(line, "2")

        # test save output
        op.save_output(out_path="test_save_output")
        test_file = os.path.join("test_save_output", "f.txt")
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as fin:
            line = fin.read()
        self.assertEqual(line, "3")

        # test pickle
        op.to_pickle(out_pkl_path="test.pkl")

    def test_debug_mode(self) -> None:
        """
        Test that debug mode prints out stuff.
        """
        settings.DEBUG_MODE = True
        op = TextOp(func=self.reverse)
        op.exec("aabb")
        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.reverse, self.reverse],
            op_class=TextOp,
        )
        pipeline.name = "TEST_PIPELINE"
        pipeline.exec("test")
