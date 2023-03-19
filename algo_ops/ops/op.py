import collections
import functools
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Any, Dict, Sequence, Optional, Tuple

import numpy as np

import algo_ops.paraloop.paraloop as paraloop
import algo_ops.ops.settings as settings
from algo_ops.pickleable_object.pickleable_object import PickleableObject
from algo_ops.plot.plot import plot_op_execution_time_distribution


class Op(ABC, PickleableObject):
    """
    Represents a single algorithm operation that can be executed. Inputs and outputs can be visualized and saved to
    file.
    """

    def __init__(self, func: Callable):
        """
        param func: The operation function
        """

        # core functionality
        self._func: Callable = func
        self.exec_func: Callable = func
        self.name: str = func.__name__
        self.input: Optional[Any] = None
        self.output: Optional[Any] = None

        # profiling
        self.execution_times: collections.deque = collections.deque(maxlen=1000)

        # evaluation functionality variables
        self.eval_func: Optional[Callable] = None
        self.incorrect_pkl_path: Optional[str] = None

    def exec(self, inp: Any) -> Any:
        """
        Executes operation function on an input. Is also self-time profiling.

        param inp: The Op input

        return
            output: The result of the executing the operation
        """
        if settings.DEBUG_MODE:
            print("Executing Op: " + str(self.name))
        self.input = inp
        t0 = time.time()
        self.output = self.exec_func(inp)
        tf = time.time()
        elapsed_time = tf - t0
        self.execution_times.append(elapsed_time)
        if settings.DEBUG_MODE:
            print(
                "Op Executed: "
                + str(self.name)
                + ", "
                + str(round(elapsed_time, 3))
                + "s"
            )
        return self.output

    @abstractmethod
    def vis_input(self) -> None:
        """
        Visualize current input.
        """
        pass

    @abstractmethod
    def vis(self) -> None:
        """
        Visualize current output.
        """
        pass

    @abstractmethod
    def save_input(self, out_path: str, basename: Optional[str] = None) -> None:
        """
        Saves current input to file.

        param out_path: Path to where input should be saved.
        param basename: File basename
        """
        pass

    @abstractmethod
    def save_output(self, out_path, basename: Optional[str] = None) -> None:
        """
        Saves current output to file.

        param out_path: Path to where output should be saved.
        param basename: File basename
        """
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Sets parameters of operation.

        param params: Dict that maps parameter name -> parameter value
        """
        self.exec_func = functools.partial(self._func, **params)

    @staticmethod
    def _format_execution_time_stats(
        execution_times: List[float], num_sf: int = 9
    ) -> str:
        """
        Formats execution time stats.

        param execution_times: List of execution times
        num_sf: The number of significant figures to display

        return
            output: (mean) +/- (std) s/calls
        """
        mean_val = np.mean(execution_times)
        std_val = np.std(execution_times)
        return (
            str(np.round(mean_val, num_sf))
            + " +/- "
            + str(np.round(std_val, num_sf))
            + " s/call"
        )

    def vis_profile(
        self, profiling_figs_path: Optional[str] = "algo_ops_profile"
    ) -> None:
        """
        Prints execution time statistics of Op. Generates stdout output and figures if profiling_figs_path is
        specified. If no profiling data is available, throw a ValueError.

        param profiling_figs_path: Path to where profiling figures should go.
        """

        # check that measurements exist
        if len(self.execution_times) == 0:
            raise ValueError(
                "There are no profiling measurements yet for Op " + str(self.name) + "."
            )

        # print summary to command line
        print(
            self.name
            + ": "
            + self._format_execution_time_stats(
                execution_times=list(self.execution_times)
            )
        )

        # make plot if needed
        if profiling_figs_path is not None:
            outfile = os.path.join(profiling_figs_path, self.name + ".png")
            plot_op_execution_time_distribution(
                execution_times=list(self.execution_times),
                op_name=self.name,
                outfile=outfile,
            )

    def _embedded_eval(self, inp: Any) -> Tuple[Any, bool]:
        """
        Helper function to embed evaluation and prediction in same function.
        Returns true if the function, when run on input, yielded correct result.
        If false, pickles, op state to file, if pickle path is specified.

        Note: The implementation of this function requires that eval_func is set to None before pickling.
        This is because the eval_func is not pickleable.

        param inp: Op input
        """
        result = self.exec(inp=inp)
        correct = self.eval_func(inp=inp, pred=result)
        if isinstance(inp, str):
            inp = inp.replace("/", "_")
        if not correct and self.incorrect_pkl_path is not None:
            outfile = os.path.join(self.incorrect_pkl_path, str(inp) + ".pkl")
            temp_eval_func = self.eval_func
            self.eval_func = None
            self.to_pickle(out_pkl_path=outfile)
            self.eval_func = temp_eval_func
        return result, correct

    def evaluate(
        self,
        inputs: Sequence[Any],
        eval_func: Callable,
        incorrect_pkl_path: Optional[str] = None,
        mechanism: str = "pool",
    ) -> List[Tuple[Any, bool]]:
        """
        Evaluates Op on a set of inputs. On each prediction, an evaluation function is run.
        If the answer is incorrect, the op state is pickled to [incorrect_pkl_path] / [inp].pkl.
        This is useful for debugging and evaluating an Op.

        param inputs: The set of inputs to evaluate on
        param eval_func: A function to evaluate a prediction on an input
        param incorrect_pkl_path: Path where incorrect prediction states should be pickled
        param mechanism: The paraloop mechanism to use (e.g. parallel, sequential)

        return:
            results: (List of results of pipeline executions, bool whether answer was correct)
        """
        if incorrect_pkl_path is not None:
            os.makedirs(incorrect_pkl_path, exist_ok=True)
        self.incorrect_pkl_path = incorrect_pkl_path
        self.eval_func = eval_func
        return paraloop.loop(
            func=self._embedded_eval, params=inputs, mechanism=mechanism
        )
