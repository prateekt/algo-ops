import collections
import functools
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Any, Dict, Sequence, Optional, Tuple

import numpy as np

import algo_ops.paraloop.paraloop as paraloop


class Op(ABC):
    """
    Represents a single algorithm operation that can be executed. Inputs and outputs can be visualized.
    """

    def __init__(self, func: Callable):
        """
        param func: The operation function
        """

        # core functionality
        self.func = func
        self.exec_func = func
        self.name = func.__name__
        self.input = None
        self.output = None

        # profiling
        self.execution_times: collections.deque = collections.deque(maxlen=1000)

        # evaluation functionality variables
        self.eval_func = None
        self.incorrect_pkl_path: Optional[str] = None

    def exec(self, inp: Any) -> Any:
        """
        Executes operation function on an input. Is also self-time profiling.

        param inp: The input
        return
            output: The result of the operation
        """
        self.input = inp
        t0 = time.time()
        self.output = self.exec_func(inp)
        tf = time.time()
        elapsed_time = tf - t0
        self.execution_times.append(elapsed_time)
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
    def save_input(self, out_path: str) -> None:
        """
        Saves current input to file.

        param out_path: Path to where input should be saved.
        """
        pass

    @abstractmethod
    def save_output(self, out_path) -> None:
        """
        Saves current output to file.

        param out_path: Path to where output should be saved.
        """
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Sets parameters of operation.

        param params: Dict that maps parameter name -> parameter value
        """
        self.exec_func = functools.partial(self.func, **params)

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

    def vis_profile(self) -> None:
        """
        Prints execution time statistics of Op.
        """
        print(
            self.name
            + ": "
            + self._format_execution_time_stats(
                execution_times=list(self.execution_times)
            )
        )

    def to_pickle(self, out_file: str) -> None:
        """
        Pickles current state of Op to file.
        """
        with open(out_file, "wb") as out:
            pickle.dump(self, out)

    @staticmethod
    def load_from_pickle(in_file: str) -> "Op":
        """
        Loads op state from a pickle file.
        """
        with open(in_file, "rb") as inp:
            op = pickle.load(inp)
        assert isinstance(op, Op)
        return op

    def _embedded_eval(self, inp: Any) -> Tuple[Any, bool]:
        """
        Helper function to embed evaluation and prediction in same function.
        Returns true if function, when run on input, yielded correct result.
        If false, pickles, op state to file, if pickle path is specified.
        """
        result = self.exec(inp=inp)
        correct = self.eval_func(inp=inp, pred=result)
        if isinstance(inp, str):
            inp = inp.replace("/", "_")
        if not correct and self.incorrect_pkl_path is not None:
            outfile = os.path.join(self.incorrect_pkl_path, str(inp) + ".pkl")
            self.to_pickle(out_file=outfile)
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
        param mechanism: The paraloop mechanism to use

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
