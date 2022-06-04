import os
from collections import OrderedDict
from typing import Callable, List, Any, Dict, Union, Optional

from algo_ops.ops.op import Op
from algo_ops.plot.plot import (
    plot_op_execution_time_distribution,
    plot_pipeline_execution_time_distribution,
)


class Pipeline(Op):
    """
    A generic pipeline is list of operations that execute serially on an input.
    The output of the previous pipeline step is the input of the next pipeline step.
    """

    @classmethod
    def _pipeline_name(cls, pipeline_ops: List[Op]) -> str:
        """
        Returns the pipeline of the pipeline as a string concatenated list of the names of its Ops.

        param pipeline_ops: The pipeline Ops

        return:
            Pipeline name
        """
        return str([str(op.name) for op in pipeline_ops])

    @classmethod
    def _pipeline_op_name(cls, op: Op) -> str:
        """
        Returns the pipeline name of an op using its memory address.

        param op: The Op

        return:
            Pipeline Op name
        """
        return str(op) + ":" + str(op.name)

    def _run(self, inp: Any) -> Any:
        """
        Run entire pipeline on an input. Each operation executes sequentially and produces an output that is passed
        to the next sequential Op. The final output of the pipeline is the result of the last Op.

        param inp: The top-level pipeline input that is fed-forward into the pipeline

        return:
            output: The output of the pipeline
        """
        current_input = inp
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            current_input = op.exec(inp=current_input)
        return current_input

    def __init__(self, ops: List[Op]):
        """
        Initialize a pipeline.

        param ops: List of ops in the pipeline
        """
        super().__init__(func=self._run)
        self.ops = OrderedDict()
        for i, op in enumerate(ops):
            assert isinstance(op, Op)
            self.ops[self._pipeline_op_name(op=op)] = op
        self.name = self._pipeline_name(pipeline_ops=self.ops.values())

    @classmethod
    def init_from_funcs(
        cls,
        funcs: List[Callable],
        op_class: Union[Any, List[Any]],
    ) -> "Pipeline":
        """
        Initializes a pipeline from a list of functions that sequentially execute in the pipeline.

        param funcs: List of pipeline functions that execute serially
            as operations in pipeline.
        param op_class: The subclass of Op that the pipeline uses (or list of subclasses, one for each func).
        """
        if not isinstance(op_class, list):
            op_class = [op_class for _ in range(len(funcs))]
        assert len(op_class) == len(funcs)
        ops: List[Op] = list()
        for i, func in enumerate(funcs):
            ops.append(op_class[i](func))
        return cls(ops=ops)

    def set_params(self, params: Dict[str, Any]) -> None:
        raise ValueError(
            "Please use set_pipeline_params when setting params of pipeline."
        )

    def find_op(self, func_name: str) -> Op:
        """
        Helper function to find an Op corresponding to a pipeline function.

        param func_name: Name of function

        return:
            Found Op (or ValueError if no Op found)
        """
        for key in self.ops.keys():
            op = self.ops[key]
            assert isinstance(op, Op)
            if op.name == func_name:
                return op
        raise ValueError("Op not found: " + func_name)

    def find_ops_by_class(self, op_class: Any) -> List[Op]:
        """
        Helper function to find ops by class.

        param op_class: The op class to find

        return:
            List of operations of that Op class in the pipeline
        """
        rtn: List[Op] = list()
        for op in self.ops.values():
            if isinstance(op, op_class):
                rtn.append(op)
        return rtn

    def set_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of a function in the pipeline.

        param func_name: The name of the function
        param params: The parameters to fix
        """
        op = self.find_op(func_name=func_name)
        op.set_params(params=params)

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        raise ValueError("Please use save_output to visualize pipeline data flow.")

    def vis_input(self) -> None:
        raise ValueError("Please use vis to visualize pipeline data flow.")

    def vis(self) -> None:
        """
        Visualize current outputs of each Op as well as pipeline input.
        """
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0 and not isinstance(op, Pipeline):
                op.vis_input()
            op.vis()

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves pipeline Op outputs to file.

        param out_path: Path to where output should go
        param basename: Basename of output file
        """
        os.makedirs(out_path, exist_ok=True)
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            op_pipeline_name = self._pipeline_op_name(op=op)
            if i == 0:
                op.save_input(out_path=out_path, basename=op_pipeline_name)
            op.save_output(out_path=out_path, basename=op_pipeline_name)

    def vis_profile(
        self, profiling_figs_path: Optional[str] = "algo_ops_profile"
    ) -> None:
        """
        Visualizes timing profiling information about pipeline Ops. Generates stdout output and figures if
        profiling_figs_path is specified. If no profiling data is available, throw a ValueError.

        param profiling_figs_path: Path to where profiling figs should go
        """
        # if pipeline has never been run, no profiling data available so cannot make profile plots
        if len(self.execution_times) == 0:
            raise ValueError(
                "No profiling data available yet since pipeline has not been run."
            )

        # print profiling per op
        print("---Profile---")
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            op.vis_profile(profiling_figs_path=profiling_figs_path)
        print(
            "Total: "
            + self._format_execution_time_stats(
                execution_times=list(self.execution_times)
            )
        )
        print("-------------")

        # make figures for pipeline if needed
        if profiling_figs_path is not None:
            # plot execution time distribution of entire pipeline
            outfile = os.path.join(profiling_figs_path, self.name + ".png")
            plot_op_execution_time_distribution(
                execution_times=list(self.execution_times),
                op_name=self.name,
                outfile=outfile,
            )

            # plot op execution time comparison as violin plot
            outfile = os.path.join(profiling_figs_path, self.name + "_violin.png")
            plot_pipeline_execution_time_distribution(
                op_execution_times={
                    op.name: list(op.execution_times) for op in self.ops.values()
                },
                pipeline_name=self.name,
                outfile=outfile,
            )
