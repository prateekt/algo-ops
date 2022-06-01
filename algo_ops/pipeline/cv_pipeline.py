import math
from typing import List, Callable

from matplotlib import pyplot as plt

import algo_ops.plot.settings as plotting_settings
from algo_ops.ops.cv import CVOp
from algo_ops.ops.op import Op
from algo_ops.pipeline.pipeline import Pipeline


class CVPipeline(Pipeline):
    """
    Implementation of an OpenCV Image Processing pipeline in the algo_ops framework that allows
    auto-dashboarding of pipeline steps.
    """

    def __init__(self, ops: List[CVOp]):
        super().__init__(ops=ops)

    @classmethod
    def init_from_funcs(cls, funcs: List[Callable], op_class=CVOp) -> "CVPipeline":
        """
        param funcs: List of pipeline functions that execute serially as operations in pipeline.
        param op_class: The subclass of Op that the pipeline uses
        param profiling_figs_path: The profiling figs path
        """
        assert op_class is CVOp, "Cannot use non-CVOp in CVPipeline."
        ops: List[CVOp] = [op_class(func=func) for func in funcs]
        return cls(ops=ops)

    def vis(
        self, num_cols: int = 4, fig_width: int = 15, fig_height: int = 6, dpi: int = 80
    ) -> None:
        """
        Plot current output images of each Op using pyplot (jupyter compatible). Defaults optimize for Jupyter
        notebook plotting. Throws ValueError if no data has been input to pipeline yet.

        param num_cols: Number of image columns to display
        param fig_width: Total width of figure
        param fig_height: Total height of figure
        param dpi: DPI of figure
        """

        # validate inputs
        if self.input is None:
            raise ValueError(
                "Cannot visualize pipeline if no input data has been run through."
            )
        if plotting_settings.SUPPRESS_PLOTS:
            print("Plot of pipeline " + str(self.name) + " suppressed.")
            return

        # make plot of pipeline CVOps data flow
        num_rows = math.ceil((len(self.ops.keys()) + 1) / num_cols)
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        plt_num = 1
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                plt.subplot(num_rows, num_cols, plt_num)
                op.vis_input()
                plt.title(str(self.name) + " Input")
                plt_num += 1
            plt.subplot(num_rows, num_cols, plt_num)
            plt_num += 1
            op.vis()
        plt.show()
