import os.path
from typing import List, Optional, Dict

import ezplotly as ep
import numpy as np
from ezplotly import EZPlotlyPlot
from matplotlib import pyplot as plt


def pyplot_image(img: np.array, title: str) -> None:
    """
    Helper function to plot image using pyplot.

    param img: Image to plot
    param title: Image title
    """
    plt.imshow(img)
    plt.title(title)


def plot_op_execution_time_distribution(
    execution_times: List[float],
    op_name: str,
    outfile: Optional[str] = None,
) -> None:
    """
    Plot distribution of execution times.

    param execution_times: List of Op execution times
    param op_name: The name of the op
    param outfile: File to write. If None, no file is written.
    """

    # make figs dir
    figs_dir = os.path.dirname(outfile)
    os.makedirs(figs_dir, exist_ok=True)

    # make plot
    tl = "Distribution of " + op_name + " Execution Times"
    fig = ep.hist(data=execution_times, xlabel="Op Execution Time (s)", title=tl)
    ep.plot_all(plots=fig, outfile=outfile)


def plot_pipeline_execution_time_distribution(
    op_execution_times: Dict[str, List[float]],
    pipeline_name: str,
    outfile: Optional[str] = None,
) -> None:
    """
    Plot execution time distribution of ops in pipeline as violin plot.

    param op_execution_times: Dict mapping op_name -> List of function call execution times
    param pipeline_name: The name of the pipeline
    param outfile: File to write. If None, no file is written.
    """

    # make figs dir
    figs_dir = os.path.dirname(outfile)
    os.makedirs(figs_dir, exist_ok=True)

    # make fig
    figs: List[EZPlotlyPlot] = list()
    for op_name in op_execution_times:
        xlabel = op_name
        ylabel = "Execution Time (s)"
        title = pipeline_name + " Op Execution Times"
        op_violin = ep.violin(
            y=op_execution_times[op_name], xlabel=xlabel, ylabel=ylabel, title=title
        )
        figs.append(op_violin)
    ep.plot_all(
        plots=figs,
        panels=[1] * len(figs),
        outfile=outfile,
    )
