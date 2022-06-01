import os
from typing import Union, Optional, Callable

import cv2
import numpy as np

import algo_ops.plot.settings as plotting_settings
from algo_ops.ops.op import Op
from algo_ops.plot.plot import pyplot_image

"""
CVOps is infrastructure to build an OpenCV pipeline as a list of
feed-forward ops that support op-level visualization tools and debugging.
"""


class CVOp(Op):
    """
    Represents a single computer vision operation that can be executed.
    Inputs and outputs can be visualized as images.
    """

    def __init__(self, func: Callable):
        super().__init__(func=func)

    def vis_input(self) -> None:
        """
        Plot current input image using pyplot (jupyter compatible)
        """
        if self.input is None:
            raise ValueError("There is no input to be visualized.")
        if plotting_settings.SUPPRESS_PLOTS:
            print("Plot of input suppressed: " + str(self.name))
        else:
            pyplot_image(img=self.input, title=self.name)

    def vis(self) -> None:
        """
        Plot current output image using pyplot (jupyter compatible)
        """
        if self.output is None:
            raise ValueError("There is no output to be visualized.")
        if plotting_settings.SUPPRESS_PLOTS:
            print("Plot of output suppressed: " + str(self.name))
        else:
            pyplot_image(img=self.output, title=self.name)

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current input image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.input is not None:
            if out_path.endswith(".png"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + "_input.png")
                else:
                    outfile = os.path.join(out_path, self.name + "_input.png")
            cv2.imwrite(outfile, self.input)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current output image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.output is not None:
            if out_path.endswith(".png"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + ".png")
                else:
                    outfile = os.path.join(out_path, self.name + ".png")
            cv2.imwrite(outfile, self.output)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def exec(self, inp: Union[np.array, str]) -> np.array:
        """
        A CV op takes in either the file name of an image or an image,
        performs an operation on the image, and returns as new image.

        param inp: The input

        return
            output: The result of the operation
        """
        if isinstance(inp, str):
            inp = cv2.imread(filename=inp)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        if not isinstance(inp, np.ndarray):
            raise ValueError("Unsupported Input: " + str(inp))
        return super().exec(inp=inp)
