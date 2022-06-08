import os
from typing import Union, Optional, Callable

import cv2
import ezplotly.settings as plotting_settings
import numpy as np

from algo_ops.ops.op import Op
from algo_ops.plot.plot import pyplot_image

"""
CVOps is infrastructure to build an OpenCV pipeline as a list of
feed-forward ops that support op-level visualization tools and debugging.
"""


class ImageResult:
    """
    Represents an image processing result.
    """

    def __init__(self, img: np.array, file_path: Optional[str] = None):
        """
        param img: The numpy image matrix
        param file_path: Path to image file (if any)
        """
        self.img = img
        self.file_path = file_path

    def plot(self, title: str) -> None:
        """
        Plot image in pyplot.

        param title: Title of image
        """
        if plotting_settings.SUPPRESS_PLOTS:
            print("Plotting of plot is suppressed " + str(title) + ".")
        else:
            pyplot_image(img=self.img, title=title)

    def save(self, out_path: str = ".", basename: Optional[str] = None):
        """
        Saves current input image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if out_path.endswith(".png"):
            outfile = out_path
        else:
            os.makedirs(out_path, exist_ok=True)
            outfile = os.path.join(out_path, basename + ".png")
        cv2.imwrite(outfile, self.img)

    def __str__(self) -> str:
        """
        return:
            str representation
        """
        return str(self.file_path)


class CVOp(Op):
    """
    Represents a single computer vision operation that can be executed. Both the input and output of a CVOp are
    images.
    """

    def __init__(self, func: Callable):
        """
        Initialize CVOp.

        param func: The function executed when the CVOp is run on an input image to produce an output image
        """
        super().__init__(func=func)
        self.input: Optional[ImageResult] = None
        self.output: Optional[ImageResult] = None

    @staticmethod
    def parse_input(inp: Union[str, np.array, ImageResult]) -> ImageResult:
        """
        Helper function to parse input and format as ImageResult.

        param inp: Either a path to an image file, a numpy image matrix, or an ImageResult object.

        return:
            ImageResult Object
        """
        if isinstance(inp, str):
            inp_file_path = inp
            inp_img = cv2.imread(filename=inp_file_path)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            input_img_result = ImageResult(img=inp_img, file_path=inp_file_path)
        elif isinstance(inp, np.ndarray):
            input_img_result = ImageResult(img=inp, file_path=None)
        elif isinstance(inp, ImageResult):
            input_img_result = inp
        else:
            raise ValueError("Unsupported input: " + str(inp))
        return input_img_result

    def exec(self, inp: Union[str, np.array, ImageResult]) -> ImageResult:
        """
        A CV operation takes in an image, performs an operation on the image, and returns a new image.

        param inp: Either a path to an image file, a numpy image matrix, or an ImageResult object.

        return
            output: The output image of the operation's execution, wrapped as an ImageResult.
        """

        # parse input into ImageResult object
        self.input = self.parse_input(inp=inp)

        # run op's function on input image to obtain output image
        input_img_result = self.input
        output_img = super().exec(inp=self.input.img)
        self.input = input_img_result

        # return output wrapped as ImageResult
        self.output = ImageResult(img=output_img, file_path=self.input.file_path)
        return self.output

    def vis_input(self) -> None:
        """
        Plot current input image using pyplot (jupyter compatible)
        """
        if self.input is None:
            raise ValueError("There is no input to be visualized.")
        else:
            self.input.plot(title=self.name)

    def vis(self) -> None:
        """
        Plot current output image using pyplot (jupyter compatible)
        """
        if self.output is None:
            raise ValueError("There is no output to be visualized.")
        else:
            self.output.plot(title=self.name)

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current input image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.input is not None:
            if basename is None:
                basename = self.name
            basename += "_input"
            self.input.save(out_path=out_path, basename=basename)
        else:
            raise ValueError("There is no input to be saved.")

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current output image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.output is not None:
            if basename is None:
                basename = self.name
            self.output.save(out_path=out_path, basename=basename)
        else:
            raise ValueError("There is no output to be saved.")
