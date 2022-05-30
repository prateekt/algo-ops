import os
from typing import Optional

from algo_ops.ops.op import Op


class TextOp(Op):
    """
    Represents a single text operation that can be executed. Inputs and outputs can be printed to stdout and saved to
    text file.
    """

    def vis_input(self) -> None:
        """
        Print current input.
        """
        print("Input: " + str(self.input))

    def vis(self) -> None:
        """
        Print current output.
        """
        print(self.name + ": " + str(self.output))

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current input text to file.

        param out_path: Path to where input file should be saved.
        """
        if self.input is not None:
            if out_path.endswith(".txt"):
                outfile = out_path
            else:
                if basename is not None:
                    outfile = os.path.join(out_path, basename + "_input.txt")
                else:
                    outfile = os.path.join(out_path, self.name + "_input.txt")
            with open(outfile, "w") as out_file:
                out_file.write(self.input)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current output to file.

        param out_path: Path to where output file should be saved.
        """
        if self.output is not None:
            if out_path.endswith(".txt"):
                outfile = out_path
            else:
                if basename is not None:
                    outfile = os.path.join(out_path, basename + ".txt")
                else:
                    outfile = os.path.join(out_path, self.name + ".txt")
            with open(outfile, "w") as out_file:
                out_file.write(self.output)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")
