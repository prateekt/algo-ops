import os
from typing import Optional

from algo_ops.ops.op import Op


class TextOp(Op):
    """
    Represents a single text operation that can be executed. Inputs and outputs are both text-based and can be
    printed to stdout and saved to text file.
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

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.input is not None:
            if out_path.endswith(".txt"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + "_input.txt")
                else:
                    outfile = os.path.join(out_path, self.name + "_input.txt")
            with open(outfile, "w") as out_file:
                out_file.write(str(self.input))
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current output to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.output is not None:
            if out_path.endswith(".txt"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + ".txt")
                else:
                    outfile = os.path.join(out_path, self.name + ".txt")
            with open(outfile, "w") as out_file:
                out_file.write(str(self.output))
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")
