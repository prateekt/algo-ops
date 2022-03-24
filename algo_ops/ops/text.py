from algo_ops.ops.op import Op


class TextOp(Op):
    """
    Represents a single text processing operation that can be executed.
    Inputs and outputs can be printed to screen and saved.
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

    def save_input(self, out_path: str) -> None:
        """
        Saves current input text to file.

        param out_path: Path to where input file should be saved.
        """
        if self.input is not None:
            with open(out_path, "w") as out_file:
                out_file.write(self.input)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str) -> None:
        """
        Saves current output to file.

        param out_path: Path to where output file should be saved.
        """
        if self.output is not None:
            with open(out_path, "w") as out_file:
                out_file.write(self.output)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")
