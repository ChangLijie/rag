from typing import Union


class Instruction:
    def __init__(self) -> None:
        self.instruction = []

    def add_new(self, command: str) -> None:
        """Add new command

        Args:
            command (str): command
        """
        self.instruction.append(command)

    def delete(
        self, idx: Union[int, None] = None, command: Union[str, None] = None
    ) -> None:
        """Delete commands.

        Args:
            idx (Union[int, None], optional): The idx of command. Defaults to None.
            command (Union[str, None], optional): command. Defaults to None.

        Raises:
            ValueError: All parameter is None!
            ValueError: Choose one parameter to delete!
            Exception: Some error happen!
        """
        if (idx is None) and (command is None):
            raise ValueError("All parameter is None!")
        elif (idx is not None) and (command is not None):
            raise ValueError("Choose one parameter to delete!")
        elif (idx is not None) and (command is None):
            self.instruction.pop(idx)
        elif (idx is None) and (command is not None):
            self.instruction.remove(command)
        else:
            raise Exception("Some error happen!")

    def get(self) -> list:
        """Return memory.

        Returns:
            list: memory.
        """
        return self.instruction
