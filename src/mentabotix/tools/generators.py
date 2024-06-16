class NameGenerator:
    """
    Class for generating unique names.

    Properties:
        basename: the base name
        counter: the counter
    """

    @property
    def basename(self) -> str:
        """

        Returns:
            str: The base name.

        """
        return self._basename

    @property
    def counter(self) -> int:
        """

        Returns:
            int: The counter.

        """
        return self._counter

    def __init__(self, basename: str):
        self._basename = basename
        self._counter = 0

    def __call__(self) -> str:
        self._counter += 1
        return f"{self._basename}{self._counter}"
