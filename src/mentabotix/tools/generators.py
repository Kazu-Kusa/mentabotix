class NameGenerator:
    def __init__(self, basename: str):
        self._basename = basename
        self._counter = 0

    def __call__(self) -> str:
        self._counter += 1
        return f"{self._basename}{self._counter}"
