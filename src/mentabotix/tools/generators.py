from enum import Enum
from typing import Callable

from numpy.random import choice


class NameGenerator:
    """
    Class for generating unique names.

    Attributes:
        basename: the base name
        counter: the counter
    """

    @property
    def basename(self) -> str:
        return self._basename

    @property
    def counter(self) -> int:
        return self._counter

    def __init__(self, basename: str):
        self._basename = basename
        self._counter = 0

    def __call__(self) -> str:
        self._counter += 1
        return f"{self._basename}{self._counter}"


class Multipliers(Enum):
    """
    Enum class for different multipliers.
    Notes:
        FLT: for float
        SRK: for shrink
        ENLG: for enlarge
    Attributes:
        FLT_UPPER: float around 1.0 but slightly higher
        FLT_MIDDLE: float exactly around 1.0
        FLT_LOWER: float around 1.0 but slightly lower

        SRK_L: slightly shrink
        SRK_LL: moderate shrink
        SRK_LLL: significantly shrink

        ENLG_L: slightly enlarge
        ENLG_LL: moderate enlarge
        ENLG_LLL: significantly enlarge
    """

    FLT_UPPER = (0.9, 0.925, 0.95, 1.0, 1.05, 1.08, 1.1, 1.17, 1.25)
    FLT_MIDDLE = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.17, 1.25)
    FLT_LOWER = (0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 1.0, 1.05, 1.08, 1.1)

    SRK_L = (0.65, 0.7, 0.725, 0.7375, 0.75, 0.7625, 0.775, 0.7875, 0.8)
    SRK_LL = (0.55, 0.6, 0.625, 0.65, 0.6625, 0.675, 0.6875, 0.7)
    SRK_LLL = (0.40, 0.5, 0.525, 0.55, 0.5625, 0.575, 0.5875, 0.6)

    ENLG_L = (1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4, 1.45, 1.5)
    ENLG_LL = (1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.65, 1.7, 1.75)
    ENLG_LLL = (1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.95, 2.0, 2.1, 2.2, 2.4)


def make_multiplier_generator(mult: "Multipliers") -> Callable[[], float]:
    """
    Generates a function that returns a random float value from the given multiplier sequence.

    Parameters:
        mult (Multipliers): The multiplier enum value.

    Returns:
        Callable[[], float]: A function that returns a random float value from the multiplier sequence.
    """
    seq = mult.value

    def _make_multiplier() -> float:
        return choice(seq)

    return _make_multiplier
