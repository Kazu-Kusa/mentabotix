from random import choice
from typing import Tuple


# region Multiplier Generator
FLOAT_SET_UPPER = (0.9, 0.925, 0.95, 1.0, 1.05, 1.08, 1.1, 1.17, 1.25)

FLOAT_SET_LOWER = (0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 1.0, 1.05, 1.08, 1.1)

FLOAT_SET_MIDDLE = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.17, 1.25)

SHRINK_SET_LLL = (0.40, 0.5, 0.525, 0.55, 0.5625, 0.575, 0.5875, 0.6)

SHRINK_SET_LL = (0.55, 0.6, 0.625, 0.65, 0.6625, 0.675, 0.6875, 0.7)

SHRINK_SET_L = (0.65, 0.7, 0.725, 0.7375, 0.75, 0.7625, 0.775, 0.7875, 0.8)

ENLARGE_SET_L = (1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4, 1.45, 1.5)
ENLARGE_SET_LL = (1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.65, 1.7, 1.75)

ENLARGE_SET_LLL = (1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.95, 2.0, 2.1, 2.2, 2.4)


def enlarge_multiplier_l() -> float:
    """
    Generate a multiplier for enlargement in a lower range.

    Returns:
        float: The randomly generated multiplier for enlargement in a lower range.
    """
    return choice(ENLARGE_SET_L)


def enlarge_multiplier_ll() -> float:
    """
    Generate a multiplier for enlargement in a middle range.

    Returns:
        float: The randomly generated multiplier for enlargement in a middle range.
    """
    return choice(ENLARGE_SET_LL)


def enlarge_multiplier_lll() -> float:
    """
    Generate a multiplier for enlargement in an upper range.

    Returns:
        float: The randomly generated multiplier for enlargement in an upper range.
    """
    return choice(ENLARGE_SET_LLL)


def shrink_multiplier_l() -> float:
    """
    Generate a multiplier for shrinkage in a lower range.

    Returns:
        float: The randomly generated multiplier for shrinkage in a lower range.
    """
    return choice(SHRINK_SET_L)


def shrink_multiplier_ll() -> float:
    """
    Generate a multiplier for shrinkage in a middle range.

    Returns:
        float: The randomly generated multiplier for shrinkage in a middle range.
    """
    return choice(SHRINK_SET_LL)


def shrink_multiplier_lll() -> float:
    """
    Generate a multiplier for shrinkage in the upper range.

    Returns:
        float: The randomly generated multiplier for shrinkage in the upper range.
    """
    return choice(SHRINK_SET_LLL)


def float_multiplier_middle() -> float:
    """
    Generate a float multiplier in a middle range.

    Returns:
        float: The randomly generated float multiplier in a middle range.
    """
    return choice(FLOAT_SET_MIDDLE)


def float_multiplier_lower() -> float:
    """
    Generate a float multiplier in a lower range.

    Returns:
        float: The randomly generated float multiplier in a lower range.
    """
    return choice(FLOAT_SET_LOWER)


def float_multiplier_upper() -> float:
    """
    Generate a float multiplier in an upper range.
    Returns:
        float: The randomly generated float multiplier in the upper range.
    """
    return choice(FLOAT_SET_UPPER)


# endregion
Location = Tuple[int | float, int | float]


def calc_p2p_dst(point_1: Location, point_2: Location) -> float:
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5


def calc_p2p_error(start: Location, target: Location) -> float | int:
    """
    drik distance
    Args:
        start:
        target:

    Returns:

    """
    return abs(target[0] - start[0]) + abs(target[1] - start[0])
