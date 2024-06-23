from typing import Sequence, Callable, TypeVar

T = TypeVar("T")


def make_weighted_selector(pool: Sequence[T], weights: Sequence[float | int]) -> Callable[[], T]:
    """
    Creates a weighted selector that randomly selects an element from a given pool based on the provided weights.

    Args:
        pool (Sequence[T]): A sequence of elements from which to select.
        weights (Sequence[float | int]): A sequence of weights corresponding to each element in the pool.

    Returns:
        Callable[[], T]: A callable that returns a randomly selected element from the pool.

    Example:
        >>> pool = ['apple', 'banana', 'orange']
        >>> weights = [0.3, 0.5, 0.2]
        >>> selector = make_weighted_selector(pool, weights)
        >>> selector()
        'banana'
    """
    from random import uniform

    total_weight = sum(weights)
    norm_weights = tuple(weight / total_weight for weight in weights)
    pack = list(zip(pool, norm_weights))

    def _selector() -> T:
        # 计算权重总和并检查是否有负权重

        # 生成一个随机数用于选择
        rand_num = uniform(0, 1)
        cum_weight = 0.0

        # 遍历归一化后的权重，累加权重直到超过随机数，从而确定选择的索引
        for selected, weight in pack:
            cum_weight += weight
            if rand_num < cum_weight:
                return selected

    return _selector
