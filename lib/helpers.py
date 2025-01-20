from functools import lru_cache
from typing import List


@lru_cache
def factorial(n):
    res = 1
    for i in range(1, n + 1):
        res *= i

    return res


def generate_data(start: float, end: float, step: float) -> List[float]:
    """

    Generates consecutive values in the specified range with specified step.

    Parameters:
        start (float): start of the range
        end (float): end of the range
        step (float): absolute difference between each consecutive pair of values

    Returns:
        List[float]: genetated values

    """
    result = []
    x = start
    while x <= end:
        result.append(x)
        x += step

    return result
