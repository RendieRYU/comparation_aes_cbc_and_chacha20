import time
from typing import Callable, Any, Tuple


def measure_time(func: Callable, *args, repeats: int = 1, **kwargs) -> Tuple[float, Any]:
    """
    Measure execution time of a function. Returns (avg_seconds, last_result).
    """
    result = None
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = (end - start) / max(1, repeats)
    return elapsed, result
