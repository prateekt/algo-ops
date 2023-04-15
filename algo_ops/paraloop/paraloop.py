import multiprocessing
from typing import Callable, List, Any, Sequence, Optional


def _sequential(
    func: Callable, params: Sequence[Any], debug: bool = False
) -> List[Any]:
    """
    Sequentially runs a function.

    param func: The function to run
    param params: The parameters to the function
    param debug: Whether to enable debug prints

    return:
        Results of function executions
    """
    results: List[Any] = list()
    for param in params:
        if debug:
            print(param)
        result = func(param)
        results.append(result)
    return results


def _pool(
    func: Callable, params: Sequence[Any], num_cores: Optional[int] = None
) -> List[Any]:
    """
    Implements paraloop using multiprocessing pool.

    param func: The function to parallelize
    param params: Inputs to function
    param num_cores: Number of cores to use

    return:
        Results of function executions
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as p:
        return p.map(func, params)


def loop(
    func: Callable,
    params: Sequence[Any],
    mechanism: str = "pool",
    num_cores: Optional[int] = None,
) -> List[Any]:
    """
    Executes function calls with list of parameters using specified mechanism.

    param func: The function
    param params: The parameters to use
    param mechanism: The mechanism
    param num_cores: Number of cores to use

    return:
        Results of function executions
    """
    if mechanism == "pool":
        return _pool(func=func, params=params, num_cores=num_cores)
    elif mechanism == "sequential":
        return _sequential(func=func, params=params, debug=False)
    elif mechanism == "debug":
        return _sequential(func=func, params=params, debug=True)
    else:
        raise ValueError("Unsupported mechanism: " + str(mechanism))
