import numpy as np
from typing import Callable

def R_mean(transition_function : Callable, N : float, params : dict, lambdas : list, mean_func = lambda n: 1) -> float:
    n_array = np.arange(1,N + 1)
    f_array = mean_func(n_array)
    return np.sum(R(transition_function, N, params, lambdas) * f_array)

def Rn_mean(transition_function : Callable, N : float, params : dict, lambdas : list) -> float:
    n_array = np.arange(1,N + 1)

    return np.sum(R(transition_function, N, params, lambdas) * n_array)

def Rn2_mean(transition_function : Callable, N : float, params : dict, lambdas : list) -> float:
    n_array = np.arange(1,N + 1)**2
    return np.sum(R(transition_function, N, params, lambdas) * n_array)

def R(transition_function : Callable, N : float, params : dict, lambdas : list) -> np.array:
    n_array = np.arange(1,N + 1)

    return np.exp(lambdas[0] * n_array + lambdas[1] * transition_function(n_array, N, params))
    