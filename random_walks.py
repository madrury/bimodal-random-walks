from typing import List, Tuple

import numpy as np


def gaussian_random_walk(n: int) -> np.array:
    dx = np.random.normal(scale=2.0, size=n)
    return dx.cumsum()

def gaussian_random_walk_2d(shape: Tuple[int, int]) -> np.array:
    dx = np.random.normal(scale=2.0, size=shape)
    return dx.cumsum(axis=1)

def bernoulli_random_walk(n: int) -> np.array:
    dx = np.random.choice([-1.0, 1.0], size=n)
    return dx.cumsum()

def bernoulli_random_walk_2d(shape: Tuple[int, int]) -> np.array:
    dx = np.random.choice([-1.0, 1.0], size=shape)
    return dx.cumsum(axis=1)

def discrete_random_walk(n: int, choices: List[int]=[-1.0, 0.0, 1.0], ps: List[float]=[1/3, 1/3, 1/3]) -> np.array:
    dx = np.random.choice(choices, size=n, p=ps)
    return dx.cumsum()


def moving_window_2d(x: np.array, window: int) -> np.array:
    """
    For example:
       > moving_window_2d([1, 2, 3, 4, 5, 6], window=3)
       [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    """   
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(x.size - window + 1, window),
        strides=(x.strides[0], x.strides[0]))


def calculate_percent_bollinger_band(walk: np.array, bandwidth: int=20) -> np.array:
    windowed = moving_window_2d(walk, bandwidth)
    window_means, window_sds = windowed.mean(axis=1), windowed.std(axis=1)
    lower, upper = window_means - 2 * window_sds, window_means + 2 * window_sds
    return (walk[bandwidth-1:] - lower) / (upper - lower)

def calculate_percent_bollinger_band_2d(walks: np.array, bandwidth: int=20) -> np.array:
    means, sds = walks.mean(axis=1), walks.std(axis=1)
    lower, upper = means - 2 * sds, means + 2 * sds
    return (walks[:, -1] - lower) / (upper - lower)