import math
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dbm_to_watt(dbm):
    return 10.0 ** ((np.asarray(dbm, dtype=np.float64) - 30.0) / 10.0)


def db_to_lin(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=np.float64) / 10.0)


def lin_to_db(x, floor_db: float = -300.0):
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-300)), floor_db)


def noise_power_dbm(cfg) -> float:
    return cfg.noise_psd_dbm_hz + 10.0 * math.log10(cfg.rb_bandwidth_hz) + cfg.noise_figure_db


def jain_fairness(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if total <= 0:
        return 0.0
    return float(total ** 2 / (len(x) * np.sum(x ** 2)))


def sample_point_in_circle(radius: float) -> np.ndarray:
    r = radius * math.sqrt(np.random.rand())
    theta = 2 * math.pi * np.random.rand()
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)


def moving_average(arr, window: int = 20) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out
