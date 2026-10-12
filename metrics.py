import numpy as np
import pandas as pd
from scipy import stats


def roc_auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y, s, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(ece)


def binary_risk_metrics(y, s, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": roc_auc(y, s),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float((pred == y).mean()),
        "brier": float(np.mean((s - y) ** 2)),
        "ece": expected_calibration_error(y, s),
        "positive_rate": float(y.mean()),
        "n_test": int(len(y)),
    }


def mean_ci(x, level: float = 0.95):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    sd = x.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return float(m), float(sd), float(half)


def summarize(df: pd.DataFrame, group_cols, metrics) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_seeds"] = len(g)
        for m in metrics:
            if m in g:
                mean, sd, half = mean_ci(g[m].values)
                row[f"{m}_mean"], row[f"{m}_std"], row[f"{m}_ci95"] = mean, sd, half
        rows.append(row)
    return pd.DataFrame(rows)
