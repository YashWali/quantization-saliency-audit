"""Concentration: Gini / Lorenz / top-k share with BCa CIs (the design spec)."""

import numpy as np
from scipy.stats import bootstrap


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * x).sum() / (n * x.sum()))


def lorenz(x: np.ndarray) -> np.ndarray:
    """Cumulative share of total, ascending (prepended 0)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    return np.concatenate([[0.0], np.cumsum(x) / x.sum()])


def top_frac_share(x: np.ndarray, frac: float) -> float:
    """Share of summed mass held by the top `frac` of units."""
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(round(frac * len(x))))
    top = np.sort(x)[::-1][:k]
    return float(top.sum() / x.sum())


def bca_ci(x, stat, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """BCa bootstrap CI for stat(x) (stat must accept an axis kwarg)."""
    res = bootstrap(
        (np.asarray(x),), stat, n_resamples=n_boot, confidence_level=level,
        method="BCa", random_state=np.random.default_rng(seed),
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def overprotection(gt_in_set, top_frac: float = 0.20,
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """RQ4: GT concentration inside one method's selected set.

    Share of the set's summed GT held by its top `top_frac` channels (by
    GT), with a BCa CI over channel resampling; compared against the
    pre-registered cfg.THRESHOLDS rq4_* rules in analysis."""
    x = np.asarray(gt_in_set, dtype=np.float64)

    def stat(sample, axis=-1):
        s = np.asarray(sample, dtype=np.float64)
        if s.ndim == 1:
            return top_frac_share(s, top_frac)
        return np.apply_along_axis(
            lambda v: top_frac_share(v, top_frac), axis, s)

    lo, hi = bca_ci(x, stat, n_boot=n_boot, seed=seed)
    return {"share": top_frac_share(x, top_frac), "ci_low": lo,
            "ci_high": hi, "top_frac": top_frac, "n": len(x)}
