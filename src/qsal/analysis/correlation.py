"""Rank correlations: Spearman, partial Spearman (rank-residual method),
paired bootstrap for metric-vs-metric differences (the design spec, RQ5)."""

import numpy as np
from scipy.stats import rankdata, spearmanr


def _rank(x):
    return rankdata(np.asarray(x))


def partial_spearman(x, y, controls=None) -> float:
    """Spearman of x,y after regressing the controls out of the ranks."""
    if controls is None:
        return float(spearmanr(x, y).statistic)
    rx, ry = _rank(x), _rank(y)
    C = np.column_stack([_rank(controls[:, j]) for j in range(controls.shape[1])])
    A = np.column_stack([np.ones(len(rx)), C])
    res_x = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def paired_bootstrap_spearman_diff(
    a, b, gt, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict:
    """CI for rho(a, gt) - rho(b, gt) by joint resampling. A metric "wins"
    only if the CI excludes 0 (pre-registered RQ5 rule)."""
    a, b, gt = map(np.asarray, (a, b, gt))
    n = len(gt)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            spearmanr(a[idx], gt[idx]).statistic
            - spearmanr(b[idx], gt[idx]).statistic
        )
    alpha = (1 - level) / 2
    return {
        "diff": float(spearmanr(a, gt).statistic - spearmanr(b, gt).statistic),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1 - alpha)),
    }
