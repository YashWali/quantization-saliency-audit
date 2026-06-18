"""Top-k set overlap vs chance (the design spec, spec §5 nullmodels).

Raw Jaccard is never reported alone [guard]: every observed value comes
with the analytic E[J] = k/(2N-k) chance baseline and a permutation
null (ratio + p-value).
"""

import numpy as np


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def expected_jaccard_null(N: int, k: int) -> float:
    """E[Jaccard] of two independent uniform k-subsets of N (k << N)."""
    return k / (2 * N - k)


def top_set(scores: np.ndarray, top_frac: float) -> set:
    k = max(1, int(round(top_frac * len(scores))))
    return set(np.argsort(scores)[::-1][:k].tolist())


def jaccard_vs_null(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    top_frac: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Observed top-k Jaccard with permutation null and analytic baseline."""
    assert len(scores_a) == len(scores_b)
    N = len(scores_a)
    k = max(1, int(round(top_frac * N)))
    sa = top_set(np.asarray(scores_a), top_frac)
    sb = top_set(np.asarray(scores_b), top_frac)
    obs = jaccard(sa, sb)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    b = np.asarray(scores_b)
    for i in range(n_perm):
        null[i] = jaccard(sa, top_set(rng.permutation(b), top_frac))
    analytic = expected_jaccard_null(N, k)
    denom = null.mean() if null.mean() > 0 else analytic
    return {
        "jaccard": obs,
        "k": k,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "analytic_null": analytic,
        "ratio_over_null": obs / denom,
        "z_over_null": float((obs - null.mean()) / (null.std() + 1e-12)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
    }
