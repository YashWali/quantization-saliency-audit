"""Stratified (within-layer) permutation nulls + multiple-comparison
helpers (spec §5 nullmodels).

overlap.jaccard_vs_null permutes a whole score vector; the spec's null is
WITHIN-LAYER (channels are exchangeable only within a stratum - depth and
module shift both scale and meaning of every score). stratified_jaccard_
vs_null permutes scores_b within each stratum, so the null preserves the
per-stratum composition of both top sets.
"""

import numpy as np

from qsal.analysis.overlap import expected_jaccard_null, jaccard


def _strata_index(strata, top_frac):
    """Per-stratum integer index arrays (in np.unique order) and each
    stratum's top-k count. Computed ONCE so the permutation loop never
    re-derives np.unique / boolean masks (the dominant cost on large,
    string-keyed strata)."""
    idx_list = [np.nonzero(strata == s)[0] for s in np.unique(strata)]
    ks = [max(1, int(round(top_frac * len(idx)))) for idx in idx_list]
    return idx_list, ks


def _top_ids_per_stratum(scores, idx_list, ks):
    """Union over strata of each stratum's top-k channel indices, using
    precomputed per-stratum index arrays."""
    out = set()
    for idx, k in zip(idx_list, ks):
        order = np.argsort(scores[idx])[::-1][:k]
        out.update(idx[order].tolist())
    return out


def stratified_jaccard_vs_null(
    scores_a,
    scores_b,
    strata,
    top_frac: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Observed within-stratum top-k Jaccard of two metrics, against a
    null that permutes scores_b WITHIN each stratum."""
    scores_a, scores_b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    strata = np.asarray(strata)
    assert len(scores_a) == len(scores_b) == len(strata)

    # Precompute stratum structure once; the permutation order below iterates
    # idx_list in a fixed np.unique order, so the RNG sequence - and thus the
    # null distribution - is unaffected by this precompute.
    idx_list, ks = _strata_index(strata, top_frac)
    sa = _top_ids_per_stratum(scores_a, idx_list, ks)
    sb = _top_ids_per_stratum(scores_b, idx_list, ks)
    obs = jaccard(sa, sb)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = scores_b.copy()
        for idx in idx_list:
            perm[idx] = rng.permutation(perm[idx])
        null[i] = jaccard(sa, _top_ids_per_stratum(perm, idx_list, ks))

    k = len(sa)
    analytic = expected_jaccard_null(len(scores_a), k)
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


def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm step-down correction (cfg.MULTIPLE_COMPARISON).

    Returns adjusted p-values (monotone, clipped at 1) and the reject
    mask at level alpha."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    raw = (m - np.arange(m)) * p[order]  # (m-i+1) * p_(i), 1-indexed
    adj_sorted = np.minimum(np.maximum.accumulate(raw), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return {"adjusted": adjusted, "reject": adjusted <= alpha}
