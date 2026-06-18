"""Step 6 - analysis toolkit on synthetic inputs with known answers
(spec §6: Jaccard + nulls, Gini, consensus net of chance, partial
correlation, BCa bootstrap)."""

import numpy as np
import pytest

import qsal.config as cfg

rng = np.random.default_rng(cfg.SEED)


# ---------------------------------------------------------------------------
# overlap.py - Jaccard + chance null
# ---------------------------------------------------------------------------


def test_jaccard_known_values():
    from qsal.analysis.overlap import jaccard

    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard({1, 2}, {3, 4}) == 0.0
    assert jaccard({1, 2}, {1, 2}) == 1.0


def test_analytic_null_matches_simulation():
    from qsal.analysis.overlap import expected_jaccard_null

    N, k = 500, 50
    sims = []
    for _ in range(2000):
        a = set(rng.choice(N, k, replace=False))
        b = set(rng.choice(N, k, replace=False))
        sims.append(len(a & b) / len(a | b))
    analytic = expected_jaccard_null(N, k)  # ~ k/(2N-k) for k<<N
    assert abs(analytic - np.mean(sims)) < 0.01


def test_permutation_null_calibrated():
    # random scores -> observed Jaccard should NOT be significant
    from qsal.analysis.overlap import jaccard_vs_null

    s1, s2 = rng.normal(size=400), rng.normal(size=400)
    res = jaccard_vs_null(s1, s2, top_frac=0.1, n_perm=500, seed=cfg.SEED)
    assert res["p_value"] > 0.01  # calibrated: no false positive
    assert 0.2 < res["ratio_over_null"] < 5.0


def test_identical_scores_maximally_significant():
    from qsal.analysis.overlap import jaccard_vs_null

    s = rng.normal(size=400)
    res = jaccard_vs_null(s, s, top_frac=0.1, n_perm=500, seed=cfg.SEED)
    assert res["jaccard"] == 1.0
    assert res["p_value"] <= 1 / 501 + 1e-12
    assert res["ratio_over_null"] > 5


# ---------------------------------------------------------------------------
# concentration.py - Gini / Lorenz / top-k fraction with BCa CIs
# ---------------------------------------------------------------------------


def test_gini_known_values():
    from qsal.analysis.concentration import gini

    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-9)
    x = np.zeros(1000)
    x[0] = 1.0
    assert gini(x) == pytest.approx(1.0, abs=2e-3)  # all mass in one unit


def test_top_frac_share():
    from qsal.analysis.concentration import top_frac_share

    x = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert top_frac_share(x, 0.1) == pytest.approx(10 / 19)


def test_bca_ci_covers_truth():
    from qsal.analysis.concentration import bca_ci

    x = rng.exponential(size=2000)
    lo, hi = bca_ci(x, np.mean, n_boot=500, seed=cfg.SEED)
    assert lo < 1.0 < hi  # true mean = 1
    assert hi - lo < 0.2


# ---------------------------------------------------------------------------
# consensus.py - m-of-5 intersection net of chance
# ---------------------------------------------------------------------------


def test_consensus_net_of_chance():
    from qsal.analysis.consensus import consensus_sets

    N, k = 1000, 100
    ids = np.arange(N)
    # 5 methods: same top-k (perfect consensus)
    same = {m: set(ids[:k]) for m in "abcde"}
    res = consensus_sets(same, N)
    assert res[5]["size"] == k
    assert res[5]["expected_chance"] < 1.0
    assert res[5]["net_size"] == pytest.approx(k, abs=1.0)
    # 5 methods: disjoint top-k (zero consensus)
    disj = {m: set(ids[i * k : (i + 1) * k]) for i, m in enumerate("abcde")}
    res = consensus_sets(disj, N)
    assert res[2]["size"] == 0


# ---------------------------------------------------------------------------
# correlation.py - Spearman, partial Spearman, paired bootstrap
# ---------------------------------------------------------------------------


def test_partial_correlation_removes_confound():
    from qsal.analysis.correlation import partial_spearman

    z = rng.normal(size=3000)  # confound
    x = z + 0.05 * rng.normal(size=3000)
    y = z + 0.05 * rng.normal(size=3000)
    raw = partial_spearman(x, y, controls=None)
    part = partial_spearman(x, y, controls=np.column_stack([z]))
    assert raw > 0.9
    assert abs(part) < 0.25  # x ~ y association is via z only


def test_paired_bootstrap_detects_clear_winner():
    from qsal.analysis.correlation import paired_bootstrap_spearman_diff

    gt = rng.normal(size=800)
    good = gt + 0.3 * rng.normal(size=800)
    bad = gt + 3.0 * rng.normal(size=800)
    res = paired_bootstrap_spearman_diff(good, bad, gt, n_boot=400,
                                         seed=cfg.SEED)
    assert res["diff"] > 0
    assert res["ci_low"] > 0  # CI excludes 0 -> "wins" per RQ5 rule


def test_paired_bootstrap_no_winner_when_equal():
    from qsal.analysis.correlation import paired_bootstrap_spearman_diff

    gt = rng.normal(size=800)
    a = gt + rng.normal(size=800)
    b = gt + rng.normal(size=800)
    res = paired_bootstrap_spearman_diff(a, b, gt, n_boot=400, seed=cfg.SEED)
    assert res["ci_low"] < 0 < res["ci_high"]  # no significant difference
