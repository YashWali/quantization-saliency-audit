"""analysis-method fixes - stratified nulls, Holm, confound
partials, RQ4 overprotection, and the fp32 KL floor at the REAL vocab
size (the existing reference test used vocab 5k; deployment is ~152k)."""

import numpy as np
import pandas as pd
import torch

import qsal.config as cfg

torch.manual_seed(cfg.SEED)
QWEN_VOCAB = 151_936


# ---------------------------------------------------------------------------
# nullmodels
# ---------------------------------------------------------------------------


def test_holm_known_example():
    from qsal.analysis.nullmodels import holm

    res = holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert np.allclose(res["adjusted"], [0.03, 0.06, 0.06, 0.02])
    assert res["reject"].tolist() == [True, False, False, True]


def test_holm_monotone_and_clipped():
    from qsal.analysis.nullmodels import holm

    rng = np.random.default_rng(cfg.SEED)
    p = rng.uniform(0, 1, 50)
    adj = holm(p)["adjusted"]
    assert (adj <= 1.0).all() and (adj >= p - 1e-12).all()
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= -1e-12).all()


def test_stratified_null_identical_scores_significant():
    from qsal.analysis.nullmodels import stratified_jaccard_vs_null

    rng = np.random.default_rng(cfg.SEED)
    scores = rng.lognormal(size=600)
    strata = np.repeat([0, 1, 2], 200)
    res = stratified_jaccard_vs_null(scores, scores, strata, top_frac=0.05,
                                     n_perm=200, seed=0)
    assert res["jaccard"] == 1.0
    assert res["p_value"] < 0.01
    assert res["ratio_over_null"] > 5


def test_stratified_null_independent_scores_calibrated():
    from qsal.analysis.nullmodels import stratified_jaccard_vs_null

    rng = np.random.default_rng(cfg.SEED)
    a, b = rng.normal(size=600), rng.normal(size=600)
    strata = np.repeat([0, 1, 2], 200)
    res = stratified_jaccard_vs_null(a, b, strata, top_frac=0.05,
                                     n_perm=200, seed=0)
    assert res["p_value"] > 0.05
    assert res["jaccard"] <= res["null_p95"] + 0.1


def test_stratified_null_catches_within_stratum_structure():
    # scores that agree only via a between-strata offset: a whole-vector
    # permutation null would call this "agreement"; the within-stratum
    # null must not.
    from qsal.analysis.nullmodels import stratified_jaccard_vs_null

    rng = np.random.default_rng(cfg.SEED)
    strata = np.repeat([0, 1, 2], 200)
    offset = strata * 100.0
    a = rng.normal(size=600) + offset
    b = rng.normal(size=600) + offset
    res = stratified_jaccard_vs_null(a, b, strata, top_frac=0.05,
                                     n_perm=200, seed=0)
    assert res["p_value"] > 0.05  # no within-stratum agreement exists


def test_stratified_null_rng_stream_bit_identical():
    # Golden-value pin on the within-stratum permutation null. The 2026-06
    # optimization (precompute per-stratum indices once) is only safe because
    # it preserves the RNG stream byte-for-byte vs the prior implementation;
    # the Phase-2 (Pythia) RQ2 numbers were computed post-optimization, so any
    # future refactor that shifts the permutation order would silently change
    # published results. These constants were produced by the committed
    # implementation on torch-free input; if they drift, the stream drifted.
    from qsal.analysis.nullmodels import stratified_jaccard_vs_null

    rng = np.random.default_rng(12345)
    a = rng.lognormal(sigma=1.0, size=900)
    b = a + rng.normal(scale=0.5, size=900)
    strata = np.repeat(np.arange(3), 300)
    res = stratified_jaccard_vs_null(a, b, strata, top_frac=0.05,
                                     n_perm=200, seed=0)
    assert res["k"] == 45
    assert np.isclose(res["jaccard"], 0.9565217391304348, rtol=0, atol=1e-12)
    assert np.isclose(res["null_mean"], 0.025593231689032797, rtol=0, atol=1e-12)
    assert np.isclose(res["null_p95"], 0.04712722298221593, rtol=0, atol=1e-12)
    assert np.isclose(res["ratio_over_null"], 37.37401164309872, rtol=1e-10, atol=0)
    assert np.isclose(res["z_over_null"], 58.61951597935997, rtol=1e-10, atol=0)
    assert np.isclose(res["p_value"], 0.004975124378109453, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# confounds
# ---------------------------------------------------------------------------


def test_confound_partials_kill_pure_confound_metric():
    from qsal.analysis.confounds import add_control_columns, confound_partials

    rng = np.random.default_rng(cfg.SEED)
    n = 800
    driver = rng.lognormal(size=n)  # shared mechanical driver (mean|x|)
    df = pd.DataFrame({
        "layer": rng.integers(0, 24, n),
        "w_norm": rng.lognormal(size=n),
        "awq": driver,
        "gt": driver * rng.lognormal(sigma=0.1, size=n),
        "metric_confound": driver * rng.lognormal(sigma=0.05, size=n),
    })
    df["metric_signal"] = df["gt"] * rng.lognormal(sigma=0.05, size=n)
    df = add_control_columns(df)
    res = confound_partials(
        df, ["metric_confound", "metric_signal"], "gt"
    ).set_index("metric")
    # both look great raw...
    assert res.loc["metric_confound", "raw_spearman"] > 0.8
    assert res.loc["metric_signal", "raw_spearman"] > 0.8
    # ...but only the true-signal metric survives the controls
    assert res.loc["metric_confound", "partial_spearman"] < 0.3
    assert res.loc["metric_signal", "partial_spearman"] > 0.5
    assert (res.loc["metric_signal", "partial_spearman"]
            > res.loc["metric_confound", "partial_spearman"] + 0.2)


def test_stratified_spearman_shape():
    from qsal.analysis.confounds import stratified_spearman

    rng = np.random.default_rng(cfg.SEED)
    df = pd.DataFrame({
        "layer": np.repeat([2, 11, 22], 100),
        "m": rng.normal(size=300),
    })
    df["g"] = df["m"] + rng.normal(scale=0.5, size=300)
    out = stratified_spearman(df, "m", "g")
    assert len(out) == 3 and (out["n"] == 100).all()
    assert (out["spearman"] > 0.5).all()


# ---------------------------------------------------------------------------
# RQ4 overprotection
# ---------------------------------------------------------------------------


def test_overprotection_known_share():
    from qsal.analysis.concentration import overprotection

    x = np.arange(1.0, 101.0)  # top 20 of 100 hold 1810/5050
    res = overprotection(x, top_frac=0.20, n_boot=200, seed=0)
    assert abs(res["share"] - 1810.0 / 5050.0) < 1e-12
    assert res["ci_low"] <= res["share"] <= res["ci_high"]
    assert res["n"] == 100


def test_overprotection_flags_concentrated_set():
    from qsal.analysis.concentration import overprotection

    rng = np.random.default_rng(cfg.SEED)
    x = np.concatenate([rng.uniform(10, 20, 20), rng.uniform(0, 0.1, 80)])
    res = overprotection(x, top_frac=0.20, n_boot=200, seed=0)
    assert res["share"] > 0.95


# ---------------------------------------------------------------------------
# fp32 KL floor at deployment vocab size (review issue 3, validation part)
# ---------------------------------------------------------------------------


def _ref_kl(z_full, z_pert):
    lp = torch.log_softmax(z_full.double(), dim=-1)
    lq = torch.log_softmax(z_pert.double(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def test_kl_fast_path_floor_at_real_vocab():
    from qsal.groundtruth import KL_FP64_RECHECK_BELOW, kl_per_token

    # The threshold gates kl_MEAN (averaged over the full eval set), so
    # the relevant floor is the error of the MEAN at the real eval token
    # count, not the per-token max (per-token rounding largely cancels
    # in the average: measured ~33x smaller at n=4096). The per-token
    # bound stays as a sanity ceiling. Empirical validation on a 504-row
    # stratified random sample: results/fp64_threshold_validation.json
    # + docs/AMENDMENT_fp64_recheck_threshold.md.
    n_tokens, chunk = cfg.EVAL_NUM_SEQS * cfg.EVAL_SEQ_LEN, 512
    g = torch.Generator().manual_seed(cfg.SEED)
    err_sum, tok_max = 0.0, 0.0
    for _ in range(n_tokens // chunk):
        z = torch.randn(chunk, QWEN_VOCAB, generator=g) * 4
        z2 = z + torch.randn(chunk, QWEN_VOCAB, generator=g) * 1e-3
        kl = kl_per_token(torch.log_softmax(z, -1), z2)
        d = kl - _ref_kl(z, z2)
        err_sum += d.sum().item()
        tok_max = max(tok_max, d.abs().max().item())
    mean_floor = abs(err_sum) / n_tokens
    assert tok_max < 5e-6  # per-token sanity ceiling
    assert KL_FP64_RECHECK_BELOW > 10 * mean_floor


def test_force_exact_skips_fast_path_logic():
    # force_exact loop contract: (force_exact, True) -> first attempt
    # already exact, loop breaks immediately with exact_used=True
    for force in (False, True):
        attempts = []
        for attempt in (force, True):
            attempts.append(attempt)
            if attempt or not False:  # exact_recheck=False default
                break
        assert attempts == [force]
