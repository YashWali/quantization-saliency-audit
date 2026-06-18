"""Step 5 - ground truth machinery (the design spec).

KL math vs an fp64 reference, suffix-forward equivalence to the full
model (prefix-activation cache), weight patching that restores exactly,
and atomic checkpoint/resume.
"""

import pytest
import torch

import qsal.config as cfg

torch.manual_seed(cfg.SEED)


# ---------------------------------------------------------------------------
# KL per token
# ---------------------------------------------------------------------------


def _ref_kl(z_full, z_pert):
    # full fp64 reference: KL(softmax(z_full) || softmax(z_pert)) per token
    lp = torch.log_softmax(z_full.double(), dim=-1)
    lq = torch.log_softmax(z_pert.double(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def test_kl_zero_for_identical_logits():
    from qsal.groundtruth import kl_per_token

    z = torch.randn(5, 100)
    kl = kl_per_token(torch.log_softmax(z, -1), z)
    assert kl.dtype == torch.float64
    assert (kl.abs() < 1e-12).all()


def test_kl_matches_fp64_reference():
    from qsal.groundtruth import kl_per_token

    g = torch.Generator().manual_seed(cfg.SEED)
    z_full = torch.randn(8, 5000, generator=g) * 4
    z_pert = z_full + torch.randn(8, 5000, generator=g) * 0.05  # near-identical
    kl = kl_per_token(torch.log_softmax(z_full, -1), z_pert)
    ref = _ref_kl(z_full, z_pert)
    assert torch.allclose(kl, ref, rtol=1e-3, atol=1e-9)
    assert (kl >= 0).all()


def test_kl_nonnegative_even_at_noise_floor():
    from qsal.groundtruth import kl_per_token, kl_per_token_fp64

    g = torch.Generator().manual_seed(cfg.SEED)
    z = torch.randn(16, 5000, generator=g) * 4
    z2 = z + torch.randn(16, 5000, generator=g) * 1e-5
    # fast fp32 path: nonnegative, accurate to the fp32 log-softmax floor
    kl = kl_per_token(torch.log_softmax(z, -1), z2)
    ref = _ref_kl(z, z2)
    assert (kl > -1e-12).all()
    assert (kl - ref).abs().max() < 1e-6
    # exact fp64 path (used to re-measure near-floor candidates):
    kl64 = kl_per_token_fp64(z, z2)
    assert (kl64 >= 0).all()
    assert (kl64 - ref).abs().max() < 1e-14


# ---------------------------------------------------------------------------
# Weight patching
# ---------------------------------------------------------------------------


def test_patched_weight_restores_exactly():
    import torch.nn as nn

    from qsal.groundtruth import patched_weight
    from qsal.quantizers import quantize_single_column

    lin = nn.Linear(256, 8, bias=False)
    orig = lin.weight.detach().clone()
    W_pert = quantize_single_column(orig, 7, bits=cfg.ISOLATED_PROBE_BITS)
    with patched_weight(lin, W_pert):
        assert torch.equal(lin.weight.detach(), W_pert)
        assert not torch.equal(lin.weight.detach(), orig)
    assert torch.equal(lin.weight.detach(), orig)


def test_joint_variants_differ_only_in_candidate_column():
    from qsal.groundtruth import joint_loi_weight, joint_loo_weight

    g = torch.Generator().manual_seed(cfg.SEED)
    W = torch.randn(8, 256, generator=g)
    c = 130
    loo = joint_loo_weight(W, c)  # whole layer quantized, incl. c
    loi = joint_loi_weight(W, c)  # whole layer quantized, c restored
    mask = torch.ones(256, dtype=torch.bool)
    mask[c] = False
    assert torch.equal(loo[:, mask], loi[:, mask])
    assert torch.equal(loi[:, c], W[:, c])
    assert not torch.equal(loo[:, c], W[:, c])


# ---------------------------------------------------------------------------
# Suffix forward == full forward (prefix-activation cache)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_suffix_forward_matches_full_forward(model):
    from qsal.groundtruth import SuffixRunner

    ids = torch.randint(100, 5000, (2, 64))
    with torch.no_grad():
        full = model(ids).logits
    runner = SuffixRunner(model, ids, cache_layers=[0, 5, 23])
    clean = torch.cat(
        [runner.clean_logits_batch(b) for b in range(len(runner.batches))]
    )
    assert torch.allclose(clean, full, rtol=1e-5, atol=1e-5)
    for L in (0, 5, 23):
        with torch.no_grad():
            suffix = runner.logits_from_layer(L)
        assert torch.allclose(suffix, full, rtol=1e-4, atol=1e-4), L


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs MPS"
)
def test_suffix_runner_works_on_mps(model):
    # regression: fp64 ops on MPS tensors (no float64 on MPS)
    from qsal.groundtruth import SuffixRunner

    ids = torch.randint(100, 5000, (2, 32))
    try:
        model.to("mps")
        runner = SuffixRunner(model, ids, cache_layers=[20])
        st = runner.stats_from_layer(20, exact_recheck=True)  # clean: KL ~ 0
        assert st["kl_exact_fp64"]  # near-floor triggers exact path
        assert st["kl_mean"] < 1e-6
        fast = runner.stats_from_layer(20)  # main-loop path: no recheck
        assert not fast["kl_exact_fp64"]
        assert fast["kl_mean"] < 1e-5  # fp32-path floor
    finally:
        model.to("cpu")


# ---------------------------------------------------------------------------
# Atomic checkpoint / resume
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_and_resume(tmp_path):
    from qsal.groundtruth import load_done, save_chunk

    rows = [
        {"channel_id": 1, "repeat": 0, "kl_iso_mean": 0.5},
        {"channel_id": 2, "repeat": 0, "kl_iso_mean": 0.1},
    ]
    save_chunk(rows, tmp_path)
    save_chunk([{"channel_id": 3, "repeat": 1, "kl_iso_mean": 0.2}], tmp_path)
    df = load_done(tmp_path)
    assert len(df) == 3
    assert set(zip(df["channel_id"], df["repeat"])) == {(1, 0), (2, 0), (3, 1)}
    assert not list(tmp_path.glob("*.tmp"))  # atomic: no temp residue


def test_load_done_empty_dir(tmp_path):
    from qsal.groundtruth import load_done

    df = load_done(tmp_path)
    assert len(df) == 0
