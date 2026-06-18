"""Step 4.2-4.5 - the five corrected metrics (the design spec).

Each metric is monotone in its driver on synthetic inputs; the two
red-team-critical corrections are pinned by tests:
  - GPTQ = 1/[H^-1]_cc, NOT the diagonal proxy H_cc.
  - SpQR  = rounding residual / [H^-1]_cc, NOT raw weight energy.
"""

import torch

import qsal.config as cfg

torch.manual_seed(cfg.SEED)


def test_awq_is_mean_abs_activation():
    from qsal.scores.awq import score

    sum_abs_x = torch.tensor([4.0, 2.0, 8.0], dtype=torch.float64)
    s = score(sum_abs_x, n_tokens=2)
    assert torch.allclose(s, torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64))


def test_gptq_uses_inverse_hessian_not_diagonal():
    from qsal.scores.gptq import score

    # channels 0,1 strongly correlated (var 1); channel 2 independent (var 0.9)
    H = torch.tensor(
        [[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 0.9]],
        dtype=torch.float64,
    )
    diag_h_inv = torch.linalg.inv(H).diagonal()
    s = score(diag_h_inv)
    # H_cc proxy would rank 0,1 above 2; OBS saliency must do the opposite
    assert s[2] > s[0] and s[2] > s[1]
    assert torch.allclose(s, 1.0 / diag_h_inv)


def test_owq_is_weight_energy_times_activation_moment():
    from qsal.scores.owq import score

    W = torch.tensor([[1.0, 0.0, 2.0], [1.0, 3.0, 0.0]], dtype=torch.float64)
    h_diag_over_n = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
    s = score(W, h_diag_over_n)
    expected = torch.tensor([2.0 * 2.0, 9.0 * 1.0, 4.0 * 0.5], dtype=torch.float64)
    assert torch.allclose(s, expected)


def test_spqr_is_rounding_residual_not_weight_energy():
    from qsal.scores.spqr import score

    # column 0: large but EXACTLY representable under int4/g128 sym
    # (proportional to integer levels) -> zero residual -> zero score.
    # column 1: small but irrational-ish -> nonzero residual.
    out, groups = 4, 128
    W = torch.zeros(out, groups, dtype=torch.float64)
    # per-row scales 1, 1, 0.5, 0.25 (powers of 2) -> exact fp arithmetic
    W[:, 0] = torch.tensor([7.0, -7.0, 3.5, 1.75])
    W[:, 1] = 0.1234567
    diag_h_inv = torch.ones(groups, dtype=torch.float64)
    s = score(W.float(), diag_h_inv, bits=cfg.DEPLOYMENT_BITS)
    assert s[0] == 0.0
    assert s[1] > 0.0
    # weight energy would say column 0 >> column 1: pinned as the wrong answer
    assert W[:, 0].pow(2).sum() > W[:, 1].pow(2).sum()


def test_spqr_divides_by_inverse_hessian_diag():
    from qsal.scores.spqr import score

    g = torch.Generator().manual_seed(cfg.SEED)
    W = torch.randn(4, 128, generator=g)
    d1 = torch.ones(128, dtype=torch.float64)
    d2 = torch.full((128,), 2.0, dtype=torch.float64)
    s1, s2 = score(W, d1), score(W, d2)
    assert torch.allclose(s1, 2.0 * s2, rtol=1e-6)


def test_unified_is_mean_squared_grad_activation_product():
    from qsal.scores.unified import score

    sum_gx2 = torch.tensor([6.0, 0.0, 3.0], dtype=torch.float64)
    s = score(sum_gx2, n_tokens=3)
    assert torch.allclose(s, torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64))


def test_all_scores_finite_and_nonnegative():
    from qsal.scores import awq, gptq, owq, spqr, unified

    g = torch.Generator().manual_seed(cfg.SEED)
    n, in_f, out_f = 64, 128, 8
    X = torch.randn(n, in_f, generator=g, dtype=torch.float64)
    W = torch.randn(out_f, in_f, generator=g)
    H = X.T @ X
    diag_h_inv = torch.linalg.inv(
        H + 1e-2 * H.diagonal().mean() * torch.eye(in_f, dtype=torch.float64)
    ).diagonal()
    vecs = [
        awq.score(X.abs().sum(0), n),
        gptq.score(diag_h_inv),
        owq.score(W.double(), H.diagonal() / n),
        spqr.score(W, diag_h_inv),
        unified.score((X * 0.01).pow(2).sum(0), n),
    ]
    for s in vecs:
        assert s.shape == (in_f,)
        assert torch.isfinite(s).all()
        assert (s >= 0).all()
