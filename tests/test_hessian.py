"""Step 4.1 - fp64 CPU damped Cholesky inverse diagonal (the design spec).

H <- H + lam*mean(diag(H))*I, then diag(H^-1) via Cholesky. Validated
against direct pinv; lam monotonicity; rank-deficient H handled.
"""

import pytest
import torch

import qsal.config as cfg

torch.manual_seed(cfg.SEED)


def _spd(n=16, rank=None):
    g = torch.Generator().manual_seed(cfg.SEED)
    X = torch.randn(rank or 4 * n, n, generator=g, dtype=torch.float64)
    return X.T @ X


def test_diag_inverse_matches_pinv():
    from qsal.hessian import damped_inverse_diag

    H = _spd(16)
    lam = 1e-3
    d, info = damped_inverse_diag(H, lam)
    Hd = H + lam * H.diagonal().mean() * torch.eye(16, dtype=torch.float64)
    expected = torch.linalg.pinv(Hd).diagonal()
    assert torch.allclose(d, expected, rtol=1e-8)
    assert d.dtype == torch.float64


def test_lambda_monotonicity():
    from qsal.hessian import damped_inverse_diag

    H = _spd(16)
    prev = None
    for lam in cfg.DAMPING_LAMBDAS:  # increasing
        d, _ = damped_inverse_diag(H, lam)
        if prev is not None:
            assert (d < prev).all()  # more damping -> smaller inverse diag
        prev = d


def test_rank_deficient_h_succeeds_with_damping():
    from qsal.hessian import damped_inverse_diag

    H = _spd(16, rank=3)  # rank 3 < 16: undamped Cholesky must fail
    with pytest.raises(Exception):
        torch.linalg.cholesky(H)
    d, info = damped_inverse_diag(H, 1e-2)
    assert torch.isfinite(d).all() and (d > 0).all()


def test_info_reports_condition_and_effective_rank():
    from qsal.hessian import damped_inverse_diag

    H = _spd(16)
    _, info = damped_inverse_diag(H, 1e-3)
    assert info["condition_number"] > 1
    assert 0 < info["effective_rank"] <= 16
    assert info["damping_lambda"] == 1e-3


def test_requires_float64():
    from qsal.hessian import damped_inverse_diag

    with pytest.raises(AssertionError):
        damped_inverse_diag(torch.eye(4, dtype=torch.float32), 1e-3)
