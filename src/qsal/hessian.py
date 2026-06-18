"""fp64 CPU damped Cholesky inverse diagonal (the design spec, spec §5).

H <- H + lam * mean(diag(H)) * I, factorize, return diag(H^-1) plus
conditioning info. Over-damping collapses 1/[H^-1]_cc toward diag(H)
(activation-family confound), hence the pre-registered lambda sweep.
"""

import torch


def damped_inverse_diag(H: torch.Tensor, lam: float, with_info: bool = True):
    """Return (diag(H_damped^-1), info) for SPD-after-damping H (fp64 CPU)."""
    assert H.dtype == torch.float64, "Hessian path is fp64-only [guard]"
    assert H.device.type == "cpu", "Hessian path runs on CPU [guard]"
    assert H.ndim == 2 and H.shape[0] == H.shape[1]

    n = H.shape[0]
    Hd = H + lam * H.diagonal().mean() * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(Hd)  # raises if not SPD: asserted success
    diag_inv = torch.cholesky_inverse(L).diagonal().clone()

    info = {"damping_lambda": lam, "n": n}
    if with_info:
        eig = torch.linalg.eigvalsh(Hd)
        info["condition_number"] = (eig[-1] / eig[0]).item()
        p = eig / eig.sum()
        info["effective_rank"] = torch.exp(-(p * p.log()).sum()).item()
    return diag_inv, info
