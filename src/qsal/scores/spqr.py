"""SpQR saliency: Sum_i (W_ic - Q(W)_ic)^2 / [H^-1]_cc - weight x Hessian.

The ROUNDING RESIDUAL under the deployment quantizer Q (INT4/g128), not
raw weight energy: raw Sum w^2 / [H^-1]_cc is algebraically GPTQ-OBS
energy and would fake agreement (red-team finding, spec §3 [guard]).
"""

import torch

import qsal.config as cfg
from qsal.quantizers import quantize_layer


def score(
    W: torch.Tensor,
    diag_h_inv: torch.Tensor,
    bits: int = cfg.DEPLOYMENT_BITS,
) -> torch.Tensor:
    Wq = quantize_layer(W.float(), bits=bits)
    residual = (W.double() - Wq.double()).pow(2).sum(0)
    return residual / diag_h_inv.double()
