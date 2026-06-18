"""GPTQ saliency: 1/[H^-1]_cc (OBS/act-order saliency) - Hessian family.

NOT H_cc: the diagonal is only GPTQ's act-order ORDERING proxy and is
near-monotone with AWQ's driver (E[x^2] vs E|x|) - using it would make
AWQ-vs-GPTQ agreement tautological (red-team finding, spec §3 [guard]).
H_cc/n is stored separately as `gptq_actorder_proxy`, reference only.
"""

import torch


def score(diag_h_inv: torch.Tensor) -> torch.Tensor:
    return 1.0 / diag_h_inv.double()
