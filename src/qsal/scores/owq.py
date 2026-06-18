"""OWQ saliency: ||W[:,c]||^2 * H_cc - weight x activation family (spec §3).

Decomposed in analysis as (weight energy) x (GPTQ-ordering factor) with
partial correlation controlling the shared factor [guard].
"""

import torch


def score(W: torch.Tensor, h_diag_over_n: torch.Tensor) -> torch.Tensor:
    return W.double().pow(2).sum(0) * h_diag_over_n.double()
