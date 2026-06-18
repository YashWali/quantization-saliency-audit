"""Unified saliency (arXiv 2601.11663): E_t[(g_{t,c} x_{t,c})^2].

Gradient x activation family - the first-order linearization of the
measured ground truth, so RQ5 asks by how much / where it fails, not
whether it wins (spec §2).
"""

import torch


def score(sum_gx2: torch.Tensor, n_tokens: int) -> torch.Tensor:
    return sum_gx2.double() / n_tokens
