"""AWQ saliency: E[|x_c|] - activation-only family (spec §3)."""

import torch


def score(sum_abs_x: torch.Tensor, n_tokens: int) -> torch.Tensor:
    return sum_abs_x.double() / n_tokens
