"""Fake quantizers (the design spec).

Per-group(128) integer fake-quant (int3/int4/..., symmetric+asymmetric) plus
fp8(e4m3)/bf16 cast round-trips. quantize_single_column quantizes one column
using its REAL 128-group scale so single-channel perturbations match
deployment quantization exactly.
"""

import torch

import qsal.config as cfg

GROUP_SIZE = cfg.QUANT_GROUP_SIZE


def _group_bounds(n: int, group_size: int):
    return [(s, min(s + group_size, n)) for s in range(0, n, group_size)]


def _quant_group_sym(Wg: torch.Tensor, bits: int):
    qmax = 2 ** (bits - 1) - 1
    scale = Wg.abs().amax(dim=1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(Wg / safe).clamp_(-qmax, qmax)
    return torch.where(scale > 0, q * safe, Wg), scale.squeeze(1)


def _quant_group_asym(Wg: torch.Tensor, bits: int):
    qmax = 2**bits - 1
    lo = Wg.amin(dim=1, keepdim=True)
    hi = Wg.amax(dim=1, keepdim=True)
    scale = (hi - lo) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    zp = torch.round(-lo / safe)
    q = (torch.round(Wg / safe) + zp).clamp_(0, qmax)
    return torch.where(scale > 0, (q - zp) * safe, Wg), scale.squeeze(1)


def group_scales(
    W: torch.Tensor,
    bits: int,
    group_size: int = GROUP_SIZE,
    symmetric: bool = True,
) -> torch.Tensor:
    """Per-(row, group) quantization scales, shape (out, n_groups)."""
    fn = _quant_group_sym if symmetric else _quant_group_asym
    return torch.stack(
        [fn(W[:, s:e], bits)[1] for s, e in _group_bounds(W.shape[1], group_size)],
        dim=1,
    )


def quantize_layer(
    W: torch.Tensor,
    bits,
    group_size: int = GROUP_SIZE,
    symmetric: bool = True,
) -> torch.Tensor:
    """Fake-quantize a full weight matrix. bits: int (e.g. 3, 4) or 'fp8'/'bf16'."""
    if bits == "fp8":
        return W.to(torch.float8_e4m3fn).to(torch.float32)
    if bits == "bf16":
        return W.to(torch.bfloat16).to(torch.float32)
    fn = _quant_group_sym if symmetric else _quant_group_asym
    out = torch.empty_like(W)
    for s, e in _group_bounds(W.shape[1], group_size):
        out[:, s:e] = fn(W[:, s:e], bits)[0]
    return out


def quantize_single_column(
    W: torch.Tensor,
    c: int,
    bits,
    group_size: int = GROUP_SIZE,
    symmetric: bool = True,
) -> torch.Tensor:
    """Quantize only column c, with the scale of c's real group (deployment
    match: column c equals quantize_layer's column c exactly)."""
    out = W.clone()
    if bits in ("fp8", "bf16"):
        out[:, c] = quantize_layer(W[:, c : c + 1], bits)[:, 0]
        return out
    fn = _quant_group_sym if symmetric else _quant_group_asym
    s = (c // group_size) * group_size
    e = min(s + group_size, W.shape[1])
    out[:, c] = fn(W[:, s:e], bits)[0][:, c - s]
    return out
