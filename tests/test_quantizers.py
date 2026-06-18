"""Step 2 - fake quantizers (the design spec).

Contract: per-group(128) fake-quant int3/int4 (sym+asym) + fp8/bf16 casts;
quantize_single_column uses the column's REAL 128-group scale so the
perturbation matches deployment.
"""

import pytest
import torch

import qsal.config as cfg

torch.manual_seed(cfg.SEED)


@pytest.fixture()
def W():
    g = torch.Generator().manual_seed(cfg.SEED)
    return torch.randn(64, 384, generator=g)  # 3 groups of 128 per row


def test_int4_roundtrip_within_per_group_bound(W):
    from qsal.quantizers import group_scales, quantize_layer

    Wq = quantize_layer(W, bits=4)
    scales = group_scales(W, bits=4)  # (out, n_groups)
    err = (W - Wq).abs().reshape(64, 3, 128)
    bound = scales.unsqueeze(-1) / 2 + 1e-8
    assert (err <= bound).all()


def test_int3_error_exceeds_int4(W):
    from qsal.quantizers import quantize_layer

    e3 = (W - quantize_layer(W, bits=3)).pow(2).mean()
    e4 = (W - quantize_layer(W, bits=4)).pow(2).mean()
    assert e3 > e4 > 0


def test_symmetric_int4_idempotent(W):
    from qsal.quantizers import quantize_layer

    Wq = quantize_layer(W, bits=4, symmetric=True)
    assert torch.equal(quantize_layer(Wq, bits=4, symmetric=True), Wq)


def test_asymmetric_beats_symmetric_on_shifted_weights():
    from qsal.quantizers import quantize_layer

    g = torch.Generator().manual_seed(cfg.SEED)
    W = torch.randn(8, 128, generator=g) + 5.0  # all-positive, shifted
    e_sym = (W - quantize_layer(W, bits=4, symmetric=True)).pow(2).mean()
    e_asym = (W - quantize_layer(W, bits=4, symmetric=False)).pow(2).mean()
    assert e_asym < e_sym


def test_single_column_touches_only_c(W):
    from qsal.quantizers import quantize_single_column

    c = 200
    Wq = quantize_single_column(W, c, bits=3)
    mask = torch.ones(W.shape[1], dtype=torch.bool)
    mask[c] = False
    assert torch.equal(Wq[:, mask], W[:, mask])
    assert not torch.equal(Wq[:, c], W[:, c])


def test_single_column_uses_real_group_scale(W):
    # deployment match: column c of Q_col(W,c) == column c of Q_layer(W)
    from qsal.quantizers import quantize_layer, quantize_single_column

    for c in (0, 127, 128, 200, 383):
        Wq_col = quantize_single_column(W, c, bits=4)
        Wq_full = quantize_layer(W, bits=4)
        assert torch.equal(Wq_col[:, c], Wq_full[:, c]), c


def test_group_scales_are_per_group(W):
    # a huge outlier in group 0 must not degrade group 2's quantization
    from qsal.quantizers import quantize_layer

    W = W.clone()
    W[:, :128] *= 1000.0
    Wq = quantize_layer(W, bits=4)
    err_g2 = (W[:, 256:] - Wq[:, 256:]).abs().max()
    assert err_g2 < 1.0  # would be ~1000x scale/2 with a row-global scale


def test_zero_group_handled(W):
    from qsal.quantizers import quantize_layer

    W = W.clone()
    W[:, 128:256] = 0.0
    Wq = quantize_layer(W, bits=4)
    assert torch.isfinite(Wq).all()
    assert torch.equal(Wq[:, 128:256], W[:, 128:256])


def test_fp8_and_bf16_casts(W):
    from qsal.quantizers import quantize_layer

    W8 = quantize_layer(W, bits="fp8")
    W16 = quantize_layer(W, bits="bf16")
    assert W8.dtype == torch.float32 and W16.dtype == torch.float32
    e8 = (W - W8).pow(2).mean()
    e16 = (W - W16).pow(2).mean()
    assert 0 < e16 < e8  # bf16 (7 mantissa bits) beats fp8 e4m3 (3 bits)


def test_ragged_last_group():
    # in_features not divisible by 128 (e.g. hypothetical) still works
    from qsal.quantizers import quantize_layer

    g = torch.Generator().manual_seed(cfg.SEED)
    W = torch.randn(4, 300, generator=g)
    Wq = quantize_layer(W, bits=4)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()
