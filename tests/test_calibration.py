"""Step 3 - calibration stats (the design spec).

Forward hooks accumulate per-input-channel Sum|x|, Sum x^2, token count and
H = Sum x x^T in fp64 (CPU buffers); a backward pass accumulates Sum (g*x)^2
for the unified metric. Data loader pins splits, records indices, and
asserts calib/eval disjointness.
"""

import pytest
import torch
import torch.nn as nn

import qsal.config as cfg


# ---------------------------------------------------------------------------
# Unit tests on a tiny known linear (exact expectations, no HF model)
# ---------------------------------------------------------------------------


def _tiny_entries():
    torch.manual_seed(cfg.SEED)
    lin = nn.Linear(8, 4, bias=False)
    return lin, [(0, "q_proj", lin)]


def test_forward_stats_match_manual_computation():
    from qsal.calibration import StatsAccumulator

    lin, entries = _tiny_entries()
    acc = StatsAccumulator(entries, with_hessian=True)
    x1 = torch.randn(3, 8)  # (tokens, in)
    x2 = torch.randn(5, 8)
    with acc:
        lin(x1)
        lin(x2)
    x = torch.cat([x1, x2]).double()
    st = acc.stats[(0, "q_proj")]
    assert st["token_count"] == 8
    assert torch.allclose(st["sum_abs_x"], x.abs().sum(0))
    assert torch.allclose(st["sum_x2"], x.pow(2).sum(0))
    assert torch.allclose(st["H"], x.T @ x)
    assert st["H"].dtype == torch.float64
    assert st["H"].device.type == "cpu"


def test_forward_stats_flatten_batch_dims():
    from qsal.calibration import StatsAccumulator

    lin, entries = _tiny_entries()
    acc = StatsAccumulator(entries, with_hessian=False)
    with acc:
        lin(torch.randn(2, 5, 8))  # (batch, seq, in)
    st = acc.stats[(0, "q_proj")]
    assert st["token_count"] == 10
    assert "H" not in st


def test_hooks_detached_after_context():
    from qsal.calibration import StatsAccumulator

    lin, entries = _tiny_entries()
    acc = StatsAccumulator(entries)
    with acc:
        lin(torch.randn(2, 8))
    n = acc.stats[(0, "q_proj")]["token_count"]
    lin(torch.randn(4, 8))  # outside context: must not accumulate
    assert acc.stats[(0, "q_proj")]["token_count"] == n


def test_grad_stats_match_autograd():
    from qsal.calibration import GradStatsAccumulator

    lin, entries = _tiny_entries()
    acc = GradStatsAccumulator(entries)
    x = torch.randn(6, 8, requires_grad=True)
    with acc:
        loss = lin(x).pow(2).sum()
        loss.backward()
    g = x.grad  # dL/dx
    expected = (g.double() * x.detach().double()).pow(2).sum(0)
    got = acc.stats[(0, "q_proj")]["sum_gx2"]
    assert torch.allclose(got, expected, rtol=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs MPS"
)
def test_accumulators_work_on_mps():
    # regression: .to("cpu", float64) casts on MPS first -> TypeError
    from qsal.calibration import GradStatsAccumulator, StatsAccumulator

    torch.manual_seed(cfg.SEED)
    lin = nn.Linear(8, 4, bias=False).to("mps")
    entries = [(0, "q_proj", lin)]
    x = torch.randn(6, 8, device="mps", requires_grad=True)
    with StatsAccumulator(entries, with_hessian=True) as acc, GradStatsAccumulator(
        entries
    ) as gacc:
        lin(x).pow(2).sum().backward()
    st = acc.stats[(0, "q_proj")]
    xd = x.detach().cpu().double()
    assert torch.allclose(st["H"], xd.T @ xd, rtol=1e-5, atol=1e-7)
    expected = (x.grad.cpu().double() * xd).pow(2).sum(0)
    got = gacc.stats[(0, "q_proj")]["sum_gx2"]
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# Variance identity on the real model (STEPS 3.5), tiny token budget
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_stats_shapes_and_variance_identity(model):
    from qsal.calibration import StatsAccumulator
    from qsal.models import enumerate_linears

    entries = list(enumerate_linears(model))[:7]  # layer 0 only
    acc = StatsAccumulator(entries, with_hessian=False)
    ids = torch.randint(100, 5000, (2, 64))
    with acc, torch.no_grad():
        model(ids)
    for (layer, name), st in acc.stats.items():
        n = st["token_count"]
        mean_abs = st["sum_abs_x"] / n
        mean_sq = st["sum_x2"] / n
        assert mean_abs.shape[0] == dict(
            q_proj=896, k_proj=896, v_proj=896, o_proj=896,
            gate_proj=896, up_proj=896, down_proj=4864,
        )[name]
        # E[x^2] >= (E|x|)^2 per channel (Jensen)
        assert (mean_sq >= mean_abs.pow(2) - 1e-12).all(), (layer, name)


# ---------------------------------------------------------------------------
# Data loading: pinned, seeded, disjoint calib/eval (STEPS 3.1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calib_eval_sequences_pinned_and_disjoint(tokenizer):
    from qsal.calibration import get_sequences

    calib, calib_meta = get_sequences(
        tokenizer, split=cfg.CALIB_SPLIT, num_seqs=4, seq_len=64, seed=cfg.SEED
    )
    ev, eval_meta = get_sequences(
        tokenizer, split=cfg.EVAL_SPLIT, num_seqs=4, seq_len=32, seed=cfg.SEED
    )
    assert calib.shape == (4, 64) and ev.shape == (4, 32)
    # determinism: same seed -> same data
    calib2, calib_meta2 = get_sequences(
        tokenizer, split=cfg.CALIB_SPLIT, num_seqs=4, seq_len=64, seed=cfg.SEED
    )
    assert torch.equal(calib, calib2)
    assert calib_meta["index_hash"] == calib_meta2["index_hash"]
    # provenance fields recorded
    for meta in (calib_meta, eval_meta):
        assert meta["dataset_id"] == cfg.DATASET_ID
        assert meta["dataset_revision"] == cfg.DATASET_REVISION
        assert "indices" in meta and "index_hash" in meta
    # zero overlap between calib and eval token content
    calib_rows = {tuple(r.tolist()) for r in calib}
    eval_rows = {tuple(r.tolist()) for r in ev}
    assert not calib_rows & eval_rows
    assert calib_meta["split"] != eval_meta["split"]
