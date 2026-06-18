"""Step 1 - shared channel grid (the design spec).

Grid contract: one row per (layer, linear, in_channel) over all decoder
linears, stable contiguous channel_id, tied lm_head excluded.
"""

import numpy as np
import pytest

import qsal.config as cfg


def test_enumerate_linears_covers_all_decoder_linears(model):
    from qsal.models import enumerate_linears

    entries = list(enumerate_linears(model))
    assert len(entries) == cfg.NUM_LAYERS * len(cfg.LINEAR_NAMES)  # 24*7
    layers = {e[0] for e in entries}
    names = {e[1] for e in entries}
    assert layers == set(range(cfg.NUM_LAYERS))
    assert names == set(cfg.LINEAR_NAMES)
    # every entry is an nn.Linear with a weight
    import torch.nn as nn

    for _, _, mod in entries:
        assert isinstance(mod, nn.Linear)


def test_grid_total_channel_count(grid):
    assert len(grid) == cfg.EXPECTED_TOTAL_CHANNELS  # 24*(896*6+4864)=245760


def test_lm_head_and_embedding_excluded(grid):
    assert "lm_head" not in set(grid["linear"])
    assert "embed_tokens" not in set(grid["linear"])
    assert set(grid["linear"]) == set(cfg.LINEAR_NAMES)


def test_per_linear_counts_match_in_features(grid):
    counts = grid.groupby(["layer", "linear"], observed=True).size()
    feats = grid.groupby(["layer", "linear"], observed=True)["in_features"].first()
    assert (counts == feats).all()
    # architecture facts: all linears except down_proj take hidden_size inputs
    for name in cfg.LINEAR_NAMES:
        expected = (
            cfg.INTERMEDIATE_SIZE if name == "down_proj" else cfg.HIDDEN_SIZE
        )
        assert (feats.xs(name, level="linear") == expected).all(), name


def test_out_features_recorded_with_gqa(grid):
    out = grid.groupby("linear", observed=True)["out_features"].unique()
    assert list(out["q_proj"]) == [cfg.HIDDEN_SIZE]
    # GQA: 2 kv-heads x 64 head_dim = 128
    assert list(out["k_proj"]) == [128]
    assert list(out["v_proj"]) == [128]
    assert list(out["o_proj"]) == [cfg.HIDDEN_SIZE]
    assert list(out["gate_proj"]) == [cfg.INTERMEDIATE_SIZE]
    assert list(out["up_proj"]) == [cfg.INTERMEDIATE_SIZE]
    assert list(out["down_proj"]) == [cfg.HIDDEN_SIZE]


def test_channel_id_unique_and_contiguous(grid):
    ids = grid["channel_id"].to_numpy()
    assert ids.dtype.kind == "i"
    assert len(np.unique(ids)) == len(grid)
    assert ids.min() == 0 and ids.max() == len(grid) - 1


def test_channel_id_roundtrip(grid):
    # (layer, linear, in_channel) -> channel_id -> same triple
    sample = grid.sample(n=200, random_state=cfg.SEED)
    indexed = grid.set_index(["layer", "linear", "in_channel"])
    for row in sample.itertuples():
        cid = indexed.loc[(row.layer, row.linear, row.in_channel), "channel_id"]
        assert cid == row.channel_id
    by_id = grid.set_index("channel_id")
    for row in sample.itertuples():
        rec = by_id.loc[row.channel_id]
        assert (rec["layer"], rec["linear"], rec["in_channel"]) == (
            row.layer,
            row.linear,
            row.in_channel,
        )


def test_input_embedding_accessor(model):
    from qsal.models import input_embedding

    assert input_embedding(model) is model.model.embed_tokens


def test_channel_id_stable_deterministic(model):
    # rebuilding the grid yields identical ids (stability guarantee)
    from qsal.models import build_channel_grid

    g1 = build_channel_grid(model)
    g2 = build_channel_grid(model)
    assert g1.equals(g2)
