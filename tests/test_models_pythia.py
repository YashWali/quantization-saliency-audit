"""Phase-2 adapter - Pythia-410m (GPT-NeoX) channel grid.

Mirrors the Qwen grid contract (tests/test_models.py) for the second model:
one row per (layer, linear, in_channel) over the 4 GPT-NeoX input-channel
linears, stable contiguous channel_id, untied embed_out excluded BY ROLE.
Loads the real pinned Pythia on CPU (grid build needs no GPU/forward).
"""

import numpy as np
import pytest
import torch.nn as nn

import qsal.config_pythia as pcfg


@pytest.fixture(scope="module")
def pythia_model():
    from qsal.models import load_model

    pcfg.set_global_seeds()
    model, _ = load_model(device="cpu", cfg=pcfg)
    return model


@pytest.fixture(scope="module")
def pythia_grid(pythia_model):
    from qsal.models import build_channel_grid

    return build_channel_grid(pythia_model, cfg=pcfg)


def test_pythia_enumerate_linears_covers_four_linears(pythia_model):
    from qsal.models import enumerate_linears

    entries = list(enumerate_linears(pythia_model, cfg=pcfg))
    assert len(entries) == pcfg.NUM_LAYERS * len(pcfg.LINEAR_NAMES)  # 24*4
    assert {e[0] for e in entries} == set(range(pcfg.NUM_LAYERS))
    assert {e[1] for e in entries} == set(pcfg.LINEAR_NAMES)
    for _, _, mod in entries:
        assert isinstance(mod, nn.Linear)


def test_pythia_grid_total_channel_count(pythia_grid):
    assert len(pythia_grid) == pcfg.EXPECTED_TOTAL_CHANNELS  # 172_032


def test_pythia_untied_head_excluded(pythia_grid):
    # GPT-NeoX is untied: embed_out (vocab head) and embed_in are excluded by
    # role, not by tying. Only the 4 decoder linears appear.
    linears = set(pythia_grid["linear"])
    assert "embed_out" not in linears
    assert "embed_in" not in linears
    assert linears == set(pcfg.LINEAR_NAMES)


def test_pythia_per_linear_in_features(pythia_grid):
    counts = pythia_grid.groupby(["layer", "linear"], observed=True).size()
    feats = pythia_grid.groupby(["layer", "linear"], observed=True)[
        "in_features"
    ].first()
    assert (counts == feats).all()
    # dense_4h_to_h takes intermediate inputs; the other three take hidden.
    for name in pcfg.LINEAR_NAMES:
        expected = (
            pcfg.INTERMEDIATE_SIZE
            if name == "dense_4h_to_h"
            else pcfg.HIDDEN_SIZE
        )
        assert (feats.xs(name, level="linear") == expected).all(), name


def test_pythia_fused_qkv_out_features(pythia_grid):
    # Fused QKV projects hidden -> 3*hidden (full MHA, no GQA).
    out = pythia_grid.groupby("linear", observed=True)["out_features"].unique()
    assert list(out["query_key_value"]) == [3 * pcfg.HIDDEN_SIZE]
    assert list(out["dense"]) == [pcfg.HIDDEN_SIZE]
    assert list(out["dense_h_to_4h"]) == [pcfg.INTERMEDIATE_SIZE]
    assert list(out["dense_4h_to_h"]) == [pcfg.HIDDEN_SIZE]


def test_pythia_input_embedding_accessor(pythia_model):
    from qsal.models import input_embedding

    emb = input_embedding(pythia_model)
    assert isinstance(emb, nn.Embedding)
    assert emb.weight.shape[0] == 50_304  # Pythia vocab (gpt_neox.embed_in)


def test_pythia_channel_id_unique_and_contiguous(pythia_grid):
    ids = pythia_grid["channel_id"].to_numpy()
    assert ids.dtype.kind == "i"
    assert len(np.unique(ids)) == len(pythia_grid)
    assert ids.min() == 0 and ids.max() == len(pythia_grid) - 1


def test_pythia_grid_stable_deterministic(pythia_model):
    from qsal.models import build_channel_grid

    g1 = build_channel_grid(pythia_model, cfg=pcfg)
    g2 = build_channel_grid(pythia_model, cfg=pcfg)
    assert g1.equals(g2)
