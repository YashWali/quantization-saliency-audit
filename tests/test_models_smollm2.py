"""Phase-3 adapter - SmolLM2-360M (Llama) channel grid + frozen config.

Mirrors the Qwen/Pythia grid contracts (test_models.py / test_models_pythia.py)
for the third model: SmolLM2 is LlamaForCausalLM, which shares Qwen2's HF module
layout, so it reuses the qwen2 taxonomy via the `llama` alias. Also asserts the
pre-registration "held-identical-to-Phase-1" invariant for config_smollm2 (only
model identity, size, and SWEEP_LAYERS may differ) and the QSAL_CONFIG selection.
Loads the real cached SmolLM2 on CPU (grid build needs no GPU/forward).
"""

import importlib

import numpy as np
import pytest
import torch.nn as nn

import qsal.config as base
import qsal.config_smollm2 as scfg


# --- pre-registration: config_smollm2 frozen values + held-identical invariant ---

def test_smollm2_frozen_model_values():
    assert scfg.MODEL_ID == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert scfg.NUM_LAYERS == 32
    assert scfg.HIDDEN_SIZE == 960
    assert scfg.INTERMEDIATE_SIZE == 2560
    assert scfg.EXPECTED_TOTAL_CHANNELS == 266_240  # 32 * (960*6 + 2560)
    # SWEEP_LAYERS MUST differ from Phase-1/2 (32 layers): depth-matched analog.
    assert scfg.SWEEP_LAYERS == (3, 15, 29)
    assert scfg.SWEEP_LAYERS != base.SWEEP_LAYERS


def test_smollm2_reuses_qwen_linear_taxonomy():
    # Llama == Qwen2 linear set; imported unchanged from config.py (not overridden).
    assert scfg.LINEAR_NAMES == base.LINEAR_NAMES


def test_smollm2_held_identical_to_phase1():
    # Everything not architecture-specific is imported from config.py and must be
    # byte-identical (the pre-registration "cannot drift" guarantee).
    for name in (
        "SEED", "CALIB_NUM_SEQS", "CALIB_SEQ_LEN", "EVAL_NUM_SEQS", "EVAL_SEQ_LEN",
        "DEPLOYMENT_BITS", "ISOLATED_PROBE_BITS", "QUANT_GROUP_SIZE",
        "DAMPING_LAMBDAS", "DAMPING_PRIMARY", "TOP_K_FRACTIONS", "TOP_PCT_UNION",
        "N_PERMUTATIONS", "N_BOOTSTRAP", "ALPHA", "THRESHOLDS",
        "DATASET_ID", "DATASET_REVISION", "FORWARD_DTYPE", "ACCUM_DTYPE",
    ):
        assert getattr(scfg, name) == getattr(base, name), name


def test_qsal_config_env_selection_logic():
    # Scripts pick the config module from QSAL_CONFIG (default config_pythia) and
    # derive the run-name (data/results namespace) from it.
    assert "config_pythia".rsplit("config_", 1)[-1] == "pythia"
    assert "config_smollm2".rsplit("config_", 1)[-1] == "smollm2"
    importlib.import_module("qsal.config_pythia")   # default
    importlib.import_module("qsal.config_smollm2")  # phase 3


# --- llama taxonomy alias ---

def test_llama_taxonomy_aliases_qwen2():
    from qsal.models import _TAXONOMY

    assert "llama" in _TAXONOMY
    assert _TAXONOMY["llama"] == _TAXONOMY["qwen2"]
    assert _TAXONOMY["llama"]["tied_embeddings"] is True


# --- channel grid against the real cached model ---

@pytest.fixture(scope="module")
def smollm2_model():
    from qsal.models import load_model

    scfg.set_global_seeds()
    model, _ = load_model(device="cpu", cfg=scfg)
    return model


@pytest.fixture(scope="module")
def smollm2_grid(smollm2_model):
    from qsal.models import build_channel_grid

    return build_channel_grid(smollm2_model, cfg=scfg)


def test_smollm2_loads_as_llama(smollm2_model):
    assert smollm2_model.config.model_type == "llama"
    assert smollm2_model.config.num_hidden_layers == 32
    assert smollm2_model.config.tie_word_embeddings is True


def test_smollm2_enumerate_linears(smollm2_model):
    from qsal.models import enumerate_linears

    entries = list(enumerate_linears(smollm2_model, cfg=scfg))
    assert len(entries) == scfg.NUM_LAYERS * len(scfg.LINEAR_NAMES)  # 32*7
    assert {e[1] for e in entries} == set(scfg.LINEAR_NAMES)
    for _, _, mod in entries:
        assert isinstance(mod, nn.Linear)


def test_smollm2_grid_total_channel_count(smollm2_grid):
    assert len(smollm2_grid) == scfg.EXPECTED_TOTAL_CHANNELS  # 266_240


def test_smollm2_tied_head_excluded(smollm2_grid):
    linears = set(smollm2_grid["linear"])
    assert "lm_head" not in linears
    assert linears == set(scfg.LINEAR_NAMES)


def test_smollm2_per_linear_in_features(smollm2_grid):
    feats = smollm2_grid.groupby("linear", observed=True)["in_features"].first()
    for name in scfg.LINEAR_NAMES:
        expected = scfg.INTERMEDIATE_SIZE if name == "down_proj" else scfg.HIDDEN_SIZE
        assert (feats[name] == expected), name


def test_smollm2_gqa_out_features(smollm2_grid):
    # GQA: 15 query heads, 5 kv heads, head_dim 64 -> q/o=960, k/v=320.
    out = smollm2_grid.groupby("linear", observed=True)["out_features"].unique()
    assert list(out["q_proj"]) == [960]
    assert list(out["k_proj"]) == [320]
    assert list(out["v_proj"]) == [320]


def test_smollm2_census_has_21_strata(smollm2_grid):
    census = smollm2_grid[smollm2_grid["layer"].isin(scfg.SWEEP_LAYERS)]
    assert len(census) == 24_960  # 3 * (960*6 + 2560)
    assert census.groupby(["layer", "linear"]).ngroups == 21  # 7 linears * 3 layers


def test_smollm2_channel_id_unique_contiguous(smollm2_grid):
    ids = smollm2_grid["channel_id"].to_numpy()
    assert len(np.unique(ids)) == len(smollm2_grid)
    assert ids.min() == 0 and ids.max() == len(smollm2_grid) - 1
