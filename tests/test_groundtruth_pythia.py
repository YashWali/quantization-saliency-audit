"""Phase-2 GT suffix machinery on Pythia (GPT-NeoX paths).

The prefix-activation cache must reproduce the full forward on GPT-NeoX exactly
as it does on Qwen - i.e. SuffixRunner must reach gpt_neox.layers /
final_layer_norm / embed_out, not the Qwen model.model.layers / norm / lm_head.
Loads the real pinned Pythia on CPU.
"""

import pytest
import torch

import qsal.config_pythia as pcfg


@pytest.fixture(scope="module")
def pythia_model():
    from qsal.models import load_model

    pcfg.set_global_seeds()
    model, _ = load_model(device="cpu", cfg=pcfg)
    return model


@pytest.mark.slow
def test_pythia_suffix_forward_matches_full_forward(pythia_model):
    from qsal.groundtruth import SuffixRunner

    ids = torch.randint(100, 50_000, (2, 64))  # within Pythia's 50,304 vocab
    with torch.no_grad():
        full = pythia_model(ids).logits
    runner = SuffixRunner(pythia_model, ids, cache_layers=[0, 5, 23])
    clean = torch.cat(
        [runner.clean_logits_batch(b) for b in range(len(runner.batches))]
    )
    assert torch.allclose(clean, full, rtol=1e-5, atol=1e-5)
    for L in (0, 5, 23):
        with torch.no_grad():
            suffix = runner.logits_from_layer(L)
        assert torch.allclose(suffix, full, rtol=1e-4, atol=1e-4), L
