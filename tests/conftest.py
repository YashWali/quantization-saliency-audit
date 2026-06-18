import pytest

import qsal.config as cfg


@pytest.fixture(scope="session")
def model():
    """Pinned Qwen2.5-0.5B-Instruct, fp32 on CPU (grid tests need no GPU)."""
    from qsal.models import load_model

    cfg.set_global_seeds()
    model, _ = load_model(device="cpu")
    return model


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.MODEL_ID, revision=cfg.MODEL_REVISION
    )


@pytest.fixture(scope="session")
def grid(model):
    from qsal.models import build_channel_grid

    return build_channel_grid(model)
