"""Model loading + shared channel grid (the design spec).

Every metric and the ground truth index into one grid:
(layer, linear, in_channel) -> stable contiguous channel_id.

The grid scores the per-layer INPUT-channel decoder linears; the vocab
embedding/output head is excluded. Three model families are supported, dispatched
on `model.config.model_type` (the adapter is confined to this file, per
the design spec):

- qwen2 (Phase 1): 7 linears under model.layers[i].{self_attn,mlp};
  lm_head excluded, justified by tied embeddings (no independent weights).
- gpt_neox (Phase 2, Pythia): 4 linears under gpt_neox.layers[i].{attention,mlp}
  (fused query_key_value - we score INPUT channels, so no q/k/v split);
  untied embed_out excluded BY ROLE (vocab head, not a decoder linear) - the
  Phase-2 analogue of Phase-1's tied-lm_head exclusion.
- llama (Phase 3, SmolLM2): LlamaForCausalLM has the identical HF module layout
  to qwen2 (7 linears under model.layers[i].{self_attn,mlp}; tied embeddings),
  so it reuses the qwen2 taxonomy verbatim (no new structural adapter).

Functions take an optional `cfg` (defaults to the Phase-1 config) so the same
code drives any model: pass `cfg=qsal.config_pythia` for Phase 2,
`cfg=qsal.config_smollm2` for Phase 3.
"""

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import qsal.config as cfg

# Per-model-family taxonomy: how to reach the decoder layers, which parent module
# holds each scored linear, and whether the vocab head is tied. Keyed by
# transformers `model_type`.
_TAXONOMY = {
    "qwen2": {
        "layers": ("model", "layers"),
        "final_norm": ("model", "norm"),
        "head": ("lm_head",),
        "input_embedding": ("model", "embed_tokens"),
        "parents": {
            "q_proj": "self_attn",
            "k_proj": "self_attn",
            "v_proj": "self_attn",
            "o_proj": "self_attn",
            "gate_proj": "mlp",
            "up_proj": "mlp",
            "down_proj": "mlp",
        },
        "tied_embeddings": True,
    },
    "gpt_neox": {
        "layers": ("gpt_neox", "layers"),
        "final_norm": ("gpt_neox", "final_layer_norm"),
        "head": ("embed_out",),
        "input_embedding": ("gpt_neox", "embed_in"),
        "parents": {
            "query_key_value": "attention",
            "dense": "attention",
            "dense_h_to_4h": "mlp",
            "dense_4h_to_h": "mlp",
        },
        "tied_embeddings": False,
    },
}

# SmolLM2 (Phase 3) is LlamaForCausalLM - identical HF module layout to Qwen2
# (model.layers[i].{self_attn,mlp}; q/k/v/o + gate/up/down; tied embeddings), so
# it reuses the qwen2 taxonomy verbatim.
_TAXONOMY["llama"] = _TAXONOMY["qwen2"]


def _taxonomy(model):
    mt = model.config.model_type
    try:
        return _TAXONOMY[mt]
    except KeyError:
        raise NotImplementedError(
            f"no channel-grid taxonomy for model_type {mt!r}"
        )


def _resolve(model, path):
    obj = model
    for attr in path:
        obj = getattr(obj, attr)
    return obj


def _decoder_layers(model, tax):
    return _resolve(model, tax["layers"])


def decoder_layers(model):
    """The decoder layer list (model.model.layers / gpt_neox.layers)."""
    return _resolve(model, _taxonomy(model)["layers"])


def final_norm(model):
    """The final pre-head norm module (model.model.norm / final_layer_norm)."""
    return _resolve(model, _taxonomy(model)["final_norm"])


def lm_head(model):
    """The vocab projection head (lm_head / embed_out)."""
    return _resolve(model, _taxonomy(model)["head"])


def input_embedding(model):
    """The input token embedding (model.embed_tokens / gpt_neox.embed_in)."""
    return _resolve(model, _taxonomy(model)["input_embedding"])


def load_model(device: str = "cpu", cfg=cfg):
    """Load the pinned model (fp32, eval mode) and tokenizer for `cfg`."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_ID, revision=cfg.MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID, revision=cfg.MODEL_REVISION, dtype=cfg.FORWARD_DTYPE
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def enumerate_linears(model, cfg=cfg):
    """Yield (layer_idx, linear_name, nn.Linear) for all scored decoder linears,
    in the stable order (layer asc, cfg.LINEAR_NAMES order)."""
    tax = _taxonomy(model)
    parents = tax["parents"]
    for layer_idx, layer in enumerate(_decoder_layers(model, tax)):
        for name in cfg.LINEAR_NAMES:
            module = getattr(getattr(layer, parents[name]), name)
            yield layer_idx, name, module


def build_channel_grid(model, cfg=cfg) -> pd.DataFrame:
    """One row per (layer, linear, in_channel) with a stable channel_id.

    The vocab embedding/output head is excluded. For tied families (qwen2) this
    is asserted via tying (a tied lm_head has no independent weights to protect);
    for untied families (gpt_neox) embed_out is excluded by role (vocab head, not
    a decoder linear) - documented in the design spec
    """
    tax = _taxonomy(model)
    if tax["tied_embeddings"]:
        assert model.config.tie_word_embeddings, "expected tied lm_head/embedding"
        assert (
            model.lm_head.weight.data_ptr()
            == model.model.embed_tokens.weight.data_ptr()
        ), "lm_head reports tied but does not share storage"
    else:
        assert (
            not model.config.tie_word_embeddings
        ), "expected untied embeddings (embed_out excluded by role)"

    rows = []
    for layer_idx, name, module in enumerate_linears(model, cfg):
        in_f, out_f = module.in_features, module.out_features
        rows.append(
            pd.DataFrame(
                {
                    "layer": layer_idx,
                    "linear": name,
                    "in_channel": range(in_f),
                    "in_features": in_f,
                    "out_features": out_f,
                }
            )
        )
    grid = pd.concat(rows, ignore_index=True)
    grid.insert(0, "channel_id", grid.index.to_numpy())
    return grid
