"""Calibration statistics (the design spec).

StatsAccumulator hooks linear inputs and accumulates, per input channel,
Sum|x|, Sum x^2, token count, and (optionally) H = Sum x x^T - all in fp64
CPU buffers, with the per-batch x^T x product also computed in fp64 on CPU
(spec §5 numerical-hygiene guard; slower but exact).

GradStatsAccumulator additionally pairs each forward input x with its
backward gradient g = dL/dx and accumulates Sum_t (g_t * x_t)^2 for the
unified metric (arXiv 2601.11663).

get_sequences pins dataset/config/revision/split, draws seeded chunk
indices, and returns a provenance meta dict (indices + hash).
"""

import hashlib
import json
import os

import numpy as np
import torch

import qsal.config as cfg

# hf-xet 1.5.0 repeatedly stalled mid-download in this environment; the
# classic CDN path is reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class StatsAccumulator:
    """Forward-hook accumulator over (layer, name, module) entries."""

    def __init__(self, entries, with_hessian: bool = True):
        self.entries = list(entries)
        self.with_hessian = with_hessian
        self.stats = {}
        for layer, name, module in self.entries:
            st = {
                "sum_abs_x": torch.zeros(module.in_features, dtype=torch.float64),
                "sum_x2": torch.zeros(module.in_features, dtype=torch.float64),
                "token_count": 0,
            }
            if with_hessian:
                st["H"] = torch.zeros(
                    module.in_features, module.in_features, dtype=torch.float64
                )
            self.stats[(layer, name)] = st
        self._handles = []

    def _make_hook(self, key):
        def hook(module, args):
            x = args[0].detach()
            # .cpu() BEFORE .double(): MPS has no float64
            x = x.reshape(-1, x.shape[-1]).cpu().double()
            st = self.stats[key]
            st["sum_abs_x"] += x.abs().sum(0)
            st["sum_x2"] += x.pow(2).sum(0)
            st["token_count"] += x.shape[0]
            if self.with_hessian:
                st["H"] += x.T @ x

        return hook

    def __enter__(self):
        for layer, name, module in self.entries:
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook((layer, name)))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class GradStatsAccumulator:
    """Accumulates Sum_t (g_{t,c} x_{t,c})^2 per input channel (unified metric).

    Forward pre-hooks push x; full backward hooks pop the matching x (LIFO -
    backward runs in reverse forward order) and consume g = dL/dx.
    """

    def __init__(self, entries):
        self.entries = list(entries)
        self.stats = {
            (layer, name): {
                "sum_gx2": torch.zeros(module.in_features, dtype=torch.float64),
                "token_count": 0,
            }
            for layer, name, module in self.entries
        }
        self._pending = {key: [] for key in self.stats}
        self._handles = []

    def _fwd_hook(self, key):
        def hook(module, args):
            x = args[0].detach()
            self._pending[key].append(x.reshape(-1, x.shape[-1]))

        return hook

    def _bwd_hook(self, key):
        def hook(module, grad_input, grad_output):
            g = grad_input[0]
            if g is None:
                return
            x = self._pending[key].pop()
            g = g.detach().reshape(-1, g.shape[-1])
            prod = (g.cpu().double() * x.cpu().double()) ** 2
            st = self.stats[key]
            st["sum_gx2"] += prod.sum(0)
            st["token_count"] += x.shape[0]

        return hook

    def __enter__(self):
        for layer, name, module in self.entries:
            key = (layer, name)
            self._handles.append(
                module.register_forward_pre_hook(self._fwd_hook(key))
            )
            self._handles.append(
                module.register_full_backward_hook(self._bwd_hook(key))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for buf in self._pending.values():
            buf.clear()
        return False


def get_sequences(tokenizer, split: str, num_seqs: int, seq_len: int, seed: int):
    """Seeded, pinned token sequences: (LongTensor[num_seqs, seq_len], meta)."""
    from datasets import load_dataset

    ds = load_dataset(
        cfg.DATASET_ID,
        cfg.DATASET_CONFIG,
        split=split,
        revision=cfg.DATASET_REVISION,
    )
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.shape[0] // seq_len
    if num_seqs > n_chunks:
        raise ValueError(f"requested {num_seqs} seqs, only {n_chunks} available")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_chunks, size=num_seqs, replace=False))
    seqs = torch.stack([ids[i * seq_len : (i + 1) * seq_len] for i in indices])
    meta = {
        "dataset_id": cfg.DATASET_ID,
        "dataset_config": cfg.DATASET_CONFIG,
        "dataset_revision": cfg.DATASET_REVISION,
        "split": split,
        "num_seqs": num_seqs,
        "seq_len": seq_len,
        "seed": seed,
        "indices": indices.tolist(),
        "index_hash": hashlib.sha256(
            json.dumps(
                [cfg.DATASET_REVISION, split, seq_len, indices.tolist()]
            ).encode()
        ).hexdigest(),
    }
    return seqs, meta
