"""Co-primary ground truth machinery (the design spec, spec §5).

- kl_per_token: KL(softmax(z_full)||softmax(z_pert)) with a positive/
  negative term split summed separately (each sum is cancellation-free),
  subtracted in fp64 - stable at the noise floor without paying a full
  fp64 vocab pass per candidate; validated against an fp64 reference in
  tests.
- SuffixRunner: prefix-activation cache. One clean forward caches the
  hidden states entering each candidate layer (+ the exact layer kwargs:
  rotary position embeddings, masks); perturbed logits then only run
  layers L..last + norm + lm_head.
- patched_weight / joint_{loi,loo}_weight: exact-restore perturbations.
- save_chunk / load_done: atomic (tmp+rename) checkpointing for resume.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

import qsal.config as cfg
from qsal.models import decoder_layers, final_norm, lm_head
from qsal.quantizers import quantize_layer

_KW_KEEP = ("attention_mask", "position_ids", "position_embeddings",
            "cache_position")


# Below this kl_mean, candidates are re-measured on the exact fp64 path.
# AMENDED 1e-4 -> 1e-6 2026-06-12 (docs/AMENDMENT_fp64_recheck_threshold
# .md): rule pre-registered before measurement; a 504-row stratified
# random sample of the 1e-6..1e-4 band then showed 0 rows shifting >5%
# (median 0.34%, p95 1.63%, max 2.85%; fast-vs-exact spearman 0.99997 -
# results/fp64_threshold_validation.json). The fast path's MEAN-level
# error at the real eval size is ~8e-8 (per-token rounding cancels in
# the average), so 1e-6 keeps a >12x margin. Guard: conclusion-
# participating rows in the skipped band are individually rechecked
# before the report is finalized.
KL_FP64_RECHECK_BELOW = 1e-6


def kl_per_token(logp_full: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
    """Per-token KL, fp64 result. logp_full = log_softmax(z_full) (fp32 ok)."""
    lq = torch.log_softmax(z_pert.float(), dim=-1)
    terms = logp_full.exp() * (logp_full - lq)
    pos = terms.clamp_min(0).sum(-1)
    neg = (-terms.clamp_max(0)).sum(-1)
    # .cpu() BEFORE .double(): MPS has no float64
    return (pos.cpu().double() - neg.cpu().double()).clamp_min(0.0)


def kl_per_token_fp64(z_full: torch.Tensor, z_pert: torch.Tensor,
                      chunk: int = 128) -> torch.Tensor:
    """Exact fp64 per-token KL from raw logits (CPU; near-floor recheck).

    Token-chunked: full-batch fp64 over a 152k vocab would allocate
    multiple ~2.5GB intermediates and swap-thrash 16GB machines."""
    zf = z_full.cpu().reshape(-1, z_full.shape[-1])
    zp = z_pert.cpu().reshape(-1, z_pert.shape[-1])
    out = torch.empty(zf.shape[0], dtype=torch.float64)
    for i in range(0, zf.shape[0], chunk):
        lp = torch.log_softmax(zf[i : i + chunk].double(), dim=-1)
        lq = torch.log_softmax(zp[i : i + chunk].double(), dim=-1)
        out[i : i + chunk] = (lp.exp() * (lp - lq)).sum(-1).clamp_min(0.0)
    return out.reshape(z_full.shape[:-1])


@contextmanager
def patched_weight(module, W_new: torch.Tensor):
    """Temporarily replace module.weight, restoring the exact original."""
    orig = module.weight.data
    module.weight.data = W_new.to(orig.device, orig.dtype)
    try:
        yield module
    finally:
        module.weight.data = orig


def joint_loo_weight(W: torch.Tensor, c: int,
                     bits=cfg.DEPLOYMENT_BITS) -> torch.Tensor:
    """Whole layer quantized at deployment bits, candidate column included."""
    return quantize_layer(W, bits=bits)


def joint_loi_weight(W: torch.Tensor, c: int,
                     bits=cfg.DEPLOYMENT_BITS) -> torch.Tensor:
    """Whole layer quantized, candidate column restored to full precision."""
    Wq = quantize_layer(W, bits=bits)
    Wq[:, c] = W[:, c]
    return Wq


class SuffixRunner:
    """Prefix-activation cache over the held-out eval set.

    Caches, per eval batch: hidden states entering each layer in
    cache_layers, the kwargs that layer received (position embeddings
    etc. - identical for all subsequent layers), clean logits, clean
    log-softmax, and clean per-token CE.
    """

    def __init__(self, model, eval_ids: torch.Tensor, cache_layers,
                 batch_size: int = 8, clean_on_device: bool = False):
        self.model = model
        self.device = next(model.parameters()).device
        self.cache_layers = sorted(set(int(x) for x in cache_layers))
        self.batches = [
            eval_ids[i : i + batch_size]
            for i in range(0, eval_ids.shape[0], batch_size)
        ]
        self._h_in = {L: [] for L in self.cache_layers}   # CPU fp32
        self._kwargs = {L: [] for L in self.cache_layers}
        # memory plan: cache ONLY log-softmax (fp32) + per-token logsumexp;
        # raw logits reconstruct exactly as logp + lse. Caching logits AND
        # logp would double the footprint (2 x ~5GB at 32x256) and swap.
        # clean_on_device=True keeps the ~5GB logp cache resident on the
        # compute device. A/B-profiled 2026-06-05 (interleaved trials,
        # identical candidates): 10% SLOWER than CPU residency on M1/16GB -
        # the per-measurement uploads overlap with GPU compute, while the
        # resident heap pressures the Metal working set. Hence default
        # False. (KL/dppl values identical either way; logit_mse differs
        # at ~1e-3 rel because the clean-logits add moves CPU<->MPS.)
        keep = (lambda t: t) if clean_on_device else (lambda t: t.cpu())
        logps, lses, ces = [], [], []

        handles = []
        layers = decoder_layers(model)
        for L in self.cache_layers:
            handles.append(
                layers[L].register_forward_pre_hook(
                    self._capture(L), with_kwargs=True
                )
            )
        try:
            with torch.no_grad():
                for ids in self.batches:
                    out = model(ids.to(self.device), use_cache=False).logits
                    logp = torch.log_softmax(out.float(), dim=-1)
                    # .clone(): the slice is a view; keeping it on-device
                    # would retain the full (B,T,V) base tensor (~1.2GB/batch)
                    lses.append(keep((out.float() - logp)[..., :1].clone()))
                    logps.append(keep(logp))
                    ces.append(self._ce(logp, ids.to(self.device)))
                    del out, logp
                    if self.device.type == "mps":
                        torch.mps.empty_cache()  # headroom for resident logp
        finally:
            for h in handles:
                h.remove()
        self._clean_logp = logps  # per batch, fp32 (device per clean_on_device)
        self._clean_lse = lses    # per batch, (B, T, 1) fp32
        self.clean_ce = torch.cat(ces)  # (seqs, T-1) fp64 CPU

    def clean_logits_batch(self, b: int) -> torch.Tensor:
        """Reconstruct the clean logits of batch b (fp32, on the clean
        cache's device)."""
        return self._clean_logp[b] + self._clean_lse[b]

    def _capture(self, L):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            self._h_in[L].append(h.detach().cpu())
            self._kwargs[L].append(
                {k: kwargs[k] for k in _KW_KEEP if k in kwargs}
            )

        return hook

    @staticmethod
    def _ce(logp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # per-token CE of next-token labels, fp64 CPU (for delta-ppl);
        # .cpu() BEFORE .double(): MPS has no float64
        lp = logp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        return -lp.cpu().double()

    def _to_dev(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, tuple):
            return tuple(self._to_dev(x) for x in obj)
        return obj

    # token-chunk width for streaming over the vocab dimension. A full
    # batch of logits is batch*T*152k*4B ~ 1.25GB and several of those
    # per measurement swap-thrash a 16GB machine; 32-token chunks cap
    # each vocab tensor at ~155MB.
    T_CHUNK = 32

    def _suffix_hidden(self, L: int, b: int) -> torch.Tensor:
        """Post-norm hidden states (B,T,hidden) for batch b, recomputing
        layers L..end from the cache. Small (no vocab dimension)."""
        h = self._h_in[L][b].to(self.device)
        kw = {k: self._to_dev(v) for k, v in self._kwargs[L][b].items()}
        with torch.no_grad():
            for layer in decoder_layers(self.model)[L:]:
                out = layer(h, **kw)
                h = out[0] if isinstance(out, tuple) else out
            return final_norm(self.model)(h)

    def logits_from_layer(self, L: int) -> torch.Tensor:
        """Full logits (CPU), chunk-projected through lm_head."""
        outs = []
        head = lm_head(self.model)
        with torch.no_grad():
            for b in range(len(self.batches)):
                h = self._suffix_hidden(L, b)
                outs.append(torch.cat([
                    head(h[:, t : t + self.T_CHUNK]).cpu()
                    for t in range(0, h.shape[1], self.T_CHUNK)
                ], dim=1))
        return torch.cat(outs)

    def _batch_stats(self, L: int, b: int, exact: bool):
        """Per-token (kl, ce) vectors + (mse_sum, n) for one batch,
        streaming lm_head/KL/CE/MSE over T_CHUNK-token slices."""
        ids = self.batches[b].to(self.device)
        h = self._suffix_hidden(L, b)
        T = h.shape[1]
        kls, ces, mse_sum, n = [], [], 0.0, 0
        head = lm_head(self.model)
        with torch.no_grad():
            for t in range(0, T, self.T_CHUNK):
                e = min(t + self.T_CHUNK, T)
                z = head(h[:, t:e])  # (B, tc, V)
                logp_c = self._clean_logp[b][:, t:e]
                if exact:
                    clean_c = logp_c + self._clean_lse[b][:, t:e]
                    kls.append(kl_per_token_fp64(clean_c, z.cpu()))
                else:
                    kls.append(kl_per_token(logp_c.to(self.device), z))
                # CE of next-token labels for positions t..e-1 (< T-1)
                lq = torch.log_softmax(z.float(), dim=-1)
                pe = min(e, T - 1)
                if t < pe:
                    lab = ids[:, t + 1 : pe + 1, None]
                    ces.append(
                        -lq[:, : pe - t].gather(-1, lab).squeeze(-1)
                        .cpu().double()
                    )
                clean_dev = (logp_c + self._clean_lse[b][:, t:e]).to(self.device)
                mse_sum += (z.float() - clean_dev).pow(2).sum().item()
                n += z.numel()
                del z, lq, clean_dev
        return (torch.cat(kls, dim=1).reshape(-1),
                torch.cat(ces, dim=1).reshape(-1), mse_sum, n)

    def stats_from_layer(self, L: int, exact_recheck: bool = False,
                         force_exact: bool = False) -> dict:
        """Streamed per-token KL/CE/logit-MSE vs clean, for the current
        (patched) weights. Returns kl mean/p95 (fp64), logit_mse, dppl.

        The fast path computes fp32 log-softmax with fp64 final reduction
        (absolute error floor ~1e-6, below forward nondeterminism). With
        exact_recheck=True, a kl_mean below KL_FP64_RECHECK_BELOW is
        re-measured on the exact fp64 path (second pass) [guard].
        force_exact=True skips the fast pass entirely (post-hoc near-floor
        re-measurement, scripts/03b_fp64_recheck.py)."""
        exact_used = False
        for attempt in (force_exact, True):
            kls, ces, mse_sum, n_logits = [], [], 0.0, 0
            for b in range(len(self.batches)):
                kl_b, ce_b, mse_b, n_b = self._batch_stats(L, b, exact=attempt)
                kls.append(kl_b)
                ces.append(ce_b)
                mse_sum += mse_b
                n_logits += n_b
            kl = torch.cat(kls)
            exact_used = attempt
            if attempt or not exact_recheck:
                break
            if kl.mean().item() >= KL_FP64_RECHECK_BELOW:
                break
        if self.device.type == "mps":
            torch.mps.empty_cache()  # allocator caching accumulates blocks
        pert_ce = torch.cat(ces)
        clean_ce = self.clean_ce.reshape(-1)
        return {
            "kl_mean": kl.mean().item(),
            "kl_p95": kl.quantile(0.95).item(),
            "logit_mse": mse_sum / n_logits,
            "dppl": torch.exp(pert_ce.mean()).item()
            - torch.exp(clean_ce.mean()).item(),
            "kl_exact_fp64": exact_used,
        }


# ---------------------------------------------------------------------------
# Atomic checkpointing
# ---------------------------------------------------------------------------


def save_chunk(rows, ckpt_dir) -> Path:
    """Atomically persist a chunk of result rows (tmp write + rename)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    while (ckpt_dir / f"chunk_{n:05d}.parquet").exists():
        n += 1
    final = ckpt_dir / f"chunk_{n:05d}.parquet"
    tmp = ckpt_dir / f"chunk_{n:05d}.parquet.tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, final)  # atomic on POSIX
    return final


def load_done(ckpt_dir) -> pd.DataFrame:
    """Load all completed rows; integrity-check (unreadable chunks dropped)."""
    ckpt_dir = Path(ckpt_dir)
    frames = []
    for p in sorted(ckpt_dir.glob("chunk_*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            print(f"WARNING: dropping unreadable checkpoint {p}")
            p.rename(str(p) + ".corrupt")
    if not frames:
        return pd.DataFrame({"channel_id": pd.Series(dtype="int64")})
    return pd.concat(frames, ignore_index=True)
