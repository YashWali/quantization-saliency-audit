"""POST-HOC robustness check (not pre-registered): symmetric confound controls
for RQ5.

The pre-registered partial-correlation analysis (04_analysis.py / Figure 4)
controls layer index, log weight norm, and the shared activation moment
E[|x_c|]. That set is asymmetric: it removes the activation scale the cheap
criteria embed, but not the gradient scale the unified gradient score embeds. A
reviewer can fairly ask whether the unified score's surviving partial (rho~0.85)
is just its own marginal gradient scale rather than genuine gradient information.

This script answers that empirically. It re-runs ONLY the calibration gradient
pass (the same seeded sequences, no Hessians, no ground-truth census) with a
local accumulator that records the per-channel gradient scale E[|g_c|] (the exact
analog of the activation control E[|x_c|]), then recomputes the metric-vs-GT
partials with a SYMMETRIC control set {layer, log||W||, E[|x|], E[|g|]}. What
survives this set for the unified score is the per-token gradient x activation
COUPLING, not either marginal scale.

Frozen pipeline and library are untouched; this is a separate artifact. Run per
model via QSAL_CONFIG (unset -> Qwen Phase-1; config_pythia; config_smollm2):
    QSAL_CONFIG=config_smollm2 python scripts/04c_symmetric_controls.py
Output: results/<run>/analysis/symmetric_controls.json
"""

import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_CFG = os.environ.get("QSAL_CONFIG", "config")
cfg = importlib.import_module("qsal." + _CFG)

from qsal.analysis.confounds import add_control_columns, confound_partials  # noqa: E402
from qsal.calibration import get_sequences  # noqa: E402
from qsal.models import (build_channel_grid, enumerate_linears,  # noqa: E402
                         input_embedding, load_model)

# Phase-1 (Qwen) analysis lives in 04_analysis.py; the others in
# phase2_04_analysis.py. Both expose load_inputs(), RESULTS, METHODS, SECONDARY.
_ANA_FILE = "04_analysis.py" if _CFG == "config" else "phase2_04_analysis.py"
_spec = importlib.util.spec_from_file_location("qsal_ana", ROOT / "scripts" / _ANA_FILE)
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GradScaleAccumulator:
    """Accumulates per-input-channel gradient scale: Sum_t |g_{t,c}| and
    Sum_t g_{t,c}^2. Same hook discipline as GradStatsAccumulator (forward-pre
    pushes x; full-backward pops the matching x — LIFO — and consumes g)."""

    def __init__(self, entries):
        self.entries = list(entries)
        self.stats = {
            (layer, name): {
                "sum_abs_g": torch.zeros(module.in_features, dtype=torch.float64),
                "sum_g2": torch.zeros(module.in_features, dtype=torch.float64),
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
            g = g.detach().reshape(-1, g.shape[-1]).cpu().double()
            st = self.stats[key]
            st["sum_abs_g"] += g.abs().sum(0)
            st["sum_g2"] += (g ** 2).sum(0)
            st["token_count"] += x.shape[0]
        return hook

    def __enter__(self):
        for layer, name, module in self.entries:
            key = (layer, name)
            self._handles.append(module.register_forward_pre_hook(self._fwd_hook(key)))
            self._handles.append(module.register_full_backward_hook(self._bwd_hook(key)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for buf in self._pending.values():
            buf.clear()
        return False


def collect_grad_scale(grid: pd.DataFrame) -> pd.DataFrame:
    """Re-run the calibration gradient pass; return per-channel E[|g|], E[g^2]
    aligned to channel_id."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading model on {device} (cfg={_CFG})")
    model, tokenizer = load_model(device=device, cfg=cfg)
    calib, _ = get_sequences(
        tokenizer, cfg.CALIB_SPLIT, cfg.CALIB_NUM_SEQS, cfg.CALIB_SEQ_LEN, cfg.SEED
    )
    entries = list(enumerate_linears(model, cfg=cfg))
    for p in model.parameters():
        p.requires_grad_(False)
    input_embedding(model).weight.requires_grad_(True)

    acc = GradScaleAccumulator(entries)
    t0 = time.time()
    with acc:
        for i in range(calib.shape[0]):
            ids = calib[i : i + 1].to(device)
            out = model(ids, labels=ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            if (i + 1) % 16 == 0:
                log(f"grad pass {i + 1}/{calib.shape[0]} ({time.time() - t0:.0f}s)")

    recs = []
    for (layer, name), st in acc.stats.items():
        n = st["token_count"]
        assert n > 0, (layer, name)
        mean_abs_g = (st["sum_abs_g"] / n).numpy()
        e_g2 = (st["sum_g2"] / n).numpy()
        sub = grid[(grid["layer"] == layer) & (grid["linear"] == name)] \
            .sort_values("in_channel")
        ic = sub["in_channel"].to_numpy()
        recs.append(pd.DataFrame({
            "channel_id": sub["channel_id"].to_numpy(),
            "mean_abs_g": mean_abs_g[ic],
            "e_g2": e_g2[ic],
        }))
    out = pd.concat(recs, ignore_index=True)
    assert np.isfinite(out[["mean_abs_g", "e_g2"]].to_numpy()).all()
    assert (out["mean_abs_g"] > 0).all()
    log(f"grad scale collected ({len(out)} channels, {time.time() - t0:.0f}s)")
    return out


def main():
    cfg.set_global_seeds()
    out_dir = ana.RESULTS / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _gt_mode = ana.load_inputs()
    grid = pd.read_parquet(ana.DATA / "channel_grid.parquet")

    grad_scale = collect_grad_scale(grid)
    df = df.merge(grad_scale, on="channel_id", how="left")

    sweep = df[df["layer"].isin(cfg.SWEEP_LAYERS)].dropna(
        subset=["gt_iso", "mean_abs_g"])
    log(f"RQ5 sweep population: {len(sweep)} channels")

    metrics = list(ana.METHODS) + list(ana.SECONDARY)
    ctrl = add_control_columns(sweep)

    baseline = confound_partials(ctrl, metrics, "gt_iso")  # pre-registered 3 controls
    symmetric = confound_partials(
        ctrl, metrics, "gt_iso",
        controls=("layer", "log_w_norm", "mean_abs_x", "mean_abs_g"))

    b = baseline.set_index("metric")
    s = symmetric.set_index("metric")
    rows = {}
    for m in metrics:
        rows[m] = {
            "raw_spearman": float(b.loc[m, "raw_spearman"]),
            "partial_baseline": float(b.loc[m, "partial_spearman"]),
            "partial_symmetric": float(s.loc[m, "partial_spearman"]),
        }
    payload = {
        "run": _CFG,
        "n_channels": int(len(sweep)),
        "baseline_controls": "layer,log_w_norm,mean_abs_x",
        "symmetric_controls": "layer,log_w_norm,mean_abs_x,mean_abs_g",
        "note": ("Post-hoc robustness check (not pre-registered). mean_abs_g = "
                 "E[|dL/dx_c|] over calibration, the gradient-scale analog of the "
                 "activation control mean_abs_x = E[|x_c|]. Re-run of the "
                 "calibration gradient pass only; ground truth and scores unchanged."),
        "metrics": rows,
    }
    out_path = out_dir / "symmetric_controls.json"
    out_path.write_text(json.dumps(payload, indent=2))

    log("metric  raw    base   symm")
    for m in metrics:
        r = rows[m]
        log(f"  {m:14s} {r['raw_spearman']:+.3f} "
            f"{r['partial_baseline']:+.3f} {r['partial_symmetric']:+.3f}")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
