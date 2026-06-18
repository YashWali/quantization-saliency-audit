"""Phase 2 (Pythia) calibration — variant of 01_collect_calibration.py.

Same procedure as Phase 1 (the design spec) but driven by config_pythia and
namespaced under data/pythia/ so the Phase-1 artifacts are never touched. Uses
the family-aware model accessors (input_embedding -> gpt_neox.embed_in), so the
only architecture-specific line in Phase 1 (model.model.embed_tokens) is handled.

  Pass A: vector stats (sum|x|, sum x^2) + unified-metric grads (sum (g*x)^2),
          via a CE-loss backward with grad only on the input embedding.
  Pass B: fp64 CPU Hessians H = sum x x^T, saved per-linear then freed.

--limit N runs Pass A/B on the first N calibration sequences only (wiring smoke;
the held full run uses all cfg.CALIB_NUM_SEQS). Outputs: data/pythia/{channel_grid
.parquet, calib_stats.pt, eval_meta.json} + data/pythia/hessians/.
"""

import json
import resource
import sys
import time
from pathlib import Path

import torch

import importlib
import os as _os

_CFG = _os.environ.get("QSAL_CONFIG", "config_pythia")
cfg = importlib.import_module("qsal." + _CFG)
_RUN = _CFG.rsplit("config_", 1)[-1]  # "pythia" / "smollm2" -> data|results dir
from qsal.calibration import GradStatsAccumulator, StatsAccumulator, get_sequences
from qsal.models import (build_channel_grid, enumerate_linears,
                         input_embedding, load_model)
from qsal.provenance import build_manifest, save_manifest

HESSIAN_LAYERS_PER_PASS = 8

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / _RUN
HESS = DATA / "hessians"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(limit: int | None = None):
    cfg.set_global_seeds()
    DATA.mkdir(parents=True, exist_ok=True)
    HESS.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading {cfg.MODEL_ID} on {device}")
    model, tokenizer = load_model(device=device, cfg=cfg)

    # ---- data (pinned, seeded, disjoint) ----
    calib, calib_meta = get_sequences(
        tokenizer, cfg.CALIB_SPLIT, cfg.CALIB_NUM_SEQS, cfg.CALIB_SEQ_LEN, cfg.SEED
    )
    eval_seqs, eval_meta = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN, cfg.SEED
    )
    calib_rows = {tuple(r.tolist()) for r in calib}
    eval_rows = {tuple(r.tolist()) for r in eval_seqs}
    assert not calib_rows & eval_rows, "calib/eval overlap"
    assert calib_meta["split"] != eval_meta["split"]
    if limit is not None:
        calib = calib[:limit]
        log(f"SMOKE: pass A/B on {calib.shape[0]} calib seqs only")
    log(f"calib {tuple(calib.shape)} from {calib_meta['split']}, "
        f"eval {tuple(eval_seqs.shape)} from {eval_meta['split']}, disjoint OK")

    # ---- channel grid ----
    grid = build_channel_grid(model, cfg=cfg)
    assert len(grid) == cfg.EXPECTED_TOTAL_CHANNELS
    grid_path = DATA / "channel_grid.parquet"
    grid.to_parquet(grid_path, index=False)
    save_manifest(grid_path, build_manifest(stage="channel_grid"))
    log(f"channel grid saved ({len(grid)} channels)")

    entries = list(enumerate_linears(model, cfg=cfg))

    # ---- Pass A: vector stats + unified-metric grads ----
    for p in model.parameters():
        p.requires_grad_(False)
    input_embedding(model).weight.requires_grad_(True)
    vec = StatsAccumulator(entries, with_hessian=False)
    grad = GradStatsAccumulator(entries)
    t0 = time.time()
    with vec, grad:
        for i in range(calib.shape[0]):
            ids = calib[i : i + 1].to(device)
            out = model(ids, labels=ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            if (i + 1) % 16 == 0:
                log(f"pass A {i + 1}/{calib.shape[0]} ({time.time() - t0:.0f}s)")
    stats = {}
    for (layer, name), st in vec.stats.items():
        g = grad.stats[(layer, name)]
        assert st["token_count"] == g["token_count"] > 0, (layer, name)
        stats[f"L{layer}.{name}"] = {
            "sum_abs_x": st["sum_abs_x"],
            "sum_x2": st["sum_x2"],
            "sum_gx2": g["sum_gx2"],
            "token_count": st["token_count"],
        }
    stats_path = DATA / "calib_stats.pt"
    torch.save(stats, stats_path)
    save_manifest(
        stats_path,
        build_manifest(stage="calib_stats", calib=calib_meta,
                       hessian_matmul_dtype="float64_cpu"),
    )
    log(f"pass A done ({time.time() - t0:.0f}s); calib_stats.pt saved")

    (DATA / "eval_meta.json").write_text(json.dumps(eval_meta, indent=2))
    save_manifest(DATA / "eval_meta.json", build_manifest(stage="eval_meta"))

    # ---- Pass B: Hessians, layer-subset passes (derive-then-free) ----
    with torch.no_grad():
        for lo in range(0, cfg.NUM_LAYERS, HESSIAN_LAYERS_PER_PASS):
            hi = min(lo + HESSIAN_LAYERS_PER_PASS, cfg.NUM_LAYERS)
            subset = [e for e in entries if lo <= e[0] < hi]
            acc = StatsAccumulator(subset, with_hessian=True)
            t0 = time.time()
            with acc:
                for i in range(calib.shape[0]):
                    model(calib[i : i + 1].to(device))
                    if (i + 1) % 32 == 0:
                        log(f"pass B layers {lo}-{hi - 1}: "
                            f"{i + 1}/{calib.shape[0]} ({time.time() - t0:.0f}s)")
            for (layer, name), st in acc.stats.items():
                h_path = HESS / f"L{layer:02d}_{name}.pt"
                torch.save(
                    {"H": st["H"], "token_count": st["token_count"]}, h_path
                )
            save_manifest(
                HESS / f"layers_{lo}-{hi - 1}",
                build_manifest(stage="hessians", layers=list(range(lo, hi)),
                               calib_index_hash=calib_meta["index_hash"],
                               hessian_matmul_dtype="float64_cpu"),
            )
            del acc
            log(f"pass B layers {lo}-{hi - 1} done ({time.time() - t0:.0f}s)")

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    log(f"ALL DONE. peak RSS {peak_gb:.2f} GB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N calib seqs (wiring smoke; "
                         "the held full run uses all)")
    args = ap.parse_args()
    sys.exit(main(limit=args.limit))
