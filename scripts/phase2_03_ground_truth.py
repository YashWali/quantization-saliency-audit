"""Phase 2 (Pythia) ground truth — single-pass runner (the design spec, choice 4).

One script, no separate recheck/guard/split-half passes (the Phase-1 lesson):

- Isolated GT (INT3, real-group scale) for EVERY candidate: full sweep-layer
  census + per-criterion union + random controls.
- Joint GT (whole linear @ INT4/g128, leave-one-in) for the union+control SET
  ONLY (set-level sign test) + the per-linear leave-one-out. Census-only
  channels get isolated GT only (choice 4: joint is set-level, not per-census).
- Phase-2-native split-half reliability: a seeded stratified subset
  (RELIABILITY_SUBSET_PER_STRATUM per sweep stratum) is measured on two disjoint
  8-seq eval halves (isolated + joint LOI), so the within-linear "joint is
  set-level only" demotion is RE-EARNED on Pythia (reported within-linear + SB by
  the analysis), not assumed from Phase 1.
- Exact-fp64 near-floor is inline (exact_recheck=True), at the pre-registered
  1e-6 threshold - no separate 03b/06 pass.

Driven by config_pythia; reads data/pythia/ + results/pythia/scores.parquet;
writes results/pythia/{ground_truth,gt_loo,gt_split_half}.parquet. Atomic chunk
checkpoints -> resumable. --limit / --max-seconds for bounded sessions; --limit
also caps the reliability subset (wiring smoke). The full run is a separate GO.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import importlib
import os as _os

_CFG = _os.environ.get("QSAL_CONFIG", "config_pythia")
cfg = importlib.import_module("qsal." + _CFG)
_RUN = _CFG.rsplit("config_", 1)[-1]  # "pythia" / "smollm2" -> data|results dir
from qsal.groundtruth import (SuffixRunner, joint_loi_weight, joint_loo_weight,
                              load_done, patched_weight, save_chunk)
from qsal.models import enumerate_linears, load_model
from qsal.phase2 import build_candidates, reliability_subset
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / _RUN
RESULTS = ROOT / "results" / _RUN
CKPT = RESULTS / "gt_chunks"
CKPT_LOO = RESULTS / "gt_loo_chunks"
CKPT_REL = RESULTS / "gt_split_half_chunks"
CHUNK_ROWS = 25
N_NOISE_CHANNELS = 8
METRICS = ["awq", "gptq", "owq_residual", "spqr", "unified"]
HALF = cfg.EVAL_NUM_SEQS // 2  # split-half: two disjoint 8-seq halves


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tranche_order(cand: pd.DataFrame) -> pd.DataFrame:
    """controls -> union -> sweep L22 -> L11 -> L2, grouped by (layer, linear)
    so each linear's LOO is measured once and reused."""
    def prio(row):
        if row["source"] == "control":
            return 0
        if row["source"] == "union":
            return 1
        return {22: 2, 11: 3, 2: 4}.get(row["layer"], 5)

    cand = cand.copy()
    cand["prio"] = cand.apply(prio, axis=1)
    return cand.sort_values(["prio", "layer", "linear", "in_channel"])


def main(limit=None, max_seconds=None):
    cfg.set_global_seeds()
    deadline = time.time() + max_seconds if max_seconds else None
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model(device=device, cfg=cfg)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet")
    eval_meta = json.loads((DATA / "eval_meta.json").read_text())

    from qsal.calibration import get_sequences

    eval_seqs, em = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN, cfg.SEED
    )
    assert em["index_hash"] == eval_meta["index_hash"], "eval set drifted"

    cand = tranche_order(build_candidates(
        grid, scores, metrics=METRICS, sweep_layers=cfg.SWEEP_LAYERS,
        n_control=cfg.NUM_CONTROL_CHANNELS, top_pct=cfg.TOP_PCT_UNION,
        seed=cfg.SEED,
    ))
    subset_ids = reliability_subset(
        grid, cfg.SWEEP_LAYERS, cfg.RELIABILITY_SUBSET_PER_STRATUM, cfg.SEED
    )
    log(f"candidates: {len(cand)} "
        f"({(cand['source'] == 'union').sum()} union, "
        f"{(cand['source'] == 'control').sum()} control, "
        f"{(cand['source'] == 'sweep').sum()} sweep); "
        f"reliability subset {len(subset_ids)}")

    modules = {(l, n): m for l, n, m in enumerate_linears(model, cfg=cfg)}

    layers_needed = sorted(cand["layer"].unique())
    t0 = time.time()
    runner = SuffixRunner(model, eval_seqs, cache_layers=layers_needed)
    log(f"clean pass + prefix cache built ({time.time() - t0:.0f}s, "
        f"{len(layers_needed)} layers)")

    # ---- noise floor: identical-perturbation repeats ----
    nf_path = RESULTS / "gt_noise_floor.parquet"
    if not nf_path.exists():
        nf_rows = []
        picks = cand.sample(n=min(N_NOISE_CHANNELS, len(cand)),
                            random_state=cfg.SEED).itertuples()
        for row in picks:
            mod = modules[(row.layer, row.linear)]
            W = mod.weight.detach().cpu()
            Wp = quantize_single_column(W, row.in_channel,
                                        bits=cfg.ISOLATED_PROBE_BITS)
            for rep in range(cfg.NOISE_FLOOR_REPEATS):
                with patched_weight(mod, Wp):
                    st = runner.stats_from_layer(row.layer, exact_recheck=True)
                nf_rows.append({"channel_id": row.channel_id, "repeat": rep,
                                **{f"kl_iso_{k}": v for k, v in st.items()}})
        nf_df = pd.DataFrame(nf_rows)
        nf_df.to_parquet(nf_path, index=False)
        save_manifest(nf_path, build_manifest(stage="gt_noise_floor_pythia"))
        spread = nf_df.groupby("channel_id")["kl_iso_kl_mean"].agg(["mean", "std"])
        log("noise floor (repeat std / mean; std~0 = deterministic):\n"
            + spread.to_string(float_format="%.3e"))

    # ---- main loop: iso for all; joint LOI for union+control only ----
    done = set(load_done(CKPT)["channel_id"])
    loo_done_df = load_done(CKPT_LOO)
    done_loo = set(zip(loo_done_df.get("layer", pd.Series(dtype=object)),
                       loo_done_df.get("linear", pd.Series(dtype=object))))
    log(f"resume: {len(done)} candidates, {len(done_loo)} LOO done")

    rows, loo_rows, measured, t0 = [], [], 0, time.time()
    todo = cand[~cand["channel_id"].isin(done)]
    for (layer, linear), group in todo.groupby(["layer", "linear"],
                                               sort=False, observed=True):
        mod = modules[(layer, linear)]
        W = mod.weight.detach().cpu()
        need_joint_linear = (group["source"].isin(["union", "control"]).any())

        if need_joint_linear and (layer, linear) not in done_loo:
            with patched_weight(mod, joint_loo_weight(W, 0)):
                st = runner.stats_from_layer(layer, exact_recheck=True)
            loo_rows.append({"layer": layer, "linear": linear,
                             **{f"kl_joint_loo_{k}": v for k, v in st.items()}})
            done_loo.add((layer, linear))
            if len(loo_rows) >= 10:
                save_chunk(loo_rows, CKPT_LOO)
                loo_rows = []

        for row in group.itertuples():
            with patched_weight(
                mod, quantize_single_column(W, row.in_channel,
                                            bits=cfg.ISOLATED_PROBE_BITS)
            ):
                iso = runner.stats_from_layer(layer, exact_recheck=True)
            rec = {
                "channel_id": row.channel_id, "source": row.source,
                "in_union": row.in_union, "in_sweep": row.in_sweep,
                "layer": layer, "linear": linear, "in_channel": row.in_channel,
                "probe_bits": cfg.ISOLATED_PROBE_BITS,
                "deploy_bits": cfg.DEPLOYMENT_BITS,
                **{f"kl_iso_{k}": v for k, v in iso.items()},
            }
            if row.source in ("union", "control"):
                with patched_weight(mod, joint_loi_weight(W, row.in_channel)):
                    loi = runner.stats_from_layer(layer, exact_recheck=True)
                rec.update({f"kl_joint_loi_{k}": v for k, v in loi.items()})
            rows.append(rec)
            measured += 1
            hit_limit = limit is not None and measured >= limit
            hit_deadline = deadline is not None and time.time() >= deadline
            if hit_limit or hit_deadline:
                save_chunk(rows, CKPT)
                if loo_rows:
                    save_chunk(loo_rows, CKPT_LOO)
                log(f"stop ({'limit' if hit_limit else 'max-seconds'}) after "
                    f"{measured} candidates - checkpointed; rerun to resume")
                return 0
            if len(rows) >= CHUNK_ROWS:
                save_chunk(rows, CKPT)
                rows = []
                rate = measured / (time.time() - t0)
                left = len(todo) - measured
                log(f"{measured}/{len(todo)} this run "
                    f"({rate:.2f} cand/s, ~{left / rate / 3600:.1f}h left)")
    if rows:
        save_chunk(rows, CKPT)
    if loo_rows:
        save_chunk(loo_rows, CKPT_LOO)

    # ---- reliability phase: subset measured on two disjoint eval halves ----
    rel_done = set(load_done(CKPT_REL)["channel_id"])
    rel_todo = (cand[cand["channel_id"].isin(subset_ids)
                     & ~cand["channel_id"].isin(rel_done)]
                .sort_values(["layer", "linear", "in_channel"]))
    if len(rel_todo):
        log(f"reliability: {len(rel_todo)} subset channels on two {HALF}-seq "
            "halves")
        del runner  # free the full-eval prefix cache before building half caches
        if device == "mps":
            torch.mps.empty_cache()
        half_a = SuffixRunner(model, eval_seqs[:HALF], cache_layers=cfg.SWEEP_LAYERS)
        half_b = SuffixRunner(model, eval_seqs[HALF:], cache_layers=cfg.SWEEP_LAYERS)
        rel_rows = []
        for row in rel_todo.itertuples():
            mod = modules[(row.layer, row.linear)]
            W = mod.weight.detach().cpu()
            Wiso = quantize_single_column(W, row.in_channel,
                                          bits=cfg.ISOLATED_PROBE_BITS)
            Wloi = joint_loi_weight(W, row.in_channel)
            vals = {}
            for tag, half in (("a", half_a), ("b", half_b)):
                with patched_weight(mod, Wiso):
                    vals[f"iso_{tag}"] = half.stats_from_layer(
                        row.layer, exact_recheck=True)["kl_mean"]
                with patched_weight(mod, Wloi):
                    vals[f"loi_{tag}"] = half.stats_from_layer(
                        row.layer, exact_recheck=True)["kl_mean"]
            rel_rows.append({"channel_id": row.channel_id, "layer": row.layer,
                             "linear": row.linear, "in_channel": row.in_channel,
                             **vals})
            if (deadline and time.time() >= deadline) or len(rel_rows) >= CHUNK_ROWS:
                save_chunk(rel_rows, CKPT_REL)
                rel_rows = []
                if deadline and time.time() >= deadline:
                    log("max-seconds reached in reliability phase - checkpointed")
                    return 0
        if rel_rows:
            save_chunk(rel_rows, CKPT_REL)

    # ---- finalize ----
    gt = load_done(CKPT)
    loo = load_done(CKPT_LOO)
    rel = load_done(CKPT_REL)
    gt_path = RESULTS / "ground_truth.parquet"
    gt.to_parquet(gt_path, index=False)
    loo.to_parquet(RESULTS / "gt_loo.parquet", index=False)
    if len(rel):
        rel.drop_duplicates("channel_id").to_parquet(
            RESULTS / "gt_split_half.parquet", index=False)
    save_manifest(gt_path, build_manifest(
        stage="ground_truth_pythia", eval_index_hash=eval_meta["index_hash"],
        probe_bits=cfg.ISOLATED_PROBE_BITS, deploy_bits=cfg.DEPLOYMENT_BITS,
        forward_device=device, joint_scope="union+control set + reliability subset",
    ))
    log(f"DONE: {len(gt)} candidates, {len(loo)} LOOs, "
        f"{len(rel)} reliability-subset channels")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N candidates this run (resumable; "
                         "wiring smoke)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="checkpoint and exit cleanly after this much time")
    args = ap.parse_args()
    sys.exit(main(limit=args.limit, max_seconds=args.max_seconds))
