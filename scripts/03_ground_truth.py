"""Measure the co-primary ground truth (the design spec).

Candidates: per-metric within-linear top-1% union + 1000 seeded random
controls + every channel of the sweep layers. Measurement order is
priority-tranched (noise-floor repeats -> controls -> union -> sweep
22 -> 11 -> 2) so the run can be interrupted after any tranche and
analyzed; atomic chunk checkpoints make it resumable.

Per candidate: kl_iso (INT3 single column, real-group scale) and
kl_joint_loi (whole linear @ INT4/g128, candidate column restored).
kl_joint_loo (whole linear quantized incl. c) is IDENTICAL for all
candidates of a linear -> measured once per linear, stored separately
and joined in analysis.

Outputs: results/ground_truth.parquet, results/gt_loo.parquet,
results/gt_noise_floor.parquet (+ provenance, sanity-gate summary).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import qsal.config as cfg
from qsal.groundtruth import (SuffixRunner, joint_loi_weight, joint_loo_weight,
                              load_done, patched_weight, save_chunk)
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = RESULTS / "gt_chunks"
CKPT_LOO = RESULTS / "gt_loo_chunks"
CHUNK_ROWS = 25
N_NOISE_CHANNELS = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_candidates(grid: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """source-tagged candidate table (deduped, union > sweep > control)."""
    rng = np.random.default_rng(cfg.SEED)
    s = scores.merge(grid, on="channel_id")
    metrics = ["awq", "gptq", "owq", "spqr", "unified"]

    union_ids = set()
    for m in metrics:
        ranks = s.groupby(["layer", "linear"], observed=True)[m].rank(
            ascending=False, pct=True
        )
        union_ids |= set(s.loc[ranks <= cfg.TOP_PCT_UNION, "channel_id"])

    sweep_ids = set(grid.loc[grid["layer"].isin(cfg.SWEEP_LAYERS), "channel_id"])
    pool = np.array(sorted(set(grid["channel_id"]) - union_ids - sweep_ids))
    control_ids = set(rng.choice(pool, size=cfg.NUM_CONTROL_CHANNELS,
                                 replace=False).tolist())

    cand = grid[grid["channel_id"].isin(union_ids | sweep_ids | control_ids)
                ].copy()
    cand["in_union"] = cand["channel_id"].isin(union_ids)
    cand["in_sweep"] = cand["channel_id"].isin(sweep_ids)
    cand["source"] = np.where(
        cand["in_union"], "union",
        np.where(cand["in_sweep"], "sweep", "control"),
    )
    return cand


def tranche_order(cand: pd.DataFrame) -> pd.DataFrame:
    """Priority: controls -> union -> sweep L22 -> L11 -> L2; grouped by
    (layer, linear) within a tranche so each linear's LOO is reused."""
    def prio(row):
        if row["source"] == "control":
            return 0
        if row["source"] == "union":
            return 1
        return {22: 2, 11: 3, 2: 4}.get(row["layer"], 5)

    cand = cand.copy()
    cand["prio"] = cand.apply(prio, axis=1)
    return cand.sort_values(["prio", "layer", "linear", "in_channel"])


def main(limit: int | None = None, max_seconds: float | None = None):
    cfg.set_global_seeds()
    deadline = time.time() + max_seconds if max_seconds else None
    RESULTS.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet")
    eval_meta = json.loads((DATA / "eval_meta.json").read_text())

    from qsal.calibration import get_sequences

    eval_seqs, em = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN, cfg.SEED
    )
    assert em["index_hash"] == eval_meta["index_hash"], "eval set drifted"

    cand = tranche_order(build_candidates(grid, scores))
    log(f"candidates: {len(cand)} "
        f"({(cand['source'] == 'union').sum()} union, "
        f"{(cand['source'] == 'control').sum()} control, "
        f"{(cand['source'] == 'sweep').sum()} sweep)")

    layers_needed = sorted(cand["layer"].unique())
    t0 = time.time()
    runner = SuffixRunner(model, eval_seqs, cache_layers=layers_needed)
    log(f"clean pass + prefix cache built ({time.time() - t0:.0f}s, "
        f"{len(layers_needed)} layers)")

    modules = {(l, n): m for l, n, m in enumerate_linears(model)}

    # ---- noise floor: N channels x NOISE_FLOOR_REPEATS identical runs ----
    nf_path = RESULTS / "gt_noise_floor.parquet"
    if not nf_path.exists():
        rng = np.random.default_rng(cfg.SEED + 1)
        nf_rows = []
        picks = cand.sample(n=N_NOISE_CHANNELS,
                            random_state=cfg.SEED).itertuples()
        for row in picks:
            mod = modules[(row.layer, row.linear)]
            W = mod.weight.detach().cpu()
            Wp = quantize_single_column(W, row.in_channel,
                                        bits=cfg.ISOLATED_PROBE_BITS)
            for rep in range(cfg.NOISE_FLOOR_REPEATS):
                with patched_weight(mod, Wp):
                    st = runner.stats_from_layer(row.layer,
                                                 exact_recheck=True)
                nf_rows.append({"channel_id": row.channel_id, "repeat": rep,
                                **{f"kl_iso_{k}": v for k, v in st.items()}})
        pd.DataFrame(nf_rows).to_parquet(nf_path, index=False)
        save_manifest(nf_path, build_manifest(stage="gt_noise_floor"))
        df = pd.DataFrame(nf_rows)
        spread = df.groupby("channel_id")["kl_iso_kl_mean"].agg(["mean", "std"])
        log("noise floor (repeat std / mean):\n"
            + spread.to_string(float_format="%.3e"))

    # ---- main measurement loop ----
    done = set(load_done(CKPT)["channel_id"])
    done_loo = set(
        zip(*[load_done(CKPT_LOO).get(k, pd.Series(dtype=object))
              for k in ("layer", "linear")])
    )
    log(f"resume: {len(done)} candidates, {len(done_loo)} LOO already done")

    rows, loo_rows, measured, t0 = [], [], 0, time.time()
    todo = cand[~cand["channel_id"].isin(done)]
    for (layer, linear), group in todo.groupby(["layer", "linear"],
                                               sort=False, observed=True):
        mod = modules[(layer, linear)]
        W = mod.weight.detach().cpu()

        if (layer, linear) not in done_loo:
            with patched_weight(mod, joint_loo_weight(W, 0)):
                st = runner.stats_from_layer(layer)
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
                iso = runner.stats_from_layer(layer)
            with patched_weight(mod, joint_loi_weight(W, row.in_channel)):
                loi = runner.stats_from_layer(layer)
            rows.append({
                "channel_id": row.channel_id, "source": row.source,
                "in_union": row.in_union, "in_sweep": row.in_sweep,
                "layer": layer, "linear": linear, "in_channel": row.in_channel,
                "probe_bits": cfg.ISOLATED_PROBE_BITS,
                "deploy_bits": cfg.DEPLOYMENT_BITS,
                **{f"kl_iso_{k}": v for k, v in iso.items()},
                **{f"kl_joint_loi_{k}": v for k, v in loi.items()},
            })
            measured += 1
            hit_limit = limit is not None and measured >= limit
            hit_deadline = deadline is not None and time.time() >= deadline
            if hit_limit or hit_deadline:
                save_chunk(rows, CKPT)
                if loo_rows:
                    save_chunk(loo_rows, CKPT_LOO)
                why = f"limit {limit}" if hit_limit else \
                    f"max-seconds {max_seconds:.0f}"
                log(f"{why} reached after {measured} candidates "
                    f"({measured / (time.time() - t0):.2f} cand/s) - "
                    "checkpointed; rerun to resume")
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

    # ---- finalize ----
    gt = load_done(CKPT)
    loo = load_done(CKPT_LOO)
    gt_path = RESULTS / "ground_truth.parquet"
    gt.to_parquet(gt_path, index=False)
    loo.to_parquet(RESULTS / "gt_loo.parquet", index=False)
    save_manifest(gt_path, build_manifest(
        stage="ground_truth", eval_index_hash=eval_meta["index_hash"],
        probe_bits=cfg.ISOLATED_PROBE_BITS, deploy_bits=cfg.DEPLOYMENT_BITS,
        forward_device=device,
    ))
    log(f"DONE: {len(gt)} candidates, {len(loo)} linear LOOs measured")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N candidates this run (resumable)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="checkpoint and exit cleanly after this much time "
                         "(resumable; e.g. 3600 for a 1h session)")
    args = ap.parse_args()
    sys.exit(main(limit=args.limit, max_seconds=args.max_seconds))
