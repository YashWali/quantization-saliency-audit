"""Split-half reliability of the per-channel JOINT ground truth (audit Note B).

The paper (§3.3) demotes the joint GT to set-level-only on the basis that the
per-channel joint signal has a split-half rank correlation of ~0.09 — i.e. one
channel's contribution inside a fully-INT4 linear is barely measurable at the
deployed eval size, because the layer-wide effect (~5e-3 KL) swamps per-channel
deltas (~1e-5). That number was computed once, informally, and is NOT
reproducible from shipped artifacts (no script, no per-half data). This script
makes it reproducible.

Method (deterministic, from shipped config — no new random draw):
  - Reconstruct the canonical 16-sequence eval set (cfg.SEED), split it into two
    disjoint 8-sequence halves A=[:8], B=[8:]. A true split-half.
  - For a seeded, stratified subsample of the CENSUS channels (layers 2/11/22 —
    the population the per-channel ranking claims are about), measure
    kl_joint_loi (whole linear @ INT4/g128, candidate column c restored to fp)
    on half A and on half B.
  - Report the split-half rank correlation of per-channel kl_joint_loi, pooled
    and within-linear (median across strata), plus the Spearman-Brown
    projection to the full 16-sequence eval (2r/(1+r)). For contrast, the same
    split-half correlation for the ISOLATED GT (kl_iso) is also computed — it
    should be high, demonstrating the asymmetry that justifies the demotion.

Resumable: atomic chunk checkpoints (results/gt_split_half_chunks). Re-run with
a larger --cap-per-stratum (or --full) for the whole census; the default
subsample is sized for a tight reliability estimate, not a full re-measurement.

Outputs: results/gt_split_half.parquet, results/analysis/split_half_reliability.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import qsal.config as cfg
from qsal.groundtruth import (SuffixRunner, load_done, patched_weight,
                              save_chunk)
from qsal.models import enumerate_linears, load_model
from qsal.quantizers import quantize_layer, quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = RESULTS / "gt_split_half_chunks"
CHUNK_ROWS = 25
HALF = cfg.EVAL_NUM_SEQS // 2  # 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_channels(grid: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """Seeded stratified subsample of census channels (cap per (layer,linear))."""
    census = grid[grid["layer"].isin(cfg.SWEEP_LAYERS)].copy()
    if cap is None:
        return census.sort_values(["layer", "linear", "in_channel"])
    rng = np.random.default_rng(cfg.SEED)
    picks = []
    for _, g in census.groupby(["layer", "linear"], observed=True):
        if len(g) <= cap:
            picks.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=cap, replace=False)
            picks.append(g.loc[np.sort(idx)])
    return pd.concat(picks).sort_values(["layer", "linear", "in_channel"])


def main(cap: int | None, max_seconds: float | None):
    cfg.set_global_seeds()
    deadline = time.time() + max_seconds if max_seconds else None
    (RESULTS / "analysis").mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    eval_meta = json.loads((DATA / "eval_meta.json").read_text())

    from qsal.calibration import get_sequences

    eval_seqs, em = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN, cfg.SEED
    )
    assert em["index_hash"] == eval_meta["index_hash"], "eval set drifted"
    half_a, half_b = eval_seqs[:HALF], eval_seqs[HALF:]
    log(f"eval split: A={half_a.shape[0]} seqs, B={half_b.shape[0]} seqs")

    chans = select_channels(grid, cap)
    layers_needed = sorted(chans["layer"].unique())
    log(f"channels: {len(chans)} over {chans['linear'].nunique()} linears "
        f"x {len(layers_needed)} layers (cap/stratum={cap})")

    t0 = time.time()
    # batch_size=HALF so each half is a single clean batch in its runner
    runner_a = SuffixRunner(model, half_a, cache_layers=layers_needed,
                            batch_size=HALF)
    runner_b = SuffixRunner(model, half_b, cache_layers=layers_needed,
                            batch_size=HALF)
    log(f"two prefix caches built ({time.time() - t0:.0f}s)")

    modules = {(l, n): m for l, n, m in enumerate_linears(model)}
    done = set(load_done(CKPT)["channel_id"])
    log(f"resume: {len(done)} channels already measured")

    todo = chans[~chans["channel_id"].isin(done)]
    rows, measured, t0 = [], 0, time.time()
    for (layer, linear), group in todo.groupby(["layer", "linear"],
                                               sort=False, observed=True):
        mod = modules[(layer, linear)]
        W = mod.weight.detach().cpu()
        Wq_base = quantize_layer(W, bits=cfg.DEPLOYMENT_BITS)  # once per linear
        for row in group.itertuples():
            c = row.in_channel
            # joint LOI: whole-linear INT4 with column c restored to fp
            Wq = Wq_base.clone()
            Wq[:, c] = W[:, c]
            with patched_weight(mod, Wq):
                a = runner_a.stats_from_layer(layer)
                b = runner_b.stats_from_layer(layer)
            # isolated (INT3 single column) for the contrast control
            with patched_weight(mod, quantize_single_column(
                    W, c, bits=cfg.ISOLATED_PROBE_BITS)):
                ia = runner_a.stats_from_layer(layer)
                ib = runner_b.stats_from_layer(layer)
            rows.append({
                "channel_id": row.channel_id, "layer": layer,
                "linear": linear, "in_channel": c,
                "loi_a": a["kl_mean"], "loi_b": b["kl_mean"],
                "iso_a": ia["kl_mean"], "iso_b": ib["kl_mean"],
            })
            measured += 1
            if (deadline and time.time() >= deadline):
                save_chunk(rows, CKPT)
                log(f"max-seconds reached after {measured} "
                    f"({measured / (time.time() - t0):.2f} ch/s) - checkpointed")
                return 0
            if len(rows) >= CHUNK_ROWS:
                save_chunk(rows, CKPT)
                rows = []
                rate = measured / (time.time() - t0)
                left = len(todo) - measured
                log(f"{measured}/{len(todo)} ({rate:.2f} ch/s, "
                    f"~{left / rate / 3600:.1f}h left)")
    if rows:
        save_chunk(rows, CKPT)

    finalize()
    return 0


def finalize():
    """Compute split-half reliability stats from all checkpointed rows."""
    df = load_done(CKPT)
    if df.empty:
        log("no rows to finalize")
        return
    df = df.drop_duplicates("channel_id")
    df.to_parquet(RESULTS / "gt_split_half.parquet", index=False)

    def sb(r):  # Spearman-Brown: half-length r -> full-length (2 halves)
        return 2 * r / (1 + r) if r > -1 else float("nan")

    def within(a, b):
        rs = []
        for _, g in df.groupby(["layer", "linear"], observed=True):
            if len(g) >= 5:
                rs.append(spearmanr(g[a], g[b]).statistic)
        return float(np.nanmedian(rs)), len(rs)

    out = {"n_channels": int(len(df)), "n_strata": int(
        df.groupby(["layer", "linear"]).ngroups), "half_seqs": HALF}
    for name, a, b in [("joint_loi", "loi_a", "loi_b"),
                       ("isolated", "iso_a", "iso_b")]:
        pooled = float(spearmanr(df[a], df[b]).statistic)
        win, n = within(a, b)
        out[name] = {
            "split_half_pooled_spearman": pooled,
            "split_half_within_linear_median": win,
            "n_strata_used": n,
            "spearman_brown_full_eval_pooled": sb(pooled),
            "spearman_brown_full_eval_within": sb(win),
        }
        log(f"{name}: split-half pooled rho={pooled:.3f} "
            f"(SB->16seq {sb(pooled):.3f}); within-linear median={win:.3f}")
    with open(RESULTS / "analysis" / "split_half_reliability.json", "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/analysis/split_half_reliability.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-stratum", type=int, default=200,
                    help="max channels sampled per (layer,linear); default 200 "
                         "(~4.2k channels for a tight estimate)")
    ap.add_argument("--full", action="store_true",
                    help="use the whole census (overrides --cap-per-stratum)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="checkpoint and exit after this long (resumable)")
    ap.add_argument("--finalize-only", action="store_true",
                    help="recompute stats from existing checkpoints and exit")
    args = ap.parse_args()
    if args.finalize_only:
        finalize()
        sys.exit(0)
    sys.exit(main(cap=None if args.full else args.cap_per_stratum,
                  max_seconds=args.max_seconds))
