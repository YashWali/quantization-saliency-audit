"""Belt-and-suspenders guard for the fp64-threshold amendment
(docs/AMENDMENT_fp64_recheck_threshold.md): individually re-measure, on
the exact fp64 path, every *individually load-bearing* row in the
skipped band (KL_FP64_RECHECK_BELOW <= kl_iso < 1e-4), and bound the
band's impact on aggregate statistics analytically.

Individually load-bearing =
  (a) member of any method's within-stratum top-1% (TOP_PCT_UNION) set
      - the sets whose GT values feed RQ3 GT-fractions and RQ4
      overprotection. (Top-5/10% sets enter RQ2 only via score-based
      MEMBERSHIP; their GT values are never read, so they are not
      load-bearing.),
  (b) member of the GT top-1% of the union+control pool (precision@k
      reference set),
  (c) all joint-LOI rows with kl_joint_loi < 1e-4 (the 64 below the
      original threshold; LOI was excluded from the amended recheck).

Aggregate guard: for sums over the band (RQ1 shares etc.) the absolute
error is bounded by p95_relΔ × Σ(band values); printed and saved.

Chunks land in results/gt_fp64_recheck_chunks/ (same schema as 03b).
Heavy (model): run AFTER 03b DONE, never concurrently.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import qsal.config as cfg
from qsal.groundtruth import (KL_FP64_RECHECK_BELOW, SuffixRunner,
                              joint_loi_weight, load_done, patched_weight,
                              save_chunk)
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT_MAIN = RESULTS / "gt_chunks"
CKPT = RESULTS / "gt_fp64_recheck_chunks"

BAND_HI = 1e-4
METHODS = ["awq", "gptq", "owq_residual", "spqr", "unified", "owq_energy"]
CHUNK_ROWS = 25
P95_REL = 0.0163  # measured: fp64_threshold_validation.json


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_band_and_sets():
    gt = load_done(CKPT_MAIN)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet").merge(
        pd.read_parquet(RESULTS / "scores_owq.parquet"), on="channel_id")
    df = grid.merge(scores, on="channel_id")
    df["stratum"] = df["layer"].astype(str) + ":" + df["linear"].astype(str)

    selected = set()
    for m in METHODS:
        for _, g in df.groupby("stratum", observed=True):
            k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
            selected |= set(g.nlargest(k, m)["channel_id"].tolist())

    pool = gt[gt["source"].isin(["union", "control"])]
    k = max(1, int(round(0.01 * len(pool))))
    gt_top = set(pool.nlargest(k, "kl_iso_kl_mean")["channel_id"])

    band = gt[(gt["kl_iso_kl_mean"] >= KL_FP64_RECHECK_BELOW)
              & (gt["kl_iso_kl_mean"] < BAND_HI)]
    done = load_done(CKPT)
    have_iso = set(done.dropna(subset=["kl_iso_fp64_kl_mean"])["channel_id"]) \
        if "kl_iso_fp64_kl_mean" in done else set()
    have_loi = set(
        done.dropna(subset=["kl_joint_loi_fp64_kl_mean"])["channel_id"]) \
        if "kl_joint_loi_fp64_kl_mean" in done else set()

    iso_targets = band[band["channel_id"].isin((selected | gt_top)
                                               - have_iso)]
    loi_targets = gt[(gt["kl_joint_loi_kl_mean"] < BAND_HI)
                     & ~gt["channel_id"].isin(have_loi)]

    # analytic bound for aggregates (everything not individually checked)
    rest = band[~band["channel_id"].isin(set(iso_targets["channel_id"]))]
    bound = {
        "n_band": int(len(band)),
        "n_individually_rechecked": int(len(iso_targets)),
        "n_loi_rechecked": int(len(loi_targets)),
        "n_bounded_only": int(len(rest)),
        "band_mass": float(rest["kl_iso_kl_mean"].sum()),
        "p95_rel_delta": P95_REL,
        "abs_error_bound_on_any_band_sum":
            float(P95_REL * rest["kl_iso_kl_mean"].sum()),
        "share_of_total_sweep_gt": float(
            rest["kl_iso_kl_mean"].sum()
            / gt[gt["in_sweep"]]["kl_iso_kl_mean"].sum()),
    }
    log(f"guard targets: {len(iso_targets)} iso (of {len(band)} band rows; "
        f"{len(rest)} bounded analytically), {len(loi_targets)} loi")
    total_sweep = gt[gt["in_sweep"]]["kl_iso_kl_mean"].sum()
    bound["error_bound_share_of_sweep_gt"] = float(
        bound["abs_error_bound_on_any_band_sum"] / total_sweep)
    log(f"aggregate bound: any band sum is within "
        f"{bound['abs_error_bound_on_any_band_sum']:.3e} nats = "
        f"{bound['error_bound_share_of_sweep_gt']:.3%} of sweep GT mass "
        f"(band itself holds {bound['share_of_total_sweep_gt']:.1%})")
    return iso_targets, loi_targets, bound


def main(max_seconds: float | None = None):
    cfg.set_global_seeds()
    deadline = time.time() + max_seconds if max_seconds else None
    iso_targets, loi_targets, bound = load_band_and_sets()
    out_json = RESULTS / "guard_recheck_summary.json"

    todo = pd.concat([
        iso_targets.assign(do_iso=True, do_loi=False),
        loi_targets.assign(do_iso=False, do_loi=True),
    ], ignore_index=True)
    if todo.empty:
        log("nothing to guard-recheck")
        with open(out_json, "w") as f:
            json.dump(bound, f, indent=2)
        return 0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    from qsal.calibration import get_sequences
    eval_seqs, _ = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN,
        cfg.SEED,
    )
    runner = SuffixRunner(model, eval_seqs,
                          cache_layers=sorted(todo["layer"].unique()))
    modules = {(l, n): m for l, n, m in enumerate_linears(model)}

    rows, measured, t0 = [], 0, time.time()
    for (layer, linear), group in todo.groupby(["layer", "linear"],
                                               sort=False, observed=True):
        mod = modules[(layer, linear)]
        W = mod.weight.detach().cpu()
        for row in group.itertuples():
            out = {"channel_id": row.channel_id}
            if row.do_iso:
                with patched_weight(
                    mod, quantize_single_column(W, row.in_channel,
                                                bits=cfg.ISOLATED_PROBE_BITS)
                ):
                    st = runner.stats_from_layer(layer, force_exact=True)
                out.update({f"kl_iso_fp64_{k}": v for k, v in st.items()})
            if row.do_loi:
                with patched_weight(mod, joint_loi_weight(W, row.in_channel)):
                    st = runner.stats_from_layer(layer, force_exact=True)
                out.update(
                    {f"kl_joint_loi_fp64_{k}": v for k, v in st.items()})
            rows.append(out)
            measured += 1
            if deadline and time.time() >= deadline:
                save_chunk(rows, CKPT)
                log(f"stop after {measured} - checkpointed; rerun to resume")
                return 0
            if len(rows) >= CHUNK_ROWS:
                save_chunk(rows, CKPT)
                rows = []
                log(f"{measured}/{len(todo)} guard rows "
                    f"({measured / (time.time() - t0):.2f} cand/s)")
    if rows:
        save_chunk(rows, CKPT)
    with open(out_json, "w") as f:
        json.dump(bound, f, indent=2)
    save_manifest(out_json, build_manifest(stage="guard_recheck", **bound))
    log(f"DONE: {measured} guard rows -> chunks; bound -> {out_json.name}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    sys.exit(main(max_seconds=args.max_seconds))
