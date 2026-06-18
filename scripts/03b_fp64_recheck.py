"""Exact-fp64 re-measurement of near-floor GT rows.

The main loop measures on the fast path (fp32 log-softmax, fp64 reduction;
absolute error floor ~1e-6). Rows whose kl_iso_kl_mean or
kl_joint_loi_kl_mean is below KL_FP64_RECHECK_BELOW (1e-4) are re-measured
here on the exact fp64 path (force_exact), giving every near-floor row a
floor-free value - including rows already present in the main GT table.
Analysis uses kl_*_fp64 columns where present.

Writes atomic chunks to results/gt_fp64_recheck_chunks/ (resumable), then
results/gt_fp64_recheck.parquet on completion.

Heavy (model + suffix runner): do NOT run concurrently with the main GT
measurement. Run after the main run completes, or between measurement runs.
Usage mirrors 03: --max-seconds for timed sessions, --limit for smokes.
"""

import sys
import time
from pathlib import Path

import torch

import qsal.config as cfg
from qsal.groundtruth import (KL_FP64_RECHECK_BELOW, SuffixRunner,
                              joint_loi_weight, load_done, patched_weight,
                              save_chunk)
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPT_MAIN = RESULTS / "gt_chunks"
CKPT = RESULTS / "gt_fp64_recheck_chunks"
CHUNK_ROWS = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(limit: int | None = None, max_seconds: float | None = None):
    cfg.set_global_seeds()
    deadline = time.time() + max_seconds if max_seconds else None
    gt = load_done(CKPT_MAIN)
    if gt.empty:
        log("no main GT chunks found; nothing to recheck")
        return 0

    need_iso = gt["kl_iso_kl_mean"] < KL_FP64_RECHECK_BELOW
    need_loi = gt["kl_joint_loi_kl_mean"] < KL_FP64_RECHECK_BELOW
    todo = gt[need_iso | need_loi].copy()
    done = set(load_done(CKPT)["channel_id"]) if CKPT.exists() else set()
    todo = todo[~todo["channel_id"].isin(done)]
    log(f"near-floor rows: {int((need_iso | need_loi).sum())} of {len(gt)} "
        f"({need_iso.sum()} iso, {need_loi.sum()} loi); "
        f"{len(todo)} left after resume")
    if todo.empty:
        return _finalize()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    from qsal.calibration import get_sequences
    eval_seqs, _ = get_sequences(
        tokenizer, cfg.EVAL_SPLIT, cfg.EVAL_NUM_SEQS, cfg.EVAL_SEQ_LEN,
        cfg.SEED,
    )
    layers_needed = sorted(todo["layer"].unique())
    runner = SuffixRunner(model, eval_seqs, cache_layers=layers_needed)
    modules = {(l, n): m for l, n, m in enumerate_linears(model)}

    rows, measured, t0 = [], 0, time.time()
    for (layer, linear), group in todo.groupby(["layer", "linear"],
                                               sort=False, observed=True):
        mod = modules[(layer, linear)]
        W = mod.weight.detach().cpu()
        for row in group.itertuples():
            out = {"channel_id": row.channel_id}
            if row.kl_iso_kl_mean < KL_FP64_RECHECK_BELOW:
                with patched_weight(
                    mod, quantize_single_column(W, row.in_channel,
                                                bits=cfg.ISOLATED_PROBE_BITS)
                ):
                    st = runner.stats_from_layer(layer, force_exact=True)
                out.update({f"kl_iso_fp64_{k}": v for k, v in st.items()})
            if row.kl_joint_loi_kl_mean < KL_FP64_RECHECK_BELOW:
                with patched_weight(mod, joint_loi_weight(W, row.in_channel)):
                    st = runner.stats_from_layer(layer, force_exact=True)
                out.update({f"kl_joint_loi_fp64_{k}": v for k, v in st.items()})
            rows.append(out)
            measured += 1
            hit = (limit is not None and measured >= limit) or (
                deadline is not None and time.time() >= deadline)
            if hit:
                save_chunk(rows, CKPT)
                log(f"stop after {measured} "
                    f"({measured / (time.time() - t0):.2f} cand/s) - "
                    "checkpointed; rerun to resume")
                return 0
            if len(rows) >= CHUNK_ROWS:
                save_chunk(rows, CKPT)
                rows = []
                log(f"{measured}/{len(todo)} rechecked "
                    f"({measured / (time.time() - t0):.2f} cand/s)")
    if rows:
        save_chunk(rows, CKPT)
    return _finalize()


def _finalize():
    df = load_done(CKPT)
    out = RESULTS / "gt_fp64_recheck.parquet"
    df.to_parquet(out, index=False)
    save_manifest(out, build_manifest(
        stage="gt_fp64_recheck", threshold=KL_FP64_RECHECK_BELOW))
    log(f"DONE: {len(df)} rows rechecked -> {out.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    sys.exit(main(limit=args.limit, max_seconds=args.max_seconds))
