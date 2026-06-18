"""Pre-registered validation sample for the fp64-recheck threshold
amendment (docs/AMENDMENT_fp64_recheck_threshold.md).

Draws n=500 seeded, decade-x-layer-stratified random rows from the
contested band 1e-6 <= kl_iso_kl_mean < 1e-4, re-measures them on the
exact fp64 path, evaluates the committed decision rule, and prints the
verdict. Chunks land in results/gt_fp64_recheck_chunks/ so the work
counts toward 03b regardless of outcome.

Heavy (model + suffix runner): do NOT run while 03/03b is running.
"""

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
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_single_column

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPT_MAIN = RESULTS / "gt_chunks"
CKPT = RESULTS / "gt_fp64_recheck_chunks"

BAND_LO, BAND_HI = 1e-6, 1e-4
N_SAMPLE = 500
SEED = 42
VIOLATION_REL = 0.05
SPEARMAN_MIN = 0.999
CHUNK_ROWS = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def draw_sample() -> pd.DataFrame:
    gt = load_done(CKPT_MAIN)
    band = gt[(gt["kl_iso_kl_mean"] >= BAND_LO)
              & (gt["kl_iso_kl_mean"] < BAND_HI)].copy()
    done = set(load_done(CKPT)["channel_id"]) if CKPT.exists() else set()
    band = band[~band["channel_id"].isin(done)]
    band["decade"] = np.where(band["kl_iso_kl_mean"] < 1e-5,
                              "1e-6_1e-5", "1e-5_1e-4")
    rng = np.random.default_rng(SEED)
    cells = band.groupby(["decade", "layer"], observed=True)
    alloc = (cells.size() / len(band) * N_SAMPLE).round().astype(int)
    alloc = alloc.clip(lower=1)
    picks = []
    for key, g in cells:
        k = min(int(alloc.loc[key]), len(g))
        picks.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    out = pd.concat(picks, ignore_index=True)
    log(f"band {BAND_LO:.0e}-{BAND_HI:.0e}: {len(band)} rows not yet "
        f"rechecked; sampled {len(out)} across {len(alloc)} cells")
    return out


def evaluate(sample_ids) -> int:
    rc = load_done(CKPT)
    gt = load_done(CKPT_MAIN)
    m = rc[rc["channel_id"].isin(sample_ids)].merge(
        gt[["channel_id", "kl_iso_kl_mean"]], on="channel_id")
    m = m.dropna(subset=["kl_iso_fp64_kl_mean"])
    fast = m["kl_iso_kl_mean"].to_numpy()
    exact = m["kl_iso_fp64_kl_mean"].to_numpy()
    rel = np.abs(exact - fast) / np.clip(exact, 1e-12, None)
    rho = float(spearmanr(fast, exact).statistic)
    violations = m[rel > VIOLATION_REL]
    log(f"evaluated {len(m)} sampled rows: "
        f"median |relΔ| {np.median(rel):.4%}, p95 {np.quantile(rel, .95):.4%}, "
        f"max {rel.max():.4%}, spearman {rho:.6f}, "
        f"violations(>{VIOLATION_REL:.0%}): {len(violations)}")
    passed = len(violations) == 0 and rho > SPEARMAN_MIN
    verdict = {
        "n_evaluated": len(m),
        "median_rel_delta": float(np.median(rel)),
        "p95_rel_delta": float(np.quantile(rel, 0.95)),
        "max_rel_delta": float(rel.max()),
        "spearman": rho,
        "n_violations": int(len(violations)),
        "violating_channel_ids": violations["channel_id"].tolist(),
        "rule_passed": bool(passed),
    }
    out = RESULTS / "fp64_threshold_validation.json"
    import json
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)
    save_manifest(out, build_manifest(
        stage="fp64_threshold_validation", seed=SEED, n_sample=N_SAMPLE,
        band=[BAND_LO, BAND_HI], violation_rel=VIOLATION_REL,
        spearman_min=SPEARMAN_MIN))
    log(f"VERDICT: rule_passed={passed} -> {out.name}")
    return 0 if passed else 1


def main():
    cfg.set_global_seeds()
    todo = draw_sample()
    if todo.empty:
        log("nothing to sample")
        return evaluate(set())

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
            with patched_weight(
                mod, quantize_single_column(W, row.in_channel,
                                            bits=cfg.ISOLATED_PROBE_BITS)
            ):
                st = runner.stats_from_layer(layer, force_exact=True)
            rows.append({"channel_id": row.channel_id,
                         **{f"kl_iso_fp64_{k}": v for k, v in st.items()}})
            measured += 1
            if len(rows) >= CHUNK_ROWS:
                save_chunk(rows, CKPT)
                rows = []
                log(f"{measured}/{len(todo)} sampled rows measured "
                    f"({measured / (time.time() - t0):.2f} cand/s)")
    if rows:
        save_chunk(rows, CKPT)
    return evaluate(set(todo["channel_id"]))


if __name__ == "__main__":
    sys.exit(main())
