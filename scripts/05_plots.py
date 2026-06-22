"""Render report figures from results/analysis/* (the design spec).

Run after scripts/04_analysis.py. Light, headless; safe alongside 03b.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qsal.config as cfg
from qsal import plots
from qsal.groundtruth import load_done

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
FIGURES = ROOT / "figures"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    FIGURES.mkdir(exist_ok=True)
    grid = pd.read_parquet(ROOT / "data" / "channel_grid.parquet")
    gt = pd.read_parquet(RESULTS / "ground_truth.parquet")
    gt = gt.merge(grid[["channel_id"]], on="channel_id")
    gt["gt_iso"] = gt["kl_iso_kl_mean"]
    rc_dir = RESULTS / "gt_fp64_recheck_chunks"
    if rc_dir.exists():
        rc = load_done(rc_dir)
        if not rc.empty and "kl_iso_fp64_kl_mean" in rc:
            iso = rc.dropna(subset=["kl_iso_fp64_kl_mean"]) \
                .drop_duplicates("channel_id", keep="last")
            m = gt.merge(iso[["channel_id", "kl_iso_fp64_kl_mean"]],
                         on="channel_id", how="left")
            got = m["kl_iso_fp64_kl_mean"].notna().to_numpy()
            gt.loc[got, "gt_iso"] = \
                m.loc[got, "kl_iso_fp64_kl_mean"].to_numpy()

    sweep = gt[gt["layer"].isin(cfg.SWEEP_LAYERS)]
    log(plots.fig_lorenz(sweep, FIGURES / "rq1_lorenz.png",
                         model="Qwen2.5-0.5B").name)

    rq2 = pd.read_parquet(ANALYSIS / "rq2_agreement.parquet")
    for frac in cfg.TOP_K_FRACTIONS:
        log(plots.fig_jaccard_heatmap(
            rq2, frac,
            FIGURES / f"rq2_jaccard_top{int(frac * 100)}pct.png").name)

    strat = pd.read_parquet(ANALYSIS / "rq5_stratified_spearman.parquet")
    partials = pd.read_parquet(ANALYSIS / "rq5_confound_partials.parquet")
    log(plots.fig_rq5_spearman(strat, partials,
                               FIGURES / "rq5_spearman.png").name)

    rq4 = pd.read_parquet(ANALYSIS / "rq4_overprotection.parquet")
    log(plots.fig_overprotection(rq4,
                                 FIGURES / "rq4_overprotection.png").name)

    nf = pd.read_parquet(RESULTS / "gt_noise_floor.parquet")
    floor = float(np.median(nf["kl_iso_kl_mean"]))
    log(plots.fig_floor_position(gt, floor,
                                 FIGURES / "gt_floor_position.png").name)
    log("DONE -> figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
