"""Phase 2 (Pythia) figures — variant of 05_plots.py.

Renders the report figures from results/pythia/analysis/* into figures/pythia/.
Run after scripts/phase2_04_analysis.py. The qsal.plots helpers are
architecture-agnostic (they take dataframes), so only paths/config change.
Single-pass: GT is exact-fp64 inline, so there is no fp64-recheck merge.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import importlib
import os as _os

_CFG = _os.environ.get("QSAL_CONFIG", "config_pythia")
cfg = importlib.import_module("qsal." + _CFG)
_RUN = _CFG.rsplit("config_", 1)[-1]  # "pythia" / "smollm2" -> data|results dir
from qsal import plots

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / _RUN
RESULTS = ROOT / "results" / _RUN
ANALYSIS = RESULTS / "analysis"
FIGURES = ROOT / "figures" / _RUN


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    gt = pd.read_parquet(RESULTS / "ground_truth.parquet")
    gt = gt.merge(grid[["channel_id"]], on="channel_id")
    gt["gt_iso"] = gt["kl_iso_kl_mean"]  # exact-fp64 inline -> no recheck merge

    sweep = gt[gt["layer"].isin(cfg.SWEEP_LAYERS)]
    log(plots.fig_lorenz(sweep, FIGURES / "rq1_lorenz.png").name)

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
    log(f"DONE -> figures/{_RUN}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
