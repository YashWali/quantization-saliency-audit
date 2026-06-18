"""Addendum analyses (oracle baseline, mass/membership recall, pooled concentration).

1. RQ4 oracle baseline: the overprotection statistic applied to a
   PERFECT selector (within-stratum top-1% by measured GT) - shows how
   much of the 0.89-0.94 range is intrinsic skew of any top set.
2. Mass-weighted recall: share of census GT mass captured by each
   criterion's within-stratum top-1%, vs the oracle's share; also with
   the dominant super-channels excluded.
3. Pooled concentration: census-wide share held by the top-k channels
   (the per-stratum RQ1 rule hides this view).

Light (pandas only). Writes results/analysis/addendum.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qsal.config as cfg
from qsal.analysis.concentration import overprotection
from qsal.groundtruth import load_done

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ["awq", "gptq", "owq_residual", "spqr", "unified"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    grid = pd.read_parquet(ROOT / "data" / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet").merge(
        pd.read_parquet(RESULTS / "scores_owq.parquet"), on="channel_id")
    gt = pd.read_parquet(RESULTS / "ground_truth.parquet")
    rc = load_done(RESULTS / "gt_fp64_recheck_chunks")
    iso = rc.dropna(subset=["kl_iso_fp64_kl_mean"]) \
        .drop_duplicates("channel_id", keep="last")
    gt = gt.merge(iso[["channel_id", "kl_iso_fp64_kl_mean"]],
                  on="channel_id", how="left")
    gt["gt_iso"] = gt["kl_iso_fp64_kl_mean"].fillna(gt["kl_iso_kl_mean"])

    df = grid.merge(scores, on="channel_id").merge(
        gt[["channel_id", "gt_iso"]], on="channel_id", how="left")
    df["stratum"] = df["layer"].astype(str) + ":" + df["linear"].astype(str)

    # --- RQ4 oracle: perfect selector graded by the same statistic ---
    measured = df.dropna(subset=["gt_iso"])
    oracle = []
    for _, g in measured.groupby("stratum", observed=True):
        k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
        oracle.append(g.nlargest(k, "gt_iso"))
    oracle = pd.concat(oracle)
    r = overprotection(oracle["gt_iso"].to_numpy(),
                       top_frac=cfg.THRESHOLDS["rq4_top_frac"],
                       n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED)
    log(f"RQ4 oracle: top-20% of a perfect selection holds "
        f"{r['share']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]")

    # --- mass-weighted recall on the census ---
    census = df[df["layer"].isin(cfg.SWEEP_LAYERS)].dropna(
        subset=["gt_iso"])
    total = census["gt_iso"].sum()
    top5 = census.nlargest(5, "gt_iso")
    top5_share = top5["gt_iso"].sum() / total
    top5_ids = set(top5["channel_id"])

    def capture(sel_metric):
        picks = []
        for _, g in census.groupby("stratum", observed=True):
            k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
            picks.append(g.nlargest(k, sel_metric))
        p = pd.concat(picks)
        ex = p[~p["channel_id"].isin(top5_ids)]
        ex_total = census[~census["channel_id"].isin(top5_ids)][
            "gt_iso"].sum()
        return (p["gt_iso"].sum() / total,
                ex["gt_iso"].sum() / ex_total,
                top5_ids.issubset(set(p["channel_id"])))

    recall = {}
    for m in METHODS + ["gt_iso"]:
        full, ex, catches5 = capture(m)
        name = "oracle" if m == "gt_iso" else m
        recall[name] = {"mass_recall_top1pct": float(full),
                        "mass_recall_excl_top5": float(ex),
                        "catches_all_top5": bool(catches5)}
        log(f"  {name}: mass recall {full:.3f} "
            f"(excl. top-5 channels {ex:.3f}; catches all 5: {catches5})")

    # --- set-MEMBERSHIP recall on the census (UNBIASED population) ---
    # precision@k in 04_analysis.py is computed on the union+control
    # pool, whose `union` half is the criteria's own top-1% selections (a
    # selection-conditioned tail metric). The unbiased recall question -
    # "of the channels measurement says are in the top 1%, how many does the
    # criterion's top 1% catch?" - must be asked on the census, where no score
    # entered selection.
    #
    # POOLED (global census top-1%) is the headline statistic: it is the
    # cross-layer question the abstract's "recovers X% of the measured top-1%"
    # refers to, and it reproduces the values the criteria's pooled rho story
    # predicts (criteria that miss cross-layer magnitude miss the super-
    # channels). WITHIN-STRATUM membership recall is reported alongside as a
    # diagnostic: it is much higher (criteria locate local top channels fine),
    # so the pooled<<within gap is the within-layer-fine / cross-layer-miss
    # finding restated in membership terms.
    k_pool = max(1, int(round(cfg.TOP_PCT_UNION * len(census))))
    truth_pool = set(census.nlargest(k_pool, "gt_iso")["channel_id"])

    def membership_recall_within(sel_metric):
        hits = truth_n = 0
        for _, g in census.groupby("stratum", observed=True):
            k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
            truth = set(g.nlargest(k, "gt_iso")["channel_id"])
            pred = set(g.nlargest(k, sel_metric)["channel_id"])
            hits += len(pred & truth)
            truth_n += len(truth)
        return hits / truth_n

    membership = {"_truth_set": {"pooled_top1pct_n": int(len(truth_pool)),
                                 "census_n": int(len(census))}}
    for m in METHODS:
        pred_pool = set(census.nlargest(k_pool, m)["channel_id"])
        pooled = len(pred_pool & truth_pool) / len(truth_pool)
        within = membership_recall_within(m)
        membership[m] = {"membership_recall_pooled_top1pct": float(pooled),
                         "membership_recall_within_stratum": float(within)}
        log(f"  {m}: census membership recall pooled {pooled:.3f} / "
            f"within-stratum {within:.3f}")

    # --- pooled concentration ---
    srt = np.sort(census["gt_iso"].to_numpy())[::-1]
    cum = np.cumsum(srt) / srt.sum()
    pooled = {
        "n_census": int(len(census)),
        "top5_channels_share": float(top5_share),
        "top_0.1pct_share": float(cum[max(1, int(0.001 * len(srt))) - 1]),
        "top_1pct_share": float(cum[max(1, int(0.01 * len(srt))) - 1]),
        "top5_list": top5[["layer", "linear", "in_channel", "gt_iso"]]
            .to_dict("records"),
    }
    log(f"pooled census: top-5 channels hold {top5_share:.3f}; "
        f"top-0.1% {pooled['top_0.1pct_share']:.3f}; "
        f"top-1% {pooled['top_1pct_share']:.3f}")

    out = {
        "rq4_oracle": {k: v for k, v in r.items()},
        "mass_weighted_recall": recall,
        "membership_recall_census": membership,
        "pooled_concentration": pooled,
    }
    with open(RESULTS / "analysis" / "addendum.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log("DONE -> results/analysis/addendum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
