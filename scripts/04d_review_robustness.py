"""POST-HOC robustness checks (not pre-registered) raised in adversarial review.

Two checks, both pure-pandas on the existing ground-truth artifacts (no model
load, no re-run of the census):

A. RQ3 oracle ceiling. The pre-registered RQ3 verdict reports the net (above
   chance) ground-truth mass that the >=m-method consensus set captures from the
   union+control pool. It is graded against a CHANCE baseline only. This adds the
   size-matched ORACLE ceiling: the net mass a perfect set of the same measured
   size captures. It contextualises the near-miss (Qwen 3-method net 0.527, the
   verdict is "fails the 0.50 bar") against what is even attainable at that size.

B. Layer-coding robustness for the RQ5 metric-vs-GT partials. The pre-registered
   partial (04_analysis.py / Figure 4) enters `layer` as a single linear rank
   regressor across the three swept depths. A reviewer correctly notes this
   under-controls an ordinal with k>2 levels. This recomputes the partials with
   `layer` dummy-coded (k-1 indicators, lowest swept layer as reference) and
   reports both, so the coding-invariance of the negligible-vs-strong contrast is
   on the record.

Frozen pipeline and library are untouched; this is a separate artifact. Run per
model via QSAL_CONFIG (unset -> Qwen Phase-1; config_pythia; config_smollm2):
    QSAL_CONFIG=config_smollm2 python scripts/04d_review_robustness.py
Output: results/<run>/analysis/review_robustness.json
"""

import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_CFG = os.environ.get("QSAL_CONFIG", "config")
cfg = importlib.import_module("qsal." + _CFG)

from qsal.analysis.confounds import add_control_columns  # noqa: E402
from qsal.analysis.consensus import consensus_sets  # noqa: E402
from qsal.analysis.correlation import partial_spearman  # noqa: E402

_ANA_FILE = "04_analysis.py" if _CFG == "config" else "phase2_04_analysis.py"
_spec = importlib.util.spec_from_file_location("qsal_ana", ROOT / "scripts" / _ANA_FILE)
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _top_set_within_strata(df, metric, frac):
    out = set()
    for _, g in df.groupby("stratum", observed=True):
        k = max(1, int(round(frac * len(g))))
        out |= set(g.nlargest(k, metric)["channel_id"].tolist())
    return out


def rq3_oracle_ceiling(df: pd.DataFrame) -> dict:
    """Net GT-mass of the >=m consensus set vs a size-matched oracle ceiling."""
    sets = {m: _top_set_within_strata(df, m, cfg.TOP_PCT_UNION)
            for m in ana.METHODS}
    cons = consensus_sets(sets, len(df))
    pool = df[df["source"].isin(["union", "control"])].dropna(subset=["gt_iso"])
    pool_ids = set(pool["channel_id"])
    pool_total = float(pool["gt_iso"].sum())
    gt_sorted = np.sort(pool["gt_iso"].to_numpy())[::-1]
    out = {}
    for m, info in cons.items():
        measured = info["members"] & pool_ids
        n = len(measured)
        chance = n / len(pool)
        obs = float(pool[pool["channel_id"].isin(measured)]["gt_iso"].sum()) / pool_total
        oracle_obs = float(gt_sorted[:n].sum()) / pool_total  # best size-n set
        net = obs - chance
        oracle_net = oracle_obs - chance
        out[str(m)] = {
            "size": int(info["size"]),
            "measured": int(n),
            "net_gt_frac": net,
            "oracle_net_gt_frac": oracle_net,
            "frac_of_ceiling": (net / oracle_net) if oracle_net > 0 else float("nan"),
        }
        log(f"  RQ3 m>={m}: consensus net {net:.3f} vs oracle ceiling "
            f"{oracle_net:.3f} ({out[str(m)]['frac_of_ceiling']:.2f} of attainable)")
    return out


def layer_coding_robustness(df: pd.DataFrame) -> dict:
    """RQ5 metric-vs-GT partials: linear-layer (published) vs dummy-coded layer."""
    sweep = df[df["layer"].isin(cfg.SWEEP_LAYERS)].dropna(subset=["gt_iso"])
    ctrl = add_control_columns(sweep)
    log_w = ctrl["log_w_norm"].astype(float).to_numpy()
    mean_x = ctrl["mean_abs_x"].astype(float).to_numpy()
    layer = ctrl["layer"].astype(float).to_numpy()
    levels = sorted(ctrl["layer"].unique())
    ref = levels[0]
    dummies = np.column_stack([(layer == lv).astype(float) for lv in levels[1:]])

    C_linear = np.column_stack([layer, log_w, mean_x])
    C_dummy = np.column_stack([dummies, log_w, mean_x])
    gt = sweep["gt_iso"].to_numpy()
    rows = {}
    for m in ana.METHODS:
        x = sweep[m].to_numpy()
        rows[m] = {
            "raw_spearman": partial_spearman(x, gt),
            "partial_linear_layer": partial_spearman(x, gt, controls=C_linear),
            "partial_dummy_layer": partial_spearman(x, gt, controls=C_dummy),
        }
        log(f"  {m:14s} raw {rows[m]['raw_spearman']:+.3f}  "
            f"linear {rows[m]['partial_linear_layer']:+.3f}  "
            f"dummy {rows[m]['partial_dummy_layer']:+.3f}")
    return {
        "n_channels": int(len(sweep)),
        "reference_layer": int(ref),
        "dummy_layers": [int(lv) for lv in levels[1:]],
        "metrics": rows,
    }


def census_recall_at_1pct(df: pd.DataFrame) -> dict:
    """Set-membership recall on the pooled census top-1% for every metric,
    primary and secondary. Reproduces the addendum's primary-criterion recalls
    and additionally persists the secondary GPTQ act-order column (cited in
    §4.3 but absent from addendum.json, which loops the five primaries only)."""
    census = df[df["layer"].isin(cfg.SWEEP_LAYERS)].dropna(subset=["gt_iso"])
    k = max(1, int(round(cfg.TOP_PCT_UNION * len(census))))
    truth = set(census.nlargest(k, "gt_iso")["channel_id"])
    out = {"census_n": int(len(census)), "top1pct_n": int(len(truth))}
    for m in list(ana.METHODS) + list(ana.SECONDARY):
        pred = set(census.nlargest(k, m)["channel_id"])
        out[m] = len(pred & truth) / len(truth)
    log("  census recall@1%: " + ", ".join(
        f"{m} {out[m]:.3f}" for m in list(ana.METHODS) + list(ana.SECONDARY)))
    return out


def main():
    cfg.set_global_seeds()
    out_dir = ana.RESULTS / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _gt_mode = ana.load_inputs()

    log("A. RQ3 oracle ceiling")
    rq3 = rq3_oracle_ceiling(df)
    log("B. layer-coding robustness (RQ5 partials)")
    layer = layer_coding_robustness(df)
    log("C. census recall@1% (all metrics, incl. secondary act-order)")
    recall = census_recall_at_1pct(df)

    payload = {
        "run": _CFG,
        "note": ("Post-hoc robustness checks (not pre-registered), raised in "
                 "adversarial review. (A) RQ3 net GT-mass vs a size-matched "
                 "oracle ceiling. (B) RQ5 metric-vs-GT partials with `layer` "
                 "dummy-coded (k-1 indicators, lowest swept layer reference) "
                 "vs the published single linear rank term. (C) census "
                 "recall@1% for every metric, persisting the secondary GPTQ "
                 "act-order column cited in the paper. No measured number "
                 "or pre-registered verdict is changed."),
        "rq3_oracle_ceiling": rq3,
        "layer_coding": layer,
        "census_recall_at_1pct": recall,
    }
    out_path = out_dir / "review_robustness.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
