"""Phase 2 (Pythia) analysis — single-pass variant of 04 + 04b (+ joint pieces).

Reproduces the Phase-1 pre-registered analysis on the Pythia artifacts
(results/pythia/), driven by config_pythia. Per-channel RQs key off the isolated
GT (gt_iso), which Phase 2 measures for the full sweep census - so RQ1/RQ2/RQ5/
RQ3/RQ4 + the mechanical baseline + the 04b oracle/mass/membership/pooled
addenda fold in unchanged.

Choice 4 (joint GT is set-level only): joint LOI exists only for the
union+control set + the reliability subset, so the joint analyses are:
  - set-level sign test: per stratum, union-mean LOI < control-mean LOI;
  - within-linear split-half reliability (from gt_split_half.parquet), reported
    within-linear + Spearman-Brown (the faithful figure; pooled is spuriously
    high) - re-earning the Phase-1 demotion on Pythia.

Single-pass: GT is already exact-fp64 near-floor (inline), so gt_mode is "final"
by construction - no fp64-recheck merge, no scores_owq merge (owq_residual is in
scores.parquet). pandas/numpy + one CPU model load (for ||W||). Writes
results/pythia/analysis/*.parquet + summary.json + addendum.json +
split_half_reliability.json.
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
from qsal.analysis.concentration import (bca_ci, gini, overprotection,
                                         top_frac_share)
from qsal.analysis.confounds import (add_control_columns, confound_partials,
                                     stratified_spearman)
from qsal.analysis.consensus import consensus_sets
from qsal.analysis.correlation import (paired_bootstrap_spearman_diff,
                                       partial_spearman)
from qsal.analysis.nullmodels import holm, stratified_jaccard_vs_null
from qsal.models import enumerate_linears, load_model
from qsal.phase2 import (joint_set_level_sign_test, spearman_brown,
                         split_half_within)
from qsal.provenance import build_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / _RUN
RESULTS = ROOT / "results" / _RUN
OUT = RESULTS / "analysis"

METHODS = ["awq", "gptq", "owq_residual", "spqr", "unified"]
SECONDARY = ["owq_energy", "gptq_actorder_proxy"]
# Attention-linear names across both families (GPT-NeoX: query_key_value/dense;
# Llama/Qwen2: q/k/v/o_proj). The union is collision-free — no MLP linear in
# either family shares these names — so module classification is correct for
# Pythia and SmolLM2 alike without a model load.
ATTN = {"query_key_value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def compute_w_norms(grid: pd.DataFrame, model) -> pd.DataFrame:
    """Per-channel ||W[:,c]|| from the loaded model (architecture-agnostic via
    enumerate_linears). Cached."""
    out_path = RESULTS / "w_norms.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    mods = {(l, n): m for l, n, m in enumerate_linears(model, cfg=cfg)}
    rows = []
    for (layer, linear), g in grid.groupby(["layer", "linear"], observed=True):
        W = mods[(layer, linear)].weight.detach().to(torch.float64)
        norms = torch.linalg.norm(W, dim=0).numpy()  # per input channel
        rows.append(pd.DataFrame({
            "channel_id": g["channel_id"].to_numpy(),
            "w_norm": norms[g["in_channel"].to_numpy()],
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out_path, index=False)
    return df


def activation_stats(grid: pd.DataFrame) -> pd.DataFrame:
    stats = torch.load(DATA / "calib_stats.pt", map_location="cpu",
                       weights_only=False)
    rows = []
    for (layer, linear), g in grid.groupby(["layer", "linear"], observed=True):
        s = stats[f"L{layer}.{linear}"]
        n = s["token_count"]
        idx = g["in_channel"].to_numpy()
        rows.append(pd.DataFrame({
            "channel_id": g["channel_id"].to_numpy(),
            "e_abs_x": (s["sum_abs_x"] / n).numpy()[idx],
            "e_x2": (s["sum_x2"] / n).numpy()[idx],
        }))
    return pd.concat(rows, ignore_index=True)


def load_inputs():
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet")
    df = grid.merge(scores, on="channel_id")
    if (RESULTS / "w_norms.parquet").exists():
        wn = pd.read_parquet(RESULTS / "w_norms.parquet")  # no model load needed
    else:
        log("loading model weights (cpu) for ||W||")
        model, _ = load_model(device="cpu", cfg=cfg)
        wn = compute_w_norms(grid, model)
    df = df.merge(wn, on="channel_id")
    df = df.merge(activation_stats(grid), on="channel_id")
    df["module"] = np.where(df["linear"].isin(ATTN), "attn", "mlp")
    df["stratum"] = df["layer"].astype(str) + ":" + df["linear"].astype(str)

    gt = pd.read_parquet(RESULTS / "ground_truth.parquet")
    keep = ["channel_id", "source", "kl_iso_kl_mean"]
    if "kl_joint_loi_kl_mean" in gt.columns:
        keep.append("kl_joint_loi_kl_mean")
    gt = gt[keep].copy()
    gt["gt_iso"] = gt["kl_iso_kl_mean"]
    # joint exists only for union+control (choice 4); NaN elsewhere
    gt["gt_loi"] = gt.get("kl_joint_loi_kl_mean", np.nan)
    # single-pass: exact-fp64 is inline at measurement time -> always final
    gt_mode = "final"
    df = df.merge(gt[["channel_id", "source", "gt_iso", "gt_loi"]],
                  on="channel_id", how="left")
    return df, gt_mode


# ---------------------------------------------------------------------------
# RQ2 - agreement vs within-stratum null
# ---------------------------------------------------------------------------

def rq2_agreement(df: pd.DataFrame) -> pd.DataFrame:
    methods = METHODS + ["owq_energy"]
    strata = df["stratum"].to_numpy()
    rows = []
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    for frac in cfg.TOP_K_FRACTIONS:
        for a, b in pairs:
            r = stratified_jaccard_vs_null(
                df[a].to_numpy(), df[b].to_numpy(), strata, frac,
                n_perm=cfg.N_PERMUTATIONS, seed=cfg.SEED)
            rows.append({"metric_a": a, "metric_b": b, "top_frac": frac, **r})
    out = pd.DataFrame(rows)
    prim = out[out.metric_a.isin(METHODS) & out.metric_b.isin(METHODS)]
    adj = np.full(len(out), np.nan)
    rej = np.full(len(out), False)
    for frac in cfg.TOP_K_FRACTIONS:
        idx = prim[prim.top_frac == frac].index
        h = holm(out.loc[idx, "p_value"].to_numpy(), alpha=cfg.ALPHA)
        adj[idx] = h["adjusted"]
        rej[idx] = h["reject"]
    out["p_holm"] = adj
    out["significant_holm"] = rej
    return out


def mechanical_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("awq_vs_ex2", "awq", "e_x2", None),
        ("owq_energy_vs_gptq", "owq_energy", "gptq", "e_x2"),
        ("owq_residual_vs_gptq", "owq_residual", "gptq", "e_x2"),
        ("spqr_vs_gptq", "spqr", "gptq", "gptq"),
        ("unified_vs_awq", "unified", "awq", "e_x2"),
    ]
    for name, a, b, ctrl in specs:
        raws, partials = [], []
        for _, g in df.groupby("stratum", observed=True):
            raws.append(partial_spearman(g[a], g[b]))
            if ctrl is not None:
                C = g[ctrl].astype(float).to_numpy()[:, None]
                partials.append(partial_spearman(g[a], g[b], controls=C))
        rows.append({
            "pair": name, "controlling": ctrl or "",
            "raw_spearman_median": float(np.median(raws)),
            "raw_spearman_q25": float(np.quantile(raws, 0.25)),
            "raw_spearman_q75": float(np.quantile(raws, 0.75)),
            "partial_spearman_median":
                float(np.median(partials)) if partials else np.nan,
            "n_strata": len(raws),
        })
    return pd.DataFrame(rows)


def rq5_prediction(df: pd.DataFrame):
    sweep = df[df["in_sweep_pool"]].dropna(subset=["gt_iso"])
    log(f"  RQ5 population: {len(sweep)} sweep-layer channels")
    metrics = METHODS + SECONDARY
    strat = pd.concat([
        stratified_spearman(sweep, m, "gt_iso",
                            strata=("layer", "linear")).assign(metric=m)
        for m in metrics
    ], ignore_index=True)
    ctrl = add_control_columns(sweep)
    partials = confound_partials(ctrl, metrics, "gt_iso")
    boots = []
    for m in metrics:
        if m == "unified":
            continue
        r = paired_bootstrap_spearman_diff(
            sweep["unified"], sweep[m], sweep["gt_iso"],
            n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED)
        boots.append({"metric_a": "unified", "metric_b": m, **r,
                      "a_wins": r["ci_low"] > 0, "b_wins": r["ci_high"] < 0})
    boots = pd.DataFrame(boots)
    pool = df[df["source"].isin(["union", "control"])].dropna(subset=["gt_iso"])
    k = max(1, int(round(0.01 * len(pool))))
    gt_top = set(pool.nlargest(k, "gt_iso")["channel_id"])
    prec = pd.DataFrame([
        {"metric": m, "k": k,
         "precision_at_k":
             len(set(pool.nlargest(k, m)["channel_id"]) & gt_top) / k}
        for m in metrics
    ])
    return strat, partials, boots, prec


def rq1_concentration(df: pd.DataFrame) -> pd.DataFrame:
    sweep = df[df["in_sweep_pool"]].dropna(subset=["gt_iso"])
    rows = []
    for (layer, linear), g in sweep.groupby(["layer", "linear"], observed=True):
        x = g["gt_iso"].to_numpy()
        top1 = top_frac_share(x, cfg.THRESHOLDS["rq1_top_pct"])
        lo, hi = bca_ci(
            x, lambda s, axis=-1: np.apply_along_axis(
                lambda v: top_frac_share(v, cfg.THRESHOLDS["rq1_top_pct"]),
                axis, np.atleast_2d(s)) if np.ndim(s) > 1
            else top_frac_share(np.asarray(s), cfg.THRESHOLDS["rq1_top_pct"]),
            n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED)
        rows.append({
            "layer": layer, "linear": linear,
            "module": "attn" if linear in ATTN else "mlp",
            "n": len(x), "gini": gini(x),
            "top1pct_share": top1, "ci_low": lo, "ci_high": hi,
            "top5pct_share": top_frac_share(x, 0.05),
            "top10pct_share": top_frac_share(x, 0.10),
            "passes": (top1 >= cfg.THRESHOLDS["rq1_explained_frac"]
                       and lo > cfg.THRESHOLDS["rq1_ci_lower_min"]),
            "falsified": hi < cfg.THRESHOLDS["rq1_ci_lower_min"],
        })
    return pd.DataFrame(rows)


def _top_set_within_strata(df, metric, frac):
    out = set()
    for _, g in df.groupby("stratum", observed=True):
        k = max(1, int(round(frac * len(g))))
        out |= set(g.nlargest(k, metric)["channel_id"].tolist())
    return out


def rq3_consensus(df: pd.DataFrame) -> dict:
    N = len(df)
    sets = {m: _top_set_within_strata(df, m, cfg.TOP_PCT_UNION) for m in METHODS}
    cons = consensus_sets(sets, N)
    pool = df[df["source"].isin(["union", "control"])].dropna(subset=["gt_iso"])
    pool_ids = set(pool["channel_id"])
    pool_total = pool["gt_iso"].sum()
    rng = np.random.default_rng(cfg.SEED)
    out = {}
    for m, info in cons.items():
        members = info.pop("members")
        measured = members & pool_ids
        in_pool = pool[pool["channel_id"].isin(measured)]
        obs_frac = in_pool["gt_iso"].sum() / pool_total
        chance_frac = len(measured) / len(pool)
        gt_vals = pool["gt_iso"].to_numpy()
        flags = pool["channel_id"].isin(measured).to_numpy()
        net_boot = np.empty(cfg.N_BOOTSTRAP)
        for i in range(cfg.N_BOOTSTRAP):
            idx = rng.integers(0, len(pool), len(pool))
            tot = gt_vals[idx].sum()
            net_boot[i] = (gt_vals[idx][flags[idx]].sum() / tot
                           - flags[idx].mean())
        out[m] = {
            **info, "measured": len(measured),
            "coverage": len(measured) / max(1, len(members)),
            "set_frac_of_N": len(members) / N,
            "gt_frac": float(obs_frac), "chance_frac": float(chance_frac),
            "net_gt_frac": float(obs_frac - chance_frac),
            "net_gt_frac_ci_low": float(np.quantile(net_boot, 0.025)),
            "net_gt_frac_ci_high": float(np.quantile(net_boot, 0.975)),
        }
    return out


def rq4_overprotection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        sel = _top_set_within_strata(df, m, cfg.TOP_PCT_UNION)
        g = df[df["channel_id"].isin(sel)].dropna(subset=["gt_iso"])
        r = overprotection(g["gt_iso"].to_numpy(),
                           top_frac=cfg.THRESHOLDS["rq4_top_frac"],
                           n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED)
        rows.append({
            "metric": m, "selected": len(sel), "measured": len(g), **r,
            "overprotects": (r["share"] >= cfg.THRESHOLDS["rq4_explained_frac"]
                             and r["ci_low"] > cfg.THRESHOLDS["rq4_ci_lower_min"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 04b addenda (all on gt_iso / census)
# ---------------------------------------------------------------------------

def addendum(df: pd.DataFrame) -> dict:
    measured = df.dropna(subset=["gt_iso"])
    oracle = pd.concat([
        g.nlargest(max(1, int(round(cfg.TOP_PCT_UNION * len(g)))), "gt_iso")
        for _, g in measured.groupby("stratum", observed=True)
    ])
    r = overprotection(oracle["gt_iso"].to_numpy(),
                       top_frac=cfg.THRESHOLDS["rq4_top_frac"],
                       n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED)

    census = df[df["layer"].isin(cfg.SWEEP_LAYERS)].dropna(subset=["gt_iso"])
    total = census["gt_iso"].sum()
    top5 = census.nlargest(5, "gt_iso")
    top5_ids = set(top5["channel_id"])
    top5_share = top5["gt_iso"].sum() / total

    def capture(metric):
        picks = pd.concat([
            g.nlargest(max(1, int(round(cfg.TOP_PCT_UNION * len(g)))), metric)
            for _, g in census.groupby("stratum", observed=True)
        ])
        ex = picks[~picks["channel_id"].isin(top5_ids)]
        ex_total = census[~census["channel_id"].isin(top5_ids)]["gt_iso"].sum()
        return (picks["gt_iso"].sum() / total, ex["gt_iso"].sum() / ex_total,
                top5_ids.issubset(set(picks["channel_id"])))

    recall = {}
    for m in METHODS + ["gt_iso"]:
        full, ex, c5 = capture(m)
        recall["oracle" if m == "gt_iso" else m] = {
            "mass_recall_top1pct": float(full),
            "mass_recall_excl_top5": float(ex), "catches_all_top5": bool(c5)}

    k_pool = max(1, int(round(cfg.TOP_PCT_UNION * len(census))))
    truth_pool = set(census.nlargest(k_pool, "gt_iso")["channel_id"])

    def within_membership(metric):
        hits = truth_n = 0
        for _, g in census.groupby("stratum", observed=True):
            k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
            truth = set(g.nlargest(k, "gt_iso")["channel_id"])
            hits += len(set(g.nlargest(k, metric)["channel_id"]) & truth)
            truth_n += len(truth)
        return hits / truth_n

    membership = {"_truth_set": {"pooled_top1pct_n": int(len(truth_pool)),
                                 "census_n": int(len(census))}}
    for m in METHODS:
        pred = set(census.nlargest(k_pool, m)["channel_id"])
        membership[m] = {
            "membership_recall_pooled_top1pct":
                len(pred & truth_pool) / len(truth_pool),
            "membership_recall_within_stratum": within_membership(m)}

    srt = np.sort(census["gt_iso"].to_numpy())[::-1]
    cum = np.cumsum(srt) / srt.sum()
    pooled = {
        "n_census": int(len(census)), "top5_channels_share": float(top5_share),
        "top_0.1pct_share": float(cum[max(1, int(0.001 * len(srt))) - 1]),
        "top_1pct_share": float(cum[max(1, int(0.01 * len(srt))) - 1]),
        "top5_list": top5[["layer", "linear", "in_channel", "gt_iso"]]
            .to_dict("records"),
    }
    return {"rq4_oracle": dict(r), "mass_weighted_recall": recall,
            "membership_recall_census": membership,
            "pooled_concentration": pooled}


# ---------------------------------------------------------------------------
# Joint GT (choice 4): set-level sign test + split-half reliability
# ---------------------------------------------------------------------------

def joint_analyses(df: pd.DataFrame) -> dict:
    out = {}
    if df["gt_loi"].notna().any():
        for mn in (3, 1):
            nc, nt = joint_set_level_sign_test(df, "gt_loi", min_per_group=mn)
            out[f"set_level_sign_min{mn}"] = {"correct": nc, "total": nt}
        log(f"  joint set-level sign (min3): "
            f"{out['set_level_sign_min3']['correct']}/"
            f"{out['set_level_sign_min3']['total']} strata union<control")
    sh_path = RESULTS / "gt_split_half.parquet"
    if sh_path.exists():
        sh = pd.read_parquet(sh_path).drop_duplicates("channel_id")
        rel = {"n_channels": int(len(sh)),
               "n_strata": int(sh.groupby(["layer", "linear"]).ngroups)}
        for name, a, b in [("joint_loi", "loi_a", "loi_b"),
                           ("isolated", "iso_a", "iso_b")]:
            from scipy.stats import spearmanr
            pooled = float(spearmanr(sh[a], sh[b]).statistic)
            win, n = split_half_within(sh, a, b)
            rel[name] = {
                "split_half_pooled_spearman": pooled,
                "split_half_within_linear_median": win, "n_strata_used": n,
                "spearman_brown_full_eval_pooled": spearman_brown(pooled),
                "spearman_brown_full_eval_within": spearman_brown(win)}
            log(f"  reliability {name}: within-linear {win:.3f} "
                f"(SB {spearman_brown(win):.3f}); pooled {pooled:.3f}")
        out["split_half_reliability"] = rel
        with open(OUT / "split_half_reliability.json", "w") as f:
            json.dump(rel, f, indent=2)
    return out


# ---------------------------------------------------------------------------

def main():
    cfg.set_global_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    df, gt_mode = load_inputs()
    df["in_sweep_pool"] = df["layer"].isin(cfg.SWEEP_LAYERS)
    log(f"inputs: {len(df)} channels, gt_mode={gt_mode}, "
        f"iso measured={int(df['gt_iso'].notna().sum())}, "
        f"joint measured={int(df['gt_loi'].notna().sum())}")

    log("RQ1 concentration...")
    rq1 = rq1_concentration(df)
    rq1.to_parquet(OUT / "rq1_concentration.parquet", index=False)
    log("RQ2 agreement...")
    rq2 = rq2_agreement(df)
    rq2.to_parquet(OUT / "rq2_agreement.parquet", index=False)
    log("mechanical baseline...")
    mech = mechanical_baseline(df)
    mech.to_parquet(OUT / "mechanical_baseline.parquet", index=False)
    log("RQ5 prediction...")
    strat, partials, boots, prec = rq5_prediction(df)
    strat.to_parquet(OUT / "rq5_stratified_spearman.parquet", index=False)
    partials.to_parquet(OUT / "rq5_confound_partials.parquet", index=False)
    boots.to_parquet(OUT / "rq5_paired_bootstrap.parquet", index=False)
    prec.to_parquet(OUT / "rq5_precision_at_k.parquet", index=False)
    log("RQ3 consensus...")
    rq3 = rq3_consensus(df)
    log("RQ4 overprotection...")
    rq4 = rq4_overprotection(df)
    rq4.to_parquet(OUT / "rq4_overprotection.parquet", index=False)
    log("addenda (oracle / mass / membership / pooled)...")
    add = addendum(df)
    with open(OUT / "addendum.json", "w") as f:
        json.dump(add, f, indent=2, default=str)
    log("joint analyses (set-level + reliability)...")
    joint = joint_analyses(df)

    summary = {
        "gt_mode": gt_mode, "n_channels": len(df),
        "n_iso_measured": int(df["gt_iso"].notna().sum()),
        "n_joint_measured": int(df["gt_loi"].notna().sum()),
        "rq1": {"strata_passing": int(rq1["passes"].sum()),
                "strata_falsified": int(rq1["falsified"].sum()),
                "n_strata": len(rq1),
                "median_top1pct_share": float(rq1["top1pct_share"].median())},
        "rq2": {"pairs_significant_holm": int(rq2["significant_holm"].sum()),
                "min_ratio_over_null":
                    float(rq2[rq2.top_frac == 0.01]["ratio_over_null"].min()),
                "max_ratio_over_null":
                    float(rq2[rq2.top_frac == 0.01]["ratio_over_null"].max())},
        "rq3": {str(m): info for m, info in rq3.items()},
        "rq4": rq4.set_index("metric")[
            ["share", "ci_low", "ci_high", "overprotects"]].to_dict("index"),
        "rq5": {"pooled_partial": partials.set_index("metric")[
                    ["raw_spearman", "partial_spearman"]].to_dict("index"),
                "unified_wins_vs": boots[boots.a_wins]["metric_b"].tolist(),
                "beats_unified": boots[boots.b_wins]["metric_b"].tolist(),
                "precision_at_k": prec.set_index("metric")[
                    "precision_at_k"].to_dict()},
        "joint": joint,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    save_manifest(OUT / "summary.json", build_manifest(
        stage=f"analysis_{_RUN}", gt_mode=gt_mode))
    log(f"DONE -> {OUT} (gt_mode={gt_mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
