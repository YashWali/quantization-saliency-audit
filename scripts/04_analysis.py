"""Pre-registered analysis (the design spec; spec §2 decision rules).

RQ1 concentration, RQ2 agreement vs within-stratum permutation nulls,
RQ3 consensus net of chance, RQ4 overprotection, RQ5 prediction with
confound partials, plus the mechanical-baseline decomposition (6.2).

pandas/numpy only - no model forward - safe to run while 03b rechecks
in another process. GT column = fast-path kl_iso_kl_mean overridden by
exact-fp64 recheck values wherever results/gt_fp64_recheck_chunks/ has
them; outputs are stamped gt_mode = "final" iff the recheck finished
(results/gt_fp64_recheck.parquet exists), else "preliminary".

owq_residual is the PRIMARY OWQ variant;
owq_energy is reported as a disclosed secondary.

Writes results/analysis/*.parquet + results/analysis/summary.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import qsal.config as cfg
from qsal.analysis.concentration import (bca_ci, gini, overprotection,
                                         top_frac_share)
from qsal.analysis.confounds import (add_control_columns, confound_partials,
                                     stratified_spearman)
from qsal.analysis.consensus import consensus_sets
from qsal.analysis.correlation import (paired_bootstrap_spearman_diff,
                                       partial_spearman)
from qsal.analysis.nullmodels import holm, stratified_jaccard_vs_null
from qsal.groundtruth import load_done
from qsal.provenance import build_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis"

METHODS = ["awq", "gptq", "owq_residual", "spqr", "unified"]
SECONDARY = ["owq_energy", "gptq_actorder_proxy"]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def compute_w_norms(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-channel ||W[:,c]|| from the pinned snapshot, lazily via
    safetensors (one tensor in memory at a time - 03b-safe). Cached."""
    out_path = RESULTS / "w_norms.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    snap = snapshot_download(cfg.MODEL_ID, revision=cfg.MODEL_REVISION)
    rows = []
    with safe_open(Path(snap) / "model.safetensors", framework="pt") as f:
        for (layer, linear), g in grid.groupby(["layer", "linear"],
                                               observed=True):
            block = "self_attn" if linear in ATTN else "mlp"
            key = f"model.layers.{layer}.{block}.{linear}.weight"
            W = f.get_tensor(key).to(torch.float64)
            norms = torch.linalg.norm(W, dim=0).numpy()
            rows.append(pd.DataFrame({
                "channel_id": g["channel_id"].to_numpy(),
                "w_norm": norms[g["in_channel"].to_numpy()],
            }))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out_path, index=False)
    return df


def activation_stats(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-channel E[|x|] and E[x^2] from calibration accumulators."""
    stats = torch.load(DATA / "calib_stats.pt", map_location="cpu",
                       weights_only=False)
    rows = []
    for (layer, linear), g in grid.groupby(["layer", "linear"],
                                           observed=True):
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
    owq2 = pd.read_parquet(RESULTS / "scores_owq.parquet")
    df = grid.merge(scores, on="channel_id").merge(owq2, on="channel_id")
    df = df.merge(compute_w_norms(grid), on="channel_id")
    df = df.merge(activation_stats(grid), on="channel_id")
    df["module"] = np.where(df["linear"].isin(ATTN), "attn", "mlp")
    df["stratum"] = (df["layer"].astype(str) + ":"
                     + df["linear"].astype(str))

    gt = pd.read_parquet(RESULTS / "ground_truth.parquet")
    gt = gt[["channel_id", "source", "kl_iso_kl_mean",
             "kl_joint_loi_kl_mean"]].copy()
    gt["gt_iso"] = gt["kl_iso_kl_mean"]
    gt["gt_loi"] = gt["kl_joint_loi_kl_mean"]
    gt["fp64_rechecked"] = False

    recheck_dir = RESULTS / "gt_fp64_recheck_chunks"
    recheck_parquet = RESULTS / "gt_fp64_recheck.parquet"
    gt_mode = "preliminary"
    rc = pd.DataFrame()
    if recheck_dir.exists():
        rc = load_done(recheck_dir)
    elif recheck_parquet.exists():
        # released package ships the folded parquet, not the chunks
        rc = pd.read_parquet(recheck_parquet)
    if not rc.empty:
            iso = rc.dropna(subset=["kl_iso_fp64_kl_mean"]) \
                if "kl_iso_fp64_kl_mean" in rc else rc.iloc[0:0]
            if len(iso):
                m = gt.merge(iso[["channel_id", "kl_iso_fp64_kl_mean"]],
                             on="channel_id", how="left")
                got = m["kl_iso_fp64_kl_mean"].notna().to_numpy()
                gt.loc[got, "gt_iso"] = \
                    m.loc[got, "kl_iso_fp64_kl_mean"].to_numpy()
                gt.loc[got, "fp64_rechecked"] = True
            log(f"fp64 recheck merged: {int(gt.fp64_rechecked.sum())} rows")
    if (RESULTS / "gt_fp64_recheck.parquet").exists():
        gt_mode = "final"

    df = df.merge(gt.drop(columns=["kl_iso_kl_mean",
                                   "kl_joint_loi_kl_mean"]),
                  on="channel_id", how="left")
    return df, gt_mode


# ---------------------------------------------------------------------------
# RQ2 - agreement vs within-stratum null (6.1)
# ---------------------------------------------------------------------------

def rq2_agreement(df: pd.DataFrame) -> pd.DataFrame:
    methods = METHODS + ["owq_energy"]
    strata = df["stratum"].to_numpy()
    rows = []
    pairs = [(a, b) for i, a in enumerate(methods)
             for b in methods[i + 1:]]
    for frac in cfg.TOP_K_FRACTIONS:
        for a, b in pairs:
            r = stratified_jaccard_vs_null(
                df[a].to_numpy(), df[b].to_numpy(), strata, frac,
                n_perm=cfg.N_PERMUTATIONS, seed=cfg.SEED)
            rows.append({"metric_a": a, "metric_b": b, "top_frac": frac,
                         **{k: v for k, v in r.items()}})
            log(f"  RQ2 {a} vs {b} @top{frac:.0%}: J={r['jaccard']:.3f} "
                f"(null {r['null_mean']:.3f}, ratio {r['ratio_over_null']:.1f}x)")
    out = pd.DataFrame(rows)
    # Holm within the primary-method family at each top_frac
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


# ---------------------------------------------------------------------------
# Mechanical baseline (6.2)
# ---------------------------------------------------------------------------

def mechanical_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement that is guaranteed by the formulas, not discovered.

    (a) AWQ = E|x| vs the activation moment E[x^2]: near-monotone under
        the empirical activation distribution.
    (b) OWQ-energy = ||W||^2 * H_cc: vs GPTQ controlling the shared
        activation factor (here E[x^2], the H ordering proxy).
    (c) SpQR and GPTQ share 1/[H^-1]_cc by derivation: partial controls
        the GPTQ score itself.
    Reported per linear (within-stratum), summarized over strata."""
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


# ---------------------------------------------------------------------------
# RQ5 - prediction (6.3)
# ---------------------------------------------------------------------------

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
                      "a_wins": r["ci_low"] > 0,
                      "b_wins": r["ci_high"] < 0})
    boots = pd.DataFrame(boots)

    # precision@k on the union+control measured pool (tail metric)
    pool = df[df["source"].isin(["union", "control"])].dropna(
        subset=["gt_iso"])
    k = max(1, int(round(0.01 * len(pool))))
    gt_top = set(pool.nlargest(k, "gt_iso")["channel_id"])
    prec = pd.DataFrame([
        {"metric": m, "k": k,
         "precision_at_k":
             len(set(pool.nlargest(k, m)["channel_id"]) & gt_top) / k}
        for m in metrics
    ])
    return strat, partials, boots, prec


# ---------------------------------------------------------------------------
# RQ1 - concentration (6.4)
# ---------------------------------------------------------------------------

def rq1_concentration(df: pd.DataFrame) -> pd.DataFrame:
    sweep = df[df["in_sweep_pool"]].dropna(subset=["gt_iso"])
    rows = []
    for (layer, linear), g in sweep.groupby(["layer", "linear"],
                                            observed=True):
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
        log(f"  RQ1 L{layer} {linear}: top-1% share {top1:.2f} "
            f"[{lo:.2f},{hi:.2f}] gini {rows[-1]['gini']:.2f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ3 - consensus (6.5) / RQ4 - overprotection (6.6)
# ---------------------------------------------------------------------------

def _top_set_within_strata(df, metric, frac):
    out = set()
    for _, g in df.groupby("stratum", observed=True):
        k = max(1, int(round(frac * len(g))))
        out |= set(g.nlargest(k, metric)["channel_id"].tolist())
    return out


def rq3_consensus(df: pd.DataFrame) -> dict:
    N = len(df)
    sets = {m: _top_set_within_strata(df, m, cfg.TOP_PCT_UNION)
            for m in METHODS}
    cons = consensus_sets(sets, N)
    pool = df[df["source"].isin(["union", "control"])].dropna(
        subset=["gt_iso"])
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
        # bootstrap CI on net fraction (channel resampling of the pool)
        gt_vals = pool["gt_iso"].to_numpy()
        flags = pool["channel_id"].isin(measured).to_numpy()
        net_boot = np.empty(cfg.N_BOOTSTRAP)
        for i in range(cfg.N_BOOTSTRAP):
            idx = rng.integers(0, len(pool), len(pool))
            tot = gt_vals[idx].sum()
            net_boot[i] = (gt_vals[idx][flags[idx]].sum() / tot
                           - flags[idx].mean())
        out[m] = {
            **info,
            "measured": len(measured),
            "coverage": len(measured) / max(1, len(members)),
            "set_frac_of_N": len(members) / N,
            "gt_frac": float(obs_frac),
            "chance_frac": float(chance_frac),
            "net_gt_frac": float(obs_frac - chance_frac),
            "net_gt_frac_ci_low": float(np.quantile(net_boot, 0.025)),
            "net_gt_frac_ci_high": float(np.quantile(net_boot, 0.975)),
        }
        log(f"  RQ3 m>={m}: size {info['size']} "
            f"(chance {info['expected_chance']:.1f}), net GT-frac "
            f"{out[m]['net_gt_frac']:.3f} "
            f"[{out[m]['net_gt_frac_ci_low']:.3f},"
            f"{out[m]['net_gt_frac_ci_high']:.3f}]")
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
            "overprotects":
                (r["share"] >= cfg.THRESHOLDS["rq4_explained_frac"]
                 and r["ci_low"] > cfg.THRESHOLDS["rq4_ci_lower_min"]),
        })
        log(f"  RQ4 {m}: top-20% of set holds {r['share']:.2f} "
            f"[{r['ci_low']:.2f},{r['ci_high']:.2f}] of set GT "
            f"({len(g)}/{len(sel)} measured)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main():
    cfg.set_global_seeds()
    OUT.mkdir(exist_ok=True)
    df, gt_mode = load_inputs()
    df["in_sweep_pool"] = df["layer"].isin(cfg.SWEEP_LAYERS)
    log(f"inputs: {len(df)} channels, gt_mode={gt_mode}, "
        f"measured={int(df['gt_iso'].notna().sum())}")

    log("RQ1 concentration...")
    rq1 = rq1_concentration(df)
    rq1.to_parquet(OUT / "rq1_concentration.parquet", index=False)

    log("RQ2 agreement (stratified permutation nulls)...")
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

    summary = {
        "gt_mode": gt_mode,
        "n_channels": len(df),
        "n_measured": int(df["gt_iso"].notna().sum()),
        "n_fp64_rechecked": int(df["fp64_rechecked"].fillna(False).sum()),
        "rq1": {
            "strata_passing": int(rq1["passes"].sum()),
            "strata_falsified": int(rq1["falsified"].sum()),
            "n_strata": len(rq1),
            "median_top1pct_share": float(rq1["top1pct_share"].median()),
        },
        "rq2": {
            "pairs_significant_holm": int(rq2["significant_holm"].sum()),
            "pairs_tested": int(rq2["significant_holm"].notna().sum()),
            "min_ratio_over_null":
                float(rq2[rq2.top_frac == 0.01]["ratio_over_null"].min()),
            "max_ratio_over_null":
                float(rq2[rq2.top_frac == 0.01]["ratio_over_null"].max()),
        },
        "rq3": {str(m): {k: v for k, v in info.items()}
                for m, info in rq3.items()},
        "rq4": rq4.set_index("metric")[
            ["share", "ci_low", "ci_high", "overprotects"]
        ].to_dict("index"),
        "rq5": {
            "pooled_partial": partials.set_index("metric")[
                ["raw_spearman", "partial_spearman"]].to_dict("index"),
            "unified_wins_vs": boots[boots.a_wins]["metric_b"].tolist(),
            "beats_unified": boots[boots.b_wins]["metric_b"].tolist(),
            "precision_at_k": prec.set_index("metric")[
                "precision_at_k"].to_dict(),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    save_manifest(OUT / "summary.json", build_manifest(
        stage="analysis", gt_mode=gt_mode))
    log(f"DONE -> {OUT} (gt_mode={gt_mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
