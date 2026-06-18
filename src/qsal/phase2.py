"""Phase-2 (single-pass) GT candidate logic.

Pure helpers shared by scripts/phase2_03_ground_truth.py, kept here so they are
unit-tested independently of the model/run. Pre-registration choices 1 & 4
(the design spec): full sweep-layer census for isolated GT; joint GT for
the union+control set only, plus a seeded stratified reliability subset.
"""

import numpy as np
from scipy.stats import spearmanr


def spearman_brown(r):
    """Half-length reliability r -> projected full-length (two halves)."""
    return 2 * r / (1 + r) if r > -1 else float("nan")


def split_half_within(df, a, b, group=("layer", "linear"), min_n=5):
    """Median within-stratum Spearman between columns a and b (the faithful
    per-channel split-half reliability; pooled is spuriously high). Returns
    (median_rho, n_strata_used)."""
    rs = []
    for _, g in df.groupby(list(group), observed=True):
        if len(g) >= min_n:
            rs.append(spearmanr(g[a], g[b]).statistic)
    return (float(np.nanmedian(rs)) if rs else float("nan")), len(rs)


def joint_set_level_sign_test(df, value="gt_loi", min_per_group=3,
                              group=("layer", "linear")):
    """Per stratum with >= min_per_group union AND control channels (non-null
    `value`), test union-mean < control-mean (the set-level joint claim that
    survives the per-channel demotion). Returns (n_correct, n_total)."""
    n_correct = n_total = 0
    d = df.dropna(subset=[value])
    for _, g in d.groupby(list(group), observed=True):
        u = g.loc[g["source"] == "union", value]
        c = g.loc[g["source"] == "control", value]
        if len(u) >= min_per_group and len(c) >= min_per_group:
            n_total += 1
            if u.mean() < c.mean():
                n_correct += 1
    return n_correct, n_total


def build_candidates(grid, scores, *, metrics, sweep_layers, n_control,
                     top_pct, seed):
    """Source-tagged candidate table (deduped; priority union > sweep > control).

    - union: per-criterion within-(layer,linear) top-`top_pct`, across ALL layers
    - sweep: every channel of `sweep_layers` (the unbiased census population)
    - control: `n_control` random channels drawn from outside union+sweep
    """
    rng = np.random.default_rng(seed)
    s = scores.merge(grid, on="channel_id")

    union_ids = set()
    for m in metrics:
        ranks = s.groupby(["layer", "linear"], observed=True)[m].rank(
            ascending=False, pct=True
        )
        union_ids |= set(s.loc[ranks <= top_pct, "channel_id"])

    sweep_ids = set(grid.loc[grid["layer"].isin(sweep_layers), "channel_id"])
    pool = np.array(sorted(set(grid["channel_id"]) - union_ids - sweep_ids))
    n_ctrl = min(n_control, len(pool))
    control_ids = (set(rng.choice(pool, size=n_ctrl, replace=False).tolist())
                   if n_ctrl > 0 else set())

    cand = grid[grid["channel_id"].isin(union_ids | sweep_ids | control_ids)
                ].copy()
    cand["in_union"] = cand["channel_id"].isin(union_ids)
    cand["in_sweep"] = cand["channel_id"].isin(sweep_ids)
    cand["source"] = np.where(
        cand["in_union"], "union",
        np.where(cand["in_sweep"], "sweep", "control"),
    )
    return cand


def reliability_subset(grid, sweep_layers, per_stratum, seed) -> set:
    """Seeded stratified channel-id subset: up to `per_stratum` channels from
    each (layer, linear) stratum of `sweep_layers` (joint split-half reliability,
    pre-registration choice 4). Takes the whole stratum if it is smaller."""
    rng = np.random.default_rng(seed)
    ids = []
    sub = grid[grid["layer"].isin(sweep_layers)]
    for _, g in sub.groupby(["layer", "linear"], observed=True):
        cids = g["channel_id"].to_numpy()
        k = min(per_stratum, len(cids))
        ids.extend(rng.choice(cids, size=k, replace=False).tolist())
    return set(ids)
