"""Figures for the report (the design spec). Reads results/analysis/*;
matplotlib only, headless (Agg)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qsal.analysis.concentration import lorenz

METHOD_LABELS = {
    "awq": "AWQ", "gptq": "GPTQ", "owq_residual": "OWQ (residual)",
    "owq_energy": "OWQ (energy)", "spqr": "SpQR", "unified": "Unified",
    "gptq_actorder_proxy": "GPTQ act-order",
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_lorenz(df_sweep: pd.DataFrame, path):
    """RQ1: Lorenz curves of GT sensitivity per sweep stratum."""
    layers = sorted(df_sweep["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.4),
                             sharey=True)
    for ax, layer in zip(np.atleast_1d(axes), layers):
        for linear, g in df_sweep[df_sweep.layer == layer].groupby(
                "linear", observed=True):
            curve = lorenz(g["gt_iso"].to_numpy())
            ax.plot(np.linspace(0, 1, len(curve)), curve, label=linear,
                    lw=1.2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"layer {layer}")
        ax.set_xlabel("channel fraction (ascending GT)")
    np.atleast_1d(axes)[0].set_ylabel("cumulative GT share")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("RQ1 - Lorenz curves of isolated-GT sensitivity", y=1.02)
    return _save(fig, path)


def fig_jaccard_heatmap(rq2: pd.DataFrame, top_frac: float, path):
    """RQ2: ratio-over-null heatmap (never raw Jaccard alone [guard])."""
    sub = rq2[rq2.top_frac == top_frac]
    methods = sorted(set(sub.metric_a) | set(sub.metric_b))
    n = len(methods)
    M = np.full((n, n), np.nan)
    A = np.full((n, n), np.nan)
    for _, r in sub.iterrows():
        i, j = methods.index(r.metric_a), methods.index(r.metric_b)
        M[i, j] = M[j, i] = r.ratio_over_null
        A[i, j] = A[j, i] = r.jaccard
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, cmap="viridis", vmin=0)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}x\nJ={A[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < np.nanmax(M) * 0.6
                        else "black")
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    ax.set_xticks(range(n), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    fig.colorbar(im, label="Jaccard ratio over stratified null")
    ax.set_title(f"RQ2 - top-{top_frac:.0%} agreement over chance")
    return _save(fig, path)


def fig_rq5_spearman(strat: pd.DataFrame, partials: pd.DataFrame, path):
    """RQ5: per-stratum Spearman distribution + pooled partial overlay."""
    metrics = [m for m in METHOD_LABELS if m in set(strat.metric)]
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [strat[strat.metric == m]["spearman"].to_numpy()
            for m in metrics]
    bp = ax.boxplot(data, tick_labels=[METHOD_LABELS[m] for m in metrics],
                    showmeans=True)
    p = partials.set_index("metric")
    for i, m in enumerate(metrics):
        if m in p.index:
            ax.plot(i + 1, p.loc[m, "partial_spearman"], "r*", ms=11,
                    zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman vs isolated GT")
    ax.set_title("RQ5 - within-stratum Spearman (box) and "
                 "confound-partial pooled (star)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def fig_overprotection(rq4: pd.DataFrame, path):
    """RQ4: top-20% GT share within each method's selected set."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rq4))
    ax.bar(x, rq4["share"], yerr=[rq4["share"] - rq4["ci_low"],
                                  rq4["ci_high"] - rq4["share"]],
           capsize=4, color="steelblue")
    ax.axhline(0.80, color="firebrick", ls="--", lw=1,
               label="overprotection bar (0.80)")
    ax.axhline(0.60, color="orange", ls=":", lw=1,
               label="CI floor (0.60)")
    ax.set_xticks(x, [METHOD_LABELS.get(m, m) for m in rq4["metric"]],
                  rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("GT share of set's top 20%")
    ax.set_title("RQ4 - GT concentration in each method's top-20% set")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_floor_position(gt: pd.DataFrame, floor: float, path):
    """Sanity: control vs union kl_iso distributions vs the noise floor."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.logspace(-8, 0, 80)
    for src, color in [("control", "gray"), ("union", "steelblue"),
                       ("sweep", "seagreen")]:
        vals = gt[gt.source == src]["gt_iso"].clip(lower=1e-8)
        ax.hist(vals, bins=bins, histtype="step", label=src, color=color,
                density=True, lw=1.3)
    if floor and floor > 0:
        ax.axvline(floor, color="firebrick", ls="--", lw=1,
                   label=f"noise floor ~{floor:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("isolated GT (KL mean)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("Isolated-GT distributions vs measurement floor")
    return _save(fig, path)
