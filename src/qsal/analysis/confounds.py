"""Confound controls (spec §5).

Every metric-vs-GT association is reported three ways:
  1. raw within-stratum Spearman (primary; depth/module stratified),
  2. pooled Spearman (reference only - depth is a confound),
  3. partial Spearman controlling layer index, log||W[:,c]|| and
     mean|x_c| (the shared drivers most criteria embed mechanically).
"""

import numpy as np
import pandas as pd

from qsal.analysis.correlation import partial_spearman

DEFAULT_CONTROLS = ("layer", "log_w_norm", "mean_abs_x")


def add_control_columns(df: pd.DataFrame, w_norm_col: str = "w_norm",
                        awq_col: str = "awq") -> pd.DataFrame:
    """Derive the standard control columns. mean|x_c| IS the AWQ score
    (E[|x_c|]), reused as a confound regressor for the other metrics."""
    out = df.copy()
    out["log_w_norm"] = np.log(out[w_norm_col].astype(float))
    out["mean_abs_x"] = out[awq_col].astype(float)
    return out


def stratified_spearman(df: pd.DataFrame, metric: str, gt: str,
                        strata=("layer",)) -> pd.DataFrame:
    """Per-stratum Spearman of metric vs gt (primary reporting unit)."""
    rows = []
    for key, g in df.groupby(list(strata), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(strata, key)),
            "n": len(g),
            "spearman": partial_spearman(g[metric], g[gt]),
        })
    return pd.DataFrame(rows)


def confound_partials(df: pd.DataFrame, metrics, gt: str,
                      controls=DEFAULT_CONTROLS) -> pd.DataFrame:
    """Raw vs partial Spearman (controls regressed out of ranks) per
    metric. A metric whose association survives the controls carries
    signal beyond the shared mechanical drivers."""
    C = np.column_stack([df[c].astype(float).to_numpy() for c in controls])
    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "raw_spearman": partial_spearman(df[m], df[gt]),
            "partial_spearman": partial_spearman(df[m], df[gt], controls=C),
            "controls": ",".join(controls),
        })
    return pd.DataFrame(rows)
