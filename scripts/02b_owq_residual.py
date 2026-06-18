"""Paper-faithful OWQ score (residual-energy variant).

The OWQ paper (2306.02272) defines column sensitivity as
lambda_j * ||dW_:,j||^2 with dW the QUANTIZATION ERROR; the original
implementation used full weight energy ||W_:,j||^2. This script computes
the paper-faithful variant

    owq_residual_c = H_cc/n * sum_i (W_ic - Q(W)_ic)^2   (Q = INT4/g128)

post-hoc (scores never enter the GT values, only the candidate union),
writes results/scores_owq.parquet (owq_residual + the original column as
owq_energy), and reports:
  - per-linear top-1% Jaccard between the two variants, and
  - coverage: how many owq_residual top-1% channels are inside the
    measured GT candidate union (built from the original five metrics).
Tail analyses for OWQ must use owq_residual and disclose the coverage gap.

CPU-only (~3GB peak: fp32 model + one fp64 Hessian at a time). Do NOT run
concurrently with the GT measurement on a 16GB machine.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import qsal.config as cfg
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_layer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HESS = DATA / "hessians"
RESULTS = ROOT / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def top_pct_ids(df: pd.DataFrame, col: str, pct: float) -> set:
    ranks = df.groupby(["layer", "linear"], observed=True)[col].rank(
        ascending=False, pct=True
    )
    return set(df.loc[ranks <= pct, "channel_id"])


def main():
    cfg.set_global_seeds()
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    scores = pd.read_parquet(RESULTS / "scores.parquet")
    model, _ = load_model(device="cpu")

    frames = []
    for layer, name, module in enumerate_linears(model):
        Hrec = torch.load(HESS / f"L{layer:02d}_{name}.pt", weights_only=True)
        h_diag_over_n = (Hrec["H"].diagonal() / Hrec["token_count"]).double()
        W = module.weight.detach()
        Wq = quantize_layer(W.float(), bits=cfg.DEPLOYMENT_BITS)
        residual = (W.double() - Wq.double()).pow(2).sum(0)
        sub = grid[(grid["layer"] == layer) & (grid["linear"] == name)]
        frames.append(pd.DataFrame({
            "channel_id": sub["channel_id"].to_numpy(),
            "owq_residual": (residual * h_diag_over_n).numpy(),
        }))
        if name == "down_proj":
            log(f"layer {layer} done")

    owq_new = pd.concat(frames, ignore_index=True).sort_values("channel_id")
    assert len(owq_new) == len(grid)
    assert np.isfinite(owq_new["owq_residual"].to_numpy()).all()

    s = scores.merge(grid, on="channel_id").merge(owq_new, on="channel_id")
    s = s.rename(columns={"owq": "owq_energy"})

    # per-linear top-1% agreement between variants
    jac_rows = []
    for (layer, name), g in s.groupby(["layer", "linear"], observed=True):
        k = max(1, int(round(cfg.TOP_PCT_UNION * len(g))))
        a = set(g.nlargest(k, "owq_energy")["channel_id"])
        b = set(g.nlargest(k, "owq_residual")["channel_id"])
        jac_rows.append({"layer": layer, "linear": name,
                         "jaccard_top1pct": len(a & b) / len(a | b)})
    jac = pd.DataFrame(jac_rows)

    # coverage of the corrected variant's top-1% by the measured union
    metrics = ["awq", "gptq", "owq_energy", "spqr", "unified"]
    union_ids = set()
    for m in metrics:
        union_ids |= top_pct_ids(s, m, cfg.TOP_PCT_UNION)
    sweep_ids = set(grid.loc[grid["layer"].isin(cfg.SWEEP_LAYERS),
                             "channel_id"])
    new_top = top_pct_ids(s, "owq_residual", cfg.TOP_PCT_UNION)
    measured = union_ids | sweep_ids  # controls excluded: they are random
    covered = len(new_top & measured)

    out = RESULTS / "scores_owq.parquet"
    s[["channel_id", "owq_energy", "owq_residual"]].to_parquet(out, index=False)
    summary = {
        "n_owq_residual_top1pct": len(new_top),
        "covered_by_measured_candidates": covered,
        "coverage_frac": covered / len(new_top),
        "in_original_union": len(new_top & union_ids),
        "jaccard_top1pct_mean": float(jac["jaccard_top1pct"].mean()),
        "jaccard_top1pct_min": float(jac["jaccard_top1pct"].min()),
    }
    (RESULTS / "owq_variant_summary.json").write_text(
        json.dumps(summary, indent=2))
    jac.to_parquet(RESULTS / "owq_variant_jaccard.parquet", index=False)
    save_manifest(out, build_manifest(
        stage="owq_residual", owq_bits=cfg.DEPLOYMENT_BITS,
        note="paper-faithful OWQ (residual energy)",
    ))
    log(f"done: {json.dumps(summary)}")


if __name__ == "__main__":
    sys.exit(main())
