"""Compute the five corrected saliency scores (the design spec).

Per linear: load H (fp64), sweep damping lambdas for diag(H^-1) and log
GPTQ-ranking sensitivity to lambda [guard 4.1]; headline scores use the
pre-registered primary lambda. Validates Cholesky diag(H^-1) against
direct pinv on one small layer [guard]. Writes results/scores.parquet
(+ gptq_actorder_proxy reference column) aligned to channel_id, no NaN.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import qsal.config as cfg
from qsal.hessian import damped_inverse_diag
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.scores import awq, gptq, owq, spqr, unified

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HESS = DATA / "hessians"
RESULTS = ROOT / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    cfg.set_global_seeds()
    RESULTS.mkdir(exist_ok=True)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    stats = torch.load(DATA / "calib_stats.pt", weights_only=True)
    log("loading model weights (cpu, fp32)")
    model, _ = load_model(device="cpu")

    # [guard] validate Cholesky-based diag(H^-1) vs direct pinv on one
    # small layer (k_proj is 896x896)
    Hrec = torch.load(HESS / "L00_k_proj.pt", weights_only=True)
    Hn = Hrec["H"] / Hrec["token_count"]
    d_chol, _ = damped_inverse_diag(Hn, cfg.DAMPING_PRIMARY, with_info=False)
    Hd = Hn + cfg.DAMPING_PRIMARY * Hn.diagonal().mean() * torch.eye(
        Hn.shape[0], dtype=torch.float64
    )
    d_pinv = torch.linalg.pinv(Hd).diagonal()
    assert torch.allclose(d_chol, d_pinv, rtol=1e-6), "Cholesky vs pinv mismatch"
    log("pinv validation OK (L00 k_proj)")

    frames, lam_sens, infos = [], [], []
    for layer, name, module in enumerate_linears(model):
        st = stats[f"L{layer}.{name}"]
        n = st["token_count"]
        Hrec = torch.load(HESS / f"L{layer:02d}_{name}.pt", weights_only=True)
        assert Hrec["token_count"] == n
        Hn = Hrec["H"] / n  # spec Appendix A: H = (1/n) Sum x x^T

        # damping sweep: diag(H^-1) per lambda + GPTQ ranking sensitivity
        diag_inv = {}
        for lam in cfg.DAMPING_LAMBDAS:
            diag_inv[lam], info = damped_inverse_diag(
                Hn, lam, with_info=(lam == cfg.DAMPING_PRIMARY)
            )
            if lam == cfg.DAMPING_PRIMARY:
                infos.append({"layer": layer, "linear": name, **info})
        lams = list(cfg.DAMPING_LAMBDAS)
        for a, b in zip(lams, lams[1:]):
            rho = spearmanr(
                (1.0 / diag_inv[a]).numpy(), (1.0 / diag_inv[b]).numpy()
            ).statistic
            lam_sens.append(
                {"layer": layer, "linear": name, "lam_a": a, "lam_b": b,
                 "gptq_rank_spearman": rho}
            )

        W = module.weight.detach()
        d_primary = diag_inv[cfg.DAMPING_PRIMARY]
        sub = grid[(grid["layer"] == layer) & (grid["linear"] == name)]
        frames.append(
            pd.DataFrame(
                {
                    "channel_id": sub["channel_id"].to_numpy(),
                    "awq": awq.score(st["sum_abs_x"], n).numpy(),
                    "gptq": gptq.score(d_primary).numpy(),
                    "owq": owq.score(W, Hn.diagonal()).numpy(),
                    "spqr": spqr.score(W, d_primary).numpy(),
                    "unified": unified.score(st["sum_gx2"], n).numpy(),
                    "gptq_actorder_proxy": Hn.diagonal().numpy(),
                }
            )
        )
        if name == "down_proj":
            log(f"layer {layer} done")

    scores = pd.concat(frames, ignore_index=True).sort_values("channel_id")
    assert len(scores) == len(grid) == cfg.EXPECTED_TOTAL_CHANNELS
    assert scores["channel_id"].is_unique
    assert np.isfinite(
        scores.drop(columns="channel_id").to_numpy()
    ).all(), "NaN/inf in scores"

    out = RESULTS / "scores.parquet"
    scores.to_parquet(out, index=False)
    pd.DataFrame(lam_sens).to_parquet(RESULTS / "lambda_sensitivity.parquet",
                                      index=False)
    pd.DataFrame(infos).to_parquet(RESULTS / "hessian_info.parquet", index=False)
    save_manifest(out, build_manifest(
        stage="scores",
        damping_lambda=cfg.DAMPING_PRIMARY,
        damping_sweep=list(cfg.DAMPING_LAMBDAS),
        spqr_bits=cfg.DEPLOYMENT_BITS,
    ))
    min_rho = min(r["gptq_rank_spearman"] for r in lam_sens)
    log(f"scores.parquet written ({len(scores)} rows); "
        f"min adjacent-lambda GPTQ rank-Spearman {min_rho:.4f}")


if __name__ == "__main__":
    sys.exit(main())
