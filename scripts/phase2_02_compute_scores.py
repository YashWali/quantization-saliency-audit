"""Phase 2 (Pythia) saliency scores — variant of 02 + 02b, single-pass.

Computes the five corrected criteria for Pythia from the Phase-2 calibration
(data/pythia/), driven by config_pythia, writing results/pythia/scores.parquet.

Single-pass design (the design spec): owq_residual is the PRIMARY OWQ from
the start (paper-faithful: H_cc/n * sum_i (W_ic - Q(W)_ic)^2, Q = INT4/g128);
owq_energy (the original full-weight-energy variant) is kept as a labelled
secondary column. So 02b is folded in here rather than run as a separate pass.

Columns: awq, gptq, owq_residual, owq_energy, spqr, unified, gptq_actorder_proxy.
Also writes lambda_sensitivity + hessian_info (the damping-sweep guard).
CPU-only; reads one fp64 Hessian at a time.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import importlib
import os as _os

_CFG = _os.environ.get("QSAL_CONFIG", "config_pythia")
cfg = importlib.import_module("qsal." + _CFG)
_RUN = _CFG.rsplit("config_", 1)[-1]  # "pythia" / "smollm2" -> data|results dir
from qsal.hessian import damped_inverse_diag
from qsal.models import enumerate_linears, load_model
from qsal.provenance import build_manifest, save_manifest
from qsal.quantizers import quantize_layer
from qsal.scores import awq, gptq, owq, spqr, unified

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / _RUN
HESS = DATA / "hessians"
RESULTS = ROOT / "results" / _RUN

# last linear of a decoder layer (cosmetic per-layer log trigger)
_LAST_LINEAR = cfg.LINEAR_NAMES[-1]   # cosmetic per-layer log trigger (family-aware)
# small square in=hidden linear for the Cholesky-vs-pinv guard (H is hidden x hidden);
# LINEAR_NAMES[0] is in=hidden for every family (q_proj / query_key_value).
_PINV_CHECK = f"L00_{cfg.LINEAR_NAMES[0]}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    cfg.set_global_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(DATA / "channel_grid.parquet")
    stats = torch.load(DATA / "calib_stats.pt", weights_only=True)
    log("loading model weights (cpu, fp32)")
    model, _ = load_model(device="cpu", cfg=cfg)

    # [guard] Cholesky diag(H^-1) vs direct pinv on one small layer
    Hrec = torch.load(HESS / f"{_PINV_CHECK}.pt", weights_only=True)
    Hn = Hrec["H"] / Hrec["token_count"]
    d_chol, _ = damped_inverse_diag(Hn, cfg.DAMPING_PRIMARY, with_info=False)
    Hd = Hn + cfg.DAMPING_PRIMARY * Hn.diagonal().mean() * torch.eye(
        Hn.shape[0], dtype=torch.float64
    )
    d_pinv = torch.linalg.pinv(Hd).diagonal()
    assert torch.allclose(d_chol, d_pinv, rtol=1e-6), "Cholesky vs pinv mismatch"
    log(f"pinv validation OK ({_PINV_CHECK})")

    frames, lam_sens, infos = [], [], []
    for layer, name, module in enumerate_linears(model, cfg=cfg):
        st = stats[f"L{layer}.{name}"]
        n = st["token_count"]
        Hrec = torch.load(HESS / f"L{layer:02d}_{name}.pt", weights_only=True)
        assert Hrec["token_count"] == n
        Hn = Hrec["H"] / n  # spec Appendix A: H = (1/n) sum x x^T

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
        h_diag_over_n = Hn.diagonal()
        Wq = quantize_layer(W.float(), bits=cfg.DEPLOYMENT_BITS)
        owq_residual = ((W.double() - Wq.double()).pow(2).sum(0)
                        * h_diag_over_n.double())
        sub = grid[(grid["layer"] == layer) & (grid["linear"] == name)]
        frames.append(
            pd.DataFrame(
                {
                    "channel_id": sub["channel_id"].to_numpy(),
                    "awq": awq.score(st["sum_abs_x"], n).numpy(),
                    "gptq": gptq.score(d_primary).numpy(),
                    "owq_residual": owq_residual.numpy(),
                    "owq_energy": owq.score(W, h_diag_over_n).numpy(),
                    "spqr": spqr.score(W, d_primary).numpy(),
                    "unified": unified.score(st["sum_gx2"], n).numpy(),
                    "gptq_actorder_proxy": h_diag_over_n.numpy(),
                }
            )
        )
        if name == _LAST_LINEAR:
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
        stage="scores_pythia",
        damping_lambda=cfg.DAMPING_PRIMARY,
        damping_sweep=list(cfg.DAMPING_LAMBDAS),
        spqr_bits=cfg.DEPLOYMENT_BITS,
        owq_primary="owq_residual",
    ))
    min_rho = min(r["gptq_rank_spearman"] for r in lam_sens)
    log(f"scores.parquet written ({len(scores)} rows); "
        f"min adjacent-lambda GPTQ rank-Spearman {min_rho:.4f}")


if __name__ == "__main__":
    sys.exit(main())
