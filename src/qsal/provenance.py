"""Provenance sidecar manifests (spec Appendix C).

Every artifact gets `<artifact>.provenance.json` recording the pinned
revisions, seed, dtypes, environment, and git SHA that produced it.
"""

import json
import subprocess
from pathlib import Path

import torch
import transformers

import qsal.config as cfg


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(**extras) -> dict:
    m = {
        "model_id": cfg.MODEL_ID,
        "model_revision": cfg.MODEL_REVISION,
        "dataset_id": cfg.DATASET_ID,
        "dataset_config": cfg.DATASET_CONFIG,
        "dataset_revision": cfg.DATASET_REVISION,
        "seed": cfg.SEED,
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "forward_dtype": str(cfg.FORWARD_DTYPE),
        "accum_dtype": str(cfg.ACCUM_DTYPE),
    }
    m.update(extras)
    return m


def save_manifest(artifact_path, manifest: dict) -> Path:
    p = Path(str(artifact_path) + ".provenance.json")
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p
