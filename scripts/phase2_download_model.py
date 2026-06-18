"""Phase 2 model pre-fetch - resilient against flaky internet.

Downloads a candidate model into the local HF cache, wrapping snapshot_download
in a retry loop with exponential backoff. snapshot_download resumes partially
-downloaded files by default, so each retry picks up where the previous attempt
dropped instead of restarting.

This does NOT touch config.py / pre-registration - it only warms the cache so
the network is not on the critical path later. After it completes it prints and
records the RESOLVED COMMIT HASH; pin that as the Phase-2 MODEL_REVISION.

Phase-2 target is EleutherAI/pythia-410m (a GPT-NeoX-family model, deliberately
architecturally distant from Qwen2.5 - see the design spec). SmolLM2-360M
-Instruct (already cached) is kept as the same-family contingency/disambiguator.

Run (background-friendly):
    .venv/bin/python scripts/phase2_download_model.py                # Pythia
    .venv/bin/python scripts/phase2_download_model.py <hf/model-id>  # any model
Progress + final status go to results/phase2_dl_<slug>.log; a completion marker
is written to results/phase2_dl_<slug>.done on success.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Prefer the plain HTTPS backend with built-in resume; the xet backend has been
# less predictable on flaky links. Set before importing huggingface_hub.
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

DEFAULT_MODEL = "EleutherAI/pythia-410m"
REVISION = "main"  # resolved to a commit hash below and recorded for pinning

# Only skip serialization formats we never use. Keep BOTH safetensors and .bin:
# some repos (e.g. Pythia) ship .bin as the only/primary weights, so excluding
# it would leave the cache unusable. transformers prefers safetensors if present.
IGNORE_PATTERNS = ["*.gguf", "*.onnx", "*.onnx_data", "onnx/*"]

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 500
BACKOFF_BASE = 5.0
BACKOFF_CAP = 60.0


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()


def main(model_id: str) -> int:
    log_path = REPO_ROOT / "results" / f"phase2_dl_{slug(model_id)}.log"
    done_path = REPO_ROOT / "results" / f"phase2_dl_{slug(model_id)}.done"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    def resolve_commit():
        try:
            return HfApi().model_info(model_id, revision=REVISION).sha
        except Exception as exc:
            log(f"could not resolve commit yet ({type(exc).__name__}: {exc})")
            return None

    log(f"START prefetch {model_id}@{REVISION} (xet disabled, resume on)")
    commit = resolve_commit()
    if commit:
        log(f"resolved {model_id}@{REVISION} -> commit {commit}")

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            local_dir = snapshot_download(
                repo_id=model_id, revision=REVISION,
                ignore_patterns=IGNORE_PATTERNS,
            )
            if not commit:
                commit = resolve_commit()
            log(f"COMPLETE on attempt {attempt}; cache path: {local_dir}")
            done_path.write_text(json.dumps({
                "model_id": model_id, "revision_ref": REVISION,
                "resolved_commit": commit, "cache_path": local_dir,
                "attempts": attempt,
            }, indent=2) + "\n")
            log(f"wrote marker {done_path}")
            log("NEXT: pin resolved_commit as the Phase-2 MODEL_REVISION.")
            return 0
        except KeyboardInterrupt:
            log("interrupted by user")
            return 130
        except Exception as exc:
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** min(attempt - 1, 6)))
            log(f"attempt {attempt} failed ({type(exc).__name__}: {exc}); "
                f"resuming in {wait:.0f}s")
            time.sleep(wait)

    log(f"GAVE UP after {MAX_ATTEMPTS} attempts")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default=DEFAULT_MODEL,
                    help=f"HF model id (default {DEFAULT_MODEL})")
    args = ap.parse_args()
    sys.exit(main(args.model_id))
