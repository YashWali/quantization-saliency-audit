# Saliency Without Validation — reproducibility code

The measurement instrument and pre-registered analysis for a per-input-channel
audit of post-training-quantization saliency criteria (AWQ, GPTQ, OWQ, SpQR, and
a unified gradient score) against measured ground truth. Each weight channel is
individually quantized and its true output-distribution impact (KL) is measured;
the criteria are then compared against measurement and against each other, net of
the activation statistic they share. Run across three models at the same scale:
Qwen2.5-0.5B, Pythia-410m (GPT-NeoX), and SmolLM2-360M (Llama family).

This repository contains everything needed to regenerate the reported numbers.

## Layout

- `src/qsal/` — library: shared channel grid, fake quantizers, the five scores,
  ground-truth machinery, and the analysis (correlation, overlap, nulls, etc.).
- `src/qsal/config.py`, `config_pythia.py`, `config_smollm2.py` — the frozen
  per-model configurations (thresholds, sweep layers, sizes).
- `scripts/` — the pipeline: calibration → scores → ground truth → analysis → plots.
- `tests/` — the test suite (`pytest`).
- `results/<model>/analysis/*.json` — the reference numbers to diff a re-run against.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q            # optional: confirm the code is correct (~1 min)
```

## Reproduce

The model is selected by the `QSAL_CONFIG` env var (default `config.py` = Qwen).

```
# Qwen2.5-0.5B (Phase 1) — model auto-downloads from Hugging Face on first use
python scripts/01_collect_calibration.py
python scripts/02_compute_scores.py
python scripts/02b_owq_residual.py
python scripts/03_ground_truth.py            # the heavy step — see "Runtime" below
python scripts/03b_fp64_recheck.py
python scripts/03c_threshold_validation.py
python scripts/03d_split_half_reliability.py
python scripts/04_analysis.py
python scripts/04b_addendum.py
python scripts/04d_review_robustness.py
python scripts/05_plots.py
python scripts/06_guard_recheck.py

# Pythia-410m / SmolLM2-360M — select the matching frozen config
QSAL_CONFIG=config_pythia  python scripts/phase2_download_model.py EleutherAI/pythia-410m
QSAL_CONFIG=config_pythia  python scripts/phase2_01_collect_calibration.py
QSAL_CONFIG=config_pythia  python scripts/phase2_02_compute_scores.py
QSAL_CONFIG=config_pythia  python scripts/phase2_03_ground_truth.py   # heavy
QSAL_CONFIG=config_pythia  python scripts/phase2_04_analysis.py
QSAL_CONFIG=config_pythia  python scripts/phase2_05_plots.py
# (use QSAL_CONFIG=config_smollm2 with HuggingFaceTB/SmolLM2-360M-Instruct)
```

Outputs land under `data/<model>/` (calibration, scores) and `results/<model>/`
(ground truth, analysis JSONs, figures).

## Runtime & hardware

The ground-truth step (`03_ground_truth.py` / `phase2_03_ground_truth.py`)
perturbs and measures every census channel one at a time — it is the expensive
stage, on the order of **several hours up to ~a day** on an Apple-silicon laptop
(M1, MPS) or a single GPU, scaling with the channel count (Qwen ≈ 36k, Pythia
≈ 21k, SmolLM2 ≈ 30k census channels) and the model's layer count. It is
**resumable**: it writes atomic per-chunk checkpoints and continues where it left
off if re-run after an interruption. For long unattended runs,
`scripts/phase2_run_overnight.sh <pythia|smollm2>` wraps the GT step in a
self-healing retry loop with a memory watchdog. Calibration, scores, analysis,
and plots are minutes each.

## Verify

After a run, compare `results/<model>/analysis/summary.json` (and
`addendum.json`, `split_half_reliability.json`) against the reference copies
shipped in this repository.

## Pre-registration & citation

The frozen analysis specification (decision rules and per-model configs, fixed
before the data) is archived on Zenodo: **DOI 10.5281/zenodo.20725591** (concept DOI; the
latest version includes the SmolLM2 pre-registration, deposited while that
model's ground-truth run was in progress). The accompanying paper is linked from
the repository description.

---

<sub>An AI coding assistant helped implement the author's decisions; all study design, analysis, and conclusions are the author's.</sub>
