"""Frozen Phase-3 run configuration - PRE-REGISTRATION (third model).

Triggered 2026-06-17: Phase 2 (Pythia-410m) replicated the algebraic/mechanical
core but DISAGREED with Phase 1 on an empirical-landscape result (which criterion
catches the pooled super-channels: unified on Qwen, AWQ/GPTQ on Pythia). Per the
pre-registered contingency in the design spec, an empirical-landscape break
(with the algebraic core intact) fires the SmolLM2 third model to disambiguate
architecture vs. training/scale.

SmolLM2-360M is Llama-family (like Qwen2.5) but a different training recipe/scale:
- agrees with Qwen  => the Qwen/Pythia divergence is ARCHITECTURAL (GPT-NeoX is the
  outlier);
- agrees with Pythia => the divergence is TRAINING/SCALE-driven, not architecture.

Design (choice 2, identical to config_pythia): EVERYTHING not architecture-specific
is held BYTE-IDENTICAL to Phase 1 by importing it from `config.py` - seeds, dtypes,
dataset+revisions, calib/eval sizes (eval-16), quant bits, damping grid, top-k, all
RQ1-RQ5 decision thresholds, fp64 hygiene. Only the model identity, its size, and
SWEEP_LAYERS (which MUST change: 32 layers, not 24) are overridden. LINEAR_NAMES is
imported unchanged because Llama reuses the Qwen2 7-linear taxonomy. The new Phase-2
knob (reliability-subset size) is held equal to Phase 2 for direct comparability.

Any post-hoc change to a value here is a pre-registration AMENDMENT and must be
documented in results/REPORT.md with its rationale.
"""

# Held IDENTICAL to Phase 1 (pre-registration choice 2) - imported, not copied.
# NOTE vs config_pythia: SWEEP_LAYERS is NOT imported (overridden below for 32
# layers); LINEAR_NAMES IS imported (Llama reuses the Qwen2 taxonomy).
from qsal.config import (  # noqa: F401  (re-exported as the frozen Phase-3 config)
    ACCUM_DTYPE,
    ALPHA,
    CALIB_NUM_SEQS,
    CALIB_SEQ_LEN,
    CALIB_SPLIT,
    CI_LEVEL,
    DAMPING_LAMBDAS,
    DAMPING_PRIMARY,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DEPLOYMENT_BITS,
    EVAL_NUM_SEQS,
    EVAL_SEQ_LEN,
    EVAL_SPLIT,
    FORWARD_DTYPE,
    ISOLATED_PROBE_BITS,
    LINEAR_NAMES,          # Llama == Qwen2: q/k/v/o + gate/up/down
    MULTIPLE_COMPARISON,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    NOISE_FLOOR_REPEATS,
    NUM_CONTROL_CHANNELS,
    QUANT_GROUP_SIZE,
    SEED,
    THRESHOLDS,
    TOP_K_FRACTIONS,
    TOP_PCT_UNION,
    set_global_seeds,
)

# ---------------------------------------------------------------------------
# Model (Phase 3 - SmolLM2-360M-Instruct; Llama family, tied embeddings).
# Verified against the cached config.json @ this revision (model_type llama,
# 32 layers, hidden 960, intermediate 2560, tie_word_embeddings True).
# ---------------------------------------------------------------------------
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"  # resolved main commit
NUM_LAYERS = 32
HIDDEN_SIZE = 960
INTERMEDIATE_SIZE = 2560
# Llama 7-linear taxonomy is reused from config.py (q/k/v/o + gate/up/down);
# tied lm_head excluded by tying (asserted in models.py), as in Phase 1.
EXPECTED_TOTAL_CHANNELS = 266_240  # 32 * (960*6 + 2560) = 32 * 8320

# ---------------------------------------------------------------------------
# SWEEP_LAYERS - the one analysis-relevant constant that MUST differ (32 layers
# vs Phase-1/Phase-2's 24). Chosen as the depth-matched analog of the frozen
# {2,11,22}/24 (fractional depths 0.083 / 0.458 / 0.917): for 32 layers the
# nearest indices are {3,15,29}/32 (0.094 / 0.469 / 0.906), preserving the
# early/mid/late span across depth x module. Frozen before any Phase-3 run.
# ---------------------------------------------------------------------------
SWEEP_LAYERS = (3, 15, 29)

# ---------------------------------------------------------------------------
# Phase-2/3 knob: per-channel joint split-half reliability subset, held EQUAL to
# config_pythia (200/stratum) so the "did the joint set-level demotion replicate?"
# within-linear reliability is directly comparable across all three models.
# SmolLM2 has 7 linears/layer x 3 sweep layers = 21 strata -> ~4,200 channels.
# ---------------------------------------------------------------------------
RELIABILITY_SUBSET_PER_STRATUM = 200
