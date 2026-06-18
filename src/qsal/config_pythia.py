"""Frozen Phase-2 run configuration - PRE-REGISTRATION (second model).

Signed off 2026-06-16 (the design spec, sections A-D). Committed BEFORE any
Phase-2 ground-truth run, exactly as Phase-1 `config.py` was. Any post-hoc change
to a value here is a pre-registration AMENDMENT and must be documented.

Design (choice 2): EVERYTHING not architecture-specific is held BYTE-IDENTICAL to
Phase 1 by importing it from `config.py` below - so "held identical" is literal
and cannot drift. Only the model identity, its size/taxonomy, and the one new
Phase-2 knob (the reliability-subset size) are overridden here.

Target: EleutherAI/pythia-410m - GPT-NeoX family, chosen for maximal
architectural distance from Qwen2.5 (full MHA, GELU MLP, LayerNorm, parallel
residual, partial rotary, untied embeddings). See the design spec

Planned scope is n=2 (Qwen + Pythia) but NOT frozen at 2: if Phase 2 disagrees
with Phase 1 on an empirical-landscape result, SmolLM2-360M is run as a
pre-registered third model (architecture-vs-training disambiguator).
"""

# Held IDENTICAL to Phase 1 (pre-registration choice 2) - imported, not copied,
# so they are provably the same values: seeds, dtypes, dataset+revisions,
# calibration/eval sizes (eval-16), quant bits, damping grid, sweep layers,
# top-k, all RQ1-RQ5 decision thresholds, fp64 hygiene.
from qsal.config import (  # noqa: F401  (re-exported as the frozen Phase-2 config)
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
    MULTIPLE_COMPARISON,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    NOISE_FLOOR_REPEATS,
    NUM_CONTROL_CHANNELS,
    QUANT_GROUP_SIZE,
    SEED,
    SWEEP_LAYERS,
    THRESHOLDS,
    TOP_K_FRACTIONS,
    TOP_PCT_UNION,
    set_global_seeds,
)

# ---------------------------------------------------------------------------
# Model (Phase 2 - Pythia-410m; GPT-NeoX). Verified against the cached
# config.json @ this revision (model_type gpt_neox, 24 layers, untied).
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-410m"
MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"  # resolved main commit
NUM_LAYERS = 24            # same as Qwen -> SWEEP_LAYERS {2,11,22} carry over
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
# The 4 GPT-NeoX input-channel linears (fused QKV; we score INPUT channels, so
# the fused query_key_value at in=hidden is correct - no need to split q/k/v).
# Module taxonomy (parent path, layer container, untied-head exclusion) lives in
# models.py keyed by model_type; embed_out is excluded BY ROLE (vocab head, not a
# decoder linear) - the Phase-2 analogue of Phase-1's tied-lm_head exclusion.
LINEAR_NAMES = (
    "query_key_value",  # attention; in = hidden (1024)
    "dense",            # attention out-proj; in = hidden (1024)
    "dense_h_to_4h",    # mlp; in = hidden (1024)
    "dense_4h_to_h",    # mlp; in = intermediate (4096)
)
EXPECTED_TOTAL_CHANNELS = 172_032  # 24 * (1024*3 + 4096) = 24 * 7168

# ---------------------------------------------------------------------------
# Phase-2-specific pre-registered knob (the only genuinely-new value)
# ---------------------------------------------------------------------------
# Per-channel joint split-half reliability subset: stratified, this many channels
# per (layer, linear) stratum, measured on two disjoint 8-seq eval halves. Set to
# 200/stratum to MATCH the Phase-1 5b measurement (scripts/03d) for a directly
# comparable "did the set-level demotion replicate?" within-linear reliability.
# Pythia has 4 linears/layer x 3 sweep layers = 12 strata -> ~2,400 channels
# (vs Phase-1's 21 strata / ~4,200; the per-stratum cap is what is held equal).
RELIABILITY_SUBSET_PER_STRATUM = 200
