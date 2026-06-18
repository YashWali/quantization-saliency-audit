"""Frozen run configuration - PRE-REGISTRATION.

This file is committed BEFORE any ground-truth run (the design spec).
It freezes every analysis-relevant decision: model/dataset revisions, seeds,
calibration/eval sizes, quantization bits, damping grid, sweep layers, top-k
fractions, and the RQ1-RQ5 decision thresholds from the design spec §2.

Any post-hoc change to a value in this file is a pre-registration AMENDMENT
and must be documented in results/REPORT.md with its rationale.

This module is the Phase-1 (Qwen) config and the default when QSAL_CONFIG is
unset. Phase 2 and Phase 3 use frozen variants (config_pythia.py,
config_smollm2.py) that import the held-identical Phase-1 values from here so
they cannot drift, overriding only the model/sizes/linear-name fields.
"""

import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_global_seeds(seed: int = SEED) -> None:
    """Set seeds for random / numpy / torch (incl. MPS where applicable)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Model (Phase 1 pilot - all claims scoped to this model; spec §1 [guard])
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # main @ 2024-09-25
NUM_LAYERS = 24
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
# Decoder linears scored per layer; tied lm_head/embedding EXCLUDED (assert tying).
LINEAR_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
EXPECTED_TOTAL_CHANNELS = 245_760  # 24 * (896*6 + 4864)

# Dtypes (spec §5 numerical hygiene [guard])
FORWARD_DTYPE = torch.float32   # GT + calibration forwards (MPS nondeterminism guard)
ACCUM_DTYPE = torch.float64     # Hessian accumulation, KL reduction, Cholesky (CPU)

# ---------------------------------------------------------------------------
# Calibration / held-out eval data (pinned; zero-overlap asserted in code)
# ---------------------------------------------------------------------------
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # main @ 2024-01-04
CALIB_SPLIT = "train"   # calibration sequences drawn here
EVAL_SPLIT = "test"     # held-out eval drawn here (disjoint split by construction)

CALIB_NUM_SEQS = 128
CALIB_SEQ_LEN = 512
# AMENDMENT 2026-06-05 (before main GT run; only 30 layer-0 smoke candidates
# had been measured, now archived): 32 -> 16, the lower end of the spec's
# pre-stated 16-32 range, for compute (~70h -> ~35h on M1). Sensitivity
# check on the smoke candidates: independent 16-seq split halves give
# spearman 0.91 / pearson(log) 0.94 on kl_iso with union/control separation
# preserved (6.1x / 6.9x vs 6.5x at 32). Document in REPORT.md.
EVAL_NUM_SEQS = 16      # spec range 16-32; pre-registered at 32; amended, see above
EVAL_SEQ_LEN = 256

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
DEPLOYMENT_BITS = 4     # joint GT + SpQR residual quantizer Q (INT4)
ISOLATED_PROBE_BITS = 3 # isolated single-channel GT (INT3 for SNR)
QUANT_GROUP_SIZE = 128  # per-group quantization; single-channel scale taken
                        # within the channel's real 128-group (matches deployment)

# Damping sweep for fp64 CPU Cholesky: H <- H + lam*mean(diag(H))*I  [guard]
DAMPING_LAMBDAS = (1e-3, 1e-2, 1e-1)
DAMPING_PRIMARY = 1e-2  # headline scores use this; ranking sensitivity reported

# ---------------------------------------------------------------------------
# Ground-truth budget (spec §5)
# ---------------------------------------------------------------------------
TOP_PCT_UNION = 0.01        # per-metric top-1% -> candidate union
NUM_CONTROL_CHANNELS = 1000 # random controls (must cluster near noise floor)
# Full-sweep layers: every channel measured; spans depth (early/mid/late).
# Population correlations (RQ5) use ONLY these layers [guard: selection bias].
SWEEP_LAYERS = (2, 11, 22)
NOISE_FLOOR_REPEATS = 3     # >=3 identical-perturbation repeats (Step 5.4 gate)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
TOP_K_FRACTIONS = (0.01, 0.05, 0.10)  # top-1/5/10% sets for RQ2/RQ3
N_PERMUTATIONS = 1000   # within-layer permutation null for Jaccard
N_BOOTSTRAP = 1000      # BCa bootstrap CIs; paired bootstrap for differences
CI_LEVEL = 0.95
MULTIPLE_COMPARISON = "holm"  # across layer-wise / pairwise families
ALPHA = 0.05

# ---------------------------------------------------------------------------
# PRE-REGISTERED DECISION RULES (spec §2) - evaluated with CIs, never point
# estimates. Each RQ lists its pass condition and falsifying outcome.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # RQ1 Concentration: "concentrated" iff on sweep layers the top-1% of
    # channels explain >=50% of summed GT sensitivity AND the BCa CI lower
    # bound > 25%. Falsified if the CI upper bound < 25%.
    "rq1_top_pct": 0.01,
    "rq1_explained_frac": 0.50,
    "rq1_ci_lower_min": 0.25,
    # RQ2 Agreement: methods "agree" at top-k iff observed Jaccard exceeds the
    # chance/mechanical baseline with a CI excluding that baseline.
    # Raw Jaccard is never reported alone [guard].
    "rq2_baseline": "chance_and_mechanical",
    # RQ3 Consensus: "tiny set explains most sensitivity" iff net-of-chance
    # GT-fraction CI lower bound > 50% at consensus-set size < 1% of channels.
    "rq3_net_gt_frac_ci_lower": 0.50,
    "rq3_max_set_frac": 0.01,
    # RQ4 Overprotection: within a method's top set, "overprotects" iff the
    # top-20% of selected channels (by GT) explain >=80% of the set's summed
    # GT with CI lower bound > 60%.
    "rq4_top_frac": 0.20,
    "rq4_explained_frac": 0.80,
    "rq4_ci_lower_min": 0.60,
    # RQ5 Best predictor (reframed): a metric "wins" only if its paired-
    # bootstrap Spearman-advantage CI excludes 0. Unified metric is the
    # 1st-order linearization of the GT, so report BY HOW MUCH and WHERE it
    # fails (PQI large-perturbation regime), not just whether it wins.
    "rq5_win_rule": "paired_bootstrap_ci_excludes_zero",
}
