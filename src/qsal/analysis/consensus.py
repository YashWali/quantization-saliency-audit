"""Consensus sets net of chance (the design spec, RQ3).

For each m, the channels picked by >= m of the M methods, with the
expected chance intersection (Poisson-binomial over per-method pick
probabilities) subtracted [guard].
"""

import numpy as np


def _prob_at_least_m(probs, m: int) -> float:
    """P(channel picked by >= m methods), independent picks (DP)."""
    dp = np.zeros(len(probs) + 1)
    dp[0] = 1.0
    for p in probs:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= 1 - p
    return float(dp[m:].sum())


def consensus_sets(method_sets: dict, N: int) -> dict:
    """{m: {size, members, expected_chance, net_size}} for m = 2..M."""
    methods = list(method_sets)
    probs = [len(method_sets[x]) / N for x in methods]
    counts = {}
    for s in method_sets.values():
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
    out = {}
    for m in range(2, len(methods) + 1):
        members = {ch for ch, c in counts.items() if c >= m}
        expected = N * _prob_at_least_m(probs, m)
        out[m] = {
            "size": len(members),
            "members": members,
            "expected_chance": expected,
            "net_size": len(members) - expected,
        }
    return out
