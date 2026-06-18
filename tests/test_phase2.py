"""Phase-2 GT candidate logic (src/qsal/phase2.py).

Pure pandas helpers for the single-pass Pythia runner: the source-tagged
candidate table (union > sweep > control) and the seeded stratified
reliability subset. Tested on synthetic grids/scores - no model needed.
"""

import numpy as np
import pandas as pd


def _toy_grid():
    rows, cid = [], 0
    for layer in range(3):
        for lin, inf in [("a", 10), ("b", 5)]:
            for ic in range(inf):
                rows.append({"channel_id": cid, "layer": layer,
                             "linear": lin, "in_channel": ic, "in_features": inf})
                cid += 1
    return pd.DataFrame(rows)


def _toy_scores(grid, metrics):
    rng = np.random.default_rng(0)
    d = {"channel_id": grid["channel_id"].to_numpy()}
    for m in metrics:
        d[m] = rng.random(len(grid))
    return pd.DataFrame(d)


def test_build_candidates_tags_dedup_and_control_count():
    from qsal.phase2 import build_candidates

    grid = _toy_grid()
    metrics = ["awq", "gptq", "owq_residual"]
    scores = _toy_scores(grid, metrics)
    cand = build_candidates(grid, scores, metrics=metrics, sweep_layers=(1,),
                            n_control=5, top_pct=0.2, seed=0)
    # all sweep-layer channels are candidates (tagged union or sweep)
    assert set(grid.loc[grid["layer"] == 1, "channel_id"]) <= set(cand["channel_id"])
    assert set(cand["source"]) <= {"union", "sweep", "control"}
    assert (cand["source"] == "control").sum() == 5
    assert cand["channel_id"].is_unique
    # controls are drawn from outside union+sweep
    ctrl = cand.loc[cand["source"] == "control"]
    assert (ctrl["layer"] != 1).all()


def test_build_candidates_union_priority_over_sweep():
    from qsal.phase2 import build_candidates

    grid = _toy_grid()
    metrics = ["awq"]
    scores = _toy_scores(grid, metrics)
    cand = build_candidates(grid, scores, metrics=metrics, sweep_layers=(1,),
                            n_control=0, top_pct=0.5, seed=0)
    # a sweep-layer channel that is also a top-1% union pick is tagged 'union'
    inter = cand[(cand["layer"] == 1) & cand["in_union"]]
    assert (inter["source"] == "union").all()


def test_reliability_subset_caps_per_stratum_and_deterministic():
    from qsal.phase2 import reliability_subset

    grid = _toy_grid()
    sub = reliability_subset(grid, sweep_layers=(1, 2), per_stratum=3, seed=0)
    # 2 layers x 2 linears = 4 strata, capped 3 each -> 12
    assert len(sub) == 12
    assert set(grid.loc[grid["channel_id"].isin(sub), "layer"]) == {1, 2}
    counts = (grid[grid["channel_id"].isin(sub)]
              .groupby(["layer", "linear"]).size())
    assert (counts <= 3).all()
    # deterministic for a fixed seed
    assert sub == reliability_subset(grid, (1, 2), 3, seed=0)


def test_reliability_subset_handles_small_strata():
    from qsal.phase2 import reliability_subset

    grid = _toy_grid()
    # per_stratum larger than a stratum's size -> take all of it (linear 'b' = 5)
    sub = reliability_subset(grid, sweep_layers=(0,), per_stratum=100, seed=1)
    assert len(sub) == 15  # linear a (10) + linear b (5)


def test_spearman_brown_monotone():
    from qsal.phase2 import spearman_brown

    # length-doubling raises reliability; 0->0, 1->1
    assert spearman_brown(0.0) == 0.0
    assert abs(spearman_brown(1.0) - 1.0) < 1e-12
    assert spearman_brown(0.24) > 0.24  # 0.24 -> ~0.387
    assert abs(spearman_brown(0.24) - 0.387) < 0.01


def test_split_half_within_uses_per_stratum_median():
    from qsal.phase2 import split_half_within

    rng = np.random.default_rng(0)
    rows = []
    # stratum X: a,b correlated; stratum Y: a,b independent
    for i in range(20):
        x = rng.random()
        rows.append({"layer": 0, "linear": "X", "a": x, "b": x + 0.01 * rng.random()})
    for i in range(20):
        rows.append({"layer": 0, "linear": "Y", "a": rng.random(), "b": rng.random()})
    df = pd.DataFrame(rows)
    med, n = split_half_within(df, "a", "b")
    assert n == 2  # two strata, each >= min_n
    assert 0.0 <= med <= 1.0


def test_joint_set_level_sign_test_counts_correct_strata():
    from qsal.phase2 import joint_set_level_sign_test

    rows = []
    # stratum with union LOI < control LOI (correct sign), enough per group
    for _ in range(3):
        rows.append({"layer": 0, "linear": "X", "source": "union", "gt_loi": 0.1})
        rows.append({"layer": 0, "linear": "X", "source": "control", "gt_loi": 0.5})
    # stratum with too few controls -> excluded
    rows.append({"layer": 0, "linear": "Y", "source": "union", "gt_loi": 0.1})
    df = pd.DataFrame(rows)
    n_correct, n_total = joint_set_level_sign_test(df)
    assert (n_correct, n_total) == (1, 1)
