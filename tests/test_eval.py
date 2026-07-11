"""Tests for the eval metrics. Fast (no model loads)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contextx.eval import metrics as M  # noqa: E402


def test_recall_precision_hit():
    rel = ["a", "b", "c"]
    ranked = ["a", "x", "b", "y", "z"]
    assert M.recall_at_k(rel, ranked, 5) == 2 / 3      # a, b found; c missing
    assert M.precision_at_k(rel, ranked, 5) == 2 / 5   # 2 of top 5 relevant
    assert M.hit_at_k(rel, ranked, 1) == 1.0           # a at rank 1
    assert M.hit_at_k(["c"], ranked, 3) == 0.0         # c not in top 3


def test_reciprocal_rank():
    assert M.reciprocal_rank(["b"], ["a", "b", "c"]) == 0.5
    assert M.reciprocal_rank(["a"], ["a", "b"]) == 1.0
    assert M.reciprocal_rank(["z"], ["a", "b"]) == 0.0


def test_ndcg_perfect_and_worst():
    rel = ["a", "b"]
    perfect = ["a", "b", "c", "d"]
    assert M.ndcg_at_k(rel, perfect, 4) == 1.0
    none = ["c", "d", "e", "f"]
    assert M.ndcg_at_k(rel, none, 4) == 0.0
    # a relevant doc lower down scores between 0 and 1
    mid = ["c", "a", "d", "b"]
    assert 0.0 < M.ndcg_at_k(rel, mid, 4) < 1.0


def test_ndcg_rewards_higher_placement():
    rel = ["a"]
    high = M.ndcg_at_k(rel, ["a", "b", "c"], 3)
    low = M.ndcg_at_k(rel, ["b", "c", "a"], 3)
    assert high > low


def test_aggregate_bootstrap_ci_brackets_mean():
    vals = [1.0, 1.0, 0.0, 1.0, 0.5]
    agg = M.aggregate({"m": vals}, n_boot=500, seed=0)
    mean, lo, hi = agg["m"]
    assert abs(mean - np.mean(vals)) < 1e-9
    assert lo <= mean <= hi
