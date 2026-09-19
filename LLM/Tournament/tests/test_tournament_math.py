from __future__ import annotations

import numpy as np

from tournament_watermark import (
    BINOMIAL_PMF,
    LAYERS,
    binomial_randomized_pit,
    brute_force_tournament_distribution,
    stable_calibrator,
    tournament_distribution,
)


def test_recursion_matches_independent_pairwise_enumeration():
    p = np.array([0.07, 0.18, 0.31, 0.44])
    g = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    recursive, _ = tournament_distribution(p, g)
    brute = brute_force_tournament_distribution(p, g)
    assert np.allclose(recursive, brute, rtol=0.0, atol=2e-15)


def test_recursion_is_normalized_and_nonnegative_for_30_layers():
    rng = np.random.default_rng(20260823)
    p = rng.dirichlet(np.ones(17))
    g = rng.integers(0, 2, size=(LAYERS, p.size), dtype=np.uint8)
    result, errors = tournament_distribution(p, g)
    assert np.all(result >= 0.0)
    assert np.isclose(result.sum(), 1.0, rtol=0.0, atol=2e-15)
    assert np.max(np.abs(errors)) < 3e-15


def test_null_score_is_binomial_30_half():
    rng = np.random.default_rng(20260824)
    scores = rng.integers(0, 2, size=(250_000, LAYERS), dtype=np.uint8).sum(axis=1)
    observed = np.bincount(scores.astype(np.int64), minlength=LAYERS + 1) / scores.size
    assert np.max(np.abs(observed - BINOMIAL_PMF)) < 0.0025


def test_randomized_pit_is_uniform_and_calibrator_has_mean_one():
    rng = np.random.default_rng(20260825)
    scores = rng.binomial(LAYERS, 0.5, size=400_000)
    pivots = binomial_randomized_pit(scores, rng.random(scores.size))
    assert abs(float(pivots.mean()) - 0.5) < 0.002
    assert abs(float(np.mean(pivots <= 0.1)) - 0.1) < 0.002
    assert abs(float(np.mean(pivots <= 0.9)) - 0.9) < 0.002
    assert abs(float(stable_calibrator(pivots).mean()) - 1.0) < 0.008


def test_randomized_pit_uses_exact_binomial_cell():
    score = np.array([0, 15, 30])
    low = binomial_randomized_pit(score, np.zeros(3))
    high = binomial_randomized_pit(score, np.nextafter(np.ones(3), 0.0))
    assert np.all(low >= 0.0)
    assert np.all(high < 1.0)
    assert np.all(high > low)
