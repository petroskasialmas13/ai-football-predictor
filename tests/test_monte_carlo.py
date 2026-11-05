"""Tests for models.monte_carlo. Run from repo root: python -m pytest -q"""
import math

import numpy as np
import pytest

from models.monte_carlo import analytical_probabilities, simulate_match

SEED = 12345


def test_reproducible_with_same_seed():
    a = simulate_match(1.6, 1.1, n_simulations=50_000, seed=SEED)
    b = simulate_match(1.6, 1.1, n_simulations=50_000, seed=SEED)
    assert a.probabilities == b.probabilities
    assert a.top_scores == b.top_scores
    assert np.array_equal(a.score_matrix, b.score_matrix)


def test_different_seed_changes_sample():
    a = simulate_match(1.6, 1.1, n_simulations=50_000, seed=1)
    b = simulate_match(1.6, 1.1, n_simulations=50_000, seed=2)
    assert a.probabilities != b.probabilities


def test_probabilities_are_consistent():
    r = simulate_match(1.45, 1.2, n_simulations=100_000, seed=SEED)
    p = r.probabilities
    assert math.isclose(p["1"] + p["X"] + p["2"], 1.0, abs_tol=1e-12)
    assert math.isclose(p["HT 1"] + p["HT X"] + p["HT 2"], 1.0, abs_tol=1e-12)
    assert math.isclose(sum(r.halftime_fulltime.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(r.total_goals_distribution.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(r.score_matrix.sum() + r.score_matrix_other, 1.0, abs_tol=1e-12)
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        assert math.isclose(p[f"Over {line}"] + p[f"Under {line}"], 1.0, abs_tol=1e-12)
    overs = [p[f"Over {line}"] for line in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert overs == sorted(overs, reverse=True)
    assert len(r.top_scores) == 10
    for key, (lo, hi) in r.intervals.items():
        assert 0 <= lo <= p[key] <= hi <= 1


@pytest.mark.parametrize("lh,la", [(1.55, 1.05), (0.6, 2.3), (2.8, 0.4)])
def test_converges_to_analytical_poisson(lh, la):
    r = simulate_match(lh, la, n_simulations=500_000, seed=SEED)
    exact = analytical_probabilities(lh, la)
    n = r.n_simulations
    for key, value in exact.items():
        se = math.sqrt(max(value * (1 - value), 1e-12) / n)
        # 5 standard errors: deterministic seed, essentially no false failures.
        assert abs(r.probabilities[key] - value) < 5 * se + 1e-9, key
    assert abs(r.mean_home_goals - lh) < 0.02
    assert abs(r.mean_away_goals - la) < 0.02


def test_intervals_shrink_with_more_simulations():
    small = simulate_match(1.5, 1.2, n_simulations=10_000, seed=SEED)
    big = simulate_match(1.5, 1.2, n_simulations=500_000, seed=SEED)
    w = lambda r: r.intervals["1"][1] - r.intervals["1"][0]
    assert w(big) < w(small) / 5


def test_halftime_split_is_bounded_by_fulltime():
    r = simulate_match(1.5, 1.2, n_simulations=100_000, seed=SEED)
    p = r.probabilities
    assert p["HT Over 0.5"] <= p["Over 0.5"]
    assert p["HT Over 1.5"] <= p["Over 1.5"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, "abc"])
def test_invalid_lambda_rejected(bad):
    with pytest.raises(ValueError):
        simulate_match(bad, 1.0, n_simulations=10_000)


@pytest.mark.parametrize("n", [9_999, 500_001, 0])
def test_simulation_count_bounds(n):
    with pytest.raises(ValueError):
        simulate_match(1.2, 1.0, n_simulations=n)


def test_zero_lambda_is_valid_and_distinct_from_missing():
    r = simulate_match(0.0, 1.0, n_simulations=10_000, seed=SEED)
    assert r.probabilities["1"] == 0.0
    assert r.mean_home_goals == 0.0


def test_500k_simulations_are_fast():
    r = simulate_match(1.5, 1.2, n_simulations=500_000, seed=SEED)
    assert r.n_simulations == 500_000
    assert r.elapsed_seconds < 5.0
