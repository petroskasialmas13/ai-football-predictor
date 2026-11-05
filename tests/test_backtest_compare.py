"""Tests for backtesting/compare.py and Monte Carlo sampling from a score matrix.

All data is synthetic, fixed-seed and contains no real statistics.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtesting.compare import (
    _blocks, accuracy, binary_brier, compare_models, log_loss, multiclass_brier, outcome_index,
)
from models.dixon_coles import DixonColesModel, matrix_probabilities, score_matrix
from models.monte_carlo import simulate_match

TEAMS = [f"Team {c}" for c in "ABCDEFGH"]
START = pd.Timestamp("2021-08-07 15:00", tz="UTC")


def synthetic_league(seasons: int = 2, seed: int = 7) -> pd.DataFrame:
    """Double round robin per season, one round per week, fixed-seed Poisson scores."""
    rng = np.random.default_rng(seed)
    strength = np.linspace(0.35, -0.35, len(TEAMS))
    rows, week = [], 0
    for _ in range(seasons):
        pairs = [(i, j) for i in range(len(TEAMS)) for j in range(len(TEAMS)) if i != j]
        rng.shuffle(pairs)
        for k in range(0, len(pairs), 4):
            day = START + pd.Timedelta(weeks=week)
            for slot, (i, j) in enumerate(pairs[k:k + 4]):
                lam = math.exp(0.25 + 0.2 + strength[i] - strength[j] * 0.8)
                mu = math.exp(0.25 + strength[j] - strength[i] * 0.8)
                rows.append({"date": day + pd.Timedelta(hours=slot), "home_team": TEAMS[i],
                             "away_team": TEAMS[j], "home_goals": int(rng.poisson(lam)),
                             "away_goals": int(rng.poisson(mu))})
            week += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    return synthetic_league()


@pytest.fixture(scope="module")
def comparison(league):
    return compare_models(league, n_matches=40, refit_days=7)


# ---------------------------------------------------------------- metrics
def test_outcome_index():
    assert outcome_index(2, 1) == 0 and outcome_index(1, 1) == 1 and outcome_index(0, 3) == 2


def test_metrics_hand_computed():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    y = np.array([0, 1])
    # Brier: (0.25+0.09+0.04) + (0.04+0.64+0.36) = 0.38 + 1.04 -> mean 0.71
    assert multiclass_brier(p, y) == pytest.approx(0.71)
    assert log_loss(p, y) == pytest.approx((-math.log(0.5) - math.log(0.2)) / 2)
    assert accuracy(p, y) == pytest.approx(0.5)
    assert binary_brier([0.7, 0.4], [1, 0]) == pytest.approx((0.09 + 0.16) / 2)


def test_log_loss_clips_zero_probability():
    p = np.array([[1.0, 0.0, 0.0]])
    assert math.isfinite(log_loss(p, np.array([1])))
    assert log_loss(p, np.array([1])) == pytest.approx(-math.log(1e-15))


def test_perfect_forecast_scores_zero():
    p = np.eye(3)
    y = np.array([0, 1, 2])
    assert multiclass_brier(p, y) == 0.0 and accuracy(p, y) == 1.0


# ---------------------------------------------------------------- blocks
def test_blocks_respect_window():
    dates = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-03", "2022-01-08", "2022-01-09", "2022-01-20"], utc=True))
    assert _blocks(dates, 7) == [[0, 1], [2, 3], [4]]
    assert _blocks(dates, 1) == [[0], [1], [2], [3], [4]]


# ---------------------------------------------------------------- comparison
def test_comparison_same_match_set(comparison):
    m = comparison.metrics
    assert list(m["Μοντέλο"]) == ["Poisson", "Dixon-Coles", "Elo", "Βάση"]
    assert (m["Αγώνες"] == comparison.evaluated).all()
    assert comparison.evaluated + comparison.skipped == 40
    for name in ("Poisson", "Dixon-Coles", "Elo", "Βάση"):
        probs = np.vstack(comparison.predictions[name].to_numpy())
        assert probs.shape == (comparison.evaluated, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_comparison_metrics_in_valid_ranges(comparison):
    m = comparison.metrics
    assert m["Ακρίβεια 1Χ2"].between(0, 1).all()
    assert m["Brier 1Χ2"].between(0, 2).all()
    assert (m["Log loss 1Χ2"] > 0).all()
    assert m["Brier Over 2.5"].between(0, 1).all()
    assert comparison.dc_refits >= 1


def test_comparison_is_deterministic(league, comparison):
    again = compare_models(league, n_matches=40, refit_days=7)
    pd.testing.assert_frame_equal(again.metrics, comparison.metrics)


def test_no_leakage_future_results_do_not_change_predictions(league):
    """Changing results on/after a cutoff must not change predictions for matches before it,
    nor the predictions for the first evaluated match on the cutoff date itself."""
    base = compare_models(league, n_matches=30, refit_days=1)
    preds = base.predictions
    target = preds.iloc[10]
    tampered = league.copy()
    future = pd.to_datetime(tampered["date"], utc=True) >= target["date"]
    tampered.loc[future, "home_goals"] = 9
    tampered.loc[future, "away_goals"] = 0
    after = compare_models(tampered, n_matches=30, refit_days=1).predictions
    merged = preds.merge(after, on=["date", "home_team", "away_team"], suffixes=("", "_t"))
    upto = merged[merged["date"] <= target["date"]]
    assert len(upto) >= 11
    for name in ("Poisson", "Dixon-Coles", "Βάση"):
        a = np.vstack(upto[name].to_numpy())
        b = np.vstack(upto[f"{name}_t"].to_numpy())
        assert np.allclose(a, b, atol=1e-12), name


def test_refit_block_never_uses_block_dates(league):
    """A block-wise fit equals a fit on data strictly before the block start (same params)."""
    df = league.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    cutoff = df["date"].sort_values().iloc[80]
    m1 = DixonColesModel().fit(df, reference_time=cutoff)
    m2 = DixonColesModel().fit(df[df["date"] < cutoff], reference_time=cutoff)
    assert m1.rho == pytest.approx(m2.rho) and m1.home_advantage == pytest.approx(m2.home_advantage)


def test_too_little_history_raises(league):
    with pytest.raises(ValueError):
        compare_models(league.head(30), n_matches=10)


def test_invalid_arguments(league):
    with pytest.raises(ValueError):
        compare_models(league, refit_days=0)
    with pytest.raises(ValueError):
        compare_models(league, n_matches=0)


# ---------------------------------------------------------------- MC from matrix
def test_mc_matrix_sampling_matches_dixon_coles_matrix():
    m = score_matrix(1.4, 1.1, -0.12)
    exact = matrix_probabilities(m)
    mc = simulate_match(1.4, 1.1, n_simulations=400_000, seed=3, score_matrix=m)
    assert mc.extra["sampling"] == "score_matrix"
    for k in ("1", "X", "2", "Over 2.5", "GG"):
        lo, hi = mc.intervals[k]
        assert abs(mc.probabilities[k] - exact[k]) < 0.004, k
    # tau correction is preserved: simulated 0-0 share matches the corrected matrix
    assert mc.score_matrix[0, 0] == pytest.approx(m[0, 0], abs=0.002)


def test_mc_matrix_sampling_reproducible():
    m = score_matrix(1.2, 0.9, -0.08)
    a = simulate_match(1.2, 0.9, n_simulations=20_000, seed=11, score_matrix=m)
    b = simulate_match(1.2, 0.9, n_simulations=20_000, seed=11, score_matrix=m)
    assert a.probabilities == b.probabilities


def test_mc_matrix_validation():
    with pytest.raises(ValueError):
        simulate_match(1.2, 0.9, n_simulations=20_000, score_matrix=np.full((3, 3), 0.2))
    with pytest.raises(ValueError):
        simulate_match(1.2, 0.9, n_simulations=20_000, score_matrix=np.array([[0.5, -0.1], [0.3, 0.3]]))


def test_mc_default_path_unchanged():
    mc = simulate_match(1.5, 1.0, n_simulations=10_000, seed=42)
    assert mc.extra["sampling"] == "poisson"
