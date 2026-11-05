"""Tests for models/ensemble.py (weighted linear opinion pool)."""
from __future__ import annotations

import numpy as np
import pytest

import config
from models.ensemble import ENSEMBLE_MODELS, EnsembleWeights, combine, fit_weights


def _probs(p1, px, p2, over=0.5, gg=0.5):
    return {"1": p1, "X": px, "2": p2, "Over 2.5": over, "GG": gg}


MODEL_PROBS = {"Poisson": _probs(0.5, 0.3, 0.2, 0.55, 0.50),
               "Dixon-Coles": _probs(0.4, 0.3, 0.3, 0.45, 0.40),
               "Elo": _probs(0.6, 0.25, 0.15, 0.60, 0.55)}


def _synthetic(n=400, seed=0):
    """Outcomes drawn from the 'good' model; 'noisy' and 'flat' are worse forecasters."""
    rng = np.random.default_rng(seed)
    good = rng.dirichlet([4, 3, 3], size=n)
    y = np.array([rng.choice(3, p=p) for p in good])
    noisy = 0.5 * good + 0.5 * rng.dirichlet([1, 1, 1], size=n)
    flat = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    over_p = rng.uniform(0.3, 0.7, size=n)
    y_over = (rng.uniform(size=n) < over_p).astype(int)
    over = {"Poisson": over_p, "Dixon-Coles": np.full(n, 0.5), "Elo": np.clip(1 - over_p, 0.05, 0.95)}
    return {"Poisson": good, "Dixon-Coles": noisy, "Elo": flat}, y, over, y_over


def test_equal_weights_average():
    out = combine(MODEL_PROBS, EnsembleWeights.equal())
    p = out["probabilities"]
    assert p["1"] == pytest.approx(0.5)
    assert p["X"] == pytest.approx((0.3 + 0.3 + 0.25) / 3)
    assert p["Over 2.5"] == pytest.approx((0.55 + 0.45 + 0.60) / 3)
    assert p["1"] + p["X"] + p["2"] == pytest.approx(1.0)


def test_derived_markets_consistent():
    p = combine(MODEL_PROBS, EnsembleWeights.equal())["probabilities"]
    assert p["1X"] == pytest.approx(p["1"] + p["X"])
    assert p["X2"] == pytest.approx(p["X"] + p["2"])
    assert p["12"] == pytest.approx(p["1"] + p["2"])
    assert p["Under 2.5"] == pytest.approx(1 - p["Over 2.5"])
    assert p["NG"] == pytest.approx(1 - p["GG"])


def test_single_model_weight_reproduces_model():
    w = EnsembleWeights({"Poisson": 0.0, "Dixon-Coles": 0.0, "Elo": 1.0},
                        {"Poisson": 0.0, "Dixon-Coles": 0.0, "Elo": 1.0}, 50, True)
    p = combine(MODEL_PROBS, w)["probabilities"]
    for k in ("1", "X", "2", "Over 2.5", "GG"):
        assert p[k] == pytest.approx(MODEL_PROBS["Elo"][k])


def test_missing_model_renormalises_weights():
    probs = {k: v for k, v in MODEL_PROBS.items() if k != "Dixon-Coles"}
    w = EnsembleWeights({"Poisson": 0.2, "Dixon-Coles": 0.6, "Elo": 0.2},
                        {"Poisson": 0.2, "Dixon-Coles": 0.6, "Elo": 0.2}, 50, True)
    out = combine(probs, w)
    assert out["w_1x2"] == pytest.approx({"Poisson": 0.5, "Elo": 0.5})
    assert out["probabilities"]["1"] == pytest.approx(0.55)


def test_goal_markets_omitted_when_a_model_lacks_them():
    probs = {m: {"1": v["1"], "X": v["X"], "2": v["2"], "Over 2.5": v["Over 2.5"]} for m, v in MODEL_PROBS.items()}
    p = combine(probs, EnsembleWeights.equal())["probabilities"]
    assert "Over 2.5" in p and "GG" not in p


def test_combine_without_models_raises():
    with pytest.raises(ValueError):
        combine({}, EnsembleWeights.equal())


def test_too_few_matches_gives_equal_weights():
    probs, y, over, y_over = _synthetic(n=config.ENSEMBLE_MIN_MATCHES - 1)
    w = fit_weights(probs, y, over, y_over)
    assert not w.fitted
    assert all(v == pytest.approx(1 / 3) for v in w.w_1x2.values())


def test_weights_are_on_simplex():
    probs, y, over, y_over = _synthetic()
    w = fit_weights(probs, y, over, y_over)
    assert w.fitted and w.n_matches == len(y)
    for ws in (w.w_1x2, w.w_goals):
        assert sum(ws.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in ws.values())


def test_best_model_gets_largest_weight():
    probs, y, over, y_over = _synthetic()
    w = fit_weights(probs, y, over, y_over)
    assert max(w.w_1x2, key=w.w_1x2.get) == "Poisson"
    assert max(w.w_goals, key=w.w_goals.get) == "Poisson"


def test_fitted_weights_do_not_worsen_log_loss_vs_equal():
    probs, y, over, y_over = _synthetic()
    w = fit_weights(probs, y, over, y_over)
    P = np.stack([probs[m] for m in ENSEMBLE_MODELS])
    def ll(weights):
        mix = np.tensordot(np.array([weights[m] for m in ENSEMBLE_MODELS]), P, axes=1)
        return -np.mean(np.log(mix[np.arange(len(y)), y]))
    assert ll(w.w_1x2) <= ll(EnsembleWeights.equal().w_1x2) + 1e-9


def test_shrinkage_pulls_small_samples_toward_equal():
    small = fit_weights(*_synthetic(n=25, seed=3))
    large = fit_weights(*_synthetic(n=2000, seed=3))
    dist = lambda w: sum((v - 1 / 3) ** 2 for v in w.w_1x2.values())
    assert dist(small) < dist(large)


def test_fit_is_deterministic():
    data = _synthetic()
    assert fit_weights(*data) == fit_weights(*data)
