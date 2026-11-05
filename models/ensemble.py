"""Ensemble of Poisson, Dixon-Coles and Elo via a weighted linear opinion pool.

    P_ens(outcome) = sum_m w_m * P_m(outcome),   w_m >= 0,  sum_m w_m = 1

Weights are learned from past *out-of-sample* predictions (e.g. the
leakage-free backtest) by minimising log loss, with a penalty that pulls the
weights toward equal weights when few matches are available:

    loss(w) = mean_i[-log P_ens(y_i)] + (ENSEMBLE_PRIOR_MATCHES / n) * sum_m (w_m - 1/M)^2

Two separate weight sets are learned:
  * 1X2 weights   -> applied to 1, X, 2 (and the derived 1X, X2, 12)
  * goals weights -> learned on Over 2.5 outcomes, applied to Over/Under 2.5 and GG/NG
    (applying them to GG/NG is a modelling assumption: GG outcomes are not backtested yet).

If fewer than ENSEMBLE_MIN_MATCHES matches are available, equal weights are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

import config

ENSEMBLE_MODELS = ("Poisson", "Dixon-Coles", "Elo")
EPS = 1e-12


@dataclass(frozen=True)
class EnsembleWeights:
    w_1x2: dict[str, float]
    w_goals: dict[str, float]
    n_matches: int
    fitted: bool          # False -> equal weights (not enough data)

    @staticmethod
    def equal(models=ENSEMBLE_MODELS, n_matches: int = 0) -> "EnsembleWeights":
        eq = {m: 1.0 / len(models) for m in models}
        return EnsembleWeights(dict(eq), dict(eq), n_matches, False)


def _optimise(loss_terms, n_models: int, n: int) -> np.ndarray:
    """Minimise mean log loss + shrinkage over the probability simplex."""
    eq = np.full(n_models, 1.0 / n_models)
    lam = config.ENSEMBLE_PRIOR_MATCHES / max(n, 1)

    def objective(w):
        return loss_terms(w) + lam * float(np.sum((w - eq) ** 2))

    res = minimize(objective, eq, method="SLSQP", bounds=[(0.0, 1.0)] * n_models,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"ftol": 1e-10, "maxiter": 200})
    w = np.clip(res.x if res.success else eq, 0.0, None)
    return w / w.sum()


def fit_weights(probs_1x2: dict[str, np.ndarray], outcomes: np.ndarray,
                probs_over: dict[str, np.ndarray] | None = None,
                over_outcomes: np.ndarray | None = None) -> EnsembleWeights:
    """Learn ensemble weights from past out-of-sample predictions.

    probs_1x2[m]: (n, 3) array of [P(1), P(X), P(2)] for model m.
    outcomes:     (n,) ints, 0 = home win, 1 = draw, 2 = away win.
    probs_over[m]: (n,) P(Over 2.5); over_outcomes: (n,) 0/1.
    """
    models = list(probs_1x2)
    if not models:
        raise ValueError("Δεν δόθηκαν μοντέλα για το ensemble.")
    y = np.asarray(outcomes, dtype=int)
    n = len(y)
    if n < config.ENSEMBLE_MIN_MATCHES:
        return EnsembleWeights.equal(models, n)

    P = np.stack([np.asarray(probs_1x2[m], dtype=float) for m in models])   # (M, n, 3)
    p_true = P[:, np.arange(n), y]                                           # (M, n)
    w1 = _optimise(lambda w: float(np.mean(-np.log(np.clip(w @ p_true, EPS, 1.0)))), len(models), n)

    if probs_over is not None and over_outcomes is not None:
        yo = np.asarray(over_outcomes, dtype=float)
        O = np.stack([np.asarray(probs_over[m], dtype=float) for m in models])  # (M, n)
        p_obs = np.where(yo[None, :] == 1, O, 1.0 - O)
        wg = _optimise(lambda w: float(np.mean(-np.log(np.clip(w @ p_obs, EPS, 1.0)))), len(models), n)
    else:
        wg = np.full(len(models), 1.0 / len(models))

    return EnsembleWeights(dict(zip(models, map(float, w1))), dict(zip(models, map(float, wg))), n, True)


def _renorm(weights: dict[str, float], available) -> dict[str, float]:
    sub = {m: weights.get(m, 0.0) for m in available}
    total = sum(sub.values())
    if total <= 0:
        return {m: 1.0 / len(sub) for m in sub}
    return {m: v / total for m, v in sub.items()}


def combine(model_probs: dict[str, dict[str, float]], weights: EnsembleWeights) -> dict:
    """Combine per-model market probabilities into ensemble probabilities.

    Models missing from ``model_probs`` are dropped and the remaining weights
    renormalised. Returns {"probabilities", "w_1x2", "w_goals"}.
    """
    available = [m for m in model_probs if m in weights.w_1x2]
    if not available:
        raise ValueError("Κανένα διαθέσιμο μοντέλο για το ensemble.")
    w1 = _renorm(weights.w_1x2, available)
    wg = _renorm(weights.w_goals, available)

    p = {k: sum(w1[m] * model_probs[m][k] for m in available) for k in ("1", "X", "2")}
    s = p["1"] + p["X"] + p["2"]
    p = {k: v / s for k, v in p.items()}
    p.update({"1X": p["1"] + p["X"], "X2": p["X"] + p["2"], "12": p["1"] + p["2"]})
    for key, complement in (("Over 2.5", "Under 2.5"), ("GG", "NG")):
        if all(key in model_probs[m] for m in available):
            v = sum(wg[m] * model_probs[m][key] for m in available)
            p.update({key: v, complement: 1.0 - v})
    return {"probabilities": p, "w_1x2": w1, "w_goals": wg}
