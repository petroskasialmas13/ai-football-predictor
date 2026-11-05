"""Vectorized Monte Carlo match simulator.

Given expected goals (lambda_home, lambda_away) produced by the Poisson model,
this module runs N independent *mathematical* simulations of a match with
NumPy (no network calls) and derives market probabilities from the simulated
scorelines, together with confidence intervals for every estimate.

Model assumptions (explicit, not measured facts):
- Home and away goals are independent Poisson variables.
- Each goal independently falls in the first half with probability
  ``first_half_share`` (binomial thinning). Under this assumption first-half
  goals are Poisson(lambda * share) and second-half goals Poisson(lambda * (1 - share)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
from scipy import stats

import config


@dataclass(frozen=True)
class MonteCarloResult:
    """All outputs of one simulation run. Probabilities are in [0, 1]."""

    n_simulations: int
    seed: int | None
    lambda_home: float
    lambda_away: float
    confidence: float
    first_half_share: float
    probabilities: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    top_scores: list[dict]
    score_matrix: np.ndarray            # P(home=i, away=j) for i, j in 0..heatmap_max
    score_matrix_other: float           # mass outside the displayed grid
    halftime_fulltime: dict[str, float]
    total_goals_distribution: dict[str, float]
    total_goals_summary: dict[str, float]
    mean_home_goals: float
    mean_away_goals: float
    elapsed_seconds: float
    extra: dict = field(default_factory=dict)


def _validate_lambda(name: str, value) -> float:
    # NULL is not 0: a missing expected-goals value must never be simulated as 0.
    if value is None:
        raise ValueError(f"Λείπει η τιμή {name} (αναμενόμενα γκολ).")
    try:
        lam = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Μη έγκυρη τιμή {name}: {value!r}") from exc
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"Η τιμή {name} πρέπει να είναι πεπερασμένος μη αρνητικός αριθμός (δόθηκε {value!r}).")
    if lam > 15:
        raise ValueError(f"Μη ρεαλιστική τιμή {name}={lam} (όριο 15).")
    return lam


def _z_value(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("Το επίπεδο εμπιστοσύνης πρέπει να είναι μεταξύ 0 και 1.")
    return float(stats.norm.ppf(0.5 + confidence / 2))


def _wilson(successes: np.ndarray | int, n: int, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval (stays inside [0, 1], robust near 0 and 1). Vectorized."""
    p = np.asarray(successes, dtype=float) / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def _outcome(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(home > away, 0, np.where(home == away, 1, 2)).astype(np.int8)


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    n_simulations: int = config.MC_DEFAULT_SIMULATIONS,
    seed: int | None = config.DEFAULT_RANDOM_SEED,
    confidence: float = config.MC_CONFIDENCE_LEVEL,
    first_half_share: float = config.MC_FIRST_HALF_GOAL_SHARE,
    top_n: int = config.MC_TOP_SCORES,
    heatmap_max_goals: int = config.MC_HEATMAP_MAX_GOALS,
    over_under_lines: tuple[float, ...] = config.OVER_UNDER_LINES,
) -> MonteCarloResult:
    """Run ``n_simulations`` independent match simulations.

    Pass a fixed ``seed`` for reproducible results (same inputs + same seed ->
    identical output). ``seed=None`` draws fresh OS entropy.
    """
    lh = _validate_lambda("lambda_home", lambda_home)
    la = _validate_lambda("lambda_away", lambda_away)
    n = int(n_simulations)
    if not config.MC_MIN_SIMULATIONS <= n <= config.MC_MAX_SIMULATIONS:
        raise ValueError(f"Οι προσομοιώσεις πρέπει να είναι από {config.MC_MIN_SIMULATIONS:,} "
                         f"έως {config.MC_MAX_SIMULATIONS:,} (δόθηκε {n:,}).")
    if not 0 < first_half_share < 1:
        raise ValueError("Το ποσοστό γκολ 1ου ημιχρόνου πρέπει να είναι μεταξύ 0 και 1.")
    z = _z_value(confidence)

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    # Draw order is fixed so a given seed always reproduces the same match set.
    home = rng.poisson(lh, n).astype(np.int32)
    away = rng.poisson(la, n).astype(np.int32)
    ht_home = rng.binomial(home, first_half_share).astype(np.int32)
    ht_away = rng.binomial(away, first_half_share).astype(np.int32)
    total = home + away
    ht_total = ht_home + ht_away
    sh_total = total - ht_total

    ft = _outcome(home, away)
    ht = _outcome(ht_home, ht_away)
    ft_counts = np.bincount(ft, minlength=3)
    ht_counts = np.bincount(ht, minlength=3)
    btts = int(np.count_nonzero((home > 0) & (away > 0)))

    counts: dict[str, int] = {
        "1": int(ft_counts[0]), "X": int(ft_counts[1]), "2": int(ft_counts[2]),
        "1X": int(ft_counts[0] + ft_counts[1]), "X2": int(ft_counts[1] + ft_counts[2]),
        "12": int(ft_counts[0] + ft_counts[2]),
    }
    for line in over_under_lines:
        over = int(np.count_nonzero(total > line))
        counts[f"Over {line}"] = over
        counts[f"Under {line}"] = n - over
    counts["GG"] = btts
    counts["NG"] = n - btts
    counts.update({"HT 1": int(ht_counts[0]), "HT X": int(ht_counts[1]), "HT 2": int(ht_counts[2])})
    for line in (0.5, 1.5):
        counts[f"HT Over {line}"] = int(np.count_nonzero(ht_total > line))
        counts[f"2H Over {line}"] = int(np.count_nonzero(sh_total > line))

    keys = list(counts)
    succ = np.array([counts[k] for k in keys])
    lo, hi = _wilson(succ, n, z)
    probabilities = {k: float(c / n) for k, c in zip(keys, succ)}
    intervals = {k: (float(a), float(b)) for k, a, b in zip(keys, lo, hi)}

    # Correct scores: encode (home, away) into one integer and count once.
    width = int(away.max()) + 1
    score_counts = np.bincount(home.astype(np.int64) * width + away)
    order = np.argsort(score_counts)[::-1][:top_n]
    s_lo, s_hi = _wilson(score_counts[order], n, z)
    top_scores = [
        {"score": f"{int(code // width)}-{int(code % width)}", "probability": float(score_counts[code] / n),
         "ci_low": float(a), "ci_high": float(b)}
        for code, a, b in zip(order, s_lo, s_hi) if score_counts[code] > 0
    ]

    g = heatmap_max_goals + 1
    in_grid = (home < g) & (away < g)
    grid = np.bincount(home[in_grid] * g + away[in_grid], minlength=g * g).reshape(g, g) / n
    other = float(1 - grid.sum())

    labels = ("1", "X", "2")
    htft_counts = np.bincount(ht.astype(np.int16) * 3 + ft, minlength=9)
    halftime_fulltime = {f"{labels[i // 3]}/{labels[i % 3]}": float(c / n) for i, c in enumerate(htft_counts)}

    tg_max = 7
    tg_counts = np.bincount(np.minimum(total, tg_max), minlength=tg_max + 1)
    total_goals_distribution = {(str(k) if k < tg_max else f"{tg_max}+"): float(c / n) for k, c in enumerate(tg_counts)}
    mean_total = float(total.mean())
    std_total = float(total.std(ddof=1))
    half_mean = z * std_total / math.sqrt(n)
    total_goals_summary = {
        "mean": mean_total, "std": std_total,
        "mean_ci_low": mean_total - half_mean, "mean_ci_high": mean_total + half_mean,
        "median": float(np.median(total)),
        "p05": float(np.percentile(total, 5)), "p95": float(np.percentile(total, 95)),
    }
    elapsed = time.perf_counter() - started

    return MonteCarloResult(
        n_simulations=n, seed=seed, lambda_home=lh, lambda_away=la, confidence=confidence,
        first_half_share=first_half_share, probabilities=probabilities, intervals=intervals,
        top_scores=top_scores, score_matrix=grid, score_matrix_other=other,
        halftime_fulltime=halftime_fulltime, total_goals_distribution=total_goals_distribution,
        total_goals_summary=total_goals_summary, mean_home_goals=float(home.mean()),
        mean_away_goals=float(away.mean()), elapsed_seconds=elapsed,
    )


def analytical_probabilities(
    lambda_home: float,
    lambda_away: float,
    first_half_share: float = config.MC_FIRST_HALF_GOAL_SHARE,
    max_goals: int = 20,
    over_under_lines: tuple[float, ...] = config.OVER_UNDER_LINES,
) -> dict[str, float]:
    """Exact (closed-form) probabilities under the same assumptions as the simulator.

    Used to validate the simulation and to show "analytical vs simulated" in the UI.
    """
    lh = _validate_lambda("lambda_home", lambda_home)
    la = _validate_lambda("lambda_away", lambda_away)

    def grid(a: float, b: float) -> np.ndarray:
        goals = np.arange(max_goals + 1)
        m = np.outer(stats.poisson.pmf(goals, a), stats.poisson.pmf(goals, b))
        return m / m.sum()

    def one_x_two(m: np.ndarray) -> tuple[float, float, float]:
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())

    ft = grid(lh, la)
    hw, dr, aw = one_x_two(ft)
    i, j = np.indices(ft.shape)
    totals = i + j
    out = {"1": hw, "X": dr, "2": aw, "1X": hw + dr, "X2": dr + aw, "12": hw + aw}
    for line in over_under_lines:
        over = float(ft[totals > line].sum())
        out[f"Over {line}"] = over
        out[f"Under {line}"] = 1 - over
    out["GG"] = float(ft[1:, 1:].sum())
    out["NG"] = 1 - out["GG"]

    s = first_half_share
    ht = grid(lh * s, la * s)
    out["HT 1"], out["HT X"], out["HT 2"] = one_x_two(ht)
    ht_lam, sh_lam = (lh + la) * s, (lh + la) * (1 - s)
    for line in (0.5, 1.5):
        k = int(line)
        out[f"HT Over {line}"] = float(stats.poisson.sf(k, ht_lam))
        out[f"2H Over {line}"] = float(stats.poisson.sf(k, sh_lam))
    return out
