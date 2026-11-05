"""Dixon-Coles (1997) bivariate goals model with time-decay weighting.

Reference
---------
Dixon, M. J. & Coles, S. G. (1997). "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market". Journal of the Royal
Statistical Society, Series C (Applied Statistics), 46(2), 265-280.

Model
-----
For a match where team ``i`` hosts team ``j`` with score (x, y):

    lambda = exp(c + h + att_i + def_j)        expected home goals
    mu     = exp(c      + att_j + def_i)        expected away goals

    P(X=x, Y=y) = tau(x, y; lambda, mu, rho) * Pois(x; lambda) * Pois(y; mu)

with the low-score dependence correction (paper, eq. 4.2):

    tau(0, 0) = 1 - lambda * mu * rho
    tau(0, 1) = 1 + lambda * rho
    tau(1, 0) = 1 + mu * rho
    tau(1, 1) = 1 - rho
    tau(x, y) = 1                     otherwise

``x`` is always the HOME score and ``y`` the AWAY score. ``tau`` leaves the
total probability mass unchanged (the four corrections sum to zero exactly),
and ``rho = 0`` reduces the model to two independent Poisson variables.

This is the paper's parametrisation (alpha_i = exp(att_i), beta_j = exp(def_j),
gamma = exp(h)) written on the log scale with an explicit intercept ``c``.
Identifiability: sum(att) = 0 and sum(def) = 0, enforced by construction
(the last team's value is minus the sum of the others), so ``exp(c)`` is the
expected away goals of an average team against an average team and
``exp(c + h)`` the corresponding home value.

Estimation
----------
Parameters maximise the time-weighted pseudo log-likelihood (paper, eq. 4.5)

    sum_k  phi(t_k) * log[ tau * Pois(x_k; lambda_k) * Pois(y_k; mu_k) ]
    phi(t) = exp(-xi * t),   t = age of match k in days at the reference time

minus an optional L2 penalty ``l2 * (sum att^2 + sum def^2)``. The penalty
is a Gaussian prior centred on the league average (att = def = 0); it pulls
teams with few (or only old, heavily down-weighted) matches toward the average
and keeps the problem well-posed when some teams never met each other. It is a
modelling choice, not part of the original paper; set ``l2=0`` for plain MLE.

The analytical gradient is supplied to L-BFGS-B, so fitting a 20-team league
takes well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

import config
from utils.data import prepare_history

# --- Defaults (modelling assumptions — calibrate with backtests) ------------
# Dixon & Coles (1997) selected xi = 0.0065 per half-week for English league
# data, i.e. 0.0065 / 3.5 ≈ 0.0019 per day (match weight halves after ~1 year).
DEFAULT_XI_PER_DAY = 0.0065 / 3.5
# Gaussian prior on att/def: penalty l2 * x^2  <=>  prior sd = 1 / sqrt(2 * l2).
# l2 = 1.0 -> sd ≈ 0.71 on the log scale (weak prior). Not a measured value.
DEFAULT_L2_PENALTY = 1.0
# Optimiser bounds. rho is additionally kept feasible (tau > 0) per match.
RHO_BOUNDS = (-0.5, 0.5)
LOG_PARAM_BOUND = 4.0          # |att|, |def|, |c|, |h| <= 4 on the log scale
TAU_FLOOR = 1e-10
MAX_ITER = 1000


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score correction factor tau (vectorised).

    ``home_goals``/``away_goals`` are the scores (x, y); ``lam``/``mu`` the
    expected home/away goals. Returns 1 for every score outside {0,1}x{0,1}.
    """
    x = np.asarray(home_goals)
    y = np.asarray(away_goals)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    shape = np.broadcast(x, y, lam, mu).shape
    out = np.ones(shape, dtype=float)
    x, y, lam, mu = (np.broadcast_to(a, shape) for a in (x, y, lam, mu))
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out if out.shape else float(out)


def rho_feasible_range(lam: float, mu: float) -> tuple[float, float]:
    """Range of rho for which all four tau values are non-negative (paper, sec. 4).

    max(-1/lambda, -1/mu) <= rho <= min(1/(lambda*mu), 1)
    """
    lower = max(-1.0 / lam if lam > 0 else -math.inf, -1.0 / mu if mu > 0 else -math.inf)
    upper = min(1.0 / (lam * mu) if lam * mu > 0 else math.inf, 1.0)
    return lower, upper


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = config.POISSON_MAX_GOALS) -> np.ndarray:
    """P(home=i, away=j) for i, j in 0..max_goals with the tau correction applied.

    The grid is renormalised to sum to 1 (the truncated tail beyond
    ``max_goals`` is negligible for realistic expected goals).
    """
    goals = np.arange(max_goals + 1)
    m = np.outer(stats.poisson.pmf(goals, lam), stats.poisson.pmf(goals, mu))
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    if (m < 0).any():
        raise ValueError(f"Μη έγκυρο rho={rho:.4f} για λ={lam:.3f}, μ={mu:.3f} (αρνητική πιθανότητα).")
    return m / m.sum()


def matrix_probabilities(matrix: np.ndarray,
                         over_under_lines: tuple[float, ...] = config.OVER_UNDER_LINES) -> dict[str, float]:
    """Market probabilities from a score matrix (same keys as engine.analyze)."""
    hw = float(np.tril(matrix, -1).sum())
    dr = float(np.trace(matrix))
    aw = float(np.triu(matrix, 1).sum())
    i, j = np.indices(matrix.shape)
    totals = i + j
    out = {"1": hw, "X": dr, "2": aw, "1X": hw + dr, "X2": dr + aw, "12": hw + aw}
    for line in over_under_lines:
        over = float(matrix[totals > line].sum())
        out[f"Over {line}"] = over
        out[f"Under {line}"] = 1.0 - over
    out["GG"] = float(matrix[1:, 1:].sum())
    out["NG"] = 1.0 - out["GG"]
    return out


@dataclass(frozen=True)
class DixonColesPrediction:
    home: str
    away: str
    lambda_home: float
    lambda_away: float
    rho: float                       # rho actually used (may be clipped, see rho_clipped)
    rho_clipped: bool
    matrix: np.ndarray               # P(home=i, away=j), i, j in 0..max_goals, sums to 1
    probabilities: dict[str, float]
    top_scores: list[tuple[str, float]]
    unknown_teams: tuple[str, ...]   # teams predicted as league-average (unknown_team="average")


@dataclass
class FitSummary:
    converged: bool
    message: str
    iterations: int
    log_likelihood: float            # weighted log-likelihood incl. constants, excl. penalty
    objective: float                 # minimised value (negative LL + penalty) / total weight
    n_matches: int
    total_weight: float
    reference_time: pd.Timestamp
    first_match: pd.Timestamp
    last_match: pd.Timestamp
    xi: float
    l2: float
    matches_per_team: dict[str, int] = field(default_factory=dict)


class DixonColesModel:
    """Dixon-Coles model with exponential time decay and optional L2 shrinkage.

    Usage::

        model = DixonColesModel(xi=0.0019, l2=1.0).fit(history, reference_time=kickoff)
        pred = model.predict("Team A", "Team B")
        pred.lambda_home, pred.lambda_away, pred.matrix, pred.probabilities

    ``fit`` uses only matches strictly before ``reference_time`` (no leakage).
    """

    def __init__(self, xi: float = DEFAULT_XI_PER_DAY, l2: float = DEFAULT_L2_PENALTY,
                 min_matches: int = config.MIN_MATCHES_TOTAL,
                 max_goals: int = config.POISSON_MAX_GOALS):
        if not (isinstance(xi, (int, float)) and math.isfinite(xi) and xi >= 0):
            raise ValueError(f"Το ξ (χρονική απόσβεση) πρέπει να είναι μη αρνητικός αριθμός (δόθηκε {xi!r}).")
        if not (isinstance(l2, (int, float)) and math.isfinite(l2) and l2 >= 0):
            raise ValueError(f"Η ποινή L2 πρέπει να είναι μη αρνητικός αριθμός (δόθηκε {l2!r}).")
        if int(min_matches) < 1:
            raise ValueError("Το min_matches πρέπει να είναι τουλάχιστον 1.")
        self.xi = float(xi)
        self.l2 = float(l2)
        self.min_matches = int(min_matches)
        self.max_goals = int(max_goals)
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self.teams: list[str] = []
        self.team_index: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.intercept: float | None = None
        self.home_advantage: float | None = None
        self.rho: float | None = None
        self.summary: FitSummary | None = None

    @property
    def is_fitted(self) -> bool:
        return self.attack is not None

    # ---------------------------------------------------------- parameterisation
    def _unpack(self, theta: np.ndarray):
        n = len(self.teams)
        c, h, rho = theta[0], theta[1], theta[2]
        att_free = theta[3:3 + n - 1]
        def_free = theta[3 + n - 1:]
        att = np.append(att_free, -att_free.sum())
        dfc = np.append(def_free, -def_free.sum())
        return c, h, rho, att, dfc

    @staticmethod
    def _reduce_grad(full: np.ndarray) -> np.ndarray:
        # d/d(free_k) of f(a) with a_last = -sum(free): grad_k - grad_last
        return full[:-1] - full[-1]

    # --------------------------------------------------------------- objective
    def _objective(self, theta, hi, ai, x, y, w, wsum, log_fact):
        c, h, rho, att, dfc = self._unpack(theta)
        log_lam = c + h + att[hi] + dfc[ai]
        log_mu = c + att[ai] + dfc[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        t = np.ones_like(lam)
        t[m00] = 1.0 - lam[m00] * mu[m00] * rho
        t[m01] = 1.0 + lam[m01] * rho
        t[m10] = 1.0 + mu[m10] * rho
        t[m11] = 1.0 - rho
        t_safe = np.maximum(t, TAU_FLOOR)   # infeasible rho -> very low likelihood, never NaN

        ll = np.log(t_safe) + x * log_lam - lam + y * log_mu - mu - log_fact
        penalty = self.l2 * (att @ att + dfc @ dfc)
        f = (-(w @ ll) + penalty) / wsum

        # d log tau / d lambda, d mu, d rho
        dl = np.zeros_like(lam)
        dm = np.zeros_like(lam)
        dr = np.zeros_like(lam)
        dl[m00] = -mu[m00] * rho / t_safe[m00]
        dm[m00] = -lam[m00] * rho / t_safe[m00]
        dr[m00] = -lam[m00] * mu[m00] / t_safe[m00]
        dl[m01] = rho / t_safe[m01]
        dr[m01] = lam[m01] / t_safe[m01]
        dm[m10] = rho / t_safe[m10]
        dr[m10] = mu[m10] / t_safe[m10]
        dr[m11] = -1.0 / t_safe[m11]

        # derivatives of the per-match log-likelihood w.r.t. log(lambda), log(mu)
        g_lam = w * (x - lam + lam * dl)
        g_mu = w * (y - mu + mu * dm)
        n = len(self.teams)
        grad_att = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
        grad_def = np.bincount(ai, g_lam, n) + np.bincount(hi, g_mu, n)
        grad_att = -grad_att + 2 * self.l2 * att
        grad_def = -grad_def + 2 * self.l2 * dfc
        grad = np.concatenate((
            [-(g_lam.sum() + g_mu.sum()), -g_lam.sum(), -(w @ dr)],
            self._reduce_grad(grad_att), self._reduce_grad(grad_def),
        )) / wsum
        return f, grad

    # --------------------------------------------------------------------- fit
    def fit(self, matches: pd.DataFrame, reference_time=None) -> "DixonColesModel":
        """Estimate all parameters by (penalised) weighted maximum likelihood.

        ``matches`` needs the columns date, home_team, away_team, home_goals,
        away_goals (as in ``utils.data.prepare_history``). Rows with missing
        values are dropped, never treated as 0.

        ``reference_time``: only matches strictly before it are used and match
        ages are measured from it. Defaults to the latest match date.
        """
        self._reset()
        df = prepare_history(matches)
        if reference_time is not None:
            ref = pd.Timestamp(reference_time)
            ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
            df = df[df.date < ref].reset_index(drop=True)   # strict boundary: no future data
        else:
            ref = df.date.max() if len(df) else pd.Timestamp.now(tz="UTC")
        if len(df) < self.min_matches:
            raise ValueError(f"Μόνο {len(df)} ολοκληρωμένοι αγώνες πριν από την ημερομηνία αναφοράς. "
                             f"Το Dixon-Coles χρειάζεται τουλάχιστον {self.min_matches}.")

        teams = sorted(set(df.home_team) | set(df.away_team))
        if len(teams) < 2:
            raise ValueError("Χρειάζονται τουλάχιστον δύο ομάδες.")
        self.teams = teams
        self.team_index = {t: k for k, t in enumerate(teams)}
        hi = df.home_team.map(self.team_index).to_numpy()
        ai = df.away_team.map(self.team_index).to_numpy()
        x = df.home_goals.to_numpy(dtype=float)
        y = df.away_goals.to_numpy(dtype=float)
        age_days = ((ref - df.date).dt.total_seconds() / 86400.0).to_numpy()
        w = np.exp(-self.xi * age_days)
        wsum = float(w.sum())
        log_fact = np.array([math.lgamma(v + 1) for v in x]) + np.array([math.lgamma(v + 1) for v in y])

        # Start: league-average goal rates, rho = 0, all teams average.
        mean_home = max(float(np.average(x, weights=w)), 0.05)
        mean_away = max(float(np.average(y, weights=w)), 0.05)
        n = len(teams)
        theta0 = np.zeros(3 + 2 * (n - 1))
        theta0[0] = math.log(mean_away)
        theta0[1] = math.log(mean_home / mean_away)
        bounds = ([(-LOG_PARAM_BOUND, LOG_PARAM_BOUND)] * 2 + [RHO_BOUNDS]
                  + [(-LOG_PARAM_BOUND, LOG_PARAM_BOUND)] * (2 * (n - 1)))

        res = optimize.minimize(self._objective, theta0, args=(hi, ai, x, y, w, wsum, log_fact),
                                jac=True, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8})
        if not res.success:
            warnings.warn(f"Dixon-Coles: ο βελτιστοποιητής δεν συνέκλινε πλήρως ({res.message}).", RuntimeWarning)

        c, h, rho, att, dfc = self._unpack(res.x)
        self.intercept, self.home_advantage, self.rho = float(c), float(h), float(rho)
        self.attack, self.defence = att, dfc
        penalty = self.l2 * (att @ att + dfc @ dfc)
        counts = pd.concat([df.home_team, df.away_team]).value_counts()
        self.summary = FitSummary(
            converged=bool(res.success), message=str(res.message), iterations=int(res.nit),
            log_likelihood=float(-(res.fun * wsum - penalty)), objective=float(res.fun),
            n_matches=int(len(df)), total_weight=wsum, reference_time=ref,
            first_match=df.date.min(), last_match=df.date.max(), xi=self.xi, l2=self.l2,
            matches_per_team={t: int(counts.get(t, 0)) for t in teams},
        )
        return self

    # ----------------------------------------------------------------- predict
    def _team_params(self, team: str, unknown_team: str) -> tuple[float, float, bool]:
        idx = self.team_index.get(team)
        if idx is not None:
            return float(self.attack[idx]), float(self.defence[idx]), False
        if unknown_team == "average":
            return 0.0, 0.0, True
        raise ValueError(f"Η ομάδα «{team}» δεν υπάρχει στα δεδομένα εκπαίδευσης του Dixon-Coles.")

    def expected_goals(self, home_team: str, away_team: str, unknown_team: str = "error") -> tuple[float, float]:
        """(lambda_home, lambda_away) — directly usable by models.monte_carlo.simulate_match."""
        pred = self.predict(home_team, away_team, unknown_team=unknown_team)
        return pred.lambda_home, pred.lambda_away

    def predict(self, home_team: str, away_team: str, unknown_team: str = "error") -> DixonColesPrediction:
        """Expected goals, tau-corrected score matrix and market probabilities.

        ``unknown_team``: "error" (default) raises ValueError for a team absent
        from the training data; "average" predicts it as a league-average team
        and lists it in ``unknown_teams``.
        """
        if not self.is_fitted:
            raise RuntimeError("Το μοντέλο δεν έχει εκπαιδευτεί· κάλεσε πρώτα fit().")
        if unknown_team not in ("error", "average"):
            raise ValueError("unknown_team πρέπει να είναι 'error' ή 'average'.")
        if home_team == away_team:
            raise ValueError("Πρέπει να επιλέξεις δύο διαφορετικές ομάδες.")
        att_h, def_h, unk_h = self._team_params(home_team, unknown_team)
        att_a, def_a, unk_a = self._team_params(away_team, unknown_team)
        lam = math.exp(self.intercept + self.home_advantage + att_h + def_a)
        mu = math.exp(self.intercept + att_a + def_h)

        lo, hi = rho_feasible_range(lam, mu)
        rho = min(max(self.rho, lo), hi)
        matrix = score_matrix(lam, mu, rho, self.max_goals)
        size = matrix.shape[1]
        flat = np.argsort(matrix.ravel())[::-1][:5]
        top = [(f"{int(k) // size}-{int(k) % size}", float(matrix.ravel()[k])) for k in flat]
        return DixonColesPrediction(
            home=home_team, away=away_team, lambda_home=lam, lambda_away=mu, rho=rho,
            rho_clipped=rho != self.rho, matrix=matrix, probabilities=matrix_probabilities(matrix),
            top_scores=top,
            unknown_teams=tuple(t for t, u in ((home_team, unk_h), (away_team, unk_a)) if u),
        )

    # --------------------------------------------------------------- reporting
    def team_ratings(self) -> pd.DataFrame:
        """Per-team multiplicative strengths (1.0 = league average).

        attack > 1: scores more than average; defence > 1: concedes more.
        """
        if not self.is_fitted:
            raise RuntimeError("Το μοντέλο δεν έχει εκπαιδευτεί· κάλεσε πρώτα fit().")
        frame = pd.DataFrame({
            "team": self.teams,
            "attack": np.exp(self.attack),
            "defence": np.exp(self.defence),
            "matches": [self.summary.matches_per_team[t] for t in self.teams],
        })
        return frame.sort_values("attack", ascending=False).reset_index(drop=True)

    def params(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Το μοντέλο δεν έχει εκπαιδευτεί· κάλεσε πρώτα fit().")
        return {"intercept": self.intercept, "home_advantage": self.home_advantage, "rho": self.rho,
                "attack": dict(zip(self.teams, self.attack.tolist())),
                "defence": dict(zip(self.teams, self.defence.tolist()))}
