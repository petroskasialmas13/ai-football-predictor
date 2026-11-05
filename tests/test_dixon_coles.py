"""Tests for models/dixon_coles.py.

All data is synthetic and reproducible: fixed seeds and fixed calendar dates.
Scores are sampled from the Dixon-Coles distribution itself with known
parameters, so parameter recovery can be checked against the truth.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from engine import generate_demo
from models.dixon_coles import (
    DixonColesModel,
    matrix_probabilities,
    rho_feasible_range,
    score_matrix,
    tau,
)
from models.monte_carlo import simulate_match
from utils.data import prepare_history

START = pd.Timestamp("2020-01-04", tz="UTC")
TEAMS = [f"Team {c}" for c in "ABCDEFGHIJ"]
TRUE_ATT = np.array([0.45, 0.30, 0.20, 0.10, 0.05, -0.05, -0.10, -0.20, -0.30, -0.45])
TRUE_DEF = np.array([-0.40, -0.25, -0.15, -0.05, 0.00, 0.05, 0.10, 0.15, 0.25, 0.30])
TRUE_C = math.log(1.10)
TRUE_H = 0.25
TRUE_RHO = -0.10


def synthetic_league(seasons: int = 4, seed: int = 7, att=TRUE_ATT, dfc=TRUE_DEF,
                     c=TRUE_C, h=TRUE_H, rho=TRUE_RHO, teams=TEAMS, start=START) -> pd.DataFrame:
    """Double round-robins; one match per day; scores drawn from the DC distribution."""
    rng = np.random.default_rng(seed)
    n = len(teams)
    rows, day = [], 0
    for _ in range(seasons):
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        rng.shuffle(pairs)
        for i, j in pairs:
            lam = math.exp(c + h + att[i] + dfc[j])
            mu = math.exp(c + att[j] + dfc[i])
            m = score_matrix(lam, mu, rho, max_goals=10)
            k = rng.choice(m.size, p=m.ravel())
            rows.append({"date": start + pd.Timedelta(days=day), "home_team": teams[i],
                         "away_team": teams[j], "home_goals": int(k // m.shape[1]),
                         "away_goals": int(k % m.shape[1])})
            day += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    return synthetic_league()


@pytest.fixture(scope="module")
def fitted(league) -> DixonColesModel:
    return DixonColesModel(xi=0.0, l2=0.0).fit(league)


# ----------------------------------------------------------------- tau / matrix
class TestTau:
    def test_values_match_dixon_coles_1997(self):
        lam, mu, rho = 1.4, 0.9, -0.13
        assert tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
        assert tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
        assert tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
        assert tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)
        for x, y in [(2, 0), (0, 2), (2, 1), (1, 2), (3, 3), (5, 0)]:
            assert tau(x, y, lam, mu, rho) == 1.0

    def test_vectorised_matches_scalar(self):
        x = np.array([0, 0, 1, 1, 2])
        y = np.array([0, 1, 0, 1, 2])
        lam = np.array([1.2, 1.5, 0.8, 2.0, 1.1])
        mu = np.array([0.7, 1.0, 1.3, 0.9, 1.4])
        vec = tau(x, y, lam, mu, 0.08)
        assert vec.shape == (5,)
        assert np.allclose(vec, [tau(a, b, l, m, 0.08) for a, b, l, m in zip(x, y, lam, mu)])

    def test_rho_zero_is_identity(self):
        for x in range(3):
            for y in range(3):
                assert tau(x, y, 1.7, 0.6, 0.0) == 1.0

    @pytest.mark.parametrize("lam,mu,rho", [(1.5, 1.1, -0.1), (0.6, 2.3, 0.2), (2.8, 0.4, -0.3)])
    def test_correction_preserves_total_mass(self, lam, mu, rho):
        # Unnormalised tau * Pois * Pois must still sum to 1 (paper: corrections cancel).
        g = np.arange(40)
        m = np.outer(stats.poisson.pmf(g, lam), stats.poisson.pmf(g, mu))
        corrected = m * tau(g[:, None], g[None, :], lam, mu, rho)
        assert corrected.sum() == pytest.approx(m.sum(), abs=1e-14)
        assert corrected.sum() == pytest.approx(1.0, abs=1e-12)

    def test_feasible_range(self):
        lo, hi = rho_feasible_range(1.5, 0.8)
        assert lo == pytest.approx(max(-1 / 1.5, -1 / 0.8))
        assert hi == pytest.approx(min(1 / (1.5 * 0.8), 1.0))
        g = np.arange(2)
        for rho in (lo, hi):
            assert (tau(g[:, None], g[None, :], 1.5, 0.8, rho) >= -1e-12).all()


class TestScoreMatrix:
    def test_sums_to_one_and_non_negative(self):
        m = score_matrix(1.6, 1.1, -0.12)
        assert m.shape == (13, 13)
        assert m.sum() == pytest.approx(1.0, abs=1e-12)
        assert (m >= 0).all()

    def test_rho_zero_equals_independent_poisson(self):
        g = np.arange(13)
        indep = np.outer(stats.poisson.pmf(g, 1.6), stats.poisson.pmf(g, 1.1))
        assert np.allclose(score_matrix(1.6, 1.1, 0.0), indep / indep.sum(), atol=1e-15)

    def test_tau_changes_only_the_four_low_scores(self):
        lam, mu, rho = 1.6, 1.1, -0.12
        dc = score_matrix(lam, mu, rho)
        po = score_matrix(lam, mu, 0.0)
        low = np.zeros_like(dc, dtype=bool)
        low[:2, :2] = True
        # Mass is conserved inside the grid, so every other cell is unchanged.
        assert np.allclose(dc[~low], po[~low], rtol=1e-12, atol=0)
        # rho < 0: more 0-0 and 1-1, fewer 1-0 and 0-1 than independent Poisson.
        assert dc[0, 0] > po[0, 0] and dc[1, 1] > po[1, 1]
        assert dc[1, 0] < po[1, 0] and dc[0, 1] < po[0, 1]
        assert dc[0, 0] == pytest.approx(po[0, 0] * (1 - lam * mu * rho), rel=1e-12)
        assert dc[0, 1] == pytest.approx(po[0, 1] * (1 + lam * rho), rel=1e-12)
        assert dc[1, 0] == pytest.approx(po[1, 0] * (1 + mu * rho), rel=1e-12)
        assert dc[1, 1] == pytest.approx(po[1, 1] * (1 - rho), rel=1e-12)

    def test_positive_rho_reverses_direction(self):
        dc, po = score_matrix(1.3, 1.0, 0.1), score_matrix(1.3, 1.0, 0.0)
        assert dc[0, 0] < po[0, 0] and dc[1, 1] < po[1, 1]
        assert dc[1, 0] > po[1, 0] and dc[0, 1] > po[0, 1]

    def test_infeasible_rho_raises(self):
        with pytest.raises(ValueError):
            score_matrix(2.5, 2.0, 0.9)   # 1 - lam*mu*rho < 0

    def test_market_probabilities_consistent(self):
        p = matrix_probabilities(score_matrix(1.6, 1.1, -0.12))
        assert p["1"] + p["X"] + p["2"] == pytest.approx(1.0, abs=1e-12)
        assert p["1X"] == pytest.approx(p["1"] + p["X"])
        assert p["X2"] == pytest.approx(p["X"] + p["2"])
        assert p["12"] == pytest.approx(p["1"] + p["2"])
        assert p["GG"] + p["NG"] == pytest.approx(1.0)
        for line in (0.5, 1.5, 2.5, 3.5, 4.5):
            assert p[f"Over {line}"] + p[f"Under {line}"] == pytest.approx(1.0)
        overs = [p[f"Over {line}"] for line in (0.5, 1.5, 2.5, 3.5, 4.5)]
        assert overs == sorted(overs, reverse=True)


# ----------------------------------------------------------------------- fit
class TestFit:
    def test_fits_and_converges(self, fitted, league):
        s = fitted.summary
        assert s.converged, s.message
        assert s.n_matches == len(league) == 360
        assert set(fitted.teams) == set(TEAMS)
        assert all(np.isfinite(v) for v in (fitted.intercept, fitted.home_advantage, fitted.rho))

    def test_identifiability_constraints(self, fitted):
        assert fitted.attack.sum() == pytest.approx(0.0, abs=1e-10)
        assert fitted.defence.sum() == pytest.approx(0.0, abs=1e-10)

    def test_recovers_known_parameters(self):
        # 3600 matches: large enough that sampling error is small (MLE is consistent;
        # with ~900 matches the per-team error is still ~0.1-0.2 on the log scale).
        data = synthetic_league(seasons=40, seed=11)
        m = DixonColesModel(xi=0.0, l2=0.0).fit(data)
        order = [m.team_index[t] for t in TEAMS]
        att, dfc = m.attack[order], m.defence[order]
        assert np.corrcoef(att, TRUE_ATT)[0, 1] > 0.98
        assert np.corrcoef(dfc, TRUE_DEF)[0, 1] > 0.98
        assert np.abs(att - TRUE_ATT).max() < 0.12
        assert np.abs(dfc - TRUE_DEF).max() < 0.12
        assert m.home_advantage == pytest.approx(TRUE_H, abs=0.08)
        assert m.intercept == pytest.approx(TRUE_C, abs=0.06)
        assert m.rho == pytest.approx(TRUE_RHO, abs=0.05)

    def test_analytic_gradient_matches_finite_differences(self, league):
        model = DixonColesModel(xi=0.002, l2=0.7).fit(league)
        df = prepare_history(league)
        hi = df.home_team.map(model.team_index).to_numpy()
        ai = df.away_team.map(model.team_index).to_numpy()
        x = df.home_goals.to_numpy(float)
        y = df.away_goals.to_numpy(float)
        age = ((df.date.max() - df.date).dt.total_seconds() / 86400).to_numpy()
        w = np.exp(-0.002 * age)
        lf = np.array([math.lgamma(a + 1) + math.lgamma(b + 1) for a, b in zip(x, y)])
        args = (hi, ai, x, y, w, w.sum(), lf)
        rng = np.random.default_rng(3)
        for rho in (-0.15, 0.0, 0.1):
            theta = rng.normal(0, 0.2, 3 + 2 * (len(model.teams) - 1))
            theta[2] = rho
            err = optimize.check_grad(lambda t: model._objective(t, *args)[0],
                                      lambda t: model._objective(t, *args)[1], theta, epsilon=1e-7)
            assert err < 1e-5

    def test_log_likelihood_matches_direct_computation(self, fitted, league):
        df = prepare_history(league)
        total = 0.0
        for r in df.itertuples():
            i, j = fitted.team_index[r.home_team], fitted.team_index[r.away_team]
            lam = math.exp(fitted.intercept + fitted.home_advantage + fitted.attack[i] + fitted.defence[j])
            mu = math.exp(fitted.intercept + fitted.attack[j] + fitted.defence[i])
            p = (tau(r.home_goals, r.away_goals, lam, mu, fitted.rho)
                 * stats.poisson.pmf(r.home_goals, lam) * stats.poisson.pmf(r.away_goals, mu))
            total += math.log(p)
        assert fitted.summary.log_likelihood == pytest.approx(total, rel=1e-9)

    def test_dixon_coles_beats_independent_poisson_likelihood(self, fitted, league):
        # The MLE over (.., rho) can never be worse than the rho = 0 sub-model.
        poisson = DixonColesModel(xi=0.0, l2=0.0)
        poisson_rho_bounds = (0.0, 0.0)
        import models.dixon_coles as dc
        original = dc.RHO_BOUNDS
        try:
            dc.RHO_BOUNDS = poisson_rho_bounds
            poisson.fit(league)
        finally:
            dc.RHO_BOUNDS = original
        assert poisson.rho == 0.0
        assert fitted.summary.log_likelihood >= poisson.summary.log_likelihood - 1e-8

    def test_reproducible(self, league):
        a = DixonColesModel().fit(league).params()
        b = DixonColesModel().fit(league).params()
        assert a == b

    def test_input_order_irrelevant(self, league):
        a = DixonColesModel(xi=0.003).fit(league)
        b = DixonColesModel(xi=0.003).fit(league.sample(frac=1, random_state=5))
        assert np.allclose(a.attack, b.attack, atol=1e-6)
        assert a.rho == pytest.approx(b.rho, abs=1e-6)

    def test_missing_scores_dropped_not_zero(self, league):
        dirty = league.copy().astype({"home_goals": "float", "away_goals": "float"})
        dirty.loc[[5, 17, 40], "home_goals"] = np.nan
        m = DixonColesModel().fit(dirty)
        assert m.summary.n_matches == len(league) - 3

    def test_demo_data_fits(self):
        history, _ = generate_demo(seed=42)
        m = DixonColesModel().fit(history)
        assert m.summary.converged
        assert m.summary.n_matches == len(prepare_history(history))
        p = m.predict(m.teams[0], m.teams[1])
        assert p.matrix.sum() == pytest.approx(1.0)


# --------------------------------------------------------------- time boundary
class TestTimeBoundary:
    def test_future_matches_never_used(self, league):
        ref = league.date.iloc[200]                              # one match per day
        base = DixonColesModel(xi=0.002).fit(league, reference_time=ref)
        assert base.summary.n_matches == 200                     # match ON ref date excluded
        assert base.summary.last_match < ref

        tampered = league.copy()
        future = tampered.date >= ref
        tampered.loc[future, "home_goals"] = 9                   # absurd future results
        tampered.loc[future, "away_goals"] = 0
        other = DixonColesModel(xi=0.002).fit(tampered, reference_time=ref)
        assert other.params() == base.params()

        truncated = DixonColesModel(xi=0.002).fit(league.iloc[:200], reference_time=ref)
        assert truncated.params() == base.params()

    def test_naive_and_aware_reference_equivalent(self, league):
        ref = league.date.iloc[150]
        a = DixonColesModel().fit(league, reference_time=ref)
        b = DixonColesModel().fit(league, reference_time=ref.tz_convert(None))
        assert a.params() == b.params()

    def test_reference_before_data_raises(self, league):
        with pytest.raises(ValueError):
            DixonColesModel().fit(league, reference_time=START)


# ------------------------------------------------------------------ time decay
class TestTimeDecay:
    @staticmethod
    def _form_change_league():
        # Team A is the weakest attack for 3 seasons, then the strongest for 1.
        early_att = TRUE_ATT[::-1].copy()
        old = synthetic_league(seasons=3, seed=21, att=early_att)
        late_start = old.date.max() + pd.Timedelta(days=1)
        new = synthetic_league(seasons=1, seed=22, start=late_start)
        return pd.concat([old, new], ignore_index=True)

    def test_recent_form_weighs_more(self):
        data = self._form_change_league()
        flat = DixonColesModel(xi=0.0, l2=0.0).fit(data)
        decayed = DixonColesModel(xi=0.01, l2=0.0).fit(data)
        a = flat.team_index["Team A"]
        assert decayed.attack[a] > flat.attack[a] + 0.2
        assert decayed.summary.total_weight < flat.summary.total_weight == len(data)

    def test_weights_follow_exponential_decay(self, league):
        xi = 0.004
        m = DixonColesModel(xi=xi).fit(league)
        ages = (league.date.max() - league.date).dt.days.to_numpy()
        assert m.summary.total_weight == pytest.approx(np.exp(-xi * ages).sum(), rel=1e-12)


# ------------------------------------------------------------------- shrinkage
class TestShrinkage:
    def test_penalty_shrinks_toward_league_average(self, league):
        ratings = [np.abs(DixonColesModel(xi=0.0, l2=l2).fit(league).attack).sum()
                   for l2 in (0.0, 1.0, 10.0)]
        assert ratings[0] > ratings[1] > ratings[2]

    def test_team_with_few_matches_is_pulled_to_average(self, league):
        extra = pd.DataFrame([
            {"date": league.date.max() + pd.Timedelta(days=1), "home_team": "Newcomer",
             "away_team": "Team J", "home_goals": 5, "away_goals": 0},
            {"date": league.date.max() + pd.Timedelta(days=2), "home_team": "Team I",
             "away_team": "Newcomer", "home_goals": 0, "away_goals": 4},
        ])
        data = pd.concat([league, extra], ignore_index=True)
        mle = DixonColesModel(xi=0.0, l2=0.0).fit(data)
        shrunk = DixonColesModel(xi=0.0, l2=1.0).fit(data)
        k = mle.team_index["Newcomer"]
        assert abs(shrunk.attack[k]) < abs(mle.attack[k])
        assert shrunk.summary.matches_per_team["Newcomer"] == 2


# --------------------------------------------------------------------- predict
class TestPredict:
    def test_prediction_structure(self, fitted):
        p = fitted.predict("Team A", "Team J")
        i, j = fitted.team_index["Team A"], fitted.team_index["Team J"]
        assert p.lambda_home == pytest.approx(math.exp(
            fitted.intercept + fitted.home_advantage + fitted.attack[i] + fitted.defence[j]))
        assert p.lambda_away == pytest.approx(math.exp(
            fitted.intercept + fitted.attack[j] + fitted.defence[i]))
        assert p.matrix.shape == (13, 13)
        assert p.matrix.sum() == pytest.approx(1.0, abs=1e-12)
        assert p.probabilities["1"] + p.probabilities["X"] + p.probabilities["2"] == pytest.approx(1.0)
        assert p.probabilities["1"] > p.probabilities["2"]        # strongest vs weakest at home
        assert len(p.top_scores) == 5
        probs = [q for _, q in p.top_scores]
        assert probs == sorted(probs, reverse=True)
        assert p.unknown_teams == ()
        assert fitted.expected_goals("Team A", "Team J") == (p.lambda_home, p.lambda_away)

    def test_every_pair_sums_to_one(self, fitted):
        for h in fitted.teams:
            for a in fitted.teams:
                if h != a:
                    assert fitted.predict(h, a).matrix.sum() == pytest.approx(1.0, abs=1e-12)

    def test_home_advantage_positive(self, fitted):
        assert fitted.home_advantage > 0
        assert fitted.predict("Team E", "Team F").lambda_home > fitted.predict("Team F", "Team E").lambda_away

    def test_unknown_team_errors_by_default(self, fitted):
        with pytest.raises(ValueError):
            fitted.predict("Nobody FC", "Team A")

    def test_unknown_team_average_mode(self, fitted):
        p = fitted.predict("Nobody FC", "Team A", unknown_team="average")
        a = fitted.team_index["Team A"]
        assert p.unknown_teams == ("Nobody FC",)
        assert p.lambda_home == pytest.approx(math.exp(fitted.intercept + fitted.home_advantage + fitted.defence[a]))
        assert p.lambda_away == pytest.approx(math.exp(fitted.intercept + fitted.attack[a]))
        both = fitted.predict("X FC", "Y FC", unknown_team="average")
        assert both.lambda_home == pytest.approx(math.exp(fitted.intercept + fitted.home_advantage))
        assert both.lambda_away == pytest.approx(math.exp(fitted.intercept))

    def test_same_team_and_bad_mode(self, fitted):
        with pytest.raises(ValueError):
            fitted.predict("Team A", "Team A")
        with pytest.raises(ValueError):
            fitted.predict("Team A", "Team B", unknown_team="guess")

    def test_rho_clipped_to_feasible_range(self, fitted):
        m = DixonColesModel()
        m.teams, m.team_index = ["H", "A"], {"H": 0, "A": 1}
        m.attack, m.defence = np.array([1.0, -1.0]), np.array([-1.0, 1.0])
        m.intercept, m.home_advantage, m.rho = 0.3, 0.3, 0.45
        p = m.predict("H", "A")
        lo, hi = rho_feasible_range(p.lambda_home, p.lambda_away)
        assert p.rho_clipped and lo <= p.rho <= hi
        assert (p.matrix >= 0).all() and p.matrix.sum() == pytest.approx(1.0)

    def test_compatible_with_monte_carlo(self, fitted):
        lam, mu = fitted.expected_goals("Team C", "Team H")
        mc = simulate_match(lam, mu, n_simulations=200_000, seed=1)
        # MC uses independent Poisson -> must agree with the DC matrix at rho = 0.
        po = matrix_probabilities(score_matrix(lam, mu, 0.0))
        for key in ("1", "X", "2"):
            assert mc.probabilities[key] == pytest.approx(po[key], abs=0.005)


# ---------------------------------------------------------- edge cases / errors
class TestEdgeCases:
    def test_not_fitted(self):
        m = DixonColesModel()
        assert not m.is_fitted
        for call in (lambda: m.predict("A", "B"), m.team_ratings, m.params):
            with pytest.raises(RuntimeError):
                call()

    def test_too_few_matches_raises(self, league):
        with pytest.raises(ValueError, match="τουλάχιστον"):
            DixonColesModel().fit(league.iloc[:10])

    def test_empty_frame_raises(self):
        empty = pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
        with pytest.raises(ValueError):
            DixonColesModel().fit(empty)

    def test_very_few_matches_with_lower_threshold_gives_finite_values(self, league):
        few = league.iloc[:12]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = DixonColesModel(min_matches=5, l2=1.0).fit(few)
        assert np.isfinite(m.attack).all() and np.isfinite(m.defence).all()
        assert np.abs(m.attack).max() < 2.0                      # shrinkage keeps it sane
        teams = m.teams
        assert m.predict(teams[0], teams[1]).matrix.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("kwargs", [{"xi": -0.001}, {"xi": float("nan")}, {"l2": -1.0},
                                        {"l2": float("inf")}, {"min_matches": 0}])
    def test_invalid_hyperparameters(self, kwargs):
        with pytest.raises(ValueError):
            DixonColesModel(**kwargs)

    def test_team_ratings_frame(self, fitted):
        r = fitted.team_ratings()
        assert list(r.columns) == ["team", "attack", "defence", "matches"]
        assert len(r) == 10 and (r.matches == 72).all()          # 4 seasons x 18 matches
        assert r.attack.is_monotonic_decreasing
        assert np.log(r.attack).sum() == pytest.approx(0.0, abs=1e-10)
