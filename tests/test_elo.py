"""Tests for the Elo rating model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.elo import (
    EloModel, EloPrediction, EloFitSummary,
    _expected_score, _split_1x2, DEFAULT_INITIAL_RATING,
    DEFAULT_K_FACTOR, DEFAULT_HFA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _demo_history(n: int = 80) -> pd.DataFrame:
    """Minimal deterministic history."""
    rng = np.random.RandomState(42)
    teams = [f"Team_{chr(65 + i)}" for i in range(8)]
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 3) % len(teams)]
        rows.append({
            "date": (base + pd.Timedelta(days=i)).isoformat(),
            "home_team": home, "away_team": away,
            "home_goals": int(rng.poisson(1.4)),
            "away_goals": int(rng.poisson(1.1)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expected score formula
# ---------------------------------------------------------------------------
class TestExpectedScore:
    def test_equal_ratings_no_hfa(self):
        assert _expected_score(1500, 1500, 0) == pytest.approx(0.5, abs=1e-9)

    def test_equal_ratings_with_hfa(self):
        e = _expected_score(1500, 1500, DEFAULT_HFA)
        assert 0.5 < e < 0.7  # home advantage

    def test_higher_home(self):
        e = _expected_score(1700, 1300, 0)
        assert e > 0.9

    def test_higher_away(self):
        e = _expected_score(1300, 1700, 0)
        assert e < 0.1

    def test_symmetry(self):
        e1 = _expected_score(1600, 1400, 0)
        e2 = _expected_score(1400, 1600, 0)
        assert e1 + e2 == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 1X2 split
# ---------------------------------------------------------------------------
class TestSplit1X2:
    def test_sums_to_one(self):
        for e in [0.1, 0.3, 0.5, 0.7, 0.9]:
            p1, pX, p2 = _split_1x2(e)
            assert p1 + pX + p2 == pytest.approx(1.0, abs=1e-9)

    def test_balanced_draw_peak(self):
        _, pX_balanced, _ = _split_1x2(0.5)
        _, pX_lopsided, _ = _split_1x2(0.9)
        assert pX_balanced > pX_lopsided

    def test_home_favourite(self):
        p1, pX, p2 = _split_1x2(0.8)
        assert p1 > p2
        assert p1 > pX

    def test_all_positive(self):
        for e in [0.01, 0.99, 0.5]:
            p1, pX, p2 = _split_1x2(e)
            assert p1 > 0 and pX > 0 and p2 > 0


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
class TestEloFit:
    def test_basic_fit(self):
        df = _demo_history()
        model = EloModel().fit(df)
        assert model.summary is not None
        assert model.summary.n_matches == 80
        assert model.summary.n_teams == 8
        assert len(model.ratings) == 8

    def test_ratings_around_initial(self):
        df = _demo_history()
        model = EloModel().fit(df)
        # zero-sum: mean should stay near initial
        vals = list(model.ratings.values())
        assert np.mean(vals) == pytest.approx(DEFAULT_INITIAL_RATING, abs=50)

    def test_too_few_matches(self):
        df = _demo_history(10)
        with pytest.raises(ValueError, match="τουλάχιστον"):
            EloModel().fit(df)

    def test_reference_time_strict(self):
        df = _demo_history(80)
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        model = EloModel().fit(df, reference_time=cutoff)
        assert model.summary.n_matches < 80
        # All used matches before cutoff
        assert model.summary.last_match < cutoff

    def test_k_factor_effect(self):
        """Higher K → more volatile ratings."""
        df = _demo_history()
        model_low = EloModel(k=8).fit(df)
        model_high = EloModel(k=64).fit(df)
        std_low = np.std(list(model_low.ratings.values()))
        std_high = np.std(list(model_high.ratings.values()))
        assert std_high > std_low

    def test_custom_initial(self):
        df = _demo_history()
        model = EloModel(initial_rating=1200).fit(df)
        assert model.summary.initial_rating == 1200
        assert np.mean(list(model.ratings.values())) == pytest.approx(1200, abs=50)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EloModel(k=-5)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class TestEloPrediction:
    @pytest.fixture
    def model(self):
        return EloModel().fit(_demo_history())

    def test_probabilities_sum_to_one(self, model):
        pred = model.predict("Team_A", "Team_B")
        assert pred.home_win_prob + pred.draw_prob + pred.away_win_prob == pytest.approx(1.0, abs=1e-6)

    def test_markets_consistent(self, model):
        pred = model.predict("Team_A", "Team_B")
        p = pred.probabilities
        assert p["1X"] == pytest.approx(p["1"] + p["X"], abs=1e-9)
        assert p["X2"] == pytest.approx(p["X"] + p["2"], abs=1e-9)
        assert p["12"] == pytest.approx(p["1"] + p["2"], abs=1e-9)
        assert p["Over 2.5"] + p["Under 2.5"] == pytest.approx(1.0, abs=1e-9)
        assert p["GG"] + p["NG"] == pytest.approx(1.0, abs=1e-9)

    def test_unknown_team_raises(self, model):
        with pytest.raises(ValueError, match="δεν βρέθηκε"):
            model.predict("Team_A", "UNKNOWN_FC")

    def test_goal_estimates_positive(self, model):
        pred = model.predict("Team_A", "Team_B")
        assert pred.expected_home_goals > 0
        assert pred.expected_away_goals > 0

    def test_ratings_in_prediction(self, model):
        pred = model.predict("Team_A", "Team_B")
        assert pred.home_rating == model.ratings["Team_A"]
        assert pred.away_rating == model.ratings["Team_B"]


# ---------------------------------------------------------------------------
# Team ratings table
# ---------------------------------------------------------------------------
class TestRatingsTable:
    def test_table_shape(self):
        model = EloModel().fit(_demo_history())
        table = model.team_ratings_table()
        assert len(table) == 8
        assert list(table.columns) == ["Ομάδα", "Elo"]

    def test_sorted_descending(self):
        model = EloModel().fit(_demo_history())
        table = model.team_ratings_table()
        elos = table["Elo"].tolist()
        assert elos == sorted(elos, reverse=True)

    def test_empty_before_fit(self):
        model = EloModel()
        table = model.team_ratings_table()
        assert len(table) == 0


# ---------------------------------------------------------------------------
# No future-data leakage
# ---------------------------------------------------------------------------
class TestEloLeakage:
    def test_future_data_does_not_affect_ratings(self):
        """Tampering with matches after the cutoff must not change ratings."""
        df = _demo_history(80)
        cutoff = pd.Timestamp("2025-02-15", tz="UTC")
        model_before = EloModel(k=32).fit(df, reference_time=cutoff)

        # Tamper with future matches
        df_tampered = df.copy()
        future_mask = pd.to_datetime(df_tampered["date"]) >= cutoff
        df_tampered.loc[future_mask, "home_goals"] = 99
        model_after = EloModel(k=32).fit(df_tampered, reference_time=cutoff)

        for team in model_before.ratings:
            assert model_before.ratings[team] == pytest.approx(
                model_after.ratings[team], abs=1e-9
            ), f"Leakage detected for {team}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestEloDeterminism:
    def test_same_input_same_output(self):
        df = _demo_history()
        m1 = EloModel().fit(df)
        m2 = EloModel().fit(df)
        for team in m1.ratings:
            assert m1.ratings[team] == pytest.approx(m2.ratings[team], abs=1e-12)
