"""Elo rating system for football match prediction.

Each team carries a numeric rating (default 1500). After every match the
loser's rating drops and the winner's rises by an amount proportional to
how surprising the result was.  Draws split the update.

Key formulas
------------
Expected score:  E_home = 1 / (1 + 10^((R_away - R_home - HFA) / 400))
where HFA is the home-field advantage expressed in rating points.

Actual score: 1 for a win, 0.5 for a draw, 0 for a loss.

Update: R_new = R_old + K * (actual - expected)

The model exposes 1X2 and Over/Under 2.5 probabilities via historical
calibration on the dataset it was fitted on.

Reference: Arpad Elo, "The Rating of Chessplayers, Past and Present" (1978);
adapted for football with a draw factor and home-field advantage following
FiveThirtyEight / ClubElo conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0
DEFAULT_HFA = 65.0          # home-field advantage in Elo points
DRAW_WEIGHT = 0.5           # actual-score value for a draw


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EloPrediction:
    """Prediction for a single match."""
    home: str
    away: str
    home_rating: float
    away_rating: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    probabilities: dict[str, float]   # 1, X, 2, 1X, X2, 12, Over/Under 2.5, GG, NG
    expected_home_goals: float        # estimated from Elo diff (rough)
    expected_away_goals: float


@dataclass
class EloFitSummary:
    """Metadata about the fitted model."""
    n_matches: int
    n_teams: int
    first_match: pd.Timestamp
    last_match: pd.Timestamp
    k_factor: float
    hfa: float
    initial_rating: float
    mean_rating: float
    std_rating: float
    # Calibration stats (from the training set)
    home_win_rate: float
    draw_rate: float
    away_win_rate: float
    over25_rate: float
    mean_total_goals: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expected_score(r_home: float, r_away: float, hfa: float) -> float:
    """Home expected score in [0, 1]."""
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - hfa) / 400.0))


def _split_1x2(e_home: float, draw_factor: float | None = None) -> tuple[float, float, float]:
    """Split an Elo expected score into (P(1), P(X), P(2)).

    The draw probability is estimated as a quadratic function of e_home that
    peaks when e_home ≈ 0.5 (evenly matched teams draw more often).  The
    coefficient *draw_factor* scales the peak draw probability; if None a
    sensible default (≈0.26 peak) is used.  The residual goes to 1 and 2 in
    proportion to e_home.
    """
    if draw_factor is None:
        draw_factor = 1.04          # gives ~26% draw at e_home=0.5
    # Quadratic that is 0 at 0 and 1, peaks at 0.5 with value draw_factor/4
    p_draw = draw_factor * e_home * (1.0 - e_home)
    p_draw = min(max(p_draw, 0.02), 0.45)   # sanity clamp
    remaining = 1.0 - p_draw
    p_home = remaining * e_home
    p_away = remaining * (1.0 - e_home)
    return (p_home, p_draw, p_away)


def _estimate_goals(p_home: float, p_draw: float, p_away: float,
                    mean_total: float) -> tuple[float, float]:
    """Very rough goal estimates from 1X2 probs and league mean total goals.

    This is intentionally simple — Elo is about ratings, not score lines.
    The Poisson and Dixon-Coles models handle goal distributions properly.
    """
    # Home share of goals increases with their win probability
    home_share = 0.5 + 0.18 * (p_home - p_away)
    home_share = min(max(home_share, 0.25), 0.75)
    return (mean_total * home_share, mean_total * (1.0 - home_share))


def _over25_from_elo(e_home: float, base_rate: float) -> float:
    """Rough Over 2.5 probability.

    Lopsided matches (high |e_home - 0.5|) tend to have more goals.
    """
    lopsidedness = abs(e_home - 0.5)
    # linear nudge: balanced → fewer goals, lopsided → more
    adj = 0.15 * lopsidedness
    p = base_rate + adj
    return min(max(p, 0.10), 0.90)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EloModel:
    """Elo-based rating and prediction model.

    Usage::

        model = EloModel(k=32, hfa=65).fit(history_df, reference_time=kickoff)
        pred = model.predict("Team A", "Team B")
        pred.probabilities["1"]   # P(home win)
    """

    def __init__(self, k: float = DEFAULT_K_FACTOR,
                 hfa: float = DEFAULT_HFA,
                 initial_rating: float = DEFAULT_INITIAL_RATING):
        if k <= 0:
            raise ValueError(f"K factor must be positive, got {k}")
        self.k = float(k)
        self.hfa = float(hfa)
        self.initial_rating = float(initial_rating)
        self.ratings: dict[str, float] = {}
        self.summary: EloFitSummary | None = None
        self._over25_base: float = 0.50
        self._gg_base: float = 0.45
        self._mean_total: float = 2.6

    # ---- fitting ----------------------------------------------------------
    def fit(self, matches: pd.DataFrame,
            reference_time: pd.Timestamp | None = None) -> "EloModel":
        """Fit (replay) Elo ratings on historical matches.

        Parameters
        ----------
        matches : DataFrame
            Must contain date, home_team, away_team, home_goals, away_goals.
        reference_time : Timestamp, optional
            Only matches strictly before this time are used.
            If None, all matches are used.
        """
        from engine import prepare_history
        df = prepare_history(matches)
        if reference_time is not None:
            ref = pd.Timestamp(reference_time)
            if ref.tzinfo is None:
                ref = ref.tz_localize("UTC")
            else:
                ref = ref.tz_convert("UTC")
            df = df[df.date < ref].copy()

        if len(df) < config.MIN_MATCHES_TOTAL:
            raise ValueError(
                f"Χρειάζονται τουλάχιστον {config.MIN_MATCHES_TOTAL} αγώνες "
                f"για Elo (υπάρχουν {len(df)}).")

        ratings: dict[str, float] = {}

        for _, row in df.iterrows():
            ht, at = row.home_team, row.away_team
            r_h = ratings.get(ht, self.initial_rating)
            r_a = ratings.get(at, self.initial_rating)

            e_h = _expected_score(r_h, r_a, self.hfa)
            # actual scores
            hg, ag = int(row.home_goals), int(row.away_goals)
            if hg > ag:
                s_h, s_a = 1.0, 0.0
            elif hg < ag:
                s_h, s_a = 0.0, 1.0
            else:
                s_h, s_a = DRAW_WEIGHT, DRAW_WEIGHT

            ratings[ht] = r_h + self.k * (s_h - e_h)
            ratings[at] = r_a + self.k * (s_a - (1.0 - e_h))

        self.ratings = ratings

        # Calibration stats from the training data
        home_wins = int(((df.home_goals > df.away_goals).sum()))
        draws = int(((df.home_goals == df.away_goals).sum()))
        away_wins = int(((df.home_goals < df.away_goals).sum()))
        n = len(df)
        total_goals = (df.home_goals + df.away_goals)
        self._mean_total = float(total_goals.mean()) if n > 0 else 2.6
        self._over25_base = float((total_goals > 2.5).mean()) if n > 0 else 0.50
        gg = float(((df.home_goals > 0) & (df.away_goals > 0)).mean()) if n > 0 else 0.45
        self._gg_base = gg

        r_vals = np.array(list(ratings.values())) if ratings else np.array([self.initial_rating])
        self.summary = EloFitSummary(
            n_matches=n, n_teams=len(ratings),
            first_match=df.date.iloc[0] if n > 0 else pd.NaT,
            last_match=df.date.iloc[-1] if n > 0 else pd.NaT,
            k_factor=self.k, hfa=self.hfa,
            initial_rating=self.initial_rating,
            mean_rating=float(r_vals.mean()),
            std_rating=float(r_vals.std()),
            home_win_rate=home_wins / n if n > 0 else 0.0,
            draw_rate=draws / n if n > 0 else 0.0,
            away_win_rate=away_wins / n if n > 0 else 0.0,
            over25_rate=self._over25_base,
            mean_total_goals=self._mean_total,
        )
        return self

    # ---- prediction -------------------------------------------------------
    def predict(self, home_team: str, away_team: str) -> EloPrediction:
        """Predict a match from current ratings.

        Raises ValueError if a team has no rating (never seen in training data).
        """
        if home_team not in self.ratings:
            raise ValueError(f"Η ομάδα «{home_team}» δεν βρέθηκε στο σύστημα Elo.")
        if away_team not in self.ratings:
            raise ValueError(f"Η ομάδα «{away_team}» δεν βρέθηκε στο σύστημα Elo.")

        r_h = self.ratings[home_team]
        r_a = self.ratings[away_team]
        e_h = _expected_score(r_h, r_a, self.hfa)

        p1, pX, p2 = _split_1x2(e_h)
        over25 = _over25_from_elo(e_h, self._over25_base)
        # GG estimate: slightly higher when match is balanced
        gg = self._gg_base + 0.05 * (1.0 - 2.0 * abs(e_h - 0.5))
        gg = min(max(gg, 0.15), 0.85)

        eh_goals, ea_goals = _estimate_goals(p1, pX, p2, self._mean_total)

        probs = {
            "1": p1, "X": pX, "2": p2,
            "1X": p1 + pX, "X2": pX + p2, "12": p1 + p2,
            "Over 2.5": over25, "Under 2.5": 1.0 - over25,
            "GG": gg, "NG": 1.0 - gg,
        }
        return EloPrediction(
            home=home_team, away=away_team,
            home_rating=r_h, away_rating=r_a,
            home_win_prob=p1, draw_prob=pX, away_win_prob=p2,
            probabilities=probs,
            expected_home_goals=round(eh_goals, 3),
            expected_away_goals=round(ea_goals, 3),
        )

    def team_ratings_table(self) -> pd.DataFrame:
        """Return a sorted DataFrame of all team ratings."""
        if not self.ratings:
            return pd.DataFrame(columns=["Ομάδα", "Elo"])
        rows = [{"Ομάδα": t, "Elo": round(r, 1)} for t, r in
                sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)]
        return pd.DataFrame(rows)
