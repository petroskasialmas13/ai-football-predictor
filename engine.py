"""Transparent, leakage-aware pre-match football probability model.
No bookmaker feed, injuries, xG, or LLM claims are made by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = ("date", "home_team", "away_team", "home_goals", "away_goals")


def clean_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w ]+", " ", text)
    return " ".join(text.split())


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in raw]
    if missing:
        raise ValueError(f"Λείπουν στήλες: {', '.join(missing)}")
    df = raw.copy()[list(REQUIRED)]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df = df.dropna(subset=list(REQUIRED))
    df = df[(df["home_team"] != "") & (df["away_team"] != "")]
    df = df[(df.home_team != df.away_team) & (df.home_goals >= 0) & (df.away_goals >= 0)]
    df = df[(df.home_goals % 1 == 0) & (df.away_goals % 1 == 0)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("date").drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return df.reset_index(drop=True)


def api_matches_to_history(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        score = (m.get("score") or {}).get("fullTime") or {}
        home, away = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        if score.get("home") is None or score.get("away") is None:
            continue
        if not home.get("name") or not away.get("name"):
            continue
        rows.append({
            "date": m.get("utcDate"), "home_team": home["name"],
            "away_team": away["name"], "home_goals": score["home"],
            "away_goals": score["away"],
        })
    return prepare_history(pd.DataFrame(rows, columns=REQUIRED))


def _recency_mean(series: pd.Series, max_games: int = 12, decay: float = 0.89) -> tuple[float | None, int]:
    values = np.asarray(series.tail(max_games), dtype=float)
    if len(values) == 0:
        return None, 0
    weights = decay ** np.arange(len(values) - 1, -1, -1)
    return float(np.average(values, weights=weights)), len(values)


def _shrink(measured: float | None, n: int, baseline: float, pseudo: float) -> float:
    if measured is None:
        return baseline
    return (n * measured + pseudo * baseline) / (n + pseudo)


def _results(df: pd.DataFrame, team: str, n: int = 6) -> list[str]:
    played = df[(df.home_team == team) | (df.away_team == team)].tail(n).iloc[::-1]
    out = []
    for _, m in played.iterrows():
        own = m.home_goals if m.home_team == team else m.away_goals
        their = m.away_goals if m.home_team == team else m.home_goals
        out.append("Ν" if own > their else "Ι" if own == their else "Η")
    return out


def _poisson_vec(lam: float, max_goals: int = 12) -> np.ndarray:
    p = np.zeros(max_goals + 1, dtype=float)
    p[0] = math.exp(-lam)
    for i in range(1, max_goals + 1):
        p[i] = p[i - 1] * lam / i
    # Tail mass is negligible for expected goals normally encountered. Normalize precisely.
    return p / p.sum()


def _team_strength(df: pd.DataFrame, team: str, venue: str, league_home: float, league_away: float) -> dict:
    if venue == "home":
        venue_games = df[df.home_team == team]
        all_games = df[(df.home_team == team) | (df.away_team == team)].copy()
        scored = venue_games.home_goals
        conceded = venue_games.away_goals
        general_scored = np.where(all_games.home_team == team, all_games.home_goals, all_games.away_goals)
        general_conceded = np.where(all_games.home_team == team, all_games.away_goals, all_games.home_goals)
        base_for, base_against = league_home, league_away
    else:
        venue_games = df[df.away_team == team]
        all_games = df[(df.home_team == team) | (df.away_team == team)].copy()
        scored = venue_games.away_goals
        conceded = venue_games.home_goals
        general_scored = np.where(all_games.home_team == team, all_games.home_goals, all_games.away_goals)
        general_conceded = np.where(all_games.home_team == team, all_games.away_goals, all_games.home_goals)
        base_for, base_against = league_away, league_home

    venue_attack, n_venue = _recency_mean(scored)
    venue_defense, _ = _recency_mean(conceded)
    # General strength is venue-neutralized by the league's overall goals per team.
    league_overall = (league_home + league_away) / 2
    general_attack, n_all = _recency_mean(pd.Series(general_scored))
    general_defense, _ = _recency_mean(pd.Series(general_conceded))
    general_attack_ratio = _shrink(general_attack, n_all, league_overall, 6) / league_overall
    general_defense_ratio = _shrink(general_defense, n_all, league_overall, 6) / league_overall
    venue_attack_ratio = _shrink(venue_attack, n_venue, base_for, 5) / base_for
    venue_defense_ratio = _shrink(venue_defense, n_venue, base_against, 5) / base_against
    venue_share = min(0.75, n_venue / (n_venue + 3))
    return {
        "attack": float(venue_share * venue_attack_ratio + (1 - venue_share) * general_attack_ratio),
        "defense": float(venue_share * venue_defense_ratio + (1 - venue_share) * general_defense_ratio),
        "all_games": int(len(all_games)), "venue_games": int(len(venue_games)),
        "recent_scored": float(general_attack if general_attack is not None else 0),
        "recent_conceded": float(general_defense if general_defense is not None else 0),
    }


def analyze(raw: pd.DataFrame, home_team: str, away_team: str, kickoff: datetime | pd.Timestamp) -> dict:
    if home_team == away_team:
        raise ValueError("Πρέπει να επιλέξεις δύο διαφορετικές ομάδες.")
    cutoff = pd.Timestamp(kickoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    df = prepare_history(raw)
    df = df[df.date < cutoff].copy()  # strict time boundary prevents future-data leakage
    if len(df) < 16:
        raise ValueError(f"Μόνο {len(df)} ολοκληρωμένοι αγώνες υπάρχουν πριν από την ημερομηνία. Χρειάζονται τουλάχιστον 16.")
    teams = set(df.home_team) | set(df.away_team)
    if home_team not in teams or away_team not in teams:
        raise ValueError("Δεν βρέθηκαν ιστορικά παιχνίδια και για τις δύο ομάδες.")
    h_n = int(((df.home_team == home_team) | (df.away_team == home_team)).sum())
    a_n = int(((df.home_team == away_team) | (df.away_team == away_team)).sum())
    if min(h_n, a_n) < 4:
        raise ValueError("Δεν επαρκούν τα δεδομένα: χρειάζονται 4 προηγούμενοι αγώνες ανά ομάδα.")

    # League means; reasonable bounds guard against a tiny or distorted sample.
    league_home = float(np.clip(df.home_goals.mean(), 0.6, 3.0))
    league_away = float(np.clip(df.away_goals.mean(), 0.45, 2.7))
    h = _team_strength(df, home_team, "home", league_home, league_away)
    a = _team_strength(df, away_team, "away", league_home, league_away)
    lh = float(np.clip(league_home * h["attack"] * a["defense"], 0.2, 4.5))
    la = float(np.clip(league_away * a["attack"] * h["defense"], 0.2, 4.5))
    matrix = np.outer(_poisson_vec(lh), _poisson_vec(la))
    matrix = matrix / matrix.sum()
    hwin = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    awin = float(np.triu(matrix, 1).sum())
    # matrix[i,j] = P(home=i, away=j), lower triangular means i>j.
    over15 = float(sum(matrix[i, j] for i in range(13) for j in range(13) if i + j > 1))
    over25 = float(sum(matrix[i, j] for i in range(13) for j in range(13) if i + j > 2))
    over35 = float(sum(matrix[i, j] for i in range(13) for j in range(13) if i + j > 3))
    btts = float(matrix[1:, 1:].sum())
    flat_ids = np.argsort(matrix.ravel())[::-1][:5]
    scores = [(f"{int(idx)//13}-{int(idx)%13}", float(matrix.ravel()[idx])) for idx in flat_ids]
    h2h = df[((df.home_team == home_team) & (df.away_team == away_team)) | ((df.home_team == away_team) & (df.away_team == home_team))].tail(5)
    home_recents = _results(df, home_team)
    away_recents = _results(df, away_team)
    data_quality = "Υψηλότερη" if min(h_n, a_n) >= 18 and len(df) >= 140 else "Μέτρια" if min(h_n, a_n) >= 9 and len(df) >= 60 else "Περιορισμένη"
    probs = {"1": hwin, "X": draw, "2": awin, "1X": hwin + draw, "X2": draw + awin, "12": hwin + awin,
             "Over 1.5": over15, "Under 1.5": 1 - over15, "Over 2.5": over25, "Under 2.5": 1 - over25,
             "Over 3.5": over35, "Under 3.5": 1 - over35, "GG": btts, "NG": 1 - btts}
    # Honest categorization: most-probable event is not necessarily a profitable bet.
    sorted_markets = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return {
        "home": home_team, "away": away_team, "kickoff": cutoff,
        "expected_home": lh, "expected_away": la, "probabilities": probs,
        "top_scores": scores, "data_quality": data_quality, "matches_used": len(df),
        "home_stats": h, "away_stats": a, "form_home": home_recents,
        "form_away": away_recents, "h2h": h2h, "ranked_markets": sorted_markets,
        "league_home_goals": league_home, "league_away_goals": league_away,
    }


def backtest(raw: pd.DataFrame, max_matches: int = 60) -> dict:
    df = prepare_history(raw)
    if len(df) < 50:
        raise ValueError("Για backtest χρειάζονται τουλάχιστον 50 ιστορικά παιχνίδια.")
    eval_matches = df.tail(max_matches)
    outcomes = []
    predictions = []
    baselines = []
    skipped = 0
    for _, match in eval_matches.iterrows():
        past = df[df.date < match.date]
        try:
            result = analyze(past, match.home_team, match.away_team, match.date)
        except ValueError:
            skipped += 1
            continue
        probs = np.array([result["probabilities"][p] for p in ("1", "X", "2")], dtype=float)
        actual = np.array([0., 0., 0.])
        actual[0 if match.home_goals > match.away_goals else 1 if match.home_goals == match.away_goals else 2] = 1
        # Baseline probabilities purely from older matches (also time-safe).
        n_home = int((past.home_goals > past.away_goals).sum())
        n_draw = int((past.home_goals == past.away_goals).sum())
        n_away = int((past.home_goals < past.away_goals).sum())
        baseline = (np.array([n_home, n_draw, n_away], dtype=float) + 1) / (len(past) + 3)
        outcomes.append(actual)
        predictions.append(probs)
        baselines.append(baseline)
    if len(outcomes) < 10:
        raise ValueError(f"Μόνο {len(outcomes)} παιχνίδια μπορούν να ελεγχθούν, μετά από {skipped} αποκλεισμούς. Θέλουμε τουλάχιστον 10.")
    y, p, b = np.array(outcomes), np.array(predictions), np.array(baselines)
    return {"count": len(y), "skipped": skipped, "accuracy": float(np.mean(p.argmax(axis=1) == y.argmax(axis=1))),
            "brier": float(np.mean(np.sum((p - y) ** 2, axis=1))),
            "baseline_brier": float(np.mean(np.sum((b - y) ** 2, axis=1))),
            "baseline_accuracy": float(np.mean(b.argmax(axis=1) == y.argmax(axis=1)))}


def generate_demo(seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    """Clearly fictional results, for running the app without any API key."""
    rng = np.random.default_rng(seed)
    teams = ["Αστήρ FC", "Κύμα FC", "Ίριδα FC", "Λέων FC", "Ορίζοντας FC", "Νίκη FC", "Κένταυρος FC", "Δόξα FC"]
    strengths = np.array([1.35, 1.15, 1.05, .95, 1.1, .92, .8, .72])
    today = pd.Timestamp.now(tz="UTC").normalize()
    base = today - pd.Timedelta(days=150)
    matches = []
    # double round robin, randomized round dates, no duplicated same-day fixture for one team.
    pairs = [(i, j) for i in range(8) for j in range(8) if i != j]
    rng.shuffle(pairs)
    completed = pairs[:52]
    for k, (i, j) in enumerate(completed):
        date = base + pd.Timedelta(days=int(k * 2.45))
        g_h = int(rng.poisson(1.48 * strengths[i] / strengths[j] ** 0.75))
        g_a = int(rng.poisson(1.12 * strengths[j] / strengths[i] ** 0.75))
        matches.append({"date": date, "home_team": teams[i], "away_team": teams[j], "home_goals": g_h, "away_goals": g_a})
    fixtures = []
    for i, j in pairs[52:]:
        fixtures.append({"home": teams[i], "away": teams[j], "date": today + pd.Timedelta(days=2 + len(fixtures))})
    return prepare_history(pd.DataFrame(matches)), fixtures
