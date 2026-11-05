"""Match-history cleaning and API payload conversion."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

REQUIRED = ("date", "home_team", "away_team", "home_goals", "away_goals")


def clean_name(name: str) -> str:
    """Accent/case-insensitive normalization used for fuzzy team matching."""
    text = unicodedata.normalize("NFKD", str(name)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w ]+", " ", text)
    return " ".join(text.split())


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a results table.

    Rows with missing (NULL) values are dropped rather than coerced to 0, so a
    missing score is never mistaken for a 0-0 result.
    """
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
    """Convert football-data.org v4 match objects to the history format.

    Only FINISHED matches with a non-null full-time score are kept.
    """
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
