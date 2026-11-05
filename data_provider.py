"""Read-only football-data.org v4 integration.

Two layers live here:

* ``get_json`` / ``competitions`` / ``competition_matches``: the original thin
  helpers (kept unchanged for backwards compatibility).
* ``FootballDataClient`` / ``fetch_real_data``: the real-data loader used to
  train the models on past seasons. It throttles requests for the free tier
  (10 requests/minute), retries HTTP 429, caches each competition-season on
  disk for one hour and converts results to the column layout of
  ``get_demo_data()`` (Date, HomeTeam, AwayTeam, FTHG, FTAG).
"""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Iterable

import pandas as pd
import requests

import config
from utils.data import prepare_history

BASE = "https://api.football-data.org/v4"

class FootballAPIError(RuntimeError):
    pass


def get_json(path: str, token: str, params: dict | None = None) -> dict:
    if not token or not token.strip():
        raise FootballAPIError("Χρειάζεται API token.")
    try:
        response = requests.get(BASE + path, params=params, headers={"X-Auth-Token": token.strip()}, timeout=18)
    except requests.RequestException as exc:
        raise FootballAPIError(f"Πρόβλημα σύνδεσης με το API: {exc}") from exc
    if response.status_code != 200:
        try:
            detail = (response.json() or {}).get("message") or (response.json() or {}).get("error", "")
        except ValueError:
            detail = ""
        if response.status_code == 429:
            raise FootballAPIError("Υπέρβαση API rate limit (429). Περίμενε πριν ξαναφορτώσεις.")
        if response.status_code in (401, 403):
            raise FootballAPIError(f"Κωδικός {response.status_code}: έλεγξε token ή συνδρομητική πρόσβαση. {detail}")
        raise FootballAPIError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def competitions(token: str) -> list[dict]:
    return (get_json("/competitions", token).get("competitions") or [])


def competition_matches(token: str, code: str) -> list[dict]:
    return get_json(f"/competitions/{code}/matches", token).get("matches") or []


# =============================================================================
# Real-data loader (football-data.org v4)
# =============================================================================
REAL_DATA_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
MATCH_KEYS = ("home_team", "away_team", "home_goals", "away_goals", "date", "status")
UPCOMING_STATUSES = ("SCHEDULED", "TIMED")

MSG_EMPTY_KEY = "Δεν δόθηκε API key. Συμπληρώστε το κλειδί σας από το football-data.org ή αφήστε κενό για demo δεδομένα."
MSG_INVALID_KEY = "Μη έγκυρο API key. Ελέγξτε το κλειδί σας."
MSG_NETWORK = ("Σφάλμα δικτύου κατά τη σύνδεση με το football-data.org. "
               "Ελέγξτε τη σύνδεσή σας και δοκιμάστε ξανά σε λίγο.")
MSG_RATE_LIMIT = ("Ξεπεράστηκε το όριο αιτημάτων του football-data.org (HTTP 429) και μετά από "
                  "{retries} επαναλήψεις. Περιμένετε ένα λεπτό και δοκιμάστε ξανά.")
MSG_NO_MATCHES = "Δεν βρέθηκαν ολοκληρωμένοι αγώνες για {code} ({seasons})."

_CODE_RE = re.compile(r"^[A-Z0-9]{2,6}$")


class FootballDataError(ValueError):
    """Any failure of the real-data loader. The message is user-facing Greek text."""


def current_season(today: date | None = None) -> int:
    """Start year of the current European season (football-data.org ``season`` filter).

    Assumption: European leagues start in July/August, so from July onwards the
    season that starts this calendar year is the current one.
    """
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_label(season: int) -> str:
    """2025 -> '2025/26'."""
    return f"{season}/{(season + 1) % 100:02d}"


def available_seasons(n: int = config.MAX_SEASONS, today: date | None = None) -> list[int]:
    """The ``n`` most recent seasons, newest first."""
    start = current_season(today)
    return [start - k for k in range(max(int(n), 1))]


def get_demo_data() -> pd.DataFrame:
    """Fictional demo results in the real-data column layout (Date, HomeTeam, AwayTeam, FTHG, FTAG)."""
    from engine import generate_demo   # local import: engine is heavy and imports this module's siblings

    history, _ = generate_demo()
    history = prepare_history(history)
    return pd.DataFrame({
        "Date": history["date"], "HomeTeam": history["home_team"], "AwayTeam": history["away_team"],
        "FTHG": history["home_goals"], "FTAG": history["away_goals"],
    })


def to_history(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the real-data layout (Date, HomeTeam, ...) to the models' history format."""
    missing = [c for c in REAL_DATA_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Λείπουν στήλες: {', '.join(missing)}")
    return prepare_history(df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals",
    }))


def _normalize_match(m: dict) -> dict:
    full_time = (m.get("score") or {}).get("fullTime") or {}
    return {
        "home_team": (m.get("homeTeam") or {}).get("name"),
        "away_team": (m.get("awayTeam") or {}).get("name"),
        "home_goals": full_time.get("home"),
        "away_goals": full_time.get("away"),
        "date": m.get("utcDate"),
        "status": m.get("status"),
    }


def _is_complete_result(m: dict) -> bool:
    return (m.get("status") == "FINISHED" and bool(m.get("home_team")) and bool(m.get("away_team"))
            and m.get("home_goals") is not None and m.get("away_goals") is not None and bool(m.get("date")))


class FootballDataClient:
    """Client for football-data.org v4 API."""

    def __init__(self, api_key: str, *, base_url: str = config.FOOTBALL_DATA_API_URL,
                 cache_dir: str | os.PathLike | None = None, cache_ttl: float = config.REAL_DATA_CACHE_TTL,
                 rate_limit_delay: float = config.API_RATE_LIMIT_DELAY,
                 max_retries: int = config.API_MAX_RETRIES,
                 timeout: float = config.FOOTBALL_DATA_TIMEOUT_SECONDS):
        if not isinstance(api_key, str) or not api_key.strip():
            raise FootballDataError(MSG_EMPTY_KEY)
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir if cache_dir is not None else config.REAL_DATA_CACHE_DIR).expanduser()
        self.cache_ttl = float(cache_ttl)
        self.rate_limit_delay = float(rate_limit_delay)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.requests_made = 0
        self._last_request: float | None = None

    # ------------------------------------------------------------------ HTTP
    def _throttle(self) -> None:
        if self._last_request is None:
            return
        wait = self.rate_limit_delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _retry_wait(self, response, attempt: int) -> float:
        # football-data.org sends the seconds until the counter resets in this
        # header (undocumented, so it is optional); otherwise back off exponentially.
        header = (response.headers or {}).get("X-RequestCounter-Reset")
        try:
            wait = float(header) + 1.0 if header is not None else self.rate_limit_delay * 2 ** attempt
        except (TypeError, ValueError):
            wait = self.rate_limit_delay * 2 ** attempt
        return min(max(wait, 0.0), config.API_RETRY_MAX_WAIT)

    @staticmethod
    def _api_message(response) -> str:
        try:
            body = response.json() or {}
        except ValueError:
            return ""
        return str(body.get("message") or body.get("error") or "") if isinstance(body, dict) else ""

    def _request(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = requests.get(url, params=params, headers={"X-Auth-Token": self.api_key},
                                        timeout=self.timeout)
            except requests.RequestException as exc:
                raise FootballDataError(f"{MSG_NETWORK} (Λεπτομέρειες: {type(exc).__name__})") from exc
            finally:
                self._last_request = time.monotonic()
                self.requests_made += 1

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise FootballDataError("Το football-data.org επέστρεψε μη έγκυρη απάντηση. Δοκιμάστε ξανά.") from exc
            if status == 429:
                if attempt < self.max_retries:
                    time.sleep(self._retry_wait(response, attempt))
                    continue
                raise FootballDataError(MSG_RATE_LIMIT.format(retries=self.max_retries))
            detail = self._api_message(response)
            if status == 403 or (status in (400, 401) and "token" in detail.lower()):
                raise FootballDataError(MSG_INVALID_KEY)
            if status == 404:
                raise FootballDataError(f"Η διοργάνωση ή η σεζόν δεν βρέθηκε στο football-data.org (HTTP 404). {detail}".strip())
            raise FootballDataError(f"Σφάλμα football-data.org (HTTP {status}). {detail} Δοκιμάστε ξανά αργότερα.".replace("  ", " ").strip())
        raise AssertionError("unreachable")

    # ----------------------------------------------------------------- cache
    def _cache_path(self, competition_code: str, season: int) -> Path:
        return self.cache_dir / f"{competition_code}_{season}.json"

    def _read_cache(self, competition_code: str, season: int) -> list[dict] | None:
        path = self._cache_path(competition_code, season)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - float(payload["fetched_at"])
            if 0 <= age < self.cache_ttl and isinstance(payload["matches"], list):
                return payload["matches"]
        except (OSError, ValueError, KeyError, TypeError):
            pass                       # missing, corrupt or stale -> refetch
        return None

    def _write_cache(self, competition_code: str, season: int, matches: list[dict]) -> None:
        path = self._cache_path(competition_code, season)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fetched_at": time.time(), "competition": competition_code,
                                       "season": season, "matches": matches}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)          # atomic: never leave a half-written cache file
        except OSError:
            pass                       # caching is best-effort

    # ------------------------------------------------------------- endpoints
    @staticmethod
    def _validate(competition_code: str, season: int) -> tuple[str, int]:
        code = str(competition_code).strip().upper()
        if not _CODE_RE.match(code):
            raise FootballDataError(f"Μη έγκυρος κωδικός διοργάνωσης: «{competition_code}».")
        try:
            season = int(season)
        except (TypeError, ValueError) as exc:
            raise FootballDataError(f"Μη έγκυρη σεζόν: «{season}».") from exc
        if not 1900 <= season <= 2100:
            raise FootballDataError(f"Μη έγκυρη σεζόν: «{season}».")
        return code, season

    def get_season_matches(self, competition_code: str, season: int) -> list[dict]:
        """All matches of a season (any status), normalised to MATCH_KEYS. Cached for ``cache_ttl``."""
        code, season = self._validate(competition_code, season)
        cached = self._read_cache(code, season)
        if cached is not None:
            return cached
        payload = self._request(f"/competitions/{code}/matches", {"season": season})
        matches = [_normalize_match(m) for m in (payload.get("matches") or []) if isinstance(m, dict)]
        self._write_cache(code, season, matches)
        return matches

    def get_matches(self, competition_code: str, season: int) -> list[dict]:
        """FINISHED matches with a full-time score: dicts with home_team, away_team,
        home_goals, away_goals, date (ISO UTC), status."""
        return [m for m in self.get_season_matches(competition_code, season) if _is_complete_result(m)]

    def get_upcoming(self, competition_code: str, season: int, now: pd.Timestamp | None = None) -> list[dict]:
        """Scheduled fixtures of a season as {"home", "away", "date"} (uses the same cache)."""
        now = now if now is not None else pd.Timestamp.now(tz="UTC")
        items = []
        for m in self.get_season_matches(competition_code, season):
            when = pd.to_datetime(m.get("date"), utc=True, errors="coerce")
            if (m.get("status") in UPCOMING_STATUSES and m.get("home_team") and m.get("away_team")
                    and not pd.isna(when) and when >= now):
                items.append({"home": m["home_team"], "away": m["away_team"], "date": when})
        return sorted(items, key=lambda item: item["date"])

    def get_standings(self, competition_code: str, season: int) -> dict:
        """Overall league table: {"competition", "season", "table": [{position, team, played, ...}]}."""
        code, season = self._validate(competition_code, season)
        payload = self._request(f"/competitions/{code}/standings", {"season": season})
        total = next((s for s in payload.get("standings") or [] if s.get("type") == "TOTAL"), None)
        table = [{
            "position": row.get("position"), "team": (row.get("team") or {}).get("name"),
            "played": row.get("playedGames"), "won": row.get("won"), "draw": row.get("draw"),
            "lost": row.get("lost"), "points": row.get("points"), "goals_for": row.get("goalsFor"),
            "goals_against": row.get("goalsAgainst"), "goal_difference": row.get("goalDifference"),
        } for row in (total or {}).get("table") or []]
        return {"competition": code, "season": season, "table": table}


def fetch_real_data(api_key: str, competition_code: str, seasons: Iterable[int], *,
                    progress_callback: Callable[[int, int, int], None] | None = None,
                    client: FootballDataClient | None = None) -> pd.DataFrame:
    """
    Fetch finished matches for the given competition/seasons.
    Returns DataFrame with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG
    (same column names as get_demo_data() so the rest of the app works unchanged).
    Caches results to ~/.cache/football_ai/{competition_code}_{season}.json (ttl: 1 hour).
    If the API key is empty or invalid, raises a clear ValueError with a Greek message.

    ``progress_callback(done, total, season)`` is called after each season.
    An empty DataFrame (with the columns above) is returned when no finished
    matches exist.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise FootballDataError(MSG_EMPTY_KEY)
    season_list = sorted({int(s) for s in seasons})
    if not season_list:
        raise FootballDataError("Επιλέξτε τουλάχιστον μία σεζόν.")
    client = client or FootballDataClient(api_key)

    rows = []
    for done, season in enumerate(season_list, start=1):
        for m in client.get_matches(competition_code, season):
            rows.append((m["date"], m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]))
        if progress_callback is not None:
            progress_callback(done, len(season_list), season)

    df = pd.DataFrame(rows, columns=list(REAL_DATA_COLUMNS))
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.astype({"FTHG": "int64", "FTAG": "int64", "HomeTeam": "object", "AwayTeam": "object"})
    df = (df.sort_values("Date", kind="stable")
            .drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
            .reset_index(drop=True))
    return df
