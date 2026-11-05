"""Tests for the football-data.org v4 real-data loader (data_provider.py).

No network access: ``requests.get`` is patched with unittest.mock, ``time.sleep``
is patched so rate-limit waits are instant, and the disk cache lives in a
per-test temporary directory. All match data below is fictional.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
import data_provider
from data_provider import (REAL_DATA_COLUMNS, FootballDataClient, available_seasons, current_season,
                           fetch_real_data, get_demo_data, season_label, to_history)


def api_match(home, away, hg, ag, date, status="FINISHED"):
    return {"utcDate": date, "status": status, "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "score": {"fullTime": {"home": hg, "away": ag}}}


def response(status=200, matches=None, headers=None, body=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = body if body is not None else {"matches": matches or []}
    return r


SEASON_2024 = [
    api_match("Alpha FC", "Beta FC", 2, 1, "2024-08-17T14:00:00Z"),
    api_match("Gamma FC", "Delta FC", 0, 0, "2024-08-18T16:30:00Z"),
    api_match("Beta FC", "Gamma FC", 1, 3, "2024-08-24T14:00:00Z"),
    api_match("Delta FC", "Alpha FC", None, None, "2024-08-25T14:00:00Z", status="POSTPONED"),
    api_match("Alpha FC", "Gamma FC", 1, 0, "2024-08-31T14:00:00Z", status="IN_PLAY"),
    api_match("Beta FC", "Delta FC", None, None, "2025-05-25T14:00:00Z", status="SCHEDULED"),
]
SEASON_2025 = [
    api_match("Beta FC", "Alpha FC", 0, 2, "2025-08-16T14:00:00Z"),
    api_match("Delta FC", "Gamma FC", 4, 1, "2025-08-17T14:00:00Z"),
]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Temp cache dir + instant sleeps for every test."""
    monkeypatch.setattr(config, "REAL_DATA_CACHE_DIR", str(tmp_path / "cache"))
    sleep = MagicMock()
    monkeypatch.setattr(data_provider.time, "sleep", sleep)
    return sleep


# --------------------------------------------------------------------- parsing
def test_valid_response_parsed_to_expected_columns():
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)) as get:
        df = fetch_real_data("KEY123", "PL", [2024])
    assert list(df.columns) == list(REAL_DATA_COLUMNS) == ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    assert len(df) == 3
    first = df.iloc[0]
    assert (first.HomeTeam, first.AwayTeam, first.FTHG, first.FTAG) == ("Alpha FC", "Beta FC", 2, 1)
    assert first.Date == pd.Timestamp("2024-08-17T14:00:00Z")
    assert str(df.Date.dt.tz) == "UTC" and df.Date.is_monotonic_increasing
    assert df.FTHG.dtype == "int64" and df.FTAG.dtype == "int64"
    # Correct endpoint, season filter and auth header.
    args, kwargs = get.call_args
    assert args[0] == "https://api.football-data.org/v4/competitions/PL/matches"
    assert kwargs["params"] == {"season": 2024}
    assert kwargs["headers"] == {"X-Auth-Token": "KEY123"}
    assert kwargs["timeout"] > 0


def test_only_finished_matches_included():
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)):
        matches = FootballDataClient("KEY").get_matches("PL", 2024)
    assert len(matches) == 3
    assert {m["status"] for m in matches} == {"FINISHED"}
    assert set(matches[0]) == {"home_team", "away_team", "home_goals", "away_goals", "date", "status"}


def test_finished_without_score_is_dropped_not_zero():
    broken = [api_match("Alpha FC", "Beta FC", None, None, "2024-08-17T14:00:00Z")]
    with patch("data_provider.requests.get", return_value=response(matches=broken)):
        df = fetch_real_data("KEY", "PL", [2024])
    assert df.empty


# ---------------------------------------------------------------------- errors
def test_http_403_raises_greek_value_error():
    body = {"message": "The resource you are looking for is restricted.", "errorCode": 403}
    with patch("data_provider.requests.get", return_value=response(403, body=body)):
        with pytest.raises(ValueError) as exc:
            fetch_real_data("BADKEY", "PL", [2024])
    assert str(exc.value) == "Μη έγκυρο API key. Ελέγξτε το κλειδί σας."


def test_http_429_retries_then_succeeds(isolated):
    replies = [response(429), response(429), response(matches=SEASON_2024)]
    with patch("data_provider.requests.get", side_effect=replies) as get:
        df = fetch_real_data("KEY", "PL", [2024])
    assert get.call_count == 3
    assert len(df) == 3
    # Exponential back-off: the two 429 waits double (6.5 s, 13 s).
    waits = [c.args[0] for c in isolated.call_args_list if c.args[0] >= config.API_RATE_LIMIT_DELAY]
    assert waits[:2] == [config.API_RATE_LIMIT_DELAY, 2 * config.API_RATE_LIMIT_DELAY]


def test_http_429_gives_up_after_three_retries():
    with patch("data_provider.requests.get", return_value=response(429)) as get:
        with pytest.raises(ValueError, match="429"):
            fetch_real_data("KEY", "PL", [2024])
    assert get.call_count == 1 + config.API_MAX_RETRIES == 4


def test_empty_api_key_raises():
    with patch("data_provider.requests.get") as get:
        for key in ("", "   "):
            with pytest.raises(ValueError, match="API key"):
                fetch_real_data(key, "PL", [2024])
    get.assert_not_called()


def test_network_error_is_greek_value_error():
    import requests
    with patch("data_provider.requests.get", side_effect=requests.ConnectionError("down")):
        with pytest.raises(ValueError, match="Σφάλμα δικτύου"):
            fetch_real_data("KEY", "PL", [2024])


def test_invalid_competition_code_rejected_before_request():
    with patch("data_provider.requests.get") as get:
        with pytest.raises(ValueError):
            fetch_real_data("KEY", "../etc", [2024])
    get.assert_not_called()


# ----------------------------------------------------------------------- cache
def test_cache_used_on_second_call(tmp_path):
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)) as get:
        a = fetch_real_data("KEY", "PL", [2024])
        b = fetch_real_data("KEY", "PL", [2024])
    assert get.call_count == 1
    pd.testing.assert_frame_equal(a, b)
    cache_file = tmp_path / "cache" / "PL_2024.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8"))["season"] == 2024


def test_cache_expires_after_ttl(tmp_path):
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)) as get:
        fetch_real_data("KEY", "PL", [2024])
        path = tmp_path / "cache" / "PL_2024.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fetched_at"] = time.time() - config.REAL_DATA_CACHE_TTL - 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        fetch_real_data("KEY", "PL", [2024])
    assert get.call_count == 2


def test_corrupt_cache_is_refetched(tmp_path):
    path = tmp_path / "cache" / "PL_2024.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)) as get:
        df = fetch_real_data("KEY", "PL", [2024])
    assert get.call_count == 1 and len(df) == 3


# ----------------------------------------------------------------- empty/multi
def test_no_finished_matches_returns_empty_frame():
    scheduled = [api_match("Alpha FC", "Beta FC", None, None, "2026-12-01T20:00:00Z", status="SCHEDULED")]
    for matches in (scheduled, []):
        with patch("data_provider.requests.get", return_value=response(matches=matches)):
            df = fetch_real_data("KEY", "DED", [2026 if matches else 2025])
        assert df.empty
        assert list(df.columns) == ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def test_multiple_seasons_concatenated():
    def fake_get(url, params=None, **kwargs):
        return response(matches={2024: SEASON_2024, 2025: SEASON_2025}[params["season"]])

    with patch("data_provider.requests.get", side_effect=fake_get) as get:
        df = fetch_real_data("KEY", "BL1", [2025, 2024])
    assert get.call_count == 2
    assert [c.kwargs["params"]["season"] for c in get.call_args_list] == [2024, 2025]
    assert len(df) == 3 + 2
    assert df.Date.is_monotonic_increasing
    assert df.iloc[-1].HomeTeam == "Delta FC" and df.iloc[-1].FTHG == 4


def test_rate_limit_delay_between_requests(isolated):
    def fake_get(url, params=None, **kwargs):
        return response(matches=SEASON_2024 if params["season"] == 2024 else SEASON_2025)

    with patch("data_provider.requests.get", side_effect=fake_get):
        fetch_real_data("KEY", "PL", [2024, 2025])
    assert isolated.call_count == 1                 # one wait, before the 2nd request only
    assert 0 < isolated.call_args.args[0] <= config.API_RATE_LIMIT_DELAY


# --------------------------------------------------------------- integration
def test_output_feeds_models_like_demo_data():
    with patch("data_provider.requests.get", return_value=response(matches=SEASON_2024)):
        real = fetch_real_data("KEY", "PL", [2024])
    demo = get_demo_data()
    assert list(demo.columns) == list(real.columns)
    history = to_history(real)
    assert list(history.columns)[:5] == ["date", "home_team", "away_team", "home_goals", "away_goals"]
    assert len(history) == 3


def test_season_helpers():
    from datetime import date
    assert current_season(date(2026, 11, 5)) == 2026
    assert current_season(date(2026, 3, 1)) == 2025
    assert available_seasons(2, date(2026, 11, 5)) == [2026, 2025]
    assert season_label(2025) == "2025/26" and season_label(2099) == "2099/00"


def test_config_values():
    assert config.FOOTBALL_DATA_API_URL == "https://api.football-data.org/v4"
    assert config.FOOTBALL_DATA_COMPETITIONS == {
        "Premier League": "PL", "Bundesliga": "BL1", "La Liga": "PD", "Serie A": "SA",
        "Ligue 1": "FL1", "Eredivisie": "DED", "Primeira Liga": "PPL", "Championship": "ELC"}
    assert config.API_RATE_LIMIT_DELAY == 6.5 and 60 / config.API_RATE_LIMIT_DELAY < 10
    assert config.MAX_SEASONS == 3
