"""Centralized configuration for the Football AI Analyst.

All tunable constants live here so models, data adapters and the UI share a
single source of truth. Values marked "modelling assumption" are NOT measured
statistics; they are defaults that should be calibrated on real data.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

# --- General ---------------------------------------------------------------
TIMEZONE = ZoneInfo("Europe/Athens")
DEFAULT_RANDOM_SEED = 42

# --- Data sources ----------------------------------------------------------
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_TIMEOUT_SECONDS = 18
FOOTBALL_DATA_TOKEN_ENV = "FOOTBALL_DATA_TOKEN"
CACHE_TTL_COMPETITIONS = 3600
CACHE_TTL_MATCHES = 900

# --- football-data.org v4: real-data loader (data_provider.fetch_real_data) --
FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"
# Free-tier league competitions (display name -> API code). Champions League is
# also free but is a cup format, so it is not offered for league modelling.
FOOTBALL_DATA_COMPETITIONS = {
    "Premier League": "PL",
    "Bundesliga": "BL1",
    "La Liga": "PD",
    "Serie A": "SA",
    "Ligue 1": "FL1",
    "Eredivisie": "DED",
    "Primeira Liga": "PPL",
    "Championship": "ELC",
}
API_RATE_LIMIT_DELAY = 6.5        # seconds between requests (free tier: 10 requests/minute)
API_MAX_RETRIES = 3               # retries after HTTP 429
API_RETRY_MAX_WAIT = 65.0         # cap (seconds) for a single 429 back-off
MAX_SEASONS = 3                   # past seasons offered for model training
REAL_DATA_CACHE_DIR = "~/.cache/football_ai"
REAL_DATA_CACHE_TTL = 3600        # seconds

# --- Poisson / team-strength model ----------------------------------------
POISSON_MAX_GOALS = 12            # score grid is 0..12 for each team (normalized)
RECENCY_MAX_GAMES = 12            # most recent matches used per team
RECENCY_DECAY = 0.89              # weight multiplier per older match
SHRINK_PSEUDO_GENERAL = 6         # pseudo-matches pulling general form to league mean
SHRINK_PSEUDO_VENUE = 5           # pseudo-matches pulling venue form to league mean
VENUE_SHARE_CAP = 0.75            # max weight of venue-specific form
LEAGUE_HOME_GOALS_BOUNDS = (0.6, 3.0)
LEAGUE_AWAY_GOALS_BOUNDS = (0.45, 2.7)
LAMBDA_BOUNDS = (0.2, 4.5)
MIN_MATCHES_TOTAL = 16
MIN_MATCHES_PER_TEAM = 4
BACKTEST_MIN_HISTORY = 50
BACKTEST_MIN_EVALUATED = 10
BACKTEST_DEFAULT_MATCHES = 60

# --- Ensemble (Poisson + Dixon-Coles + Elo) --------------------------------
ENSEMBLE_MIN_MATCHES = 20         # below this many past predictions -> equal weights
# Modelling assumption: strength of the pull toward equal weights, expressed as
# "pseudo-matches" (penalty = ENSEMBLE_PRIOR_MATCHES / n * sum (w - 1/M)^2).
ENSEMBLE_PRIOR_MATCHES = 10

# --- Markets ---------------------------------------------------------------
OVER_UNDER_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)

# --- Monte Carlo simulation -----------------------------------------------
MC_MIN_SIMULATIONS = 10_000
MC_MAX_SIMULATIONS = 500_000
MC_DEFAULT_SIMULATIONS = 100_000
MC_SIMULATION_STEP = 10_000
MC_CONFIDENCE_LEVEL = 0.95
MC_TOP_SCORES = 10
MC_HEATMAP_MAX_GOALS = 6          # heatmap shows 0..6 goals per team
# Modelling assumption: probability that any single goal falls in the first
# half. Each goal is assigned independently (binomial split). Calibrate this
# from your own historical half-time data before relying on HT markets.
MC_FIRST_HALF_GOAL_SHARE = 0.45
