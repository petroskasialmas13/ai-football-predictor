"""Leakage-free comparative backtest: Poisson engine vs Dixon-Coles vs league baseline.

For every evaluated match only matches played strictly before it are used.

Dixon-Coles refitting is done in blocks to keep runtime reasonable: evaluated
matches are grouped into consecutive blocks spanning at most ``refit_days``
days; the model is fitted once per block on data strictly before the FIRST
match of the block. Every match in a block is on/after that date, so no data
from on/after a predicted match ever enters its prediction. (Because the time
decay is exponential, measuring ages from the block start instead of the match
date gives identical parameters; the only cost is that results inside the
block's own window are not yet used — a slightly conservative choice.)

All models are scored on the SAME set of matches: if any model cannot predict a
match (e.g. too little history), the match is skipped for all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import warnings

import numpy as np
import pandas as pd

import config
from engine import analyze
from models.dixon_coles import DixonColesModel
from utils.data import prepare_history

LOG_LOSS_EPS = 1e-15
MODEL_NAMES = ("Poisson", "Dixon-Coles", "Βάση")


def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2


def multiclass_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean over matches of sum_k (p_k - y_k)^2 (range 0..2, lower is better)."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(outcomes, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = LOG_LOSS_EPS) -> float:
    """Mean of -log(p of the actual outcome), probabilities clipped to [eps, 1]."""
    probs = np.asarray(probs, dtype=float)
    p_true = probs[np.arange(len(probs)), np.asarray(outcomes, dtype=int)]
    return float(np.mean(-np.log(np.clip(p_true, eps, 1.0))))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(np.asarray(probs), axis=1) == np.asarray(outcomes, dtype=int)))


def binary_brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


@dataclass
class ComparisonResult:
    metrics: pd.DataFrame                 # one row per model
    predictions: pd.DataFrame             # one row per evaluated match (all models)
    evaluated: int
    skipped: int
    skip_reasons: dict[str, int]
    dc_refits: int
    dc_not_converged: int
    elapsed_seconds: float
    refit_days: int
    notes: list[str] = field(default_factory=list)


def _baseline(past: pd.DataFrame) -> tuple[np.ndarray, float]:
    """League frequencies from older matches only, with +1 (Laplace) smoothing."""
    n_home = int((past.home_goals > past.away_goals).sum())
    n_draw = int((past.home_goals == past.away_goals).sum())
    n_away = int((past.home_goals < past.away_goals).sum())
    p1x2 = (np.array([n_home, n_draw, n_away], dtype=float) + 1) / (len(past) + 3)
    n_over = int(((past.home_goals + past.away_goals) > 2.5).sum())
    p_over = (n_over + 1) / (len(past) + 2)
    return p1x2, float(p_over)


def _blocks(dates: pd.Series, refit_days: int) -> list[list[int]]:
    """Group positional indices of chronologically sorted dates into refit blocks."""
    blocks: list[list[int]] = []
    start = None
    for pos, d in enumerate(dates):
        if start is None or d >= start + pd.Timedelta(days=refit_days):
            blocks.append([])
            start = d
        blocks[-1].append(pos)
    return blocks


def compare_models(raw: pd.DataFrame, n_matches: int = config.BACKTEST_DEFAULT_MATCHES,
                   refit_days: int = 7, dc_kwargs: dict | None = None) -> ComparisonResult:
    if int(refit_days) < 1:
        raise ValueError("Το διάστημα επανεκπαίδευσης πρέπει να είναι τουλάχιστον 1 ημέρα.")
    if int(n_matches) < 1:
        raise ValueError("Ο αριθμός αγώνων ελέγχου πρέπει να είναι τουλάχιστον 1.")
    started = time.perf_counter()
    df = prepare_history(raw)
    if len(df) < config.BACKTEST_MIN_HISTORY:
        raise ValueError(f"Για backtest χρειάζονται τουλάχιστον {config.BACKTEST_MIN_HISTORY} ιστορικά παιχνίδια "
                         f"(υπάρχουν {len(df)}).")
    eval_df = df.tail(int(n_matches)).reset_index(drop=True)
    dc_kwargs = dc_kwargs or {}

    rows: list[dict] = []
    reasons: dict[str, int] = {}
    refits = not_converged = 0

    def skip(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for block in _blocks(eval_df.date, int(refit_days)):
        block_start = eval_df.date.iloc[block[0]]
        dc = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dc = DixonColesModel(**dc_kwargs).fit(df, reference_time=block_start)
            refits += 1
            if not dc.summary.converged:
                not_converged += 1
        except ValueError:
            dc = None

        for pos in block:
            m = eval_df.iloc[pos]
            past = df[df.date < m.date]           # strictly before this match
            if dc is None:
                skip("Dixon-Coles: ανεπαρκή δεδομένα")
                continue
            try:
                dc_pred = dc.predict(m.home_team, m.away_team)
            except ValueError:
                skip("Dixon-Coles: ομάδα χωρίς προηγούμενους αγώνες")
                continue
            try:
                po = analyze(past, m.home_team, m.away_team, m.date)
            except ValueError:
                skip("Poisson: ανεπαρκή δεδομένα")
                continue
            base, base_over = _baseline(past)
            rows.append({
                "date": m.date, "home_team": m.home_team, "away_team": m.away_team,
                "home_goals": int(m.home_goals), "away_goals": int(m.away_goals),
                "outcome": outcome_index(int(m.home_goals), int(m.away_goals)),
                "over25": int(m.home_goals + m.away_goals > 2.5),
                "Poisson": [po["probabilities"][k] for k in ("1", "X", "2")],
                "Poisson_over25": po["probabilities"]["Over 2.5"],
                "Dixon-Coles": [dc_pred.probabilities[k] for k in ("1", "X", "2")],
                "Dixon-Coles_over25": dc_pred.probabilities["Over 2.5"],
                "Βάση": base.tolist(), "Βάση_over25": base_over,
            })

    skipped = sum(reasons.values())
    if len(rows) < config.BACKTEST_MIN_EVALUATED:
        raise ValueError(f"Μόνο {len(rows)} αγώνες μπορούν να ελεγχθούν από όλα τα μοντέλα "
                         f"({skipped} αποκλείστηκαν). Χρειάζονται τουλάχιστον {config.BACKTEST_MIN_EVALUATED}.")
    preds = pd.DataFrame(rows)
    y = preds["outcome"].to_numpy()
    y_over = preds["over25"].to_numpy()
    metric_rows = []
    for name in MODEL_NAMES:
        p = np.vstack(preds[name].to_numpy())
        metric_rows.append({
            "Μοντέλο": name, "Αγώνες": len(preds),
            "Ακρίβεια 1Χ2": accuracy(p, y), "Brier 1Χ2": multiclass_brier(p, y),
            "Log loss 1Χ2": log_loss(p, y), "Brier Over 2.5": binary_brier(preds[f"{name}_over25"], y_over),
        })
    return ComparisonResult(
        metrics=pd.DataFrame(metric_rows), predictions=preds, evaluated=len(preds), skipped=skipped,
        skip_reasons=reasons, dc_refits=refits, dc_not_converged=not_converged,
        elapsed_seconds=time.perf_counter() - started, refit_days=int(refit_days),
    )
