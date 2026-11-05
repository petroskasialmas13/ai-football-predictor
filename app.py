"""AI Football Predictor — personal, read-only pre-match analytics dashboard."""
from __future__ import annotations

from datetime import datetime, date, time, timedelta
from difflib import get_close_matches
from io import BytesIO
from zoneinfo import ZoneInfo
import math
import os
import re

import pandas as pd
import streamlit as st

import config
from engine import analyze, clean_name, generate_demo, api_matches_to_history, prepare_history
from data_provider import competitions, competition_matches, FootballAPIError
from models.monte_carlo import MonteCarloResult, analytical_probabilities, simulate_match
from models.dixon_coles import RHO_BOUNDS, DixonColesModel
from backtesting.compare import compare_models
import warnings
import numpy as np

ATHENS = ZoneInfo("Europe/Athens")

def api_token_default() -> str:
    env_token = os.getenv("FOOTBALL_DATA_TOKEN", "")
    if env_token:
        return env_token
    try:
        return str(st.secrets.get("FOOTBALL_DATA_TOKEN", ""))
    except Exception:
        return ""

st.set_page_config(page_title="AI Football Predictor", page_icon="⚽", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.5rem; max-width: 1250px;}
  @media (max-width: 640px) { .block-container {padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 0.8rem;} h1 {font-size: 1.75rem !important;} }
  div[data-testid="stMetric"] {border: 1px solid #334155; border-radius: 12px; padding: 12px 16px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_competitions(token: str) -> list[dict]:
    return competitions(token)


@st.cache_data(ttl=900, show_spinner=False)
def cached_matches(token: str, competition_code: str) -> list[dict]:
    return competition_matches(token, competition_code)


def athens_dt(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock).replace(tzinfo=ATHENS)


def match_options(api_games: list[dict]) -> list[dict]:
    today = pd.Timestamp.now(tz="UTC")
    items = []
    for m in api_games:
        if m.get("status") in ("FINISHED", "CANCELLED", "POSTPONED", "SUSPENDED", "IN_PLAY", "PAUSED", "LIVE"):
            continue
        h = (m.get("homeTeam") or {}).get("name")
        a = (m.get("awayTeam") or {}).get("name")
        when = pd.to_datetime(m.get("utcDate"), utc=True, errors="coerce")
        if not h or not a or pd.isna(when) or when < today:
            continue
        items.append({"home": h, "away": a, "date": when, "id": m.get("id")})
    return sorted(items, key=lambda item: item["date"])


def read_batch(input_text: str, names: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    lookup = {clean_name(name): name for name in names}
    pairs, errors = [], []
    for index, raw in enumerate(input_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\s+(?:-|–|—|vs\.?|v\.?|εναντίον)\s+", line, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            errors.append(f"Γραμμή {index}: γράψε Ομάδα Α - Ομάδα Β")
            continue
        resolved = []
        for term in parts:
            norm = clean_name(term)
            if norm in lookup:
                resolved.append(lookup[norm])
            else:
                similar = get_close_matches(norm, list(lookup), n=1, cutoff=0.87)
                if similar:
                    resolved.append(lookup[similar[0]])
                else:
                    errors.append(f"Γραμμή {index}: δεν βρέθηκε ομάδα «{term.strip()}» στο επιλεγμένο πρωτάθλημα.")
                    break
        if len(resolved) == 2:
            if resolved[0] == resolved[1]:
                errors.append(f"Γραμμή {index}: δύο ίδιες ομάδες.")
            else:
                pairs.append((resolved[0], resolved[1]))
    return list(dict.fromkeys(pairs)), errors


@st.cache_data(max_entries=32, show_spinner=False)
def cached_simulation(lambda_home: float, lambda_away: float, n_simulations: int, seed: int,
                      score_matrix: np.ndarray | None = None) -> MonteCarloResult:
    # Pure NumPy simulation (no API calls); fixed seed -> identical output for identical inputs.
    return simulate_match(lambda_home, lambda_away, n_simulations=n_simulations, seed=seed,
                          score_matrix=score_matrix)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_dixon_coles(history_df: pd.DataFrame, cutoff_iso: str) -> DixonColesModel:
    # Fitted only on matches strictly before the kickoff (enforced inside fit()).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # convergence is reported via summary.converged
        return DixonColesModel().fit(history_df, reference_time=pd.Timestamp(cutoff_iso))


def add_dixon_coles(result: dict, history_df: pd.DataFrame) -> dict:
    """Attach a Dixon-Coles prediction (or a Greek error message) to a Poisson result dict."""
    try:
        model = cached_dixon_coles(history_df, pd.Timestamp(result["kickoff"]).isoformat())
        result["dixon_coles"] = {"prediction": model.predict(result["home"], result["away"]),
                                 "summary": model.summary, "home_advantage": model.home_advantage,
                                 "rho": model.rho}
        result.pop("dixon_coles_error", None)
    except ValueError as exc:
        result.pop("dixon_coles", None)
        result["dixon_coles_error"] = str(exc)
    return result


def greek_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


MC_COMPARISON_MARKETS = ("1", "X", "2", "1X", "X2", "12",
                         "Over 0.5", "Under 0.5", "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
                         "Over 3.5", "Under 3.5", "Over 4.5", "Under 4.5", "GG", "NG")
MC_HALFTIME_MARKETS = {"HT 1": "Ημίχρονο 1", "HT X": "Ημίχρονο X", "HT 2": "Ημίχρονο 2",
                       "HT Over 0.5": "Over 0.5 στο 1ο ημίχρονο", "HT Over 1.5": "Over 1.5 στο 1ο ημίχρονο",
                       "2H Over 0.5": "Over 0.5 στο 2ο ημίχρονο", "2H Over 1.5": "Over 1.5 στο 2ο ημίχρονο"}
MC_CI_MARKETS = ("1", "X", "2", "Over 2.5", "Under 2.5", "GG", "NG")


def mc_comparison_table(poisson: dict, analytic: dict, mc: MonteCarloResult,
                        exact_label: str = "Poisson (αναλυτικά)") -> pd.DataFrame:
    """Analytical Poisson vs simulated probabilities.

    Uses the engine's Poisson values where the engine provides them (same numbers as the other tabs),
    otherwise the closed-form values for the same λ and assumptions (Over/Under 0.5 and 4.5).
    """
    rows = []
    for market in MC_COMPARISON_MARKETS:
        exact = poisson.get(market, analytic.get(market))
        sim = mc.probabilities.get(market)
        if exact is None or sim is None:
            continue
        lo, hi = mc.intervals[market]
        rows.append({"Αγορά": market, exact_label: f"{exact:.2%}", "Monte Carlo": f"{sim:.2%}",
                     "Διαφορά (π.μ.)": f"{100 * (sim - exact):+.2f}",
                     "Αναλυτική τιμή εντός διαστήματος;": "✅ Ναι" if lo <= exact <= hi else "⚠️ Όχι"})
    return pd.DataFrame(rows)


def _heat_color(value: float, vmax: float) -> str:
    # Manual white → green gradient, so no extra plotting dependency is needed.
    t = 0.0 if vmax <= 0 else min(max(value / vmax, 0.0), 1.0)
    r, g, b = (int(255 + (c - 255) * t) for c in (22, 101, 52))
    text = "#ffffff" if t > 0.55 else "#0f172a"
    return f"background-color: rgb({r},{g},{b}); color: {text}"


def score_heatmap(mc: MonteCarloResult):
    size = mc.score_matrix.shape[0]
    frame = pd.DataFrame(mc.score_matrix * 100,
                         index=[f"Γηπεδούχος {i}" for i in range(size)],
                         columns=[f"Φιλοξ. {j}" for j in range(size)])
    vmax = float(frame.values.max())
    styler = frame.style
    mapper = getattr(styler, "map", None) or styler.applymap  # pandas < 2.1 fallback
    return mapper(lambda v: _heat_color(v, vmax)).format("{:.2f}%")


def monte_carlo_tab(result: dict) -> None:
    dc = result.get("dixon_coles")
    options = ["Poisson", "Dixon-Coles"] if dc else ["Poisson"]
    source = st.radio("Πηγή αναμενόμενων γκολ", options, horizontal=True,
                      key=f"mc_source_{result['home']}_{result['away']}")
    if not dc:
        st.caption("Το Dixon-Coles δεν είναι διαθέσιμο για αυτόν τον αγώνα (δες την καρτέλα «Dixon-Coles»).")
    matrix = None
    if source == "Dixon-Coles":
        pred = dc["prediction"]
        p = pred.probabilities
        lh, la = float(pred.lambda_home), float(pred.lambda_away)
        matrix = pred.matrix
        exact_label = "Dixon-Coles (αναλυτικά)"
        st.caption("Μαθηματικές προσομοιώσεις με NumPy. Κάθε προσομοιωμένο τελικό σκορ κληρώνεται απευθείας από "
                   "τον πίνακα σκορ του Dixon-Coles, οπότε η διόρθωση για τα χαμηλά σκορ (0-0, 1-0, 0-1, 1-1) "
                   "διατηρείται. Δεν γίνονται κλήσεις API.")
    else:
        p = result["probabilities"]
        lh, la = float(result["expected_home"]), float(result["expected_away"])
        exact_label = "Poisson (αναλυτικά)"
        st.caption("Μαθηματικές προσομοιώσεις με NumPy πάνω στα ίδια αναμενόμενα γκολ (λ) του μοντέλου Poisson. "
                   "Δεν γίνονται κλήσεις API και δεν προστίθεται νέα πληροφορία για τις ομάδες· η προσομοίωση "
                   "επαληθεύει τις αναλυτικές πιθανότητες και προσθέτει ημίχρονα και διαστήματα εμπιστοσύνης.")
    n_sims = st.slider("Αριθμός προσομοιώσεων", min_value=config.MC_MIN_SIMULATIONS,
                       max_value=config.MC_MAX_SIMULATIONS, value=config.MC_DEFAULT_SIMULATIONS,
                       step=config.MC_SIMULATION_STEP, format="%d",
                       key=f"mc_sims_{result['home']}_{result['away']}")
    seed = config.DEFAULT_RANDOM_SEED
    try:
        with st.spinner(f"Εκτέλεση {greek_int(int(n_sims))} προσομοιώσεων..."):
            mc = cached_simulation(round(lh, 6), round(la, 6), int(n_sims), seed, matrix)
            analytic = analytical_probabilities(lh, la)
            if matrix is not None:
                # Full-time exact values come from the DC matrix; analytic Poisson half-time values
                # do not include the tau correction, so they are not shown as "exact" for DC.
                analytic = {k: v for k, v in analytic.items() if not k.startswith(("HT", "2H"))}
    except ValueError as exc:
        st.error(f"Η προσομοίωση δεν ολοκληρώθηκε: {exc}")
        return

    conf = f"{mc.confidence:.0%}"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Προσομοιώσεις", greek_int(mc.n_simulations))
    m2.metric("Μ.Ο. γκολ γηπεδούχου (προσομ.)", f"{mc.mean_home_goals:.2f}", help=f"λ μοντέλου: {lh:.2f}")
    m3.metric("Μ.Ο. γκολ φιλοξενούμενου (προσομ.)", f"{mc.mean_away_goals:.2f}", help=f"λ μοντέλου: {la:.2f}")
    m4.metric("Χρόνος υπολογισμού", f"{mc.elapsed_seconds:.2f} δευτ.")
    st.caption(f"Σταθερό seed τυχαιότητας: {seed} (ίδια δεδομένα → ίδια αποτελέσματα). "
               f"Επίπεδο εμπιστοσύνης: {conf} (διάστημα Wilson).")

    st.markdown(f"**Σύγκριση: {exact_label} έναντι Monte Carlo**")
    st.dataframe(mc_comparison_table(p, analytic, mc, exact_label), hide_index=True, use_container_width=True)
    st.caption("π.μ. = ποσοστιαίες μονάδες. Μικρές διαφορές είναι αναμενόμενος στατιστικός θόρυβος και "
               "μειώνονται όσο αυξάνονται οι προσομοιώσεις. Ένα «⚠️ Όχι» μπορεί να εμφανιστεί τυχαία "
               f"(περίπου στο {1 - mc.confidence:.0%} των αγορών) και δεν σημαίνει σφάλμα του μοντέλου.")

    left, right = st.columns(2)
    with left:
        st.markdown("**10 πιθανότερα ακριβή σκορ (Monte Carlo)**")
        st.dataframe(pd.DataFrame([{"Σκορ": s["score"], "Πιθανότητα": f"{s['probability']:.2%}",
                                    f"Διάστημα {conf}": f"{s['ci_low']:.2%} – {s['ci_high']:.2%}"}
                                   for s in mc.top_scores]), hide_index=True, use_container_width=True)
    with right:
        st.markdown(f"**Διαστήματα εμπιστοσύνης {conf} για βασικές αγορές**")
        st.dataframe(pd.DataFrame([{"Αγορά": k, "Monte Carlo": f"{mc.probabilities[k]:.2%}",
                                    "Κάτω όριο": f"{mc.intervals[k][0]:.2%}",
                                    "Άνω όριο": f"{mc.intervals[k][1]:.2%}",
                                    "Εύρος (π.μ.)": f"{100 * (mc.intervals[k][1] - mc.intervals[k][0]):.2f}"}
                                   for k in MC_CI_MARKETS]), hide_index=True, use_container_width=True)
        tg = mc.total_goals_summary
        st.caption(f"Σύνολο γκολ: μέσος όρος {tg['mean']:.2f} (διάστημα {conf}: {tg['mean_ci_low']:.2f} – "
                   f"{tg['mean_ci_high']:.2f}), διάμεσος {tg['median']:.0f}, το 90% των προσομοιώσεων "
                   f"έχει από {tg['p05']:.0f} έως {tg['p95']:.0f} γκολ.")
    st.caption("Τα διαστήματα εμπιστοσύνης μετρούν μόνο τον θόρυβο της προσομοίωσης, όχι το σφάλμα "
               "του μοντέλου ή των δεδομένων.")

    st.markdown("**Χάρτης θερμότητας ακριβούς σκορ** (γραμμές: γκολ γηπεδούχου · στήλες: γκολ φιλοξενούμενου)")
    st.dataframe(score_heatmap(mc), use_container_width=True)
    st.caption(f"Σκορ εκτός πίνακα (πάνω από {mc.score_matrix.shape[0] - 1} γκολ για μία ομάδα): "
               f"{mc.score_matrix_other:.2%} των προσομοιώσεων.")

    st.markdown("**Ημίχρονο**")
    h1, h2 = st.columns(2)
    with h1:
        ht_rows = []
        for k, label in MC_HALFTIME_MARKETS.items():
            row = {"Αγορά": label}
            if k in analytic:
                row["Poisson (αναλυτικά)"] = f"{analytic[k]:.2%}"
            row["Monte Carlo"] = f"{mc.probabilities[k]:.2%}"
            row[f"Διάστημα {conf}"] = f"{mc.intervals[k][0]:.2%} – {mc.intervals[k][1]:.2%}"
            ht_rows.append(row)
        st.dataframe(pd.DataFrame(ht_rows), hide_index=True, use_container_width=True)
    with h2:
        htft = sorted(mc.halftime_fulltime.items(), key=lambda kv: kv[1], reverse=True)
        st.dataframe(pd.DataFrame([{"Ημίχρονο/Τελικό": k, "Πιθανότητα": f"{v:.2%}"} for k, v in htft]),
                     hide_index=True, use_container_width=True)
    st.warning(f"Παραδοχή μοντέλου (όχι μετρημένο στατιστικό): κάθε γκολ πέφτει στο 1ο ημίχρονο με πιθανότητα "
               f"{mc.first_half_share:.0%}, ανεξάρτητα από τα υπόλοιπα. Οι αγορές ημιχρόνου χρειάζονται "
               "βαθμονόμηση με πραγματικά δεδομένα ημιχρόνων πριν θεωρηθούν αξιόπιστες.")


DC_COMPARISON_MARKETS = ("1", "X", "2", "1X", "X2", "12", "Over 1.5", "Under 1.5",
                         "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5", "GG", "NG")


def dixon_coles_tab(result: dict) -> None:
    if "dixon_coles" not in result:
        st.warning("Το μοντέλο Dixon-Coles δεν μπόρεσε να υπολογιστεί για αυτόν τον αγώνα: "
                   f"{result.get('dixon_coles_error', 'άγνωστη αιτία')} Τα αποτελέσματα Poisson παραμένουν διαθέσιμα.")
        return
    dc = result["dixon_coles"]
    pred, summary = dc["prediction"], dc["summary"]
    p = result["probabilities"]
    st.caption("Το Dixon-Coles εκτιμά μαζί, για όλες τις ομάδες, επιθετική και αμυντική δύναμη, πλεονέκτημα έδρας "
               "και μια διόρθωση για τα χαμηλά σκορ. Οι πρόσφατοι αγώνες μετράνε περισσότερο. "
               "Χρησιμοποιούνται μόνο αγώνες πριν από την ώρα έναρξης.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Αναμ. γκολ γηπεδούχου (DC)", f"{pred.lambda_home:.2f}", f"{pred.lambda_home - result['expected_home']:+.2f} vs Poisson",
              delta_color="off")
    m2.metric("Αναμ. γκολ φιλοξενούμενου (DC)", f"{pred.lambda_away:.2f}", f"{pred.lambda_away - result['expected_away']:+.2f} vs Poisson",
              delta_color="off")
    m3.metric("Πλεονέκτημα έδρας", f"×{math.exp(dc['home_advantage']):.2f}")
    m4.metric("ρ (διόρθωση χαμηλών σκορ)", f"{pred.rho:+.3f}")

    rows = []
    for k in DC_COMPARISON_MARKETS:
        rows.append({"Αγορά": k, "Poisson": f"{p[k]:.1%}", "Dixon-Coles": f"{pred.probabilities[k]:.1%}",
                     "Διαφορά (π.μ.)": f"{100 * (pred.probabilities[k] - p[k]):+.1f}",
                     "Δίκαιη απόδοση DC": round(1 / pred.probabilities[k], 2) if pred.probabilities[k] > 0 else None})
    st.markdown("**Poisson έναντι Dixon-Coles**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**5 πιθανότερα ακριβή σκορ (Dixon-Coles)**")
        st.dataframe(pd.DataFrame([{"Σκορ": s, "Πιθανότητα": f"{v:.1%}"} for s, v in pred.top_scores]), hide_index=True)
    with right:
        st.markdown("**Τι σημαίνουν οι παράμετροι**")
        st.write(f"- **Πλεονέκτημα έδρας ×{math.exp(dc['home_advantage']):.2f}:** ο γηπεδούχος σκοράρει τόσες φορές "
                 "περισσότερα γκολ από ό,τι θα σκόραρε η ίδια ομάδα σε ουδέτερο γήπεδο.")
        st.write(f"- **ρ = {pred.rho:+.3f}:** αρνητικό ρ σημαίνει ότι τα 0-0 και 1-1 είναι λίγο συχνότερα (και τα 1-0/0-1 "
                 "λίγο σπανιότερα) από ό,τι προβλέπει το απλό Poisson. Κοντά στο 0 = σχεδόν καμία διόρθωση.")
        if pred.rho_clipped:
            st.caption(f"Το ρ του μοντέλου ({dc['rho']:+.3f}) περιορίστηκε σε {pred.rho:+.3f} για αυτόν τον αγώνα, "
                       "ώστε να μην προκύψουν αρνητικές πιθανότητες.")
    st.caption(f"Εκπαίδευση σε {summary.n_matches} αγώνες ({summary.first_match:%d/%m/%Y} – {summary.last_match:%d/%m/%Y}). "
               f"Χρονική απόσβεση ξ = {summary.xi:.5f}/ημέρα (ένας αγώνας πριν από ~{math.log(2) / summary.xi:.0f} ημέρες "
               "μετράει το μισό). Το ξ είναι η τιμή της βιβλιογραφίας (Dixon & Coles 1997, αγγλικά πρωταθλήματα) "
               "και δεν έχει ρυθμιστεί σε αυτά τα δεδομένα.")
    if min(abs(dc["rho"] - bound) for bound in RHO_BOUNDS) < 1e-3:
        st.warning(f"Το ρ έφτασε στο όριο που επιτρέπει ο αλγόριθμος ({dc['rho']:+.3f}). Συνήθως αυτό σημαίνει ότι "
                   "οι αγώνες δεν είναι αρκετοί για να εκτιμηθεί αξιόπιστα η διόρθωση χαμηλών σκορ· "
                   "οι πιθανότητες για 0-0/1-1 και για την ισοπαλία μπορεί να είναι παραμορφωμένες.")
    if not summary.converged:
        st.warning(f"Ο αλγόριθμος εκτίμησης δεν συνέκλινε πλήρως ({summary.message}). Αντιμετώπισε τα αποτελέσματα με επιφύλαξη.")


def scorecard(result: dict) -> None:
    p = result["probabilities"]
    st.subheader(f"{result['home']}  —  {result['away']}")
    st.caption(f"Χρόνος πρόβλεψης: πριν από {result['kickoff'].tz_convert(ATHENS).strftime('%d/%m/%Y %H:%M')} ώρα Ελλάδας | "
               f"Δεδομένα: {result['matches_used']} προηγούμενοι αγώνες | Ποιότητα δείγματος: {result['data_quality']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("1 · Γηπεδούχος", f"{p['1']:.1%}")
    col2.metric("X · Ισοπαλία", f"{p['X']:.1%}")
    col3.metric("2 · Φιλοξενούμενος", f"{p['2']:.1%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Αναμενόμενα γκολ γηπεδούχου", f"{result['expected_home']:.2f}")
    c2.metric("Αναμενόμενα γκολ φιλοξενούμενου", f"{result['expected_away']:.2f}")
    c3.metric("Over 2.5", f"{p['Over 2.5']:.1%}")
    c4.metric("GG", f"{p['GG']:.1%}")

    tab1, tab2, tab_dc, tab3, tab4 = st.tabs(["Προγνωστικά", "Πώς προκύπτουν", "Dixon-Coles", "Σύγκριση αποδόσεων", "Monte Carlo"])
    with tab1:
        rows = []
        markets = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5", "GG", "NG"]
        for m in markets:
            prob = p[m]
            rows.append({"Αγορά": m, "Πιθανότητα μοντέλου": f"{prob:.1%}", "Δίκαιη απόδοση (χωρίς γκανιότα)": round(1 / prob, 2) if prob else None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**5 πιθανότερα ακριβή σκορ**")
        st.dataframe(pd.DataFrame([{"Σκορ": s, "Πιθανότητα": f"{prob:.1%}"} for s, prob in result["top_scores"]]), hide_index=True)
        best_1x2 = max(("1", "X", "2"), key=lambda k: p[k])
        st.info(f"Το συχνότερο αποτέλεσμα στο μοντέλο είναι το **{best_1x2}** ({p[best_1x2]:.1%}). "
                "Μεγαλύτερη πιθανότητα δεν σημαίνει αυτομάτως συμφέρουσα στοιχηματική απόδοση.")
    with tab2:
        h = result["home_stats"]
        a = result["away_stats"]
        st.markdown(f"**Τελευταία 6:** {result['home']} → {' · '.join(result['form_home']) or '—'} · "
                    f"{result['away']} → {' · '.join(result['form_away']) or '—'}  (Ν=Νίκη, Ι=Ισοπαλία, Η=Ήττα)")
        st.write(f"{result['home']}: {h['all_games']} διαθέσιμοι αγώνες, {h['venue_games']} εντός. "
                 f"{result['away']}: {a['all_games']} διαθέσιμοι αγώνες, {a['venue_games']} εκτός.")
        st.write(f"Μέσοι όροι πρωταθλήματος: γηπεδούχος {result['league_home_goals']:.2f} γκολ, "
                 f"φιλοξενούμενος {result['league_away_goals']:.2f} γκολ.")
        st.write("Ο μηχανισμός συνδυάζει πρόσφατες επιδόσεις, επιθετική/αμυντική ισχύ, έδρα και εξομάλυνση "
                 "προς τη μέση τιμή του πρωταθλήματος. Μετά υπολογίζει τις πιθανότητες με μοντέλο Poisson.")
        if not result["h2h"].empty:
            st.markdown("**Διαθέσιμες μεταξύ τους αναμετρήσεις (πληροφοριακά, όχι ξεχωριστός βαρύς συντελεστής)**")
            recent = result["h2h"].copy().sort_values("date", ascending=False)
            recent["date"] = recent["date"].dt.strftime("%d/%m/%Y")
            st.dataframe(recent.rename(columns={"date": "Ημερομηνία", "home_team": "Γηπεδούχος", "away_team": "Φιλοξενούμενος", "home_goals": "Γκολ 1", "away_goals": "Γκολ 2"}), hide_index=True, use_container_width=True)
        else:
            st.caption("Δεν βρέθηκαν μεταξύ τους ματς στο διαθέσιμο δείγμα.")
        st.warning("Δεν έχουν ελεγχθεί τραυματισμοί, τιμωρίες, ενδεκάδες, xG, διαιτητής, καιρός ή ζωντανές αποδόσεις. "
                   "Η ποιότητα δείγματος αφορά την επάρκεια των αποτελεσμάτων, όχι βεβαιότητα πρόβλεψης.")
    with tab3:
        st.caption("Συμπλήρωσε δεκαδικές αποδόσεις από όποια πηγή θέλεις. Δεν γίνεται σύνδεση με bookmaker.")
        cols = st.columns(3)
        odds = {market: cols[i].number_input(f"Απόδοση {market}", min_value=0.0, max_value=1000.0, step=0.05, value=0.0, key=f"odd_{result['home']}_{result['away']}_{market}") for i, market in enumerate(("1", "X", "2"))}
        rows = []
        for market, odd in odds.items():
            if odd >= 1.01:
                rows.append({"Αγορά": market, "Μοντέλο": f"{p[market]:.1%}", "Απόδοση": odd,
                             "Ενδεικτικό EV ανά 1€": f"{(p[market] * odd - 1):+.1%}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("EV = εκτίμηση μοντέλου × απόδοση − 1. Θετικό EV δεν εγγυάται κέρδος· "
                       "τα σφάλματα πιθανοτήτων μπορούν να ανατρέψουν πλήρως το συμπέρασμα.")
        else:
            st.info("Βάλε τουλάχιστον μία απόδοση ≥ 1.01 για να δεις τη σύγκριση.")
    with tab_dc:
        dixon_coles_tab(result)
    with tab4:
        monte_carlo_tab(result)


st.title("⚽ AI Football Predictor")
st.write("Προσωπικός μηχανισμός ποδοσφαιρικών πιθανοτήτων — χωρίς ισχυρισμούς για «σίγουρα» σημεία.")
with st.sidebar:
    st.header("Πηγή δεδομένων")
    source = st.radio("Επιλογή", ["Football-data.org API · πραγματικά δεδομένα", "Demo · φανταστικά δεδομένα", "CSV · δικά μου ιστορικά"], label_visibility="collapsed")
    st.caption("Το demo περιέχει **κατασκευασμένες ομάδες/αποτελέσματα** και ΔΕΝ είναι πραγματικά προγνωστικά.")

history = pd.DataFrame()
fixtures = []
verified = False
competition_label = ""
try:
    if source.startswith("Demo"):
        history, fixtures = generate_demo()
        competition_label = "Φανταστικό πρωτάθλημα (Demo)"
        st.warning("🧪 **Demo:** Όλες οι ομάδες και οι βαθμολογίες εδώ είναι τεχνητές. Για πραγματικούς αγώνες διάλεξε API ή ανέβασε CSV.")
    elif source.startswith("Football-data"):
        with st.sidebar:
            token = st.text_input("Football-data.org API token", value=api_token_default(), type="password", help="Χρησιμοποιείται για αιτήματα προς το football-data.org. Μην κοινοποιείς το token.")
        if not token.strip():
            st.info("Για πραγματικά δεδομένα, πρόσθεσε API token από football-data.org στο πλευρικό μενού. Αν θέλεις πρώτα να εξερευνήσεις την εφαρμογή, άλλαξε σε Demo.")
            st.stop()
        with st.spinner("Φόρτωση διαθέσιμων πρωταθλημάτων..."):
            comps = cached_competitions(token)
        if not comps:
            st.error("Ο λογαριασμός δεν επέστρεψε διαθέσιμες διοργανώσεις.")
            st.stop()
        comp = st.selectbox("Διοργάνωση", options=comps, format_func=lambda c: f"{c.get('name', '')} · {c.get('code', '')}")
        competition_label = comp.get("name", "Διοργάνωση")
        with st.spinner("Ανάκτηση αποτελεσμάτων και επόμενων αγώνων..."):
            games = cached_matches(token, str(comp["code"]))
        history = api_matches_to_history(games)
        fixtures = match_options(games)
        verified = True
        st.caption("Πραγματικά αποτελέσματα από football-data.org. Η διαθεσιμότητα των διοργανώσεων, "
                   "η ενημέρωση και το ιστορικό εξαρτώνται από το πακέτο σου.")
    else:
        file = st.file_uploader("Ανέβασε CSV ιστορικών αγώνων", type=["csv"])
        st.caption("Στήλες: date,home_team,away_team,home_goals,away_goals. Μία γραμμή ανά ολοκληρωμένο ματς. Η ημερομηνία είναι UTC.")
        if file is None:
            sample = pd.DataFrame([{ "date": "2026-01-01T17:00:00Z", "home_team": "ΟΜΑΔΑ Α", "away_team": "ΟΜΑΔΑ Β", "home_goals": 2, "away_goals": 1 }])
            st.download_button("Κατέβασε υπόδειγμα CSV", sample.to_csv(index=False).encode("utf-8-sig"), file_name="history_template.csv", mime="text/csv")
            st.stop()
        history = prepare_history(pd.read_csv(file))
        competition_label = "Προσωπικά δεδομένα CSV"
except (FootballAPIError, ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

if history.empty:
    st.error("Δεν υπάρχουν ολοκληρωμένα ματς με σκορ στην επιλεγμένη πηγή.")
    st.stop()

names = sorted(set(history.home_team) | set(history.away_team))
st.caption(f"{competition_label} · {len(history)} διαθέσιμα ολοκληρωμένα ματς · {len(names)} ομάδες")

st.header("1. Επέλεξε αναμέτρηση")
if fixtures:
    fmt = lambda f: f"{f['home']} – {f['away']} · {pd.Timestamp(f['date']).tz_convert(ATHENS).strftime('%d/%m %H:%M')}"
    selection = st.radio("Τρόπος επιλογής", ["Επιλογή από πρόγραμμα", "Χειροκίνητα"], horizontal=True)
else:
    selection = "Χειροκίνητα"

if selection == "Επιλογή από πρόγραμμα":
    selected = st.selectbox("Προσεχής αναμέτρηση", fixtures, format_func=fmt)
    home, away = selected["home"], selected["away"]
    kickoff = pd.Timestamp(selected["date"])
    if verified:
        st.caption("Η αναμέτρηση και η ημερομηνία βρέθηκαν στο API.")
else:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Γηπεδούχος", names, key="home_name")
    away = c2.selectbox("Φιλοξενούμενος", names, index=min(1, len(names)-1), key="away_name")
    today = datetime.now(ATHENS)
    c3, c4 = st.columns(2)
    requested_day = c3.date_input("Ημερομηνία αναμέτρησης", value=today.date() + timedelta(days=1))
    requested_time = c4.time_input("Ώρα Ελλάδας", value=time(20, 0))
    kickoff = pd.Timestamp(athens_dt(requested_day, requested_time))
    if verified:
        st.warning("Χειροκίνητη επιλογή: δεν έχει επιβεβαιωθεί ότι αυτός ο αγώνας είναι προγραμματισμένος.")

if st.button("🔎 Ανάλυση αγώνα", type="primary", use_container_width=True):
    try:
        st.session_state["prediction"] = add_dixon_coles(analyze(history, home, away, kickoff), history)
        st.session_state["prediction_key"] = (source, competition_label, home, away, str(pd.Timestamp(kickoff)))
    except ValueError as exc:
        st.session_state.pop("prediction", None)
        st.error(str(exc))

# Don't silently show stale results after a source or competition switch.
if "prediction" in st.session_state and st.session_state.get("prediction_key") == (source, competition_label, home, away, str(pd.Timestamp(kickoff))):
    scorecard(st.session_state["prediction"])

st.divider()
st.header("2. Ανάλυση πολλών αγώνων")
st.caption("Μία αναμέτρηση ανά γραμμή, π.χ. «Αστήρ FC - Νίκη FC». "
           "Οι αγώνες υπολογίζονται ως υποθετικά προγραμματισμένοι στην παρακάτω κοινή ημερομηνία, "
           "εκτός αν δοθούν από το επίσημο πρόγραμμα στον επάνω επιλογέα.")
batch = st.text_area("Λίστα αγώνων", placeholder="Ομάδα Α - Ομάδα Β\nΟμάδα Γ - Ομάδα Δ", height=105)
b1, b2 = st.columns(2)
batch_day = b1.date_input("Ημερομηνία για τη λίστα", value=datetime.now(ATHENS).date() + timedelta(days=1), key="batchday")
batch_clock = b2.time_input("Ώρα Ελλάδας για τη λίστα", value=time(20, 0), key="batchclock")
if st.button("Υπολογισμός λίστας"):
    pairs, errors = read_batch(batch, names)
    for error in errors:
        st.error(error)
    results = []
    for team_a, team_b in pairs:
        try:
            r = analyze(history, team_a, team_b, athens_dt(batch_day, batch_clock))
            row = {"Αγώνας": f"{team_a} - {team_b}", "Ποιότητα δείγματος": r["data_quality"],
                   "λ έδρας": round(r["expected_home"], 2), "λ εκτός": round(r["expected_away"], 2)}
            row.update({key: round(100*r["probabilities"][key], 1) for key in ("1", "X", "2", "1X", "X2", "Over 2.5", "GG")})
            row["Πιθανότερο σκορ"] = r["top_scores"][0][0]
            results.append(row)
        except ValueError as exc:
            st.warning(f"{team_a} - {team_b}: {exc}")
    if results:
        st.session_state["batch_result"] = pd.DataFrame(results)
        st.session_state["batch_key"] = (source, competition_label, batch, str(batch_day), str(batch_clock))
    else:
        st.session_state.pop("batch_result", None)
    if not results and not batch.strip():
        st.info("Γράψε πρώτα έναν ή περισσότερους αγώνες.")

if "batch_result" in st.session_state and st.session_state.get("batch_key") == (source, competition_label, batch, str(batch_day), str(batch_clock)):
    table = st.session_state["batch_result"]
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Αποθήκευση CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name="football_predictions.csv", mime="text/csv")

st.divider()
st.header("3. Έλεγχος μοντέλου σε παλιούς αγώνες")
st.write("Το backtest ελέγχει τις εκτιμήσεις μόνο με πληροφορίες που ήταν γνωστές πριν από κάθε αγώνα. "
         "Συγκρίνει **τρία μοντέλα στους ίδιους ακριβώς αγώνες**: Poisson (το βασικό μοντέλο της εφαρμογής), "
         "Dixon-Coles και μια απλή «Βάση» (ιστορικές συχνότητες 1/Χ/2 του πρωταθλήματος).")
bt_matches = st.slider("Πόσοι από τους πιο πρόσφατους αγώνες θα ελεγχθούν", min_value=30, max_value=200,
                       value=config.BACKTEST_DEFAULT_MATCHES, step=10)
BT_REFIT_DAYS = 7
bt_key = (source, competition_label, len(history), bt_matches)
if st.button(f"▶ Εκτέλεση συγκριτικού backtest ({bt_matches} αγώνες)"):
    try:
        with st.spinner("Εκτέλεση backtest... Το Dixon-Coles εκπαιδεύεται ξανά για κάθε εβδομάδα αγώνων."):
            st.session_state["comparison"] = compare_models(history, n_matches=bt_matches, refit_days=BT_REFIT_DAYS)
        st.session_state["comparison_key"] = bt_key
    except ValueError as exc:
        st.session_state.pop("comparison", None)
        st.warning(str(exc))

if "comparison" in st.session_state and st.session_state.get("comparison_key") == bt_key:
    cmp_res = st.session_state["comparison"]
    table = cmp_res.metrics.copy()
    best_brier = table.loc[table["Brier 1Χ2"].idxmin(), "Μοντέλο"]
    best_logloss = table.loc[table["Log loss 1Χ2"].idxmin(), "Μοντέλο"]
    table["Ακρίβεια 1Χ2"] = table["Ακρίβεια 1Χ2"].map(lambda v: f"{v:.1%}")
    for col in ("Brier 1Χ2", "Log loss 1Χ2", "Brier Over 2.5"):
        table[col] = table[col].map(lambda v: f"{v:.4f}")
    st.dataframe(table, hide_index=True, use_container_width=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Αγώνες που ελέγχθηκαν", cmp_res.evaluated)
    q2.metric("Αγώνες που παραλείφθηκαν", cmp_res.skipped)
    q3.metric("Εκπαιδεύσεις Dixon-Coles", cmp_res.dc_refits)
    q4.metric("Χρόνος εκτέλεσης", f"{cmp_res.elapsed_seconds:.1f} δευτ.")
    if cmp_res.skip_reasons:
        st.caption("Λόγοι παράλειψης: " + " · ".join(f"{reason}: {count}" for reason, count in cmp_res.skip_reasons.items()))
    if cmp_res.dc_not_converged:
        st.warning(f"Σε {cmp_res.dc_not_converged} από τις {cmp_res.dc_refits} εκπαιδεύσεις το Dixon-Coles δεν συνέκλινε πλήρως· "
                   "τα αποτελέσματά του εδώ να διαβάζονται με επιφύλαξη.")
    st.info(f"Σε αυτό το δείγμα, το χαμηλότερο Brier 1Χ2 το έχει το **{best_brier}** "
            f"και το χαμηλότερο log loss το **{best_logloss}**.")
    st.markdown(
        "**Πώς διαβάζεται ο πίνακας**\n"
        "- **Brier 1Χ2:** πόσο απέχουν οι πιθανότητες από αυτό που έγινε. Από 0 (τέλειο) έως 2 (χειρότερο). "
        "**Χαμηλότερο = καλύτερο.**\n"
        "- **Log loss 1Χ2:** τιμωρεί πολύ αυστηρά τις «σίγουρες» προβλέψεις που βγήκαν λάθος. **Χαμηλότερο = καλύτερο.**\n"
        "- **Brier Over 2.5:** το ίδιο με το Brier, για την αγορά Over/Under 2.5 (0 έως 1). **Χαμηλότερο = καλύτερο.**\n"
        "- **Ακρίβεια 1Χ2:** πόσο συχνά το πιθανότερο σημείο ήταν το σωστό. Μόνη της μπορεί να παραπλανήσει, "
        "γιατί δεν δείχνει αν οι πιθανότητες ήταν σωστά βαθμονομημένες."
    )
    st.caption(f"Το Dixon-Coles εκπαιδεύεται ξανά ανά {cmp_res.refit_days} ημέρες, κάθε φορά μόνο με αγώνες αυστηρά πριν από "
               "την αρχή του διαστήματος. Το Poisson και η Βάση υπολογίζονται ξεχωριστά για κάθε αγώνα με ό,τι ήταν γνωστό πριν από αυτόν. "
               "Πρόκειται για ιστορικό έλεγχο συγκεκριμένου δείγματος, όχι για απόδειξη μελλοντικών κερδών.")

with st.expander("Μεθοδολογία και όρια"):
    st.markdown("""
    - **Δεδομένα:** πραγματικά ολοκληρωμένα ματς από το API ή το προσωπικό CSV. Το demo περιέχει μόνο φανταστικά δεδομένα.
    - **Χωρίς χρονική διαρροή:** χρησιμοποιούνται αποτελέσματα αυστηρά πριν από τη δηλωμένη ώρα έναρξης.
    - **Εκτιμώμενα γκολ λ (όχι πραγματικά xG):** σταθμισμένη πρόσφατη επίδοση, έδρα, ισχύς αντιπάλου και εξομάλυνση στους μέσους όρους της λίγκας.
    - **Poisson:** υπολογίζονται όλες οι βασικές πιθανότητες από την κατανομή πιθανών σκορ.
    - **Dixon-Coles:** δεύτερο, ανεξάρτητο μοντέλο με επίθεση/άμυνα ανά ομάδα, κοινό πλεονέκτημα έδρας, χρονική απόσβεση (πιο πρόσφατα ματς μετράνε περισσότερο) και διόρθωση ρ για τα χαμηλά σκορ (0-0, 1-0, 0-1, 1-1). Η τιμή απόσβεσης ξ είναι προεπιλογή από τη βιβλιογραφία, δεν έχει ρυθμιστεί ακόμη στα δικά σου δεδομένα.
    - **Monte Carlo:** προσομοιώνει χιλιάδες αγώνες είτε με τα αναμενόμενα γκολ του Poisson είτε με τον πίνακα σκορ του Dixon-Coles.
    - **Συγκριτικό backtest:** Poisson, Dixon-Coles και απλή βάση στους ίδιους αγώνες, μόνο με πληροφορίες πριν από κάθε αγώνα (Brier, log loss, ακρίβεια).
    - **Αβεβαιότητα:** μικρό δείγμα, αλλαγές προπονητή, μεταγραφές, απουσίες, πρόγραμμα και ισχύς αντιπάλων δεν μοντελοποιούνται πλήρως.
    - **Αποδόσεις:** χωρίς αυτόματη λήψη στοιχηματικών αποδόσεων· μπορείς να καταχωρίσεις δικές σου για ενδεικτικό EV.
    - **Πρόκειται για στατιστικό baseline, όχι για εκπαιδευμένο βαθύ νευρωνικό δίκτυο ή αυτόματο LLM.** Απαιτεί αναβαθμίσεις, περισσότερα δεδομένα και calibration πριν θεωρηθεί αξιόπιστο παραγωγικό betting model.
    """)
