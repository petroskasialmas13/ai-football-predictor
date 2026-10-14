"""AI Football Predictor — personal, read-only pre-match analytics dashboard."""
from __future__ import annotations

from datetime import datetime, date, time, timedelta
from difflib import get_close_matches
from io import BytesIO
from zoneinfo import ZoneInfo
import os
import re

import pandas as pd
import streamlit as st

from engine import analyze, backtest, clean_name, generate_demo, api_matches_to_history, prepare_history
from data_provider import competitions, competition_matches, FootballAPIError

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

    tab1, tab2, tab3 = st.tabs(["Προγνωστικά", "Πώς προκύπτουν", "Σύγκριση αποδόσεων"])
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
        st.session_state["prediction"] = analyze(history, home, away, kickoff)
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
         "Συγκρίνει το μοντέλο με έναν απλό ιστορικό μέσο όρο του πρωταθλήματος.")
if st.button("▶ Εκτέλεση backtest έως 60 αγώνων"):
    try:
        test = backtest(history)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Αγώνες ελέγχου", test["count"])
        q2.metric("1Χ2 accuracy", f"{test['accuracy']:.1%}")
        q3.metric("Brier μοντέλου (χαμηλότερο καλύτερο)", f"{test['brier']:.3f}")
        q4.metric("Brier απλής βάσης", f"{test['baseline_brier']:.3f}")
        st.caption(f"Ακρίβεια απλής βάσης: {test['baseline_accuracy']:.1%}. "
                   "Πρόκειται για ιστορικό έλεγχο συγκεκριμένου δείγματος, όχι για απόδειξη μελλοντικών κερδών.")
    except ValueError as exc:
        st.warning(str(exc))

with st.expander("Μεθοδολογία και όρια"):
    st.markdown("""
    - **Δεδομένα:** πραγματικά ολοκληρωμένα ματς από το API ή το προσωπικό CSV. Το demo περιέχει μόνο φανταστικά δεδομένα.
    - **Χωρίς χρονική διαρροή:** χρησιμοποιούνται αποτελέσματα αυστηρά πριν από τη δηλωμένη ώρα έναρξης.
    - **Εκτιμώμενα γκολ λ (όχι πραγματικά xG):** σταθμισμένη πρόσφατη επίδοση, έδρα, ισχύς αντιπάλου και εξομάλυνση στους μέσους όρους της λίγκας.
    - **Poisson:** υπολογίζονται όλες οι βασικές πιθανότητες από την κατανομή πιθανών σκορ.
    - **Αβεβαιότητα:** μικρό δείγμα, αλλαγές προπονητή, μεταγραφές, απουσίες, πρόγραμμα και ισχύς αντιπάλων δεν μοντελοποιούνται πλήρως.
    - **Αποδόσεις:** χωρίς αυτόματη λήψη στοιχηματικών αποδόσεων· μπορείς να καταχωρίσεις δικές σου για ενδεικτικό EV.
    - **Πρόκειται για στατιστικό baseline, όχι για εκπαιδευμένο βαθύ νευρωνικό δίκτυο ή αυτόματο LLM.** Απαιτεί αναβαθμίσεις, περισσότερα δεδομένα και calibration πριν θεωρηθεί αξιόπιστο παραγωγικό betting model.
    """)
