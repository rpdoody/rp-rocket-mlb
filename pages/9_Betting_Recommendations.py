"""Page: Betting Recommendations — contextual MLB projections and market edges."""

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from page_utils import (
    _MLB_TO_RETRO,
    _fetch_espn_odds,
    _fetch_pitcher_stats,
    _fetch_team_standings,
    _fetch_todays_schedule,
    _load_game_context_cache,
    init_session_state,
    render_sidebar,
)
from src.ingestion.weather import fetch_forecast
from src.models.contextual_projection import project_contextual_game
from src.ui.recommendation_cards import (
    _build_game_recs,
    _prob_bar_html,
    _projection_summary,
    _rec_card_html,
    _short,
)

ET = ZoneInfo("America/New_York")


def eastern_today() -> datetime.date:
    return datetime.datetime.now(ET).date()


def format_game_time_et(game_datetime: str) -> str:
    if not game_datetime:
        return "TBD"

    try:
        game_time = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if game_time.tzinfo is None:
            game_time = game_time.replace(tzinfo=datetime.timezone.utc)

        return game_time.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
    except (TypeError, ValueError):
        return "TBD"


def game_date_from_datetime(game_datetime: str) -> str:
    if not game_datetime:
        return ""

    try:
        game_time = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if game_time.tzinfo is None:
            game_time = game_time.replace(tzinfo=datetime.timezone.utc)

        return game_time.astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return ""


def normalize_team_name(team_name: str | None) -> str:
    return "".join(char.lower() for char in (team_name or "") if char.isalnum())


def is_matching_odds_game(game: dict, odds_game: dict) -> bool:
    return normalize_team_name(game.get("away_name")) == normalize_team_name(
        odds_game.get("away_team")
    ) and normalize_team_name(game.get("home_name")) == normalize_team_name(
        odds_game.get("home_team")
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_todays_schedule(game_date_iso: str):
    game_date = datetime.date.fromisoformat(game_date_iso)

    try:
        return _fetch_todays_schedule(game_date)
    except TypeError:
        return _fetch_todays_schedule()


@st.cache_data(ttl=900, show_spinner=False)
def cached_standings():
    return _fetch_team_standings()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_pitcher_stats(pitcher_name: str):
    return _fetch_pitcher_stats(pitcher_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_espn_odds():
    return _fetch_espn_odds()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(venue_name: str, game_date_iso: str):
    return fetch_forecast(venue_name, game_date_iso)


init_session_state()
render_sidebar(show_year_filter=False)

today_et = eastern_today()
games_today = cached_todays_schedule(today_et.isoformat())

st.title("🎯 Games & Betting Recommendations")
st.caption(
    "Contextual run projections include historical offense/defense, probable "
    "starters, bullpen workload proxy, park factor, weather, day/night context, "
    "and home field. ✅ BET = edge > 3% · ➡ LEAN = 0–3% · "
    "⛔ PASS = negative edge."
)

if not games_today:
    st.info(
        "No MLB games are scheduled today, or the MLB Stats API is unavailable. "
        "Check back on a game day."
    )
    st.stop()

if st.button(
    "Load betting recommendations",
    key="load_betting_recommendations",
    type="primary",
    width="content",
):
    st.session_state["show_betting_recommendations"] = True

if not st.session_state.get("show_betting_recommendations", False):
    st.caption(
        f"{len(games_today)} game{'s' if len(games_today) != 1 else ''} scheduled today. "
        "Load recommendations to calculate projections and compare available odds."
    )
    st.stop()

with st.spinner("Building contextual projections and comparing odds…"):
    standings = cached_standings()
    game_context = _load_game_context_cache()
    espn_odds = cached_espn_odds()

status_labels = {
    "Final": "🏁 Final",
    "Game Over": "🏁 Final",
    "In Progress": "🔴 LIVE",
    "Scheduled": "🕐 Scheduled",
    "Pre-Game": "⏳ Pre-Game",
    "Warmup": "⏳ Pre-Game",
    "Delayed": "⚠️ Delayed",
    "Suspended": "⚠️ Suspended",
    "Postponed": "🚫 Postponed",
    "Cancelled": "🚫 Cancelled",
}

for idx, game in enumerate(games_today):
    away_full = game.get("away_name", "Away")
    home_full = game.get("home_name", "Home")
    away_sp = game.get("away_probable_pitcher", "TBD") or "TBD"
    home_sp = game.get("home_probable_pitcher", "TBD") or "TBD"
    status = game.get("status", "Scheduled")
    venue = game.get("venue_name", "—")
    game_time = format_game_time_et(game.get("game_datetime", ""))
    away_retro = _MLB_TO_RETRO.get(away_full, away_full)
    home_retro = _MLB_TO_RETRO.get(home_full, home_full)
    weather_date = game_date_from_datetime(game.get("game_datetime", "")) or today_et.isoformat()

    away_pitcher_stats = cached_pitcher_stats(away_sp)
    home_pitcher_stats = cached_pitcher_stats(home_sp)
    weather = cached_weather(venue, weather_date) if venue else None

    projection = project_contextual_game(
        game=game,
        hist_stnd=standings,
        game_context=game_context,
        away_retro=away_retro,
        home_retro=home_retro,
        away_pitcher_stats=away_pitcher_stats,
        home_pitcher_stats=home_pitcher_stats,
        weather=weather,
    )

    score_str = ""
    if (
        str(status).lower() in {"final", "game over", "in progress", "live", "completed"}
        and game.get("away_score") is not None
        and game.get("home_score") is not None
    ):
        score_str = f" &nbsp;·&nbsp; **{game['away_score']}–{game['home_score']}**"

    espn_game = next(
        (odds for odds in espn_odds if is_matching_odds_game(game, odds)),
        None,
    )
    recs = _build_game_recs(game, espn_game, projection, standings)
    home_prob = projection.home_win_probability

    with st.container(border=True):
        header_left, header_right = st.columns([3, 2])

        with header_left:
            st.markdown(
                f"#### {away_full} @ {home_full}{score_str}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>🏟️ {venue} &nbsp;·&nbsp; "
                f"{status_labels.get(status, status)} &nbsp;·&nbsp; "
                f"🕐 {game_time}</small><br>"
                f"<small>SP: <b>{away_sp}</b> (away) &nbsp;/&nbsp; "
                f"<b>{home_sp}</b> (home)</small>",
                unsafe_allow_html=True,
            )

        with header_right:
            st.markdown(
                _prob_bar_html(home_prob, home_full, away_full),
                unsafe_allow_html=True,
            )

        st.caption(_projection_summary(projection))

        if projection.warnings:
            with st.expander("Projection data notes", expanded=False):
                for warning in sorted(set(projection.warnings)):
                    st.caption(f"• {warning}")

        if not recs:
            st.caption("⏳ Odds not yet available for this game.")
            continue

        st.divider()
        col_ml, col_rl, col_ou = st.columns(3)

        with col_ml:
            st.markdown("##### 💵 Moneyline")

            if "ml" not in recs:
                st.caption("— odds unavailable —")
            else:
                market = recs["ml"]
                side = market[market["best"]]
                other = market["away" if market["best"] == "home" else "home"]
                explanation = f"Est: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"

                st.markdown(
                    _rec_card_html("ML", side, explanation),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Other side: {_short(other['team'])} {other['odds_str']} "
                    f"(edge {other['edge'] * 100:+.1f}%)"
                )

        with col_rl:
            st.markdown("##### 📏 Run Line (±1.5)")

            if "rl" not in recs:
                st.caption("— odds unavailable —")
            else:
                market = recs["rl"]
                side = market[market["best"]]
                other = market["away" if market["best"] == "home" else "home"]
                explanation = f"Est cover: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"

                st.markdown(
                    _rec_card_html("RL", side, explanation),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Other side: {other['pick']} {other['odds_str']} "
                    f"(edge {other['edge'] * 100:+.1f}%)"
                )

        with col_ou:
            st.markdown("##### 📊 Over/Under")

            if "ou" not in recs:
                st.caption("— odds unavailable —")
            else:
                market = recs["ou"]
                side = market[market["best"]]
                other = market["under" if market["best"] == "over" else "over"]
                explanation = (
                    f"Model total: {market['exp_total']:.1f} · "
                    f"Posted: {market['posted']} · "
                    f"Impl: {side['impl']:.0%}"
                )

                st.markdown(
                    _rec_card_html("OU", side, explanation),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Other side: {other['pick']} {other['odds_str']} "
                    f"(edge {other['edge'] * 100:+.1f}%)"
                )
