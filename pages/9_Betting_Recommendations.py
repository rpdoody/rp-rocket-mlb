"""Page: Betting Recommendations — contextual MLB projections and market edges."""

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.top_nav import inject_app_style, render_top_nav
from page_utils import (
    _MLB_TO_RETRO,
    _fetch_espn_odds,
    _fetch_pitcher_stats,
    _fetch_team_standings,
    _fetch_todays_schedule,
    _load_game_context_cache,
    init_session_state,
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

inject_app_style()
render_top_nav()

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


def is_final_game(game: dict) -> bool:
    status = str(game.get("status", "")).strip().lower()

    return (
        status in {"final", "game over", "completed"}
        or status.startswith("final")
    )


def grade_recommendation(
    market_key: str,
    side: dict,
    game: dict,
    posted_total: float | None = None,
) -> tuple[str, str]:
    """Return (label, emoji) for a displayed recommendation."""

    if not is_final_game(game):
        return "PENDING", "⏳"

    away_score = game.get("away_score")
    home_score = game.get("home_score")

    if away_score is None or home_score is None:
        return "PENDING", "⏳"

    away_score = float(away_score)
    home_score = float(home_score)

    pick = str(side.get("pick") or side.get("team") or "")
    pick_lower = pick.lower()

    home_name = str(game.get("home_name", "")).lower()
    away_name = str(game.get("away_name", "")).lower()
    home_short = home_name.split()[-1] if home_name else ""
    away_short = away_name.split()[-1] if away_name else ""

    picked_home = (
        (home_name and home_name in pick_lower)
        or (home_short and home_short in pick_lower)
    )
    picked_away = (
        (away_name and away_name in pick_lower)
        or (away_short and away_short in pick_lower)
    )

    if market_key == "ml":
        if picked_home:
            return ("WIN", "✅") if home_score > away_score else ("LOSS", "❌")

        if picked_away:
            return ("WIN", "✅") if away_score > home_score else ("LOSS", "❌")

        return "PENDING", "⏳"

    if market_key == "rl":
        if picked_home:
            margin = home_score - away_score
        elif picked_away:
            margin = away_score - home_score
        else:
            return "PENDING", "⏳"

        if "+1.5" in pick:
            line = 1.5
        elif "-1.5" in pick or "−1.5" in pick:
            line = -1.5
        else:
            return "PENDING", "⏳"

        adjusted_margin = margin + line

        if adjusted_margin > 0:
            return "WIN", "✅"
        if adjusted_margin < 0:
            return "LOSS", "❌"
        return "PUSH", "↔️"

    if market_key == "ou":
        if posted_total is None:
            return "PENDING", "⏳"

        final_total = away_score + home_score

        if pick_lower.startswith("over"):
            if final_total > posted_total:
                return "WIN", "✅"
            if final_total < posted_total:
                return "LOSS", "❌"
            return "PUSH", "↔️"

        if pick_lower.startswith("under"):
            if final_total < posted_total:
                return "WIN", "✅"
            if final_total > posted_total:
                return "LOSS", "❌"
            return "PUSH", "↔️"

    return "PENDING", "⏳"

def empty_record() -> dict[str, int]:
    return {"wins": 0, "losses": 0, "pushes": 0, "pending": 0}


def record_text(record: dict[str, int]) -> str:
    return f"{record['wins']}-{record['losses']}-{record['pushes']}"


@st.cache_data(ttl=60, show_spinner=False)
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

today_et = eastern_today()

st.title("🎯 Games & Betting Recommendations")

selected_date = st.date_input(
    "Game date",
    value=today_et,
    max_value=today_et,
    format="MM/DD/YYYY",
    key="betting_recommendations_date",
)

games_today = cached_todays_schedule(selected_date.isoformat())
st.caption(
    "Contextual run projections include historical offense/defense, probable "
    "starters, bullpen workload proxy, park factor, weather, day/night context, "
    "and home field. ✅ BET = edge > 3% · ➡ LEAN = 0–3% · "
    "⛔ PASS = negative edge."
)

record_slot = st.empty()
if st.button("↻ Refresh final scores", key="refresh_scores"):
    cached_todays_schedule.clear()
    st.rerun()

if not games_today:
    st.info(
        "No MLB games are scheduled today, or the MLB Stats API is unavailable. "
        "Check back on a game day."
    )
    st.stop()


with st.spinner("Building contextual projections and comparing odds…"):
    standings = cached_standings()
    game_context = _load_game_context_cache()
    espn_odds = cached_espn_odds()
daily_records = {
    "ml": empty_record(),
    "rl": empty_record(),
    "ou": empty_record(),
}
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

    for market_key in ("ml", "rl", "ou"):
        if market_key not in recs:
            continue

        market = recs[market_key]
        side = market[market["best"]]

        # Count only actionable, positive-edge recommendations.
        if side["edge"] <= 0:
            continue

        result, _ = grade_recommendation(
            market_key,
            side,
            game,
            posted_total=market.get("posted") if market_key == "ou" else None,
        )

        if result == "WIN":
            daily_records[market_key]["wins"] += 1
        elif result == "LOSS":
            daily_records[market_key]["losses"] += 1
        elif result == "PUSH":
            daily_records[market_key]["pushes"] += 1
        else:
            daily_records[market_key]["pending"] += 1
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
                    f"Other side: {_short(other.get('team', other.get('pick', 'Other side')))} "
                    f"{other['odds_str']} "
                    f"(edge {other['edge'] * 100:+.1f}%)"
                )

                if side["edge"] <= 0:
                    st.caption("⛔ No official recommendation to grade.")
                else:
                    result, emoji = grade_recommendation("ml", side, game)
                    st.caption(f"{emoji} Result: **{result}**")

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
                if side["edge"] <= 0:
                    st.caption("⛔ No official recommendation to grade.")
                else:
                    result, emoji = grade_recommendation("rl", side, game)
                    st.caption(f"{emoji} Result: **{result}**")

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

                if side["edge"] <= 0:
                    st.caption("⛔ No official recommendation to grade.")
                else:
                    result, emoji = grade_recommendation(
                        "ou",
                        side,
                        game,
                        posted_total=market.get("posted"),
                    )
                    st.caption(f"{emoji} Result: **{result}**")

with record_slot.container():
    st.markdown("### 📊 Today’s Record")
    ml_record, rl_record, ou_record = st.columns(3)

    ml_record.metric(
        "💵 Moneyline",
        record_text(daily_records["ml"]),
        delta=f"{daily_records['ml']['pending']} pending",
        delta_color="off",
    )
    rl_record.metric(
        "📏 Run Line",
        record_text(daily_records["rl"]),
        delta=f"{daily_records['rl']['pending']} pending",
        delta_color="off",
    )
    ou_record.metric(
        "📊 Over/Under",
        record_text(daily_records["ou"]),
        delta=f"{daily_records['ou']['pending']} pending",
        delta_color="off",
    )

    st.caption("Record format: W-L-P. Only positive-edge recommendations are included.")
