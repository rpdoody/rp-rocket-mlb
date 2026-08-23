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
    _short,
)

ET = ZoneInfo("America/New_York")
QUALIFYING_EDGE = 0.03


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
    return (
        normalize_team_name(game.get("away_name"))
        == normalize_team_name(odds_game.get("away_team"))
        and normalize_team_name(game.get("home_name"))
        == normalize_team_name(odds_game.get("home_team"))
    )


def is_final_game(game: dict) -> bool:
    return str(game.get("status", "")).strip().lower() in {
        "final",
        "game over",
        "completed",
    }


def grade_recommendation(
    market_key: str,
    side: dict,
    game: dict,
    posted_total: float | None = None,
) -> tuple[str, str]:
    if not is_final_game(game):
        return "PENDING", "⏳"

    away_score = game.get("away_score")
    home_score = game.get("home_score")

    if away_score is None or home_score is None:
        return "PENDING", "⏳"

    away_score = float(away_score)
    home_score = float(home_score)

    if market_key == "ml":
        pick = str(side.get("pick", "")).lower()
        home_name = str(game.get("home_name", "")).lower()
        away_name = str(game.get("away_name", "")).lower()

        if home_name and home_name in pick:
            return ("WIN", "✅") if home_score > away_score else ("LOSS", "❌")
        if away_name and away_name in pick:
            return ("WIN", "✅") if away_score > home_score else ("LOSS", "❌")
        return "PENDING", "⏳"

    if market_key == "rl":
        pick = str(side.get("pick", ""))
        pick_lower = pick.lower()
        home_name = str(game.get("home_name", "")).lower()
        away_name = str(game.get("away_name", "")).lower()

        if home_name and home_name in pick_lower:
            picked_home = True
        elif away_name and away_name in pick_lower:
            picked_home = False
        else:
            return "PENDING", "⏳"

        if "+1.5" in pick:
            line = 1.5
        elif "-1.5" in pick or "−1.5" in pick:
            line = -1.5
        else:
            return "PENDING", "⏳"

        adjusted_margin = (
            home_score - away_score + line
            if picked_home
            else away_score - home_score + line
        )

        if adjusted_margin > 0:
            return "WIN", "✅"
        if adjusted_margin < 0:
            return "LOSS", "❌"
        return "PUSH", "↔️"

    if market_key == "ou" and posted_total is not None:
        final_total = away_score + home_score
        pick = str(side.get("pick", "")).lower()

        if pick.startswith("over"):
            if final_total > posted_total:
                return "WIN", "✅"
            if final_total < posted_total:
                return "LOSS", "❌"
            return "PUSH", "↔️"

        if pick.startswith("under"):
            if final_total < posted_total:
                return "WIN", "✅"
            if final_total > posted_total:
                return "LOSS", "❌"
            return "PUSH", "↔️"

    return "PENDING", "⏳"


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

st.title("🎯 Sports Picks")
st.caption(
    "Only qualifying recommendations are displayed. "
    f"A pick qualifies when its model edge is greater than {QUALIFYING_EDGE:.0%}."
)

if not games_today:
    st.info("No MLB games are scheduled today, or the MLB Stats API is unavailable.")
    st.stop()

if st.button(
    "Load qualifying picks",
    key="load_qualifying_picks",
    type="primary",
    width="content",
):
    st.session_state["show_qualifying_picks"] = True

if not st.session_state.get("show_qualifying_picks", False):
    st.caption(
        f"{len(games_today)} game{'s' if len(games_today) != 1 else ''} scheduled. "
        "Load picks to find qualifying recommendations."
    )
    st.stop()

with st.spinner("Scanning the slate for qualifying picks…"):
    standings = cached_standings()
    game_context = _load_game_context_cache()
    espn_odds = cached_espn_odds()

qualifying_picks = []

market_meta = {
    "ml": {"label": "Moneyline", "icon": "💵"},
    "rl": {"label": "Run Line", "icon": "📏"},
    "ou": {"label": "Over/Under", "icon": "📊"},
}

for game in games_today:
    away_full = game.get("away_name", "Away")
    home_full = game.get("home_name", "Home")
    away_sp = game.get("away_probable_pitcher", "TBD") or "TBD"
    home_sp = game.get("home_probable_pitcher", "TBD") or "TBD"
    venue = game.get("venue_name", "—")
    game_time = format_game_time_et(game.get("game_datetime", ""))
    weather_date = (
        game_date_from_datetime(game.get("game_datetime", "")) or today_et.isoformat()
    )

    projection = project_contextual_game(
        game=game,
        hist_stnd=standings,
        game_context=game_context,
        away_retro=_MLB_TO_RETRO.get(away_full, away_full),
        home_retro=_MLB_TO_RETRO.get(home_full, home_full),
        away_pitcher_stats=cached_pitcher_stats(away_sp),
        home_pitcher_stats=cached_pitcher_stats(home_sp),
        weather=cached_weather(venue, weather_date) if venue else None,
    )

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

        # This is the page-level filter: do not display non-qualifiers.
        if side["edge"] <= QUALIFYING_EDGE:
            continue

        result, emoji = grade_recommendation(
            market_key,
            side,
            game,
            posted_total=market.get("posted") if market_key == "ou" else None,
        )

        qualifying_picks.append(
            {
                "matchup": f"{away_full} @ {home_full}",
                "time": game_time,
                "venue": venue,
                "away_sp": away_sp,
                "home_sp": home_sp,
                "market": market_meta[market_key]["label"],
                "icon": market_meta[market_key]["icon"],
                "pick": side.get("pick", side.get("team", "—")),
                "odds": side.get("odds_str", "—"),
                "edge": side["edge"],
                "estimated_probability": side.get("est_prob"),
                "implied_probability": side.get("impl"),
                "model_total": market.get("exp_total"),
                "posted_total": market.get("posted"),
                "result": result,
                "result_emoji": emoji,
            }
        )

st.divider()

if not qualifying_picks:
    st.info(
        f"No picks currently meet the {QUALIFYING_EDGE:.0%} model-edge threshold. "
        "This is a pass slate under the current rules."
    )
else:
    st.success(
        f"{len(qualifying_picks)} qualifying pick"
        f"{'s' if len(qualifying_picks) != 1 else ''} found."
    )

    for pick in sorted(qualifying_picks, key=lambda item: item["edge"], reverse=True):
        with st.container(border=True):
            left, right = st.columns([4, 1])

            with left:
                st.markdown(
                    f"### {pick['icon']} {pick['market']} — {pick['matchup']}"
                )
                st.markdown(
                    f"**Pick:** {pick['pick']} ({pick['odds']})  \n"
                    f"**Edge:** {pick['edge']:.1%}  \n"
                    f"**Model probability:** {pick['estimated_probability']:.1%}  \n"
                    f"**Implied probability:** {pick['implied_probability']:.1%}"
                )

                if pick["market"] == "Over/Under":
                    st.caption(
                        f"Model total: {pick['model_total']:.1f} · "
                        f"Posted total: {pick['posted_total']}"
                    )

                st.caption(
                    f"🕐 {pick['time']} · 🏟️ {pick['venue']} · "
                    f"SP: {pick['away_sp']} vs. {pick['home_sp']}"
                )

            with right:
                st.markdown(f"## {pick['result_emoji']}")
                st.caption(pick["result"])