"""
Entry point for the Rocket Report MLB dashboard.

  - st.set_page_config()  called exactly once here
  - home_page()           landing page with per-game betting recommendations
  - st.navigation()       6-page top navigation (mobile-friendly)
"""

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from page_utils import (
    _fetch_espn_odds,
    _fetch_pitcher_stats,
    _fetch_todays_schedule,
    _load_model_results,
    _load_precomputed,
)
from src.ingestion.weather import fetch_forecast
from src.top_nav import inject_app_style, render_top_nav

ET = ZoneInfo("America/New_York")


def eastern_now() -> datetime.datetime:
    return datetime.datetime.now(ET)


def eastern_today() -> datetime.date:
    return eastern_now().date()


def format_game_time_et(game_datetime: str) -> str:
    """Format an MLB ISO timestamp in US Eastern time, including DST."""
    if not game_datetime:
        return "TBD"
    try:
        dt_utc = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        return dt_utc.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
    except (TypeError, ValueError):
        return "TBD"


def game_date_et(game_datetime: str, fallback_date: datetime.date) -> str:
    """Resolve a scheduled game date in Eastern time for weather retrieval."""
    if not game_datetime:
        return fallback_date.isoformat()
    try:
        dt_utc = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        return dt_utc.astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return fallback_date.isoformat()


def normalize_team_name(team_name: str | None) -> str:
    """Create a comparison-safe representation of team names across feeds."""
    return "".join(char.lower() for char in (team_name or "") if char.isalnum())


def is_matching_odds_game(game: dict, odds_game: dict) -> bool:
    """Require exact normalized home and away team matches."""
    return normalize_team_name(game.get("home_name")) == normalize_team_name(
        odds_game.get("home_team")
    ) and normalize_team_name(game.get("away_name")) == normalize_team_name(
        odds_game.get("away_team")
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_todays_schedule(game_date_iso: str) -> list[dict]:
    """Load the MLB schedule for an explicit ET calendar date."""
    game_date = datetime.date.fromisoformat(game_date_iso)

    try:
        return _fetch_todays_schedule(game_date)
    except TypeError:
        return _fetch_todays_schedule()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_espn_odds(game_date_iso: str) -> list[dict]:
    """Load ESPN odds for the selected schedule date.

    Requires page_utils._fetch_espn_odds(game_date: date | None) to support
    its optional date argument. A compatibility fallback preserves the app
    while that helper is being updated.
    """
    game_date = datetime.date.fromisoformat(game_date_iso)
    try:
        return _fetch_espn_odds(game_date)
    except TypeError:
        return _fetch_espn_odds()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_pitcher_stats(pitcher_name: str) -> dict:
    return _fetch_pitcher_stats(pitcher_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(venue_name: str, game_date_iso: str) -> dict | None:
    return fetch_forecast(venue_name, game_date_iso)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_model_results():
    return _load_model_results()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_precomputed():
    return _load_precomputed()


from importlib import import_module


_top_nav = import_module("top_nav")
inject_app_style = _top_nav.inject_app_style
render_top_nav = _top_nav.render_top_nav

st.set_page_config(
    page_title="RP Rocket Report - MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_app_style()

st.markdown(
    """
    <style>
    .stApp { background-color: #f9fafb; color: #111827; }
    h1, h2, h3 { color: #002D72; }
    </style>
    """,
    unsafe_allow_html=True,
)

def home_page() -> None:
    """Landing page for RP Rocket Report."""

    render_top_nav()

    hdr_left, hdr_right = st.columns([1, 5])

    with hdr_left:
        logo = ROOT / "data_files" / "IMG_0185.PNG"
        if logo.exists():
            st.image(str(logo), width=180)

    with hdr_right:
        st.markdown(
            "<h1 style='margin-bottom:0;color:#002D72'>RP Rocket Report</h1>"
            "<p style='color:#6b7280;margin-top:2px'>MLB Predictions</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "Explore today’s schedule, model projections, matchup research, "
        "performance tracking, and qualifying Sports Picks."
    )
    st.markdown("---")
    st.markdown("### Explore")

    tiles = [
        ("📅", "Today", "Full schedule with detailed game drill-down", "pages/1_Today.py"),
        ("⚾", "Sports Picks", "Today’s qualifying MLB bets", "pages/2_Sports_Picks.py"),
        ("📊", "Stats", "Standings · Batting · Pitching · Leaders", "pages/3_Stats.py"),
        (
            "🆚",
            "Matchup Analysis",
            "H2H history · Rolling win-rate charts",
            "pages/4_Matchup_Analysis.py",
        ),
        (
            "🤖",
            "Models",
            "XGBoost features · Evaluation · Savant research",
            "pages/5_Models.py",
        ),
        (
            "📈",
            "Performance",
            "Pick history · Model P&L · Kelly bankroll",
            "pages/6_Performance.py",
        ),
        ("🎯", "Pick 6", "Six-pick slate and card overview", "pages/7_Pick_6.py"),
        (
            "🎯",
            "Betting Recommendations",
            "All game recommendations",
            "pages/9_Betting_Recommendations.py",
        ),
        ("ℹ️", "About", "Methodology, data sources & tech stack", "pages/8_Info.py"),
    ]

    for row_tiles in (tiles[:3], tiles[3:6], tiles[6:]):
        columns = st.columns(3)

        for column, (icon, title, description, path) in zip(columns, row_tiles):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>',
                        unsafe_allow_html=True,
                    )
                    st.page_link(path, label=f"**{title}**")
                    st.caption(description)


# These must be outside home_page() — no indentation.
pages = [
    st.Page(home_page, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Today.py", title="Today", icon="📅"),
    st.Page("pages/2_Sports_Picks.py", title="Sports Picks", icon="⚾"),
    st.Page("pages/3_Stats.py", title="Stats", icon="📊"),
    st.Page("pages/4_Matchup_Analysis.py", title="Matchup Analysis", icon="🆚"),
    st.Page("pages/5_Models.py", title="Models", icon="🤖"),
    st.Page("pages/6_Performance.py", title="Performance", icon="📈"),
    st.Page("pages/7_Pick_6.py", title="Pick 6", icon="🎯"),
    st.Page("pages/8_Info.py", title="About", icon="ℹ️"),
    st.Page(
        "pages/9_Betting_Recommendations.py",
        title="Betting Recommendations",
        icon="🎯",
    ),
]

pg = st.navigation(pages, position="hidden")
pg.run()