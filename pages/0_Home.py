from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.top_nav import inject_app_style, render_top_nav

inject_app_style()
render_top_nav()

st.title("⚾ RP Rocket Report")
st.caption("MLB Predictions")

logo = ROOT / "data_files" / "IMG_0185.PNG"
if logo.exists():
    st.image(str(logo), width=180)

st.markdown(
    "Explore today’s schedule, model projections, matchup research, "
    "performance tracking, and qualifying Sports Picks."
)

st.divider()
st.subheader("Explore")

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
    ("🤖", "Models", "XGBoost features · Evaluation · Savant research", "pages/5_Models.py"),
    ("📈", "Performance", "Pick history · Model P&L · Kelly bankroll", "pages/6_Performance.py"),
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
