import streamlit as st


NAV_ITEMS = [
    ("🏠 Home", "pages/0_Home.py"),
    ("📅 Today", "pages/1_Today.py"),
    ("⚾ Sports Picks", "pages/2_Sports_Picks.py"),
    ("📊 Stats", "pages/3_Stats.py"),
    ("🆚 Matchups", "pages/4_Matchup_Analysis.py"),
    ("🤖 Models", "pages/5_Models.py"),
    ("📈 Performance", "pages/6_Performance.py"),
    ("🎯 Pick 6", "pages/7_Pick_6.py"),
    ("🎯 Betting", "pages/9_Betting_Recommendations.py"),
    ("ℹ️ About", "pages/8_Info.py"),
]


def inject_app_style() -> None:
    """Apply shared styling and hide Streamlit's default navigation."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none;
        }

        .stApp {
            background-color: #f9fafb;
            color: #111827;
        }

        div[data-testid="stHorizontalBlock"] .stPageLink a {
            align-items: center;
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            color: #002D72 !important;
            display: flex;
            font-size: 0.78rem;
            font-weight: 650;
            justify-content: center;
            min-height: 42px;
            padding: 8px 5px;
            text-align: center;
            white-space: nowrap;
            width: 100%;
        }

        div[data-testid="stHorizontalBlock"] .stPageLink a:hover {
            background: #eaf2ff;
            border-color: #002D72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> None:
    """Render the same page navigation bar at the top of every page."""
    columns = st.columns(len(NAV_ITEMS), gap="small")

    for column, (label, page_path) in zip(columns, NAV_ITEMS):
        with column:
            st.page_link(page_path, label=label)

    st.divider()