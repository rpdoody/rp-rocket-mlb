import streamlit as st


NAV_ITEMS = [
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
    """Apply shared styling and suppress Streamlit's built-in navigation UI."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"] {
            display: none;
        }

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
            line-height: 1.15;
            min-height: 42px;
            padding: 8px 5px;
            text-align: center;
            white-space: nowrap;
            width: 100%;
        }

        div[data-testid="stHorizontalBlock"] .stPageLink a:hover {
            background: #eaf2ff;
            border-color: #002D72;
            color: #001f4d !important;
        }

        div[data-testid="stHorizontalBlock"] .stPageLink a p {
            margin: 0;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> None:
    """Display the shared navigation at the top of every application page."""
    nav_columns = st.columns(len(NAV_ITEMS) + 1, gap="small")

    with nav_columns[0]:
        st.page_link("Home", label="🏠 Home")

    for column, (label, page_path) in zip(nav_columns[1:], NAV_ITEMS):
        with column:
            st.page_link(page_path, label=label)

    st.divider()