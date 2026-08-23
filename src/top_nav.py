import streamlit as st


NAV_ITEMS = [
    ("Home", "pages/0_Home.py"),
    ("Today", "pages/1_Today.py"),
    ("Sports Picks", "pages/2_Sports_Picks.py"),
    ("Stats", "pages/3_Stats.py"),
    ("Matchups", "pages/4_Matchup_Analysis.py"),
    ("Models", "pages/5_Models.py"),
    ("Performance", "pages/6_Performance.py"),
    ("Pick 6", "pages/7_Pick_6.py"),
    ("Betting", "pages/9_Betting_Recommendations.py"),
    ("About", "pages/8_Info.py"),
]


def inject_app_style() -> None:
    """Apply shared application and navigation styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
            color: #172033;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.top-nav-marker) {
            background: #002d72;
            border: 1px solid #00265f;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.14);
            margin-bottom: 1.25rem;
            padding: 6px 10px;
        }

        div[data-testid="stHorizontalBlock"]:has(.top-nav-marker) .stPageLink a {
            align-items: center;
            background: transparent;
            border: none;
            color: #d9e3f1 !important;
            display: flex;
            font-size: 0.88rem;
            font-weight: 600;
            justify-content: center;
            line-height: 1;
            min-height: 42px;
            padding: 13px 8px 11px;
            text-align: center;
            text-decoration: none;
            transition: background 0.15s ease, color 0.15s ease;
            white-space: nowrap;
            width: 100%;
        }

        div[data-testid="stHorizontalBlock"]:has(.top-nav-marker) .stPageLink a:hover {
            background: rgba(255, 255, 255, 0.10);
            color: #ffffff !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.top-nav-marker) .stPageLink a p {
            margin: 0;
        }

        .top-nav-brand {
            color: #ffffff !important;
            font-size: 0.96rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            padding: 12px 8px;
            white-space: nowrap;
        }

        .top-nav-brand span {
            color: #9fc5ff !important;
            font-size: 0.72rem;
            font-weight: 600;
            margin-left: 6px;
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"]:has(.top-nav-marker) .stPageLink a {
                font-size: 0.76rem;
                padding: 11px 4px 9px;
            }

            .top-nav-brand {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> None:
    """Render a compact, professional shared navigation header."""
    st.markdown('<span class="top-nav-marker"></span>', unsafe_allow_html=True)

    brand_col, *nav_cols = st.columns(
        [1.45] + [1] * len(NAV_ITEMS),
        gap="small",
        vertical_alignment="center",
    )

    with brand_col:
        st.markdown(
            '<div class="top-nav-brand">RP ROCKET <span>MLB</span></div>',
            unsafe_allow_html=True,
        )

    for column, (label, page_path) in zip(nav_cols, NAV_ITEMS):
        with column:
            st.page_link(page_path, label=label)

    st.divider()
