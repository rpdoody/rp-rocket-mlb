import streamlit as st


def inject_app_style() -> None:
    """Apply shared styling and hide Streamlit's native navigation UI."""
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(pages: list[st.Page]) -> None:
    """Render page links from the registered Streamlit Page objects."""
    columns = st.columns(len(pages), gap="small")

    for column, page in zip(columns, pages):
        with column:
            st.page_link(page, label=f"{page.icon} {page.title}")

    st.divider()