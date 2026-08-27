"""Page: Performance — pick history, model performance, and bankroll tracking."""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from page_utils import (
    _american_to_implied_prob,
    _kelly_fraction,
    init_session_state,
)
from src.top_nav import inject_app_style, render_top_nav

inject_app_style()
render_top_nav()

init_session_state()

st.title("📈 Performance")
st.caption("Pick history, model performance, and bankroll tracking.")

tab_history, tab_perf, tab_bankroll = st.tabs(["Pick History", "Model Performance", "Bankroll"])

backtests = st.session_state.get("eval_backtests")


def get_dataframe_height(
    df: pd.DataFrame,
    row_height: int = 35,
    header_height: int = 38,
    padding: int = 2,
    max_height: int | None = 600,
) -> int:
    """Calculate a practical dataframe height while preventing oversized tables."""

    calculated_height = (len(df) * row_height) + header_height + padding

    if max_height is None:
        return calculated_height

    return min(calculated_height, max_height)


def title_case(value: object) -> str:
    """Convert an optional display value to title case."""

    if value is None or pd.isna(value):
        return "—"

    return str(value).replace("_", " ").title()


def readable_pick_type(value: object) -> str:
    """Format stored pick-type values for dashboard display."""

    mapping = {
        "ml": "Moneyline",
        "moneyline": "Moneyline",
        "rl": "Run Line",
        "spread": "Run Line",
        "totals": "Totals",
        "ou": "Over/Under",
        "over_under": "Over/Under",
    }

    normalized = str(value or "").strip().lower()
    return mapping.get(normalized, title_case(value))


def normalize_result(value: object) -> str:
    """Normalize result labels for filters and summary metrics."""

    normalized = str(value or "").strip().lower()

    if normalized in {"win", "won"}:
        return "win"

    if normalized in {"loss", "lost"}:
        return "loss"

    if normalized in {"push", "tie"}:
        return "push"

    if normalized in {"pending", "open", "unsettled"}:
        return "pending"

    return normalized or "pending"


def format_american_odds(value: object) -> str:
    """Format an odds value safely as an American odds string."""

    try:
        odds = int(float(value))
    except (TypeError, ValueError):
        return "—"

    return f"+{odds}" if odds > 0 else str(odds)


def build_history_dataframe(
    evaluation_backtests: object,
) -> pd.DataFrame:
    """Convert stored backtest bet objects into one normalized dataframe."""

    if not evaluation_backtests:
        return pd.DataFrame()

    rows: list[dict] = []

    for model_name, backtest in evaluation_backtests.items():
        bets = getattr(backtest, "bets", [])

        for bet in bets:
            rows.append(
                {
                    "model": title_case(model_name),
                    "date": pd.to_datetime(
                        getattr(bet, "date", None),
                        errors="coerce",
                    ),
                    "game_id": getattr(bet, "game_id", None),
                    "pick_type": readable_pick_type(getattr(bet, "pick_type", None)),
                    "confidence": title_case(getattr(bet, "confidence", None)),
                    "predicted_prob": pd.to_numeric(
                        getattr(bet, "predicted_prob", None),
                        errors="coerce",
                    ),
                    "edge": pd.to_numeric(
                        getattr(bet, "edge", None),
                        errors="coerce",
                    ),
                    "american_odds": pd.to_numeric(
                        getattr(bet, "american_odds", None),
                        errors="coerce",
                    ),
                    "result": normalize_result(getattr(bet, "result", None)),
                    "profit_units": pd.to_numeric(
                        getattr(bet, "profit_units", None),
                        errors="coerce",
                    ),
                }
            )

    history = pd.DataFrame(rows)

    if history.empty:
        return history

    history["profit_units"] = history["profit_units"].fillna(0.0)
    history = history.dropna(subset=["date"]).copy()

    return history.sort_values(
        ["date", "model"],
        ascending=[True, True],
    ).reset_index(drop=True)


def load_live_pick_history() -> pd.DataFrame:
    """Load the current-season captured pick ledger, if it exists."""

    ledger_path = (
        Path(__file__).parent.parent
        / "data_files"
        / "processed"
        / f"pick_history_{datetime.date.today().year}.parquet"
    )

    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        live_history = pd.read_parquet(ledger_path).copy()

        if live_history.empty:
            return pd.DataFrame()

        required_columns = [
            "model",
            "date",
            "game_id",
            "pick_type",
            "confidence",
            "predicted_prob",
            "edge",
            "american_odds",
            "result",
            "profit_units",
        ]

        live_history["model"] = "Live Model"

        live_history["date"] = pd.to_datetime(
            live_history["game_date"]
            if "game_date" in live_history.columns
            else pd.Series(pd.NaT, index=live_history.index),
            errors="coerce",
        )

        live_history["pick_type"] = (
            live_history["market"]
            if "market" in live_history.columns
            else pd.Series("—", index=live_history.index)
        ).map(readable_pick_type)

        live_history["confidence"] = (
            live_history["confidence"]
            if "confidence" in live_history.columns
            else pd.Series("LOW", index=live_history.index)
        ).map(title_case)

        live_history["result"] = (
            live_history["result"]
            if "result" in live_history.columns
            else pd.Series("pending", index=live_history.index)
        ).map(normalize_result)

        live_history["profit_units"] = pd.to_numeric(
            live_history["profit_units"]
            if "profit_units" in live_history.columns
            else pd.Series(0.0, index=live_history.index),
            errors="coerce",
        ).fillna(0.0)

        live_history["predicted_prob"] = pd.to_numeric(
            live_history["predicted_prob"]
            if "predicted_prob" in live_history.columns
            else pd.Series(pd.NA, index=live_history.index),
            errors="coerce",
        )

        live_history["edge"] = pd.to_numeric(
            live_history["edge"]
            if "edge" in live_history.columns
            else pd.Series(pd.NA, index=live_history.index),
            errors="coerce",
        )

        live_history["american_odds"] = pd.to_numeric(
            live_history["american_odds"]
            if "american_odds" in live_history.columns
            else pd.Series(pd.NA, index=live_history.index),
            errors="coerce",
        )

        if "game_id" not in live_history.columns:
            live_history["game_id"] = pd.NA

        return live_history[required_columns].dropna(subset=["date"]).copy()

    except Exception as exc:
        st.warning(f"Could not load live pick history: {exc}")
        return pd.DataFrame()


backtest_history = build_history_dataframe(backtests)
live_history = load_live_pick_history()

history_df = pd.concat(
    [backtest_history, live_history],
    ignore_index=True,
    sort=False,
)


# ── Pick History ──────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Pick History")
    st.markdown(
        "Settled modeled picks with recorded pregame market prices. "
        "This ledger uses saved backtest and pregame-odds records; it does not "
        "retroactively invent historical odds for games whose prices were not captured."
    )

    if history_df.empty:
        st.info(
            "No pick history is available yet. Run the daily pipeline or "
            "backtest scripts to generate saved picks, odds, and outcomes."
        )
    else:
        earliest_pick_date = history_df["date"].min()
        latest_pick_date = history_df["date"].max()

        st.caption(
            f"Saved pick-history coverage: "
            f"{earliest_pick_date.strftime('%B %d, %Y')} through "
            f"{latest_pick_date.strftime('%B %d, %Y')}."
        )

        min_date = earliest_pick_date.date()
        max_date = latest_pick_date.date()

        selected_dates = st.date_input(
            "Pick date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="performance_date_range",
        )

        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            model_options = ["All"] + sorted(history_df["model"].dropna().unique().tolist())
            selected_model = st.selectbox(
                "Model",
                model_options,
                key="performance_model_filter",
            )

        with fc2:
            selected_result = st.selectbox(
                "Result",
                ["All", "win", "loss", "push", "pending"],
                format_func=lambda value: "All" if value == "All" else value.title(),
                key="performance_result_filter",
            )

        with fc3:
            confidence_options = ["All"] + sorted(
                history_df["confidence"].dropna().unique().tolist()
            )
            selected_confidence = st.selectbox(
                "Confidence",
                confidence_options,
                key="performance_confidence_filter",
            )

        with fc4:
            pick_type_options = ["All"] + sorted(history_df["pick_type"].dropna().unique().tolist())
            selected_pick_type = st.selectbox(
                "Pick type",
                pick_type_options,
                key="performance_pick_type_filter",
            )

        filtered = history_df.copy()

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered = filtered[filtered["date"].dt.date.between(start_date, end_date)].copy()

        if selected_model != "All":
            filtered = filtered[filtered["model"] == selected_model]

        if selected_result != "All":
            filtered = filtered[filtered["result"] == selected_result]

        if selected_confidence != "All":
            filtered = filtered[filtered["confidence"] == selected_confidence]

        if selected_pick_type != "All":
            filtered = filtered[filtered["pick_type"] == selected_pick_type]

        settled = filtered[filtered["result"].isin(["win", "loss", "push"])].copy()

        total_picks = len(filtered)
        settled_bets = len(settled)
        wins = int((settled["result"] == "win").sum())
        losses = int((settled["result"] == "loss").sum())
        pushes = int((settled["result"] == "push").sum())
        pending = int((filtered["result"] == "pending").sum())

        win_rate = wins / (wins + losses) if (wins + losses) else 0.0
        total_units = float(filtered["profit_units"].sum())
        avg_edge = float(filtered["edge"].mean()) if not filtered.empty else 0.0
        roi_per_bet = total_units / settled_bets if settled_bets else 0.0

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        m1.metric("Picks", total_picks)
        m2.metric("Record", f"{wins}–{losses}–{pushes}")
        m3.metric("Win Rate", f"{win_rate:.1%}")
        m4.metric("Units", f"{total_units:+.2f}")
        m5.metric("ROI / Bet", f"{roi_per_bet:+.2%}")
        m6.metric("Pending", pending)

        st.caption(f"Settled bets: {settled_bets:,} · Average model edge: {avg_edge:.1%}")

        st.divider()

        if not filtered.empty:
            cumulative = filtered.sort_values(
                ["model", "date"],
                ascending=[True, True],
            ).copy()

            cumulative["cumulative_units"] = cumulative.groupby("model")["profit_units"].cumsum()

            pnl_figure = px.line(
                cumulative,
                x="date",
                y="cumulative_units",
                color="model",
                title="Cumulative P&L (Units)",
                labels={
                    "date": "Date",
                    "cumulative_units": "Cumulative Units",
                    "model": "Model",
                },
            )

            pnl_figure.add_hline(
                y=0,
                line_dash="dot",
                line_color="gray",
            )

            st.plotly_chart(pnl_figure, width="stretch")

        st.markdown("#### Detailed Ledger")

        if filtered.empty:
            st.info("No picks match the current filters.")
        else:
            display_df = filtered[
                [
                    "date",
                    "model",
                    "pick_type",
                    "confidence",
                    "predicted_prob",
                    "edge",
                    "american_odds",
                    "result",
                    "profit_units",
                ]
            ].copy()

            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["predicted_prob"] = display_df["predicted_prob"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            )
            display_df["edge"] = display_df["edge"].map(
                lambda value: f"{value:+.1%}" if pd.notna(value) else "—"
            )
            display_df["american_odds"] = display_df["american_odds"].map(format_american_odds)
            display_df["result"] = display_df["result"].str.title()
            display_df["profit_units"] = display_df["profit_units"].map(
                lambda value: f"{value:+.2f}"
            )

            display_df = (
                display_df.rename(
                    columns={
                        "date": "Date",
                        "model": "Model",
                        "pick_type": "Pick Type",
                        "confidence": "Confidence",
                        "predicted_prob": "Pred. Prob.",
                        "edge": "Edge",
                        "american_odds": "Odds",
                        "result": "Result",
                        "profit_units": "P&L (Units)",
                    }
                )
                .sort_values("Date", ascending=False)
                .reset_index(drop=True)
            )

            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                height=get_dataframe_height(display_df),
            )

            st.download_button(
                "Download filtered pick history CSV",
                data=filtered.to_csv(index=False),
                file_name="filtered_pick_history.csv",
                mime="text/csv",
            )


# ── Model Performance ─────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Model Performance")
    st.markdown("Backtest-derived profitability and calibration-ready performance metrics.")

    if not backtests:
        st.info(
            "No model-performance data is available. Run the evaluation "
            "or training scripts to generate backtest artifacts."
        )
    else:
        leaderboard_rows: list[dict] = []

        for backtest in backtests.values():
            try:
                leaderboard_rows.append(backtest.summary())
            except (AttributeError, TypeError, ValueError):
                continue

        leaderboard = pd.DataFrame(leaderboard_rows)

        if leaderboard.empty:
            st.info("No readable model summaries are available.")
        else:
            if "model" in leaderboard.columns:
                leaderboard["model"] = leaderboard["model"].map(title_case)

            if "pick_type" in leaderboard.columns:
                leaderboard["pick_type"] = leaderboard["pick_type"].map(readable_pick_type)

            if "period" in leaderboard.columns:
                leaderboard["period"] = (
                    leaderboard["period"]
                    .astype(str)
                    .str.replace(
                        r" \d{2}:\d{2}:\d{2}",
                        "",
                        regex=True,
                    )
                    .str.strip()
                )

            if "roi" in leaderboard.columns:
                leaderboard = leaderboard.sort_values(
                    "roi",
                    ascending=False,
                )

            st.markdown("### Leaderboard")

            leaderboard_columns = {
                "model": "Model",
                "pick_type": "Pick Type",
                "total_bets": "Bets",
                "wins": "Wins",
                "losses": "Losses",
                "pushes": "Pushes",
                "win_rate": "Win Rate",
                "total_units": "Units",
                "max_drawdown": "Max Drawdown",
                "roi": "ROI",
                "period": "Period",
            }

            existing_columns = [
                column for column in leaderboard_columns if column in leaderboard.columns
            ]

            st.dataframe(
                leaderboard[existing_columns].rename(columns=leaderboard_columns),
                hide_index=True,
                width="stretch",
                height=get_dataframe_height(leaderboard[existing_columns]),
            )

        if history_df.empty:
            st.info("No individual backtest bets are available for charts.")
        else:
            st.markdown("### Cumulative P&L by Model")

            performance_df = history_df.sort_values(
                ["model", "date"],
                ascending=[True, True],
            ).copy()

            performance_df["cumulative_units"] = performance_df.groupby("model")[
                "profit_units"
            ].cumsum()

            performance_figure = px.line(
                performance_df,
                x="date",
                y="cumulative_units",
                color="model",
                title="Cumulative Units by Model",
                labels={
                    "date": "Date",
                    "cumulative_units": "Cumulative Units",
                    "model": "Model",
                },
            )

            performance_figure.add_hline(
                y=0,
                line_dash="dot",
                line_color="gray",
            )

            st.plotly_chart(performance_figure, width="stretch")

            st.markdown("### Performance by Confidence Tier")

            settled_performance = performance_df[
                performance_df["result"].isin(["win", "loss", "push"])
            ].copy()

            if settled_performance.empty:
                st.info("No settled bets are available for confidence-tier analysis.")
            else:
                tier_summary = (
                    settled_performance.groupby(
                        ["model", "confidence"],
                        dropna=False,
                    )
                    .agg(
                        bets=("profit_units", "count"),
                        wins=(
                            "result",
                            lambda values: (values == "win").sum(),
                        ),
                        losses=(
                            "result",
                            lambda values: (values == "loss").sum(),
                        ),
                        pushes=(
                            "result",
                            lambda values: (values == "push").sum(),
                        ),
                        total_units=("profit_units", "sum"),
                    )
                    .reset_index()
                )

                tier_summary["win_rate"] = (
                    tier_summary["wins"] / (tier_summary["wins"] + tier_summary["losses"])
                ).fillna(0.0)

                tier_summary["roi_per_bet"] = (
                    tier_summary["total_units"] / tier_summary["bets"]
                ).fillna(0.0)

                st.dataframe(
                    tier_summary.rename(
                        columns={
                            "model": "Model",
                            "confidence": "Confidence",
                            "bets": "Bets",
                            "wins": "Wins",
                            "losses": "Losses",
                            "pushes": "Pushes",
                            "win_rate": "Win Rate",
                            "total_units": "Units",
                            "roi_per_bet": "ROI / Bet",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                    height=get_dataframe_height(tier_summary),
                )

                tier_figure = px.bar(
                    tier_summary,
                    x="confidence",
                    y="roi_per_bet",
                    color="model",
                    barmode="group",
                    title="ROI per Bet by Confidence Tier",
                    labels={
                        "confidence": "Confidence",
                        "roi_per_bet": "ROI per Bet",
                        "model": "Model",
                    },
                )

                tier_figure.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="gray",
                )

                st.plotly_chart(tier_figure, width="stretch")


# ── Bankroll ──────────────────────────────────────────────────────────────────
with tab_bankroll:
    st.subheader("Bankroll Management")
    st.markdown(
        "Use the Kelly calculator to size a wager, then simulate the "
        "historical backtest results in dollar terms."
    )

    kelly_column, simulation_column = st.columns(
        [1, 1],
        gap="large",
    )

    with kelly_column:
        st.markdown("#### Kelly Calculator")

        bankroll_size = st.number_input(
            "Bankroll ($)",
            min_value=10,
            max_value=1_000_000,
            value=200,
            step=10,
            key="performance_bankroll_size",
        )

        unit_size = st.number_input(
            "Unit size ($)",
            min_value=1,
            max_value=10_000,
            value=6,
            step=1,
            key="performance_unit_size",
        )

        kelly_confidence = st.selectbox(
            "Confidence tier",
            options=["HIGH", "MEDIUM", "LOW"],
            key="performance_kelly_confidence",
        )

        tier_fractions = {
            "HIGH": 0.50,
            "MEDIUM": 0.25,
            "LOW": 0.125,
        }

        tier_fraction = tier_fractions[kelly_confidence]

        american_odds = st.number_input(
            "American odds (e.g. -110, +150)",
            min_value=-2000,
            max_value=2000,
            value=-110,
            step=5,
            key="performance_kelly_odds",
        )

        implied_probability = _american_to_implied_prob(american_odds)

        st.caption(f"Implied probability: {implied_probability:.1%}")

        default_probability = min(
            max(round(implied_probability + 0.04, 2), 0.01),
            0.99,
        )

        estimated_probability = st.slider(
            "Estimated win probability",
            min_value=0.01,
            max_value=0.99,
            value=default_probability,
            step=0.01,
            key="performance_kelly_probability",
        )

        full_kelly = max(
            _kelly_fraction(
                prob=estimated_probability,
                american_odds=american_odds,
            ),
            0.0,
        )

        applied_kelly = full_kelly * tier_fraction
        bet_size = bankroll_size * applied_kelly
        units_to_bet = bet_size / max(unit_size, 1)

        st.divider()

        kc1, kc2 = st.columns(2)
        kc1.metric("Full Kelly", f"{full_kelly:.2%}")
        kc2.metric("Tier Fraction", f"{tier_fraction:.1%}")

        kc3, kc4 = st.columns(2)
        kc3.metric("Applied Kelly", f"{applied_kelly:.2%}")
        kc4.metric("Bet Size", f"${bet_size:,.2f}")

        st.metric("Units to Bet", f"{units_to_bet:.2f}")

        with st.expander("Kelly Formula"):
            st.latex(
                r"f^* = \frac{bp-q}{b}"
                r"\quad\text{where } b=\text{decimal payout},\;"
                r"p=\hat{p},\;q=1-p"
            )

    with simulation_column:
        st.markdown("#### Historical Bankroll Simulation")

        if history_df.empty:
            st.info("No settled pick history is available for a bankroll simulation.")
        else:
            model_names = sorted(history_df["model"].unique().tolist())

            simulation_model = st.selectbox(
                "Select model",
                options=model_names,
                key="performance_simulation_model",
            )

            starting_bankroll = st.number_input(
                "Starting bankroll ($)",
                min_value=100,
                max_value=1_000_000,
                value=int(bankroll_size),
                step=100,
                key="performance_simulation_start",
            )

            simulation_unit = st.number_input(
                "Simulation unit size ($)",
                min_value=1,
                max_value=10_000,
                value=int(unit_size),
                step=1,
                key="performance_simulation_unit",
            )

            simulation_df = history_df[history_df["model"] == simulation_model].copy()

            simulation_df = simulation_df[
                simulation_df["result"].isin(["win", "loss", "push"])
            ].copy()

            simulation_df = simulation_df.sort_values("date").reset_index(drop=True)

            if simulation_df.empty:
                st.info("No settled bets are available for this model.")
            else:
                simulation_df["pnl_dollars"] = simulation_df["profit_units"] * simulation_unit

                simulation_df["bankroll"] = (
                    starting_bankroll + simulation_df["pnl_dollars"].cumsum()
                )

                simulation_df["running_peak"] = simulation_df["bankroll"].cummax()

                simulation_df["drawdown"] = (
                    simulation_df["bankroll"] - simulation_df["running_peak"]
                ) / simulation_df["running_peak"].clip(lower=1)

                simulation_figure = go.Figure()

                simulation_figure.add_trace(
                    go.Scatter(
                        x=simulation_df["date"],
                        y=simulation_df["bankroll"],
                        mode="lines",
                        name="Bankroll",
                        line=dict(
                            color="#2563eb",
                            width=2,
                        ),
                    )
                )

                simulation_figure.add_hline(
                    y=starting_bankroll,
                    line_dash="dot",
                    line_color="gray",
                )

                simulation_figure.update_layout(
                    title=f"{simulation_model} — Bankroll Growth",
                    xaxis_title="Date",
                    yaxis_title="Bankroll ($)",
                    yaxis_tickprefix="$",
                    height=380,
                )

                st.plotly_chart(
                    simulation_figure,
                    width="stretch",
                )

                final_bankroll = simulation_df["bankroll"].iloc[-1]
                total_pnl = final_bankroll - starting_bankroll
                peak_bankroll = simulation_df["running_peak"].max()
                max_drawdown = abs(simulation_df["drawdown"].min())

                sc1, sc2, sc3 = st.columns(3)

                sc1.metric(
                    "Final Bankroll",
                    f"${final_bankroll:,.0f}",
                    delta=f"${total_pnl:+,.0f}",
                )

                sc2.metric(
                    "Peak Bankroll",
                    f"${peak_bankroll:,.0f}",
                )

                sc3.metric(
                    "Max Drawdown",
                    f"{max_drawdown:.1%}",
                )
