"""Page: ML models, feature exploration, evaluation, and Savant research."""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from page_utils import READABLE_COLS, _load_precomputed, init_session_state
from retrosheet import MODERN_START, load_gameinfo, season_standings
from src.evaluation.calibration import calibration_plot_data
from src.top_nav import inject_app_style, render_top_nav

inject_app_style()
render_top_nav()

init_session_state()

MIN_YEAR = 2020
CURRENT_YEAR = datetime.date.today().year

_pre = _load_precomputed()

_pre_gameinfo = _pre.get("gameinfo", pd.DataFrame())
_pre_gameinfo_max_year = (
    int(_pre_gameinfo["season"].max())
    if not _pre_gameinfo.empty and "season" in _pre_gameinfo.columns
    else MIN_YEAR
)

_live_gameinfo = load_gameinfo(
    min_year=MODERN_START,
    max_year=CURRENT_YEAR,
)

_live_gameinfo_max_year = (
    int(_live_gameinfo["season"].max())
    if not _live_gameinfo.empty and "season" in _live_gameinfo.columns
    else _pre_gameinfo_max_year
)

GAMEINFO_MAX_YEAR = max(_pre_gameinfo_max_year, _live_gameinfo_max_year)

features_df = _pre.get("model_features", pd.DataFrame()).copy()

if not features_df.empty and "season" in features_df.columns:
    features_df = features_df[
        features_df["season"].between(MIN_YEAR, CURRENT_YEAR)
    ].copy()

init_session_state(features_df=features_df)


def get_dataframe_height(
    df: pd.DataFrame,
    row_height: int = 35,
    header_height: int = 38,
    padding: int = 2,
    max_height: int | None = 600,
) -> int:
    calculated_height = (len(df) * row_height) + header_height + padding

    if max_height is None:
        return calculated_height

    return min(calculated_height, max_height)


def _build_feature_matrix(
    gi: pd.DataFrame,
    standings: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-game model features from final-game results and team standings."""

    if gi.empty or standings.empty:
        return pd.DataFrame()

    required_game_columns = {"hometeam", "visteam", "wteam"}
    required_stat_columns = {
        "WPct",
        "RS_per_G",
        "RA_per_G",
        "RD_per_G",
        "PythWPct",
    }

    if not required_game_columns.issubset(gi.columns):
        return pd.DataFrame()

    if not required_stat_columns.issubset(standings.columns):
        return pd.DataFrame()

    gi_valid = gi[
        gi["hometeam"].isin(standings.index)
        & gi["visteam"].isin(standings.index)
    ].copy()

    if gi_valid.empty:
        return pd.DataFrame()

    stat_columns = [
        "WPct",
        "RS_per_G",
        "RA_per_G",
        "RD_per_G",
        "PythWPct",
    ]

    home_stats = standings[stat_columns].add_prefix("home_")
    visitor_stats = standings[stat_columns].add_prefix("vis_")

    merged = (
        gi_valid.merge(
            home_stats,
            left_on="hometeam",
            right_index=True,
            how="inner",
        )
        .merge(
            visitor_stats,
            left_on="visteam",
            right_index=True,
            how="inner",
        )
    )

    if merged.empty:
        return pd.DataFrame()

    merged["WPct_diff"] = merged["home_WPct"] - merged["vis_WPct"]
    merged["PythWPct_diff"] = (
        merged["home_PythWPct"] - merged["vis_PythWPct"]
    )
    merged["RS_advantage"] = (
        merged["home_RS_per_G"] - merged["vis_RS_per_G"]
    )
    merged["RA_advantage"] = (
        merged["vis_RA_per_G"] - merged["home_RA_per_G"]
    )
    merged["home_win"] = (
        merged["wteam"] == merged["hometeam"]
    ).astype(int)

    merged = merged.rename(
        columns={
            "visteam": "visitor",
            "hometeam": "home_team",
            "home_RS_per_G": "home_RS_G",
            "home_RA_per_G": "home_RA_G",
            "home_RD_per_G": "home_RD_G",
            "vis_RS_per_G": "vis_RS_G",
            "vis_RA_per_G": "vis_RA_G",
            "vis_RD_per_G": "vis_RD_G",
        }
    )

    keep_columns = [
        "date",
        "visitor",
        "home_team",
        "home_WPct",
        "vis_WPct",
        "home_RS_G",
        "home_RA_G",
        "vis_RS_G",
        "vis_RA_G",
        "home_RD_G",
        "vis_RD_G",
        "home_PythWPct",
        "vis_PythWPct",
        "WPct_diff",
        "PythWPct_diff",
        "RS_advantage",
        "RA_advantage",
        "daynight",
        "attendance",
        "temp",
        "windspeed",
        "home_win",
        "total_runs",
    ]

    keep_columns = [
        column for column in keep_columns if column in merged.columns
    ]

    return merged[keep_columns].reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def cached_gameinfo(year: int) -> pd.DataFrame:
    return load_gameinfo(min_year=year, max_year=year)


@st.cache_data(ttl=900, show_spinner=False)
def cached_current_standings(year: int) -> pd.DataFrame:
    return season_standings(min_year=year, max_year=year)


tab_feat, tab_models, tab_eval, tab_savant = st.tabs(
    [
        "Betting Features",
        "ML Models",
        "Model Evaluation",
        "Savant Research",
    ]
)


# ── Betting Features ──────────────────────────────────────────────────────────
with tab_feat:
    st.subheader("Engineered Betting Features")
    st.markdown(
        "Feature matrix built from season-level stats — designed as inputs for ML models."
    )

    historical_standings = _pre.get("standings", pd.DataFrame()).copy()

    if (
        not historical_standings.empty
        and "season" in historical_standings.columns
    ):
        historical_standings["season"] = pd.to_numeric(
            historical_standings["season"],
            errors="coerce",
        )

    current_standings = cached_current_standings(CURRENT_YEAR).copy()

    if (
        not current_standings.empty
        and "season" in current_standings.columns
    ):
        current_standings["season"] = pd.to_numeric(
            current_standings["season"],
            errors="coerce",
        )

        if not historical_standings.empty:
            historical_standings = historical_standings[
                historical_standings["season"].ne(CURRENT_YEAR)
            ]

        all_standings = pd.concat(
            [historical_standings, current_standings],
            ignore_index=True,
            sort=False,
        )
    else:
        all_standings = historical_standings.copy()

    if all_standings.empty or "season" not in all_standings.columns:
        st.error("No season standings data is available.")
        st.stop()

    available_feature_years = sorted(
        [
            int(year)
            for year in all_standings["season"].dropna().unique()
            if MIN_YEAR <= int(year) <= GAMEINFO_MAX_YEAR
        ],
        reverse=True,
    )

    if not available_feature_years:
        st.error(
            "No overlapping seasons exist between standings data and game-level data."
        )
        st.stop()

    feat_season = st.selectbox(
        "Season",
        available_feature_years,
        key="feat_season",
    )

    with st.spinner("Loading game-level feature data…"):
        gi = cached_gameinfo(int(feat_season)).copy()

    if gi.empty:
        st.info(f"No game-level data is available for {feat_season}.")
        st.stop()

    if "date" not in gi.columns:
        st.error("The game data does not contain a date column.")
        st.stop()

    gi["date"] = pd.to_datetime(
        gi["date"],
        errors="coerce",
    )

    latest_game_date = gi["date"].max()
    earliest_game_date = gi["date"].min()

    if pd.notna(latest_game_date):
        st.caption(
            f"Game-level data currently runs from "
            f"{earliest_game_date.strftime('%B %d, %Y')} through "
            f"{latest_game_date.strftime('%B %d, %Y')}."
        )
    else:
        st.caption(
            f"Game-level data is available through {GAMEINFO_MAX_YEAR}."
        )

    season_standings_df = all_standings[
        all_standings["season"].eq(int(feat_season))
    ].copy()

    if season_standings_df.empty or "team" not in season_standings_df.columns:
        st.info(f"No standings data is available for {feat_season}.")
        st.stop()

    season_standings_df = season_standings_df.set_index("team")

    with st.spinner("Building feature matrix…"):
        feat_df = _build_feature_matrix(
            gi,
            season_standings_df,
        )

    if feat_df.empty:
        st.info(
            "No games with complete standings coverage are available for this season."
        )
    else:
        feat_df["date"] = pd.to_datetime(
            feat_df["date"],
            errors="coerce",
        )

        st.markdown(
            f"**{len(feat_df):,} games** in {feat_season} with full feature coverage."
        )

        valid_dates = feat_df["date"].dropna()

        if valid_dates.empty:
            st.error("Feature rows do not contain valid game dates.")
            st.stop()

        min_feature_date = valid_dates.min().date()
        max_feature_date = valid_dates.max().date()

        date_range = st.date_input(
            "Filter game dates",
            value=(min_feature_date, max_feature_date),
            min_value=min_feature_date,
            max_value=max_feature_date,
            key="feature_date_range",
        )

        filtered_feat = feat_df.copy()

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range

            filtered_feat = filtered_feat[
                filtered_feat["date"].dt.date.between(
                    start_date,
                    end_date,
                )
            ].copy()

        st.caption(
            f"Showing {len(filtered_feat):,} games from the selected date range. "
            "The table lists the 50 most recent games first."
        )

        display_feat = (
            filtered_feat.sort_values(
                "date",
                ascending=False,
            )
            .head(50)
            .copy()
        )

        display_feat["date"] = display_feat["date"].dt.date

        feat_rename = {
            **READABLE_COLS,
            "visitor": "Visitor",
            "home_team": "Home",
            "home_WPct": "Home Win %",
            "vis_WPct": "Visitor Win %",
            "home_RS_G": "Home RS/G",
            "home_RA_G": "Home RA/G",
            "vis_RS_G": "Visitor RS/G",
            "vis_RA_G": "Visitor RA/G",
            "home_RD_G": "Home RD/G",
            "vis_RD_G": "Visitor RD/G",
            "home_PythWPct": "Home Pyth W%",
            "vis_PythWPct": "Visitor Pyth W%",
            "WPct_diff": "Win % Diff",
            "PythWPct_diff": "Pyth W% Diff",
            "RS_advantage": "RS Advantage",
            "RA_advantage": "RA Advantage",
            "daynight": "Day/Night",
            "attendance": "Attendance",
            "temp": "Temperature",
            "windspeed": "Wind Speed",
            "home_win": "Home Win?",
            "total_runs": "Total Runs",
        }

        st.dataframe(
            display_feat.rename(columns=feat_rename),
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(display_feat),
        )

        numerical_features = [
            "home_WPct",
            "vis_WPct",
            "WPct_diff",
            "PythWPct_diff",
            "home_RS_G",
            "home_RA_G",
            "vis_RS_G",
            "vis_RA_G",
            "RS_advantage",
            "RA_advantage",
            "home_win",
            "total_runs",
        ]

        numerical_features = [
            column
            for column in numerical_features
            if column in filtered_feat.columns
        ]

        if len(numerical_features) > 1:
            readable_names = {
                "home_WPct": "Home Win %",
                "vis_WPct": "Visitor Win %",
                "WPct_diff": "Win % Diff",
                "PythWPct_diff": "Pyth W% Diff",
                "home_RS_G": "Home RS/G",
                "home_RA_G": "Home RA/G",
                "vis_RS_G": "Visitor RS/G",
                "vis_RA_G": "Visitor RA/G",
                "RS_advantage": "RS Advantage",
                "RA_advantage": "RA Advantage",
                "home_win": "Home Win?",
                "total_runs": "Total Runs",
            }

            correlation = filtered_feat[numerical_features].corr()
            correlation.index = [
                readable_names.get(column, column)
                for column in correlation.index
            ]
            correlation.columns = [
                readable_names.get(column, column)
                for column in correlation.columns
            ]

            fig = px.imshow(
                correlation,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto",
            )

            st.plotly_chart(fig, width="stretch")

        with st.expander("Feature data dictionary"):
            dictionary_rows = [
                ("home_WPct", "Home Win %, season standings"),
                ("vis_WPct", "Visitor Win %, season standings"),
                ("WPct_diff", "Home Win % minus visitor Win %"),
                ("PythWPct_diff", "Home Pythagorean Win % minus visitor"),
                ("home_RS_G", "Home runs scored per game"),
                ("home_RA_G", "Home runs allowed per game"),
                ("vis_RS_G", "Visitor runs scored per game"),
                ("vis_RA_G", "Visitor runs allowed per game"),
                ("RS_advantage", "Home RS/G minus visitor RS/G"),
                ("RA_advantage", "Visitor RA/G minus home RA/G"),
                ("home_win", "Indicator: 1 if the home team won"),
                ("total_runs", "Total runs scored in the game"),
            ]

            dictionary_df = pd.DataFrame(
                dictionary_rows,
                columns=["column", "description"],
            )

            st.dataframe(
                dictionary_df.rename(columns=READABLE_COLS),
                width="stretch",
                hide_index=True,
            )

        st.markdown("#### Home Win % over time")

        trend_max_year = min(CURRENT_YEAR, GAMEINFO_MAX_YEAR)

        if MIN_YEAR > trend_max_year:
            st.info(
                f"Home-field and weather trend data are available only through "
                f"{GAMEINFO_MAX_YEAR}."
            )
        else:
            with st.spinner("Loading historical trend data…"):
                gi_all = load_gameinfo(MIN_YEAR, trend_max_year).copy()

            if not gi_all.empty and {
                "wteam",
                "hometeam",
                "season",
            }.issubset(gi_all.columns):
                gi_all["home_win"] = (
                    gi_all["wteam"] == gi_all["hometeam"]
                ).astype(int)

                home_field_advantage = (
                    gi_all.groupby("season")["home_win"]
                    .mean()
                    .reset_index()
                )
                home_field_advantage.columns = [
                    "season",
                    "home_win_pct",
                ]

                fig2 = px.line(
                    home_field_advantage,
                    x="season",
                    y="home_win_pct",
                    title="Home Field Advantage — Win % by Season",
                    labels={
                        "home_win_pct": "Home Win %",
                        "season": "Season",
                    },
                )

                fig2.add_hline(
                    y=0.5,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="50%",
                )

                st.plotly_chart(fig2, width="stretch")

                if (
                    "temp" in gi_all.columns
                    and "total_runs" in gi_all.columns
                    and gi_all["temp"].notna().sum() > 100
                ):
                    temp_df = gi_all.dropna(
                        subset=["temp", "total_runs"]
                    ).copy()

                    temp_df = temp_df[temp_df["temp"] > 0]

                    if not temp_df.empty:
                        fig3 = px.scatter(
                            temp_df.sample(
                                min(5000, len(temp_df)),
                                random_state=42,
                            ),
                            x="temp",
                            y="total_runs",
                            trendline="lowess",
                            title="Temperature vs Total Runs (Sample)",
                            labels={
                                "temp": "Temp (°F)",
                                "total_runs": "Total Runs",
                            },
                            opacity=0.4,
                        )

                        st.plotly_chart(fig3, width="stretch")

        with st.expander("Download feature CSV"):
            st.download_button(
                label="Download features.csv",
                data=filtered_feat.to_csv(index=False),
                file_name=f"features_{feat_season}.csv",
                mime="text/csv",
            )


# ── ML Models ─────────────────────────────────────────────────────────────────
with tab_models:
    st.subheader("ML Betting Models")

    st.markdown(
        """
Three XGBoost classifiers trained on **2020+ Retrosheet** game data.
All models use a **chronological train/test split** to avoid lookahead.

| Model | Target | Features |
|---|---|---|
| **Moneyline** | P(home team wins) | Team stats, SP stats, weather |
| **Spread** | P(home covers −1.5) | Same as moneyline |
| **Over/Under** | P(total > expected) | Same + expected-total offset |
"""
    )

    results = st.session_state.get("ml_results")

    if results is None:
        st.info(
            "Pre-trained results not found. Run `scripts/train_models.py` to generate them."
        )
    else:
        model_labels = {
            "moneyline": "🏆 Moneyline (P home win)",
            "spread": "📏 Spread (P home covers −1.5)",
            "totals": "📈 Over/Under (P went over)",
        }

        st.markdown("### Model Performance (Test Set)")

        c1, c2, c3 = st.columns(3)

        for column, key in zip(
            [c1, c2, c3],
            ["moneyline", "spread", "totals"],
        ):
            metrics = results[key]["metrics"]

            column.markdown(f"**{model_labels[key]}**")
            column.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            column.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            column.metric("Brier", f"{metrics['brier_score']:.4f}")
            column.metric("Log Loss", f"{metrics['log_loss']:.4f}")

        with st.expander("ℹ️ Metric guide"):
            st.markdown(
                """
- **ROC-AUC:** 0.5 is random; 1.0 is perfect.
- **Accuracy:** Fraction of correct binary picks on the test set.
- **Brier Score:** Mean squared error of predicted probabilities; lower is better.
- **Log Loss:** Cross-entropy loss; lower is better.
"""
            )

        st.markdown("### Feature Importances")

        feature_labels = {
            "WPct_diff": "Win % Diff",
            "PythWPct_diff": "Pythagorean W% Diff",
            "sp_ERA_gap": "SP ERA Gap",
            "home_WPct": "Home Win %",
            "away_WPct": "Away Win %",
            "home_PythWPct": "Home Pyth W%",
            "away_PythWPct": "Away Pyth W%",
            "home_RS_G": "Home RS / G",
            "home_RA_G": "Home RA / G",
            "away_RS_G": "Away RS / G",
            "away_RA_G": "Away RA / G",
            "home_RD_G": "Home RD / G",
            "away_RD_G": "Away RD / G",
            "ERA_diff": "ERA Diff",
            "WHIP_diff": "WHIP Diff",
            "temp": "Temperature (°F)",
            "windspeed": "Wind Speed",
            "is_day": "Day Game?",
            "exp_total": "Expected Total Runs",
        }

        def importance_chart(
            model_key: str,
            top_n: int = 20,
        ) -> None:
            importance = results[model_key]["importances"].head(top_n).copy()

            importance["label"] = importance["feature"].map(
                lambda value: feature_labels.get(value, value)
            )

            fig = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="label",
                orientation="h",
                title=f"Top {top_n} Features — {model_labels[model_key]}",
                labels={
                    "importance": "Importance",
                    "label": "Feature",
                },
                color="importance",
                color_continuous_scale="Blues",
            )

            fig.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
            )

            st.plotly_chart(fig, width="stretch")

        imp_tab_ml, imp_tab_sp, imp_tab_ou = st.tabs(
            ["Moneyline", "Spread", "Over/Under"]
        )

        with imp_tab_ml:
            importance_chart("moneyline")

        with imp_tab_sp:
            importance_chart("spread")

        with imp_tab_ou:
            importance_chart("totals")

        st.markdown("### Calibration: Predicted vs Actual Win Rate")

        def calibration_chart(
            model_key: str,
            probability_column: str,
            actual_column: str,
            label: str,
        ) -> None:
            test_df = results[model_key]["test_df"][
                [probability_column, actual_column]
            ].copy()

            test_df["bin"] = pd.cut(
                test_df[probability_column],
                bins=10,
            )

            calibration = (
                test_df.groupby("bin", observed=False)
                .agg(
                    mean_pred=(probability_column, "mean"),
                    actual_rate=(actual_column, "mean"),
                    count=(probability_column, "count"),
                )
                .reset_index()
                .dropna()
            )

            fig = px.scatter(
                calibration,
                x="mean_pred",
                y="actual_rate",
                size="count",
                title=f"{model_labels[model_key]} — Calibration",
                labels={
                    "mean_pred": f"Mean predicted {label}",
                    "actual_rate": "Actual rate",
                },
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(dash="dot", color="gray"),
            )

            st.plotly_chart(fig, width="stretch")

        cal_tab_ml, cal_tab_sp, cal_tab_ou = st.tabs(
            ["Moneyline", "Spread", "Over/Under"]
        )

        with cal_tab_ml:
            calibration_chart(
                "moneyline",
                "pred_prob",
                "home_win",
                "home win probability",
            )

        with cal_tab_sp:
            calibration_chart(
                "spread",
                "pred_prob",
                "home_cover",
                "P(cover −1.5)",
            )

        with cal_tab_ou:
            calibration_chart(
                "totals",
                "pred_prob_over",
                "went_over",
                "P(over)",
            )

        st.markdown("### Backtest Sample — Recent Test-Set Games")

        bt_model = st.selectbox(
            "Model",
            ["moneyline", "spread", "totals"],
            format_func=lambda value: model_labels[value],
            key="bt_model_sel",
        )

        n_show = st.slider(
            "Games to display",
            25,
            200,
            50,
            key="bt_n_show",
        )

        bt_df = results[bt_model]["test_df"].tail(n_show).copy()

        try:
            bt_df["date"] = pd.to_datetime(bt_df["date"]).dt.date
        except (ValueError, TypeError, KeyError):
            pass

        if bt_model == "moneyline" and "pred_win" in bt_df.columns:
            bt_df["pred_win"] = bt_df["pred_win"].map(
                {1: "Home", 0: "Away"}
            )

        if bt_model == "spread" and "pred_cover" in bt_df.columns:
            bt_df["pred_cover"] = bt_df["pred_cover"].map(
                {1: "Home −1.5", 0: "Away +1.5"}
            )

        if "correct" in bt_df.columns:
            bt_df["correct"] = bt_df["correct"].astype(bool).map(
                {True: "✔", False: ""}
            )

        for column in ("home_win", "home_cover", "went_over"):
            if column in bt_df.columns:
                bt_df[column] = bt_df[column].astype(bool).map(
                    {True: "✔", False: ""}
                )

        if bt_model == "moneyline":
            display_columns = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "hruns": "H Runs",
                "vruns": "V Runs",
                "home_win": "Actually Won?",
                "pred_prob": "Pred. Prob",
                "pred_win": "Model Pick",
                "correct": "Correct?",
            }
        elif bt_model == "spread":
            display_columns = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "home_margin": "Margin",
                "home_cover": "Actually Covered?",
                "pred_prob": "P(cover −1.5)",
                "pred_cover": "Model Pick",
                "correct": "Correct?",
            }
        else:
            display_columns = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "total_runs": "Total Runs",
                "exp_total": "Exp Total",
                "went_over": "Went Over?",
                "pred_prob_over": "P(over)",
                "pick_side": "Pick",
                "correct": "Correct?",
            }

        existing_columns = [
            column
            for column in display_columns
            if column in bt_df.columns
        ]

        display_bt = (
            bt_df[existing_columns]
            .reset_index(drop=True)
            .rename(columns=display_columns)
        )

        st.dataframe(
            display_bt,
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(display_bt),
        )

        with st.expander("Download Backtest CSVs"):
            c_dl1, c_dl2, c_dl3 = st.columns(3)

            for column, key, label in zip(
                [c_dl1, c_dl2, c_dl3],
                ["moneyline", "spread", "totals"],
                ["moneyline", "spread", "totals"],
            ):
                column.download_button(
                    label=f"Download {label}.csv",
                    data=results[key]["test_df"].to_csv(index=False),
                    file_name=f"backtest_{label}.csv",
                    mime="text/csv",
                )


# ── Model Evaluation ──────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Model Evaluation")

    st.markdown(
        "Walk-forward backtests, calibration analyses, and profitability reports. "
        "Results are pre-loaded from parquet files generated by the evaluation scripts."
    )

    evaluation_backtests = st.session_state.get("eval_backtests")

    if evaluation_backtests:
        leaderboard = [
            backtest.summary()
            for backtest in evaluation_backtests.values()
        ]

        leaderboard_df = pd.DataFrame(leaderboard).sort_values(
            "roi",
            ascending=False,
        )

        if "period" in leaderboard_df.columns:
            leaderboard_df["period"] = (
                leaderboard_df["period"]
                .astype(str)
                .str.strip()
                .str.slice(0, 10)
            )

        st.markdown("### Backtest Leaderboard")

        st.dataframe(
            leaderboard_df.rename(
                columns={
                    "model": "Model",
                    "pick_type": "Pick Type",
                    "period": "Period",
                    "total_bets": "Bets",
                    "wins": "Wins",
                    "losses": "Losses",
                    "pushes": "Pushes",
                    "win_rate": "Win Rate",
                    "total_units": "Units",
                    "max_drawdown": "Max Drawdown",
                    "roi": "ROI",
                }
            ),
            hide_index=True,
            width="stretch",
            height=get_dataframe_height(leaderboard_df),
        )

        st.markdown("### Calibration Charts")

        for name, backtest in evaluation_backtests.items():
            actual = [
                1 if bet.result == "win" else 0
                for bet in backtest.bets
            ]

            probabilities = [
                bet.predicted_prob
                for bet in backtest.bets
            ]

            cal_data = calibration_plot_data(
                np.array(actual),
                np.array(probabilities),
            )

            fig = px.scatter(
                x=cal_data["mean_predicted"],
                y=cal_data["fraction_positive"],
                size=[1] * len(cal_data["mean_predicted"]),
                title=f"{name.capitalize()} Calibration",
                labels={
                    "x": "Mean Predicted Probability",
                    "y": "Actual Rate",
                },
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(dash="dot", color="gray"),
            )

            st.plotly_chart(fig, width="stretch")

    else:
        st.info(
            "No evaluation data yet. Run `scripts/run_evaluation.py` "
            "or `scripts/train_models.py`."
        )


# ── Savant Research ───────────────────────────────────────────────────────────
with tab_savant:
    mc_ranking = _pre.get("mc_ranking")
    mc_trials = _pre.get("mc_trials")
    savant_metrics = _pre.get("savant_metrics")

    st.subheader("Savant Feature Research — Monte Carlo Selection")

    st.markdown(
        "Results of a Monte Carlo search over Baseball Savant advanced metrics. "
        "Random subsets of Statcast columns are evaluated across three bet targets. "
        "Features are ranked by how often they appeared in the top 10% of trials by ROC-AUC."
    )

    if mc_ranking is None or mc_ranking.empty:
        st.info(
            "Monte Carlo results not found. "
            "Run `python scripts/monte_carlo_features.py --trials 1000` "
            "to generate them."
        )
    else:
        baseline_auc = {
            "moneyline": 0.6253,
            "spread": 0.6304,
            "totals": 0.6157,
        }

        n_valid = len(mc_trials) if mc_trials is not None else None

        top_cutoff = (
            mc_trials["mean_auc"].quantile(0.90)
            if mc_trials is not None
            and "mean_auc" in mc_trials.columns
            else None
        )

        col_m, col_s, col_t, col_v = st.columns(4)

        col_m.metric(
            "Baseline Moneyline AUC",
            f"{baseline_auc['moneyline']:.4f}",
        )
        col_s.metric(
            "Baseline Spread AUC",
            f"{baseline_auc['spread']:.4f}",
        )
        col_t.metric(
            "Baseline Totals AUC",
            f"{baseline_auc['totals']:.4f}",
        )
        col_v.metric(
            "Valid Trials",
            f"{n_valid:,}" if n_valid is not None else "—",
        )

        if savant_metrics is not None and not savant_metrics.empty:
            st.markdown("---")
            st.markdown("#### Savant-Enriched Model Performance")

            performance_columns = st.columns(3)

            for index, model_name in enumerate(
                ["moneyline", "spread", "totals"]
            ):
                row = savant_metrics[
                    savant_metrics["model"] == model_name
                ]

                if row.empty:
                    continue

                auc = float(row.iloc[0]["roc_auc"])

                performance_columns[index].metric(
                    label=f"{model_name.capitalize()} AUC (Savant)",
                    value=f"{auc:.4f}",
                    delta=(
                        f"{auc - baseline_auc[model_name]:+.4f} "
                        "vs baseline"
                    ),
                )

        if mc_trials is not None and not mc_trials.empty:
            st.markdown("---")

            auc_columns = [
                column
                for column in mc_trials.columns
                if column.endswith("_auc") and column != "mean_auc"
            ]

            if auc_columns:
                auc_long = mc_trials[
                    auc_columns + ["mean_auc"]
                ].melt(
                    var_name="target",
                    value_name="auc",
                )

                auc_long["target"] = (
                    auc_long["target"]
                    .str.replace("_auc", "", regex=False)
                    .str.capitalize()
                )

                fig_dist = px.box(
                    auc_long[auc_long["target"] != "Mean"],
                    x="target",
                    y="auc",
                    color="target",
                    points=False,
                    title="Trial AUC by Bet Target",
                    labels={
                        "target": "Bet Target",
                        "auc": "ROC-AUC",
                    },
                )

                for target_name, baseline in [
                    ("Moneyline", baseline_auc["moneyline"]),
                    ("Spread", baseline_auc["spread"]),
                    ("Totals", baseline_auc["totals"]),
                ]:
                    fig_dist.add_hline(
                        y=baseline,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"{target_name} baseline",
                        annotation_position="right",
                    )

                if top_cutoff is not None:
                    fig_dist.add_hline(
                        y=top_cutoff,
                        line_dash="dash",
                        line_color="#7c3aed",
                        annotation_text="Top 10% cutoff",
                        annotation_position="left",
                    )

                st.plotly_chart(fig_dist, width="stretch")

        bat_ranks = mc_ranking[
            mc_ranking["type"] == "batter"
        ].head(20).copy()

        pit_ranks = mc_ranking[
            mc_ranking["type"] == "pitcher"
        ].head(20).copy()

        bat_ranks["appearance_pct"] = (
            bat_ranks["appearance_rate"] * 100
        )

        pit_ranks["appearance_pct"] = (
            pit_ranks["appearance_rate"] * 100
        )

        st.markdown("#### Top Batter Features")

        fig_bat = px.bar(
            bat_ranks.sort_values("appearance_pct"),
            x="appearance_pct",
            y="feature",
            orientation="h",
            title="Batter Feature Selection Frequency",
            labels={
                "appearance_pct": "Appearance Rate in Top Trials (%)",
                "feature": "Savant Column",
            },
            color="appearance_pct",
            color_continuous_scale="Blues",
        )

        fig_bat.update_layout(
            coloraxis_showscale=False,
            height=500,
        )

        st.plotly_chart(fig_bat, width="stretch")

        st.markdown("#### Top Pitcher Features")

        fig_pit = px.bar(
            pit_ranks.sort_values("appearance_pct"),
            x="appearance_pct",
            y="feature",
            orientation="h",
            title="Pitcher Feature Selection Frequency",
            labels={
                "appearance_pct": "Appearance Rate in Top Trials (%)",
                "feature": "Savant Column",
            },
            color="appearance_pct",
            color_continuous_scale="Reds",
        )

        fig_pit.update_layout(
            coloraxis_showscale=False,
            height=500,
        )

        st.plotly_chart(fig_pit, width="stretch")

        with st.expander("Full Feature Ranking Table"):
            display_rank = mc_ranking.copy()

            display_rank["appearance_rate"] = (
                display_rank["appearance_rate"] * 100
            ).round(1)

            st.dataframe(
                display_rank.rename(
                    columns={
                        "feature": "Savant Column",
                        "type": "Type",
                        "top_trial_appearances": "Appearances in Top Trials",
                        "appearance_rate": "Rate (%)",
                    }
                ),
                hide_index=True,
                width="stretch",
                height=get_dataframe_height(display_rank),
            )
