"""Daily picks generation pipeline.

Run the full morning pipeline (8 AM – 11:25 AM window):
    1. Fetch schedule + probable pitchers
    2. Fetch consensus odds
    3. Fetch weather
    4. Build feature matrix
    5. Run all 3 models
    6. Filter by edge / confidence thresholds
    7. Store picks to CSV  (parquet append coming later)

The morning consensus odds snapshot is saved so the 4 PM afternoon_refresh
job can compare lines and detect significant movement.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ingestion.mlb_stats import fetch_probable_pitchers
from src.ingestion.odds import fetch_current_odds, get_consensus_line
from src.ingestion.weather import fetch_weather_for_games
from src.models.features import (
    SPREAD_FEATURES,
    TOTAL_FEATURES,
    UNDERDOG_FEATURES,
)
from src.models.underdog_model import predict_underdog
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals

logger = logging.getLogger(__name__)

MLB_TZ = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data_files" / "processed"

# Minimum thresholds to publish a pick.
MIN_EDGE_UNDERDOG = 0.03
MIN_EDGE_SPREAD = 0.03
MIN_EDGE_TOTALS = 0.025
MIN_CONFIDENCE = 0.30


def run_daily_pipeline(
    target_date: Optional[date] = None,
) -> dict[str, list[dict]]:
    """Run the daily MLB picks pipeline for a specified MLB schedule date.

    The default date is resolved in America/New_York so GitHub Actions'
    UTC clock does not accidentally shift an MLB slate to the next day.

    Args:
        target_date: MLB schedule date to process. Defaults to today in ET.

    Returns:
        Dictionary containing eligible picks by market type.
    """
    target_date = target_date or datetime.now(MLB_TZ).date()
    logger.info("Running daily picks pipeline for %s", target_date)

    # ---- Step 1: Fetch target-date schedule and probable pitchers ----
    schedule = fetch_probable_pitchers(target_date)

    if schedule.empty:
        logger.warning("No games found for %s", target_date)
        return _empty_picks()

    _require_columns(
        schedule,
        ["game_id", "away_team", "home_team"],
        context="schedule",
    )

    logger.info("Found %d scheduled games for %s", len(schedule), target_date)

    # ---- Step 2: Fetch and normalize available market odds ----
    odds_raw = fetch_current_odds()
    if odds_raw.empty:
        logger.warning("No odds returned for %s; no picks will be published", target_date)
        return _empty_picks()

    consensus = get_consensus_line(odds_raw)
    if consensus.empty:
        logger.warning("No consensus odds available for %s", target_date)
        return _empty_picks()

    game_odds = _pivot_odds(consensus)

    # ---- Step 3: Fetch game-specific weather ----
    weather = fetch_weather_for_games(schedule)

    # ---- Step 4: Build model feature matrix ----
    features = _build_todays_features(
        schedule=schedule,
        odds=game_odds,
        weather=weather,
    )

    if features.empty:
        logger.warning("Feature matrix is empty for %s", target_date)
        return _empty_picks()

    # ---- Step 5: Run market-specific prediction models ----
    picks: dict[str, list[dict]] = {}

    _validate_model_features(
        features=features,
        required_features=UNDERDOG_FEATURES,
        model_name="underdog",
    )
    underdog_predictions = predict_underdog(
        model_path=str(MODEL_DIR / "underdog_xgb_v1.joblib"),
        game_features=features,
    )
    underdog_picks = _filter_picks(
        predictions=underdog_predictions,
        min_edge=MIN_EDGE_UNDERDOG,
        min_confidence=MIN_CONFIDENCE,
    )
    picks["underdog"] = _format_picks(
        df=underdog_picks,
        pick_type="underdog",
        target_date=target_date,
    )

    _validate_model_features(
        features=features,
        required_features=SPREAD_FEATURES,
        model_name="spread",
    )
    spread_predictions = predict_spread(
        model_path=str(MODEL_DIR / "spread_xgb_v1.joblib"),
        game_features=features,
    )
    spread_picks = _filter_picks(
        predictions=spread_predictions,
        min_edge=MIN_EDGE_SPREAD,
        min_confidence=MIN_CONFIDENCE,
    )
    picks["spread"] = _format_picks(
        df=spread_picks,
        pick_type="spread",
        target_date=target_date,
    )

    _validate_model_features(
        features=features,
        required_features=TOTAL_FEATURES,
        model_name="totals",
    )
    totals_predictions = predict_totals(
        model_path=str(MODEL_DIR / "totals_xgb_v1.joblib"),
        game_features=features,
    )
    totals_picks = _filter_picks(
        predictions=totals_predictions,
        min_edge=MIN_EDGE_TOTALS,
        min_confidence=MIN_CONFIDENCE,
    )
    picks["over_under"] = _format_picks(
        df=totals_picks,
        pick_type="over_under",
        target_date=target_date,
    )

    # ---- Step 6: Persist picks and audit snapshot ----
    _store_picks(
        picks=picks,
        target_date=target_date,
    )

    # ---- Step 7: Summary ----
    total_picks = sum(len(pick_list) for pick_list in picks.values())

    logger.info(
        "Generated %d picks for %s: %d underdog, %d spread, %d O/U",
        total_picks,
        target_date,
        len(picks["underdog"]),
        len(picks["spread"]),
        len(picks["over_under"]),
    )

    return picks


def _empty_picks() -> dict[str, list[dict]]:
    """Return the standard empty pipeline response."""
    return {
        "underdog": [],
        "spread": [],
        "over_under": [],
    }


def _pivot_odds(consensus: pd.DataFrame) -> pd.DataFrame:
    """Pivot consensus odds from long format into one row per game.

    Expected consensus input fields:
        game_id, away_team, home_team, market, outcome_name,
        median_price, median_point.

    Returns:
        One row per game containing moneyline, spread, and total fields.
    """
    required_columns = [
        "game_id",
        "away_team",
        "home_team",
        "market",
        "outcome_name",
        "median_price",
    ]
    _require_columns(consensus, required_columns, context="consensus odds")

    games = consensus[
        ["game_id", "away_team", "home_team"]
    ].drop_duplicates(
        subset=["game_id"]
    ).copy()

    # ---- Moneyline ----
    moneyline = consensus.loc[
        consensus["market"].eq("h2h")
    ].copy()

    if not moneyline.empty:
        home_moneyline = moneyline.loc[
            moneyline["outcome_name"].eq(moneyline["home_team"]),
            ["game_id", "median_price"],
        ].rename(
            columns={"median_price": "home_moneyline"}
        )

        away_moneyline = moneyline.loc[
            moneyline["outcome_name"].eq(moneyline["away_team"]),
            ["game_id", "median_price"],
        ].rename(
            columns={"median_price": "away_moneyline"}
        )

        games = games.merge(
            home_moneyline.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

        games = games.merge(
            away_moneyline.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    # ---- Run line / spread ----
    spreads = consensus.loc[
        consensus["market"].eq("spreads")
    ].copy()

    if not spreads.empty:
        _require_columns(
            spreads,
            ["median_point"],
            context="spread consensus odds",
        )

        home_spread = spreads.loc[
            spreads["outcome_name"].eq(spreads["home_team"]),
            ["game_id", "median_point", "median_price"],
        ].rename(
            columns={
                "median_point": "home_spread_point",
                "median_price": "home_spread_price",
            }
        )

        away_spread = spreads.loc[
            spreads["outcome_name"].eq(spreads["away_team"]),
            ["game_id", "median_point", "median_price"],
        ].rename(
            columns={
                "median_point": "away_spread_point",
                "median_price": "away_spread_price",
            }
        )

        games = games.merge(
            home_spread.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

        games = games.merge(
            away_spread.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    # ---- Totals ----
    totals = consensus.loc[
        consensus["market"].eq("totals")
    ].copy()

    if not totals.empty:
        _require_columns(
            totals,
            ["median_point"],
            context="totals consensus odds",
        )

        over_total = totals.loc[
            totals["outcome_name"].eq("Over"),
            ["game_id", "median_point", "median_price"],
        ].rename(
            columns={
                "median_point": "posted_total",
                "median_price": "over_price",
            }
        )

        under_total = totals.loc[
            totals["outcome_name"].eq("Under"),
            ["game_id", "median_price"],
        ].rename(
            columns={"median_price": "under_price"}
        )

        games = games.merge(
            over_total.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

        games = games.merge(
            under_total.drop_duplicates(subset=["game_id"]),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    return games


def _build_todays_features(
    schedule: pd.DataFrame,
    odds: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge schedule, odds, and weather into the model feature matrix.

    Odds are joined by game_id plus teams, rather than team names alone.
    This avoids accidental odds assignments for doubleheaders or duplicate
    matchups in source data.
    """
    _require_columns(
        schedule,
        ["game_id", "away_team", "home_team"],
        context="schedule",
    )
    _require_columns(
        odds,
        ["game_id", "away_team", "home_team"],
        context="pivoted odds",
    )

    features = schedule.merge(
        odds,
        on=["game_id", "away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )

    if weather is not None and not weather.empty:
        weather_columns = [
            "game_id",
            "temp_f",
            "wind_mph",
            "wind_dir_deg",
            "precip_prob_pct",
            "is_dome",
        ]

        available_weather_columns = [
            column
            for column in weather_columns
            if column in weather.columns
        ]

        if "game_id" not in available_weather_columns:
            logger.warning(
                "Weather data returned without game_id; skipping weather merge"
            )
        else:
            features = features.merge(
                weather[available_weather_columns].drop_duplicates(
                    subset=["game_id"]
                ),
                on="game_id",
                how="left",
                validate="one_to_one",
            )

    # Add team, pitcher, bullpen, lineup, park, and rolling-stat feature
    # merges here. The resulting DataFrame must contain all columns in:
    # UNDERDOG_FEATURES, SPREAD_FEATURES, and TOTAL_FEATURES.

    return features


def _validate_model_features(
    features: pd.DataFrame,
    required_features: list[str],
    model_name: str,
) -> None:
    """Fail loudly if a model's required features are absent or unusable."""
    missing_columns = [
        column
        for column in required_features
        if column not in features.columns
    ]

    all_null_columns = [
        column
        for column in required_features
        if column in features.columns and features[column].isna().all()
    ]

    if missing_columns or all_null_columns:
        raise ValueError(
            f"{model_name} feature matrix is invalid. "
            f"Missing columns: {missing_columns}. "
            f"Columns containing only nulls: {all_null_columns}."
        )


def _filter_picks(
    predictions: pd.DataFrame,
    min_edge: float,
    min_confidence: float,
) -> pd.DataFrame:
    """Filter predictions to publishable, complete selections."""
    required_columns = [
        "game_id",
        "pick_value",
        "predicted_prob",
        "edge",
        "confidence_score",
    ]
    _require_columns(
        predictions,
        required_columns,
        context="model predictions",
    )

    filtered = predictions.dropna(
        subset=required_columns
    ).copy()

    filtered["edge"] = pd.to_numeric(
        filtered["edge"],
        errors="coerce",
    )
    filtered["confidence_score"] = pd.to_numeric(
        filtered["confidence_score"],
        errors="coerce",
    )
    filtered["predicted_prob"] = pd.to_numeric(
        filtered["predicted_prob"],
        errors="coerce",
    )

    filtered = filtered.dropna(
        subset=["edge", "confidence_score", "predicted_prob"]
    )

    filtered = filtered.loc[
        (filtered["edge"] >= min_edge)
        & (filtered["confidence_score"] >= min_confidence)
    ].copy()

    return filtered.sort_values(
        by=["confidence_score", "edge"],
        ascending=[False, False],
    )


def _format_picks(
    df: pd.DataFrame,
    pick_type: str,
    target_date: date,
) -> list[dict]:
    """Convert model output into serializable pick records.

    Records retain generated time and available market details so that a
    later audit can evaluate the original pick rather than a changed line.
    """
    generated_at_et = datetime.now(MLB_TZ).isoformat()

    picks: list[dict] = []

    for _, row in df.iterrows():
        picks.append(
            {
                "date": target_date.isoformat(),
                "game_id": row.get("game_id"),
                "away_team": row.get("away_team"),
                "home_team": row.get("home_team"),
                "pick_type": pick_type,
                "pick_value": row.get("pick_value", ""),
                "predicted_prob": _rounded_value(
                    row.get("predicted_prob"),
                    decimals=3,
                ),
                "confidence": row.get("confidence", "low"),
                "confidence_score": _rounded_value(
                    row.get("confidence_score"),
                    decimals=3,
                ),
                "edge": _rounded_value(
                    row.get("edge"),
                    decimals=3,
                ),
                "market": row.get("market", pick_type),
                "selected_price": row.get(
                    "selected_price",
                    row.get("price"),
                ),
                "selected_line": row.get(
                    "selected_line",
                    row.get("line"),
                ),
                "market_implied_probability": _rounded_value(
                    row.get("market_implied_probability"),
                    decimals=3,
                ),
                "market_timestamp": row.get("market_timestamp"),
                "scheduled_start_et": row.get(
                    "game_datetime_et",
                    row.get("game_time_et"),
                ),
                "generated_at_et": generated_at_et,
                "model_name": f"xgb_{pick_type}_v1",
                "is_locked": False,
            }
        )

    return picks


def _store_picks(
    picks: dict[str, list[dict]],
    target_date: date,
) -> None:
    """Save the current daily pick set and append it to an audit ledger.

    The date-specific CSV is the latest publishable slate for that date.
    The append-only parquet ledger preserves every generated snapshot for
    later auditing, line-movement analysis, and pick-lock implementation.
    """
    all_picks = [
        pick
        for pick_list in picks.values()
        for pick in pick_list
    ]

    if not all_picks:
        logger.info("No qualifying picks to save for %s", target_date)
        return

    output_df = pd.DataFrame(all_picks)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = PROCESSED_DIR / f"picks_{target_date.isoformat()}.csv"
    output_df.to_csv(
        daily_path,
        index=False,
    )
    logger.info("Current daily picks saved to %s", daily_path)

    ledger_path = PROCESSED_DIR / "pick_ledger.parquet"

    if ledger_path.exists():
        existing_ledger = pd.read_parquet(ledger_path)
        ledger = pd.concat(
            [existing_ledger, output_df],
            ignore_index=True,
            sort=False,
        )
    else:
        ledger = output_df.copy()

    ledger.to_parquet(
        ledger_path,
        index=False,
    )
    logger.info("Pick audit ledger updated at %s", ledger_path)


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    context: str,
) -> None:
    """Raise a clear error when a required source-data field is absent."""
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{context} is missing required columns: {missing_columns}"
        )


def _rounded_value(
    value: object,
    decimals: int = 3,
) -> Optional[float]:
    """Return a rounded float or None for null/non-numeric values."""
    if value is None or pd.isna(value):
        return None

    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    run_daily_pipeline()
