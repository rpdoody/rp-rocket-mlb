"""Capture qualifying pregame model recommendations into a pick-history ledger."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from page_utils import (
    _MLB_TO_RETRO,
    _fetch_espn_odds,
    _fetch_pitcher_stats,
    _fetch_todays_schedule,
    _load_game_context_cache,
    _load_precomputed,
)

from src.ingestion.weather import fetch_forecast
from src.models.contextual_projection import project_contextual_game
from src.ui.recommendation_cards import _build_game_recs

ET = ZoneInfo("America/New_York")
MIN_EDGE = 0.03


def current_season() -> int:
    return datetime.datetime.now(ET).year


def output_path() -> Path:
    return ROOT / "data_files" / "processed" / f"pick_history_{current_season()}.parquet"


def normalize_team_name(team_name: str | None) -> str:
    """Normalize team labels for schedule/odds matching."""
    return "".join(character.lower() for character in (team_name or "") if character.isalnum())


def find_espn_game(game: dict, espn_games: list[dict]) -> dict | None:
    """Match a StatsAPI schedule game to its ESPN odds event."""
    away_name = normalize_team_name(game.get("away_name"))
    home_name = normalize_team_name(game.get("home_name"))

    for espn_game in espn_games:
        if (
            normalize_team_name(espn_game.get("away_team")) == away_name
            and normalize_team_name(espn_game.get("home_team")) == home_name
        ):
            return espn_game

    return None


def parse_american_odds(raw_value: object) -> int | None:
    """Parse a signed American odds value without inventing a missing price."""
    try:
        value = str(raw_value).strip().replace("+", "")

        if value.lower() in {"", "—", "none", "nan"}:
            return None

        return int(float(value))
    except (TypeError, ValueError):
        return None


def game_date_eastern(game: dict, fallback_date: datetime.date) -> str:
    """Return the game date in Eastern time for ledger grouping."""
    raw_datetime = game.get("game_datetime", "")

    if not raw_datetime:
        return fallback_date.isoformat()

    try:
        parsed = datetime.datetime.fromisoformat(raw_datetime.replace("Z", "+00:00"))

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)

        return parsed.astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return fallback_date.isoformat()


def game_is_pregame(game: dict) -> bool:
    """Avoid recording picks after a game has begun or ended."""
    status = str(game.get("status", "")).strip().lower()

    excluded_statuses = {
        "final",
        "game over",
        "completed",
        "in progress",
        "live",
        "postponed",
        "cancelled",
        "suspended",
    }

    return status not in excluded_statuses


def confidence_tier(edge: float) -> str:
    """Use the app's recommendation thresholds for ledger confidence."""
    if edge > 0.06:
        return "HIGH"

    if edge > 0.03:
        return "MEDIUM"

    return "LOW"


def kelly_fraction(
    probability: float,
    american_odds: int,
    confidence: str,
) -> float:
    """Return confidence-scaled Kelly fraction, capped at a conservative 5%."""
    decimal_profit = american_odds / 100.0 if american_odds > 0 else 100.0 / abs(american_odds)

    if decimal_profit <= 0:
        return 0.0

    full_kelly = max(
        (decimal_profit * probability - (1.0 - probability)) / decimal_profit,
        0.0,
    )

    fractions = {
        "HIGH": 0.50,
        "MEDIUM": 0.25,
        "LOW": 0.125,
    }

    return min(full_kelly * fractions[confidence], 0.05)


def load_existing_ledger() -> pd.DataFrame:
    """Read the existing current-season ledger, if available."""
    path = output_path()

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"Could not read existing pick-history ledger: {exc}")
        return pd.DataFrame()


def build_ledger_row(
    game: dict,
    market_name: str,
    market: dict,
    captured_at: datetime.datetime,
    capture_date: datetime.date,
) -> dict | None:
    """Convert the best recommendation in one market into a ledger row."""
    best_side_name = market.get("best")

    if best_side_name not in market:
        return None

    side = market[best_side_name]
    edge = float(side.get("edge", 0.0))

    if edge <= MIN_EDGE:
        return None

    american_odds = parse_american_odds(side.get("odds_str"))

    if american_odds is None:
        return None

    if market_name == "ml":
        pick = str(side.get("team", "")).strip()
    else:
        pick = str(side.get("pick", "")).strip()

    game_id = game.get("game_id")

    if not game_id or not pick:
        return None

    confidence = confidence_tier(edge)

    return {
        "ledger_key": f"{game_id}_{market_name}",
        "game_id": int(game_id),
        "season": current_season(),
        "game_date": game_date_eastern(game, capture_date),
        "game_datetime": game.get("game_datetime", ""),
        "captured_at_utc": captured_at.isoformat(),
        "away_team": game.get("away_name", ""),
        "home_team": game.get("home_name", ""),
        "market": market_name,
        "pick": pick,
        "american_odds": american_odds,
        "predicted_prob": float(side.get("est_prob", 0.0)),
        "implied_prob": float(side.get("impl", 0.0)),
        "edge": edge,
        "confidence": confidence,
        "kelly_fraction": kelly_fraction(
            probability=float(side.get("est_prob", 0.0)),
            american_odds=american_odds,
            confidence=confidence,
        ),
        "posted_total": (
            float(market["posted"])
            if market_name == "ou" and market.get("posted") is not None
            else None
        ),
        "expected_total": (
            float(market["exp_total"])
            if market_name == "ou" and market.get("exp_total") is not None
            else None
        ),
        "result": "pending",
        "profit_units": 0.0,
        "graded_at_utc": None,
        "odds_source": "ESPN",
    }


def main() -> None:
    capture_date = datetime.datetime.now(ET).date()
    captured_at = datetime.datetime.now(datetime.timezone.utc)

    schedule = _fetch_todays_schedule(capture_date)
    odds_games = _fetch_espn_odds(capture_date)

    if not schedule:
        print(f"No MLB schedule found for {capture_date.isoformat()}.")
        return

    if not odds_games:
        print(f"No ESPN odds found for {capture_date.isoformat()}.")
        return

    historical_standings = _load_precomputed()["standings"]
    game_context = _load_game_context_cache()

    new_rows: list[dict] = []

    for game in schedule:
        if not game_is_pregame(game):
            continue

        espn_game = find_espn_game(game, odds_games)

        if not espn_game:
            continue

        away_team = game.get("away_name", "")
        home_team = game.get("home_name", "")
        away_pitcher = game.get("away_probable_pitcher") or "TBD"
        home_pitcher = game.get("home_probable_pitcher") or "TBD"

        game_date = game_date_eastern(game, capture_date)
        venue = game.get("venue_name", "")

        try:
            weather = fetch_forecast(venue, game_date) if venue else None
        except Exception as exc:
            print(f"Weather unavailable for {away_team} at {home_team}: {exc}")
            weather = None

        projection = project_contextual_game(
            game=game,
            hist_stnd=historical_standings,
            game_context=game_context,
            away_retro=_MLB_TO_RETRO.get(away_team, away_team),
            home_retro=_MLB_TO_RETRO.get(home_team, home_team),
            away_pitcher_stats=_fetch_pitcher_stats(away_pitcher),
            home_pitcher_stats=_fetch_pitcher_stats(home_pitcher),
            weather=weather,
        )

        recs = _build_game_recs(
            game=game,
            espn_game=espn_game,
            projection=projection,
            historical_data=historical_standings,
        )

        for market_name in ("ml", "rl", "ou"):
            market = recs.get(market_name)

            if not market:
                continue

            row = build_ledger_row(
                game=game,
                market_name=market_name,
                market=market,
                captured_at=captured_at,
                capture_date=capture_date,
            )

            if row is not None:
                new_rows.append(row)

    if not new_rows:
        print(
            f"No qualifying picks captured for {capture_date.isoformat()}. "
            f"Threshold: edge > {MIN_EDGE:.1%}."
        )
        return

    existing = load_existing_ledger()
    fresh = pd.DataFrame(new_rows)

    combined = pd.concat(
        [existing, fresh],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.sort_values(["ledger_key", "captured_at_utc"])
        .drop_duplicates(subset=["ledger_key"], keep="first")
        .sort_values(["game_date", "game_id", "market"])
        .reset_index(drop=True)
    )

    path = output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)

    print(f"Saved ledger: {path}")
    print(f"New qualifying picks: {len(fresh):,}")
    print(f"Total ledger rows: {len(combined):,}")


if __name__ == "__main__":
    main()
