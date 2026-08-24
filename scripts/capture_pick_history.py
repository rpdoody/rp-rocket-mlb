"""Capture qualifying MLB model picks with pregame odds into a durable ledger."""

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
    _fetch_team_standings,
    _fetch_todays_schedule,
    _load_game_context_cache,
)
from src.ingestion.weather import fetch_forecast
from src.models.contextual_projection import project_contextual_game
from src.ui.recommendation_cards import _build_game_recs

ET = ZoneInfo("America/New_York")
SEASON = 2026
MIN_EDGE = 0.03
OUTPUT_PATH = ROOT / "data_files" / "processed" / f"pick_history_{SEASON}.parquet"


def eastern_today() -> datetime.date:
    return datetime.datetime.now(ET).date()


def normalize_team_name(team_name: str | None) -> str:
    return "".join(
        character.lower()
        for character in (team_name or "")
        if character.isalnum()
    )


def is_matching_odds_game(game: dict, odds_game: dict) -> bool:
    return (
        normalize_team_name(game.get("away_name"))
        == normalize_team_name(odds_game.get("away_team"))
        and normalize_team_name(game.get("home_name"))
        == normalize_team_name(odds_game.get("home_team"))
    )


def is_pregame_game(game: dict) -> bool:
    status = str(game.get("status", "")).strip().lower()

    return status not in {
        "final",
        "game over",
        "completed",
        "in progress",
        "live",
        "postponed",
        "cancelled",
        "suspended",
    }


def game_date_from_datetime(
    game_datetime: str,
    fallback_date: datetime.date,
) -> str:
    if not game_datetime:
        return fallback_date.isoformat()

    try:
        game_time = datetime.datetime.fromisoformat(
            game_datetime.replace("Z", "+00:00")
        )

        if game_time.tzinfo is None:
            game_time = game_time.replace(
                tzinfo=datetime.timezone.utc
            )

        return game_time.astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return fallback_date.isoformat()


def parse_american_odds(odds_text: object) -> int | None:
    try:
        value = str(odds_text).strip().replace("+", "")
        if value.lower() in {"", "none", "nan", "—"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def confidence_from_edge(edge: float) -> str:
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
    if american_odds >= 0:
        decimal_payout = american_odds / 100.0
    else:
        decimal_payout = 100.0 / abs(american_odds)

    full_kelly = max(
        (
            decimal_payout * probability
            - (1.0 - probability)
        )
        / decimal_payout,
        0.0,
    )

    confidence_multiplier = {
        "HIGH": 0.50,
        "MEDIUM": 0.25,
        "LOW": 0.125,
    }

    return full_kelly * confidence_multiplier[confidence]


def load_existing() -> pd.DataFrame:
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        print(f"Warning: could not read existing ledger: {exc}")
        return pd.DataFrame()


def build_pick_row(
    game: dict,
    market_key: str,
    market: dict,
    side: dict,
    capture_time: datetime.datetime,
    fallback_date: datetime.date,
) -> dict | None:
    american_odds = parse_american_odds(side.get("odds_str"))

    if american_odds is None:
        return None

    edge = float(side.get("edge", 0.0))

    if edge <= MIN_EDGE:
        return None

    confidence = confidence_from_edge(edge)

    if market_key == "ml":
        pick = str(side.get("team") or side.get("pick") or "")
    else:
        pick = str(side.get("pick") or side.get("team") or "")

    game_id = game.get("game_id")

    if not game_id or not pick:
        return None

    game_date = game_date_from_datetime(
        game.get("game_datetime", ""),
        fallback_date,
    )

    return {
        "ledger_key": f"{game_id}_{market_key}",
        "game_id": int(game_id),
        "season": SEASON,
        "game_date": game_date,
        "game_datetime": game.get("game_datetime", ""),
        "captured_at_utc": capture_time.isoformat(),
        "away_team": game.get("away_name", ""),
        "home_team": game.get("home_name", ""),
        "market": market_key,
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
            float(market.get("posted"))
            if market_key == "ou" and market.get("posted") is not None
            else None
        ),
        "expected_total": (
            float(market.get("exp_total"))
            if market_key == "ou" and market.get("exp_total") is not None
            else None
        ),
        "result": "pending",
        "profit_units": 0.0,
        "graded_at_utc": None,
        "odds_source": "ESPN",
    }


def main() -> None:
    capture_date = eastern_today()
    capture_time = datetime.datetime.now(datetime.timezone.utc)

    try:
        schedule = _fetch_todays_schedule(capture_date)
    except TypeError:
        schedule = _fetch_todays_schedule()

    odds_events = _fetch_espn_odds(capture_date)

    if not schedule:
        print(f"No MLB schedule available for {capture_date}.")
        return

    if not odds_events:
        print(f"No ESPN odds events available for {capture_date}.")
        return

    standings = _fetch_team_standings()
    game_context = _load_game_context_cache()

    rows: list[dict] = []

    for game in schedule:
        if not is_pregame_game(game):
            continue

        odds_game = next(
            (
                odds
                for odds in odds_events
                if is_matching_odds_game(game, odds)
            ),
            None,
        )

        if odds_game is None:
            continue

        away_team = game.get("away_name", "")
        home_team = game.get("home_name", "")
        away_pitcher = game.get("away_probable_pitcher", "TBD") or "TBD"
        home_pitcher = game.get("home_probable_pitcher", "TBD") or "TBD"
        venue = game.get("venue_name", "")

        game_date = game_date_from_datetime(
            game.get("game_datetime", ""),
            capture_date,
        )

        weather = (
            fetch_forecast(venue, game_date)
            if venue
            else None
        )

        projection = project_contextual_game(
            game=game,
            hist_stnd=standings,
            game_context=game_context,
            away_retro=_MLB_TO_RETRO.get(away_team, away_team),
            home_retro=_MLB_TO_RETRO.get(home_team, home_team),
            away_pitcher_stats=_fetch_pitcher_stats(away_pitcher),
            home_pitcher_stats=_fetch_pitcher_stats(home_pitcher),
            weather=weather,
        )

        recommendations = _build_game_recs(
            game=game,
            espn_game=odds_game,
            projection=projection,
            historical_data=standings,
        )

        for market_key in ("ml", "rl", "ou"):
            if market_key not in recommendations:
                continue

            market = recommendations[market_key]
            side = market[market["best"]]

            row = build_pick_row(
                game=game,
                market_key=market_key,
                market=market,
                side=side,
                capture_time=capture_time,
                fallback_date=capture_date,
            )

            if row is not None:
                rows.append(row)

    if not rows:
        print(
            "No qualifying picks were captured. "
            f"Minimum edge is {MIN_EDGE:.0%}."
        )
        return

    fresh = pd.DataFrame(rows)
    existing = load_existing()

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Total ledger picks: {len(combined):,}")
    print(f"New qualifying picks: {len(fresh):,}")


if __name__ == "__main__":
    main()
2. Grade pending picks
Create:

text
scripts/grade_pick_history.py
Paste:

python
"""Grade pending pick-history rows using final live MLB game results."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEASON = 2026
LEDGER_PATH = ROOT / "data_files" / "processed" / f"pick_history_{SEASON}.parquet"
RESULTS_PATH = ROOT / "data_files" / "processed" / f"live_gameinfo_{SEASON}.parquet"


def normalize_team_name(name: object) -> str:
    return "".join(
        character.lower()
        for character in str(name or "")
        if character.isalnum()
    )


def short_name(name: object) -> str:
    return str(name or "").strip().lower().split()[-1] if str(name or "").strip() else ""


def american_profit_units(american_odds: int) -> float:
    if american_odds > 0:
        return american_odds / 100.0
    return 100.0 / abs(american_odds)


def load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        print(f"No {label} file found at {path}.")
        return pd.DataFrame()

    return pd.read_parquet(path)


def match_result(
    pick_row: pd.Series,
    results: pd.DataFrame,
) -> pd.Series | None:
    game_id = pd.to_numeric(
        pick_row.get("game_id"),
        errors="coerce",
    )

    if pd.notna(game_id) and "game_id" in results.columns:
        by_id = results[
            pd.to_numeric(
                results["game_id"],
                errors="coerce",
            ).eq(int(game_id))
        ]

        if not by_id.empty:
            return by_id.iloc[-1]

    away_token = normalize_team_name(pick_row.get("away_team"))
    home_token = normalize_team_name(pick_row.get("home_team"))
    game_date = str(pick_row.get("game_date", ""))

    candidates = results[
        results["game_date"].eq(game_date)
        & results["away_key"].eq(away_token)
        & results["home_key"].eq(home_token)
    ]

    if candidates.empty:
        return None

    return candidates.iloc[-1]


def grade_moneyline(
    pick: str,
    away_team: str,
    home_team: str,
    away_score: float,
    home_score: float,
) -> str:
    pick_lower = pick.lower()

    away_short = short_name(away_team)
    home_short = short_name(home_team)

    if normalize_team_name(home_team) in normalize_team_name(pick) or (
        home_short and home_short in pick_lower
    ):
        return "win" if home_score > away_score else "loss"

    if normalize_team_name(away_team) in normalize_team_name(pick) or (
        away_short and away_short in pick_lower
    ):
        return "win" if away_score > home_score else "loss"

    return "pending"


def grade_run_line(
    pick: str,
    away_team: str,
    home_team: str,
    away_score: float,
    home_score: float,
) -> str:
    pick_lower = pick.lower()
    away_short = short_name(away_team)
    home_short = short_name(home_team)

    picked_home = (
        normalize_team_name(home_team) in normalize_team_name(pick)
        or (home_short and home_short in pick_lower)
    )

    picked_away = (
        normalize_team_name(away_team) in normalize_team_name(pick)
        or (away_short and away_short in pick_lower)
    )

    if "+1.5" in pick:
        run_line = 1.5
    elif "-1.5" in pick or "−1.5" in pick:
        run_line = -1.5
    else:
        return "pending"

    if picked_home:
        adjusted_margin = home_score - away_score + run_line
    elif picked_away:
        adjusted_margin = away_score - home_score + run_line
    else:
        return "pending"

    if adjusted_margin > 0:
        return "win"
    if adjusted_margin < 0:
        return "loss"
    return "push"


def grade_total(
    pick: str,
    posted_total: float | None,
    away_score: float,
    home_score: float,
) -> str:
    if posted_total is None or pd.isna(posted_total):
        return "pending"

    final_total = away_score + home_score
    pick_lower = pick.lower()

    if pick_lower.startswith("over"):
        if final_total > posted_total:
            return "win"
        if final_total < posted_total:
            return "loss"
        return "push"

    if pick_lower.startswith("under"):
        if final_total < posted_total:
            return "win"
        if final_total > posted_total:
            return "loss"
        return "push"

    return "pending"


def grade_pick(
    pick_row: pd.Series,
    game_result: pd.Series,
) -> str:
    away_score = float(game_result["vruns"])
    home_score = float(game_result["hruns"])

    market = str(pick_row.get("market", "")).lower()
    pick = str(pick_row.get("pick", ""))

    if market == "ml":
        return grade_moneyline(
            pick,
            str(pick_row.get("away_team", "")),
            str(pick_row.get("home_team", "")),
            away_score,
            home_score,
        )

    if market == "rl":
        return grade_run_line(
            pick,
            str(pick_row.get("away_team", "")),
            str(pick_row.get("home_team", "")),
            away_score,
            home_score,
        )

    if market == "ou":
        return grade_total(
            pick,
            pd.to_numeric(
                pick_row.get("posted_total"),
                errors="coerce",
            ),
            away_score,
            home_score,
        )

    return "pending"


def main() -> None:
    ledger = load_parquet(LEDGER_PATH, "pick-history ledger")

    if ledger.empty:
        return

    results = load_parquet(RESULTS_PATH, "live gameinfo")

    if results.empty:
        return

    results = results.copy()
    results["game_date"] = pd.to_datetime(
        results["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.date.astype(str)

    results["away_key"] = results["visteam"].map(normalize_team_name)
    results["home_key"] = results["hometeam"].map(normalize_team_name)

    ledger = ledger.copy()

    if "result" not in ledger.columns:
        ledger["result"] = "pending"

    if "profit_units" not in ledger.columns:
        ledger["profit_units"] = 0.0

    if "graded_at_utc" not in ledger.columns:
        ledger["graded_at_utc"] = None

    pending_mask = ledger["result"].fillna("pending").str.lower().eq("pending")

    graded_count = 0

    for index in ledger.index[pending_mask]:
        result_game = match_result(ledger.loc[index], results)

        if result_game is None:
            continue

        result = grade_pick(ledger.loc[index], result_game)

        if result == "pending":
            continue

        odds = int(
            pd.to_numeric(
                ledger.at[index, "american_odds"],
                errors="coerce",
            )
        )

        if result == "win":
            profit_units = american_profit_units(odds)
        elif result == "loss":
            profit_units = -1.0
        else:
            profit_units = 0.0

        ledger.at[index, "result"] = result
        ledger.at[index, "profit_units"] = profit_units
        ledger.at[index, "graded_at_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        graded_count += 1

    ledger.to_parquet(LEDGER_PATH, index=False)

    print(f"Wrote: {LEDGER_PATH}")
    print(f"Newly graded picks: {graded_count:,}")


if __name__ == "__main__":
    main()
