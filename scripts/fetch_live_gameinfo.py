"""Fetch completed MLB regular-season games into live gameinfo parquet."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import statsapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ET = ZoneInfo("America/New_York")
SEASON = 2026
OUTPUT_PATH = ROOT / "data_files" / "processed" / f"live_gameinfo_{SEASON}.parquet"

MLB_ABBR_TO_RETRO = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CWS": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "OAK",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SDP": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "SFG": "SFN",
    "STL": "SLN",
    "TB": "TBA",
    "TBR": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
    "WAS": "WAS",
}

MLB_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Angels of Anaheim": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def retro_code(team_abbr: str) -> str:
    """Translate an MLB Stats API abbreviation to a Retrosheet code."""

    value = str(team_abbr).strip().upper()
    return MLB_ABBR_TO_RETRO.get(value, value)


def is_completed_game(game: dict) -> bool:
    """Return true for a completed MLB regular-season game with final scores."""

    status = str(game.get("status", "")).strip().lower()

    is_final = (
        status in {"final", "game over", "completed"}
        or status.startswith("final")
    )

    return (
        game.get("game_type") == "R"
        and is_final
        and game.get("away_score") is not None
        and game.get("home_score") is not None
    )


def fetch_completed_games(
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[dict]:
    """Fetch schedule games and return only completed regular-season games."""

    games = statsapi.schedule(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        sportId=1,
    ) or []

    return [game for game in games if is_completed_game(game)]


def _schedule_team_abbr(game: dict, side: str) -> str:
    """Resolve an away/home MLB abbreviation from a schedule record."""

    direct_keys = (
        f"{side}_abbr",
        f"{side}_abbreviation",
        f"{side}_team_abbr",
    )

    for key in direct_keys:
        value = game.get(key)
        if value:
            return str(value).strip().upper()

    name_keys = (
        f"{side}_name",
        f"{side}_team_name",
    )

    for key in name_keys:
        value = game.get(key)
        if value:
            abbreviation = MLB_NAME_TO_ABBR.get(str(value).strip())
            if abbreviation:
                return abbreviation

    return ""


def normalize_game(game: dict) -> dict:
    """Convert one completed schedule game to the app's gameinfo format."""

    game_date = datetime.date.fromisoformat(game["game_date"])

    away_abbr = _schedule_team_abbr(game, "away")
    home_abbr = _schedule_team_abbr(game, "home")

    if not away_abbr or not home_abbr:
        raise ValueError(
            f"Unable to identify team abbreviations for game {game.get('game_id')}."
        )

    away_code = retro_code(away_abbr)
    home_code = retro_code(home_abbr)

    away_runs = int(game["away_score"])
    home_runs = int(game["home_score"])

    if away_runs == home_runs:
        raise ValueError(
            f"Completed game {game['game_id']} has an invalid tied final score."
        )

    return {
        "game_id": int(game["game_id"]),
        "gid": f"MLB{game['game_id']}",
        "season": game_date.year,
        "date": int(game_date.strftime("%Y%m%d")),
        "visteam": away_code,
        "hometeam": home_code,
        "vruns": away_runs,
        "hruns": home_runs,
        "wteam": home_code if home_runs > away_runs else away_code,
        "lteam": away_code if home_runs > away_runs else home_code,
        "total_runs": away_runs + home_runs,
        "daynight": "",
        "attendance": pd.NA,
        "temp": pd.NA,
        "windspeed": pd.NA,
        "game_type": game.get("game_type", "R"),
        "status": game.get("status", ""),
        "source": "mlb_stats_api",
        "retrieved_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }


def load_existing() -> pd.DataFrame:
    """Read the existing season output if it is a valid Parquet file."""

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        print(
            f"Warning: ignoring unreadable {OUTPUT_PATH.name}: {exc}"
        )
        return pd.DataFrame()


def main() -> None:
    """Fetch all completed regular-season games from opening day through today."""

    today_et = datetime.datetime.now(ET).date()

    season_start = datetime.date(SEASON, 3, 1)
    season_end = min(today_et, datetime.date(SEASON, 12, 31))

    if season_end < season_start:
        print(f"No completed {SEASON} regular-season games are available yet.")
        return

    print(
        f"Fetching completed MLB regular-season games: "
        f"{season_start} through {season_end}"
    )

    games = fetch_completed_games(season_start, season_end)

    rows = []
    errors = []

    for game in games:
        try:
            rows.append(normalize_game(game))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"Skipped game {game.get('game_id', 'unknown')}: {exc}"
            )

    if errors:
        print("\n".join(errors[:20]))

    if not rows:
        print("No completed games returned from the MLB Stats API.")
        return

    fresh = pd.DataFrame(rows)

    required_columns = [
        "game_id",
        "season",
        "date",
        "visteam",
        "hometeam",
        "vruns",
        "hruns",
        "wteam",
        "lteam",
    ]

    missing_columns = [
        column for column in required_columns if column not in fresh.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Refusing to write incomplete game data; missing {missing_columns}."
        )

    team_columns = ["visteam", "hometeam", "wteam", "lteam"]

    blank_team_mask = (
        fresh[team_columns]
        .replace("", pd.NA)
        .isna()
        .any(axis=1)
    )

    if blank_team_mask.any():
        raise ValueError(
            "Refusing to write games with blank team codes:\n"
            + fresh.loc[blank_team_mask].to_string(index=False)
        )

    existing = load_existing()

    combined = pd.concat(
        [existing, fresh],
        ignore_index=True,
        sort=False,
    )

    combined["game_id"] = pd.to_numeric(
        combined["game_id"],
        errors="coerce",
    )

    combined["date"] = pd.to_numeric(
        combined["date"],
        errors="coerce",
    )

    combined = combined.dropna(subset=["game_id", "date"]).copy()

    combined["game_id"] = combined["game_id"].astype(int)
    combined["date"] = combined["date"].astype(int)

    combined = (
        combined.sort_values(["game_id", "retrieved_at_utc"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    coverage_dates = pd.to_datetime(
        combined["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Games: {len(combined):,}")
    print(f"First game: {coverage_dates.min().date()}")
    print(f"Last game: {coverage_dates.max().date()}")


if __name__ == "__main__":
    main()
