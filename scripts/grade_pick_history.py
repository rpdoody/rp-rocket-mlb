"""Grade pending pick-history ledger rows using completed MLB game results."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from page_utils import _MLB_TO_RETRO

ET = ZoneInfo("America/New_York")


def current_season() -> int:
    return datetime.datetime.now(ET).year


def ledger_path() -> Path:
    return (
        ROOT
        / "data_files"
        / "processed"
        / f"pick_history_{current_season()}.parquet"
    )


def results_path() -> Path:
    return (
        ROOT
        / "data_files"
        / "processed"
        / f"live_gameinfo_{current_season()}.parquet"
    )


def normalize_text(value: object) -> str:
    """Normalize text for loose matching of team labels and pick strings."""
    return "".join(
        character.lower()
        for character in str(value or "")
        if character.isalnum()
    )


def team_short_name(full_name: object) -> str:
    """Return the final word of a full team name."""
    words = str(full_name or "").strip().split()
    return words[-1].lower() if words else ""


def retro_code_for_full_name(full_name: object) -> str:
    """Map an MLB full name into its Retrosheet team-code equivalent."""
    short_name = _MLB_TO_RETRO.get(str(full_name or ""), "")

    # Reverse the page_utils mapping indirectly from known short-name matches
    # in the result matching process; fallback to normalized full team name.
    return short_name


def american_profit_units(american_odds: int) -> float:
    """Profit in units when risking one unit at American odds."""
    return (
        american_odds / 100.0
        if american_odds > 0
        else 100.0 / abs(american_odds)
    )


def team_matches_pick(
    pick: str,
    full_team_name: str,
) -> bool:
    """Determine whether a displayed pick identifies a given MLB team."""
    normalized_pick = normalize_text(pick)
    normalized_full = normalize_text(full_team_name)
    short = team_short_name(full_team_name)

    return (
        normalized_full in normalized_pick
        or (short and short in normalized_pick)
    )


def grade_moneyline(
    pick: str,
    away_team: str,
    home_team: str,
    away_runs: int,
    home_runs: int,
) -> str:
    """Grade a moneyline selection."""
    if team_matches_pick(pick, home_team):
        return "win" if home_runs > away_runs else "loss"

    if team_matches_pick(pick, away_team):
        return "win" if away_runs > home_runs else "loss"

    return "pending"


def grade_run_line(
    pick: str,
    away_team: str,
    home_team: str,
    away_runs: int,
    home_runs: int,
) -> str:
    """Grade a ±1.5 run-line selection."""
    normalized_pick = pick.replace("−", "-")

    if "+1.5" in normalized_pick:
        run_line = 1.5
    elif "-1.5" in normalized_pick:
        run_line = -1.5
    else:
        return "pending"

    if team_matches_pick(pick, home_team):
        adjusted_margin = home_runs - away_runs + run_line
    elif team_matches_pick(pick, away_team):
        adjusted_margin = away_runs - home_runs + run_line
    else:
        return "pending"

    return "win" if adjusted_margin > 0 else "loss"


def grade_total(
    pick: str,
    posted_total: object,
    away_runs: int,
    home_runs: int,
) -> str:
    """Grade an over/under selection, including a whole-number push."""
    try:
        total_line = float(posted_total)
    except (TypeError, ValueError):
        return "pending"

    final_total = away_runs + home_runs
    normalized_pick = str(pick or "").strip().lower()

    if normalized_pick.startswith("over"):
        if final_total > total_line:
            return "win"
        if final_total < total_line:
            return "loss"
        return "push"

    if normalized_pick.startswith("under"):
        if final_total < total_line:
            return "win"
        if final_total > total_line:
            return "loss"
        return "push"

    return "pending"


def grade_row(
    ledger_row: pd.Series,
    result_row: pd.Series,
) -> str:
    """Apply the matching market grader to one ledger row."""
    market = str(ledger_row.get("market", "")).strip().lower()
    pick = str(ledger_row.get("pick", "")).strip()

    away_runs = int(result_row["vruns"])
    home_runs = int(result_row["hruns"])

    if market == "ml":
        return grade_moneyline(
            pick,
            str(ledger_row.get("away_team", "")),
            str(ledger_row.get("home_team", "")),
            away_runs,
            home_runs,
        )

    if market == "rl":
        return grade_run_line(
            pick,
            str(ledger_row.get("away_team", "")),
            str(ledger_row.get("home_team", "")),
            away_runs,
            home_runs,
        )

    if market == "ou":
        return grade_total(
            pick,
            ledger_row.get("posted_total"),
            away_runs,
            home_runs,
        )

    return "pending"


def load_ledger() -> pd.DataFrame:
    """Load the active-season pick ledger."""
    path = ledger_path()

    if not path.exists() or path.stat().st_size == 0:
        print(f"No pick-history ledger found: {path}")
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"Could not read pick-history ledger: {exc}")
        return pd.DataFrame()


def load_final_results() -> pd.DataFrame:
    """Load final results only from the live gameinfo dataset."""
    path = results_path()

    if not path.exists() or path.stat().st_size == 0:
        print(f"No live game-results file found: {path}")
        return pd.DataFrame()

    try:
        results = pd.read_parquet(path).copy()
    except Exception as exc:
        print(f"Could not read live game-results file: {exc}")
        return pd.DataFrame()

    required_columns = {
        "game_id",
        "date",
        "visteam",
        "hometeam",
        "vruns",
        "hruns",
        "status",
    }

    missing = required_columns - set(results.columns)

    if missing:
        print(
            "Live game-results file is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
        return pd.DataFrame()

    results["status"] = results["status"].astype(str).str.strip().str.lower()
    results = results[results["status"].eq("final")].copy()

    results["game_id"] = pd.to_numeric(
        results["game_id"],
        errors="coerce",
    )

    results["game_date"] = pd.to_datetime(
        results["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.date.astype(str)

    results["vruns"] = pd.to_numeric(
        results["vruns"],
        errors="coerce",
    )

    results["hruns"] = pd.to_numeric(
        results["hruns"],
        errors="coerce",
    )

    return results.dropna(
        subset=["game_id", "game_date", "vruns", "hruns"]
    ).copy()


def find_result(
    ledger_row: pd.Series,
    final_results: pd.DataFrame,
) -> pd.Series | None:
    """Find the final outcome by MLB game ID, with date/team fallback."""
    game_id = pd.to_numeric(
        ledger_row.get("game_id"),
        errors="coerce",
    )

    if pd.notna(game_id):
        by_game_id = final_results[
            final_results["game_id"].eq(int(game_id))
        ]

        if not by_game_id.empty:
            return by_game_id.iloc[-1]

    game_date = str(ledger_row.get("game_date", ""))
    away_team = str(ledger_row.get("away_team", ""))
    home_team = str(ledger_row.get("home_team", ""))

    # Fallback only when a game ID is unavailable/mismatched.
    # Compare final-word team labels against Retrosheet codes loosely.
    candidates = final_results[
        final_results["game_date"].eq(game_date)
    ].copy()

    if candidates.empty:
        return None

    away_token = team_short_name(away_team)
    home_token = team_short_name(home_team)

    candidates["away_token"] = candidates["visteam"].astype(str).str.lower()
    candidates["home_token"] = candidates["hometeam"].astype(str).str.lower()

    # Game ID should handle normal operation. This fallback is intentionally
    # conservative and only matches direct code/short-token overlap.
    for _, candidate in candidates.iterrows():
        if (
            away_token in candidate["away_token"]
            or candidate["away_token"] in normalize_text(away_team)
        ) and (
            home_token in candidate["home_token"]
            or candidate["home_token"] in normalize_text(home_team)
        ):
            return candidate

    return None


def main() -> None:
    ledger = load_ledger()

    if ledger.empty:
        return

    final_results = load_final_results()

    if final_results.empty:
        print("No completed final results available to grade yet.")
        return

    ledger = ledger.copy()

    if "result" not in ledger.columns:
        ledger["result"] = "pending"

    if "profit_units" not in ledger.columns:
        ledger["profit_units"] = 0.0

    if "graded_at_utc" not in ledger.columns:
        ledger["graded_at_utc"] = None

    pending_mask = (
        ledger["result"]
        .fillna("pending")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("pending")
    )

    graded_count = 0

    for index in ledger.index[pending_mask]:
        result_row = find_result(
            ledger.loc[index],
            final_results,
        )

        if result_row is None:
            continue

        outcome = grade_row(
            ledger.loc[index],
            result_row,
        )

        if outcome == "pending":
            continue

        odds = pd.to_numeric(
            ledger.at[index, "american_odds"],
            errors="coerce",
        )

        if pd.isna(odds):
            continue

        if outcome == "win":
            profit_units = american_profit_units(int(odds))
        elif outcome == "loss":
            profit_units = -1.0
        else:
            profit_units = 0.0

        ledger.at[index, "result"] = outcome
        ledger.at[index, "profit_units"] = profit_units
        ledger.at[index, "graded_at_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        graded_count += 1

    path = ledger_path()
    ledger.to_parquet(path, index=False)

    pending_after = int(
        ledger["result"]
        .fillna("pending")
        .astype(str)
        .str.lower()
        .eq("pending")
        .sum()
    )

    print(f"Updated ledger: {path}")
    print(f"Newly graded picks: {graded_count:,}")
    print(f"Still pending: {pending_after:,}")


if __name__ == "__main__":
    main()