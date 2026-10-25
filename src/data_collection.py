"""
data_collection.py
==================

Step 1 of the NBA Injury Risk pipeline: pull raw data from the NBA Stats API
(stats.nba.com) through the `nba_api` package and save it as CSV files in
data/raw/.

What this script produces (all in data/raw/):

    player_game_logs.csv    One row per player per game played
                            (regular season + play-in + playoffs).
                            -> workload: minutes, usage proxies, game density.

    schedule.csv            One row per game actually played, with date,
                            tip-off time, home/away teams and arena city.
                            -> rest days, back-to-backs, travel, and (later)
                               detecting games a player MISSED.

    player_season_bio.csv   One row per player per season: age, height and
                            weight AS OF THAT SEASON.
                            -> physical profile without look-ahead leakage.

    player_details.csv      One row per player: birthdate, position, first
                            NBA season, draft info.
                            -> exact age on any game date, position group,
                               years of experience.

Usage (run from the repository root):

    python src/data_collection.py                     # full pull (~10-15 min)
    python src/data_collection.py --skip-player-details   # fast pull (~30 s)
    python src/data_collection.py --refresh-player-details  # ignore cache

Design notes
------------
* Bulk endpoints over per-player endpoints. `LeagueGameLog` returns every
  player-game of a season in a single request. Calling `PlayerGameLog` once
  per player would be ~500 requests per season, slower and much more likely
  to get rate-limited. The only per-player endpoint used is
  `CommonPlayerInfo`, because it's the only source of birthdate and position.
* Light cleaning only. This step selects and renames columns, adds
  SEASON / SEASON_TYPE labels and standardizes dates, but does not engineer
  features. Raw data stays close to the source so the feature step can be
  re-run without hitting the API again.
* Every API call goes through one retry wrapper (`call_endpoint`) so rate
  limiting and network errors are handled in one place.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguedashplayerbiostats,
    leaguegamelog,
    scheduleleaguev2,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Paths are resolved from this file's location, so the script works no matter
# which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Seasons use the NBA's "YYYY-YY" string format, which the API expects.
DEFAULT_SEASONS = ["2022-23", "2023-24"]

# Season types to pull game logs for. Play-in and playoff minutes are real
# physical load, so leaving them out would undercount fatigue for good teams.
# The values are the exact strings the API's SeasonType parameter accepts.
GAME_LOG_SEASON_TYPES = ["Regular Season", "PlayIn", "Playoffs"]

# The first 3 characters of an NBA GAME_ID encode the game type.
#   001 preseason, 002 regular season, 003 All-Star, 004 playoffs,
#   005 play-in, 006 NBA Cup (In-Season Tournament) championship game.
# Preseason and All-Star are dropped: starters play limited minutes in
# preseason, and the All-Star game isn't team-scheduled load.
GAME_TYPE_BY_PREFIX = {
    "002": "Regular Season",
    "004": "Playoffs",
    "005": "PlayIn",
    "006": "NBA Cup Final",
}

# Rate limiting. stats.nba.com has no published limit, but it silently
# throttles or drops connections when hit too fast. ~0.6 s between requests
# is the commonly used safe pace in the nba_api community.
REQUEST_DELAY_SECONDS = 0.6
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
# Base wait before the first retry; it doubles each attempt (exponential
# backoff): 2 s, 4 s, 8 s, 16 s, ...
RETRY_BACKOFF_SECONDS = 2.0

# Save player details to disk every N players, so a crash or block halfway
# through the ~700 calls doesn't lose progress.
PLAYER_DETAILS_CHECKPOINT_EVERY = 50

logger = logging.getLogger("data_collection")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def call_endpoint(endpoint_cls, delay: float = REQUEST_DELAY_SECONDS, **params) -> list[pd.DataFrame]:
    """Call an nba_api endpoint with retries and return its DataFrames.

    Why a wrapper: stats.nba.com is unreliable for scripted access. Typical
    failures are:
      * ReadTimeout / ConnectionError when you request too fast (throttling).
      * An HTML or empty response instead of JSON, which makes nba_api raise
        a JSONDecodeError (a subclass of ValueError).
      * A KeyError inside nba_api if the payload is missing expected keys.
    All of these are usually temporary, so we retry with exponential backoff
    plus a little random "jitter" so retries don't land in lockstep.

    Args:
        endpoint_cls: An nba_api endpoint class, e.g. leaguegamelog.LeagueGameLog.
        delay: Seconds to sleep after a successful call (politeness delay).
        **params: Parameters forwarded to the endpoint constructor.

    Returns:
        A list of DataFrames, one per result set the endpoint returns.

    Raises:
        RuntimeError: if every retry fails.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = endpoint_cls(timeout=REQUEST_TIMEOUT_SECONDS, **params)
            frames = response.get_data_frames()
            time.sleep(delay)  # Pace ourselves even when the call succeeds.
            return frames
        except (requests.exceptions.RequestException, ValueError, KeyError) as error:
            last_error = error
            wait = RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1) + random.uniform(0, 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s: %s. Retrying in %.1fs",
                endpoint_cls.__name__, attempt, MAX_RETRIES,
                type(error).__name__, error, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"{endpoint_cls.__name__} failed after {MAX_RETRIES} attempts "
        f"with params {params}. Last error: {last_error!r}"
    )


def require_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    """Fail fast with a clear message if the API response changed shape.

    The NBA renames or drops fields without notice. A clear error here is
    much easier to debug than a confusing KeyError three steps later.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} response is missing expected columns: {missing}")


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to data/raw/<filename> and log what was written."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved %s (%d rows x %d cols)", path.relative_to(PROJECT_ROOT), *df.shape)
    return path


# ---------------------------------------------------------------------------
# 1. Player game logs
# ---------------------------------------------------------------------------
# Columns kept from LeagueGameLog. Dropped columns and why:
#   SEASON_ID        -> replaced by readable SEASON + SEASON_TYPE columns.
#   TEAM_NAME        -> redundant with TEAM_ID / TEAM_ABBREVIATION.
#   FG_PCT, FG3_PCT, FT_PCT -> derivable from makes/attempts, and NaN when
#                       attempts are 0. Recompute them later if needed.
#   FANTASY_PTS, VIDEO_AVAILABLE -> not relevant to injury risk.
GAME_LOG_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS",
]


def fetch_player_game_logs(seasons: list[str], delay: float) -> pd.DataFrame:
    """Pull every player's box score line for each season and season type.

    Endpoint: LeagueGameLog (stats.nba.com/stats/leaguegamelog) with
    PlayerOrTeam='P'.
      Why this one: one request returns ALL player-games for a season and
      season type (~26,000 rows for a regular season) in about 1 second.
      The per-player alternative (PlayerGameLog) would need ~500 requests
      per season.

    Raw data: 32 columns. Identifiers (PLAYER_ID, TEAM_ID, GAME_ID),
    GAME_DATE as 'YYYY-MM-DD', MATCHUP like 'DEN vs. LAL' (home) or
    'LAL @ DEN' (away), MIN as a whole number, and the standard box score.

    Quirks to know:
      * Only players who appeared in the game get a row. A DNP (injury,
        rest, coach's decision) produces NO row. That's important for us:
        missed games must be found by comparing these logs with the team
        schedule, which is why we also pull schedule.csv.
      * MIN is rounded to whole minutes, and a few rows have MIN = 0
        (players who checked in for a few seconds). We keep them, since
        they did appear.
      * The NBA Cup championship game (Dec 2023) is not returned under any
        season type, so those ~30 player-games are missing. It is in
        schedule.csv, so team rest-day calculations are still correct.
    """
    frames = []
    for season in seasons:
        for season_type in GAME_LOG_SEASON_TYPES:
            logger.info("Fetching player game logs: %s %s", season, season_type)
            df = call_endpoint(
                leaguegamelog.LeagueGameLog,
                delay=delay,
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
            )[0]
            if df.empty:
                # E.g. play-in didn't exist before 2020-21. Not an error.
                logger.warning("  -> no rows returned for %s %s", season, season_type)
                continue
            require_columns(df, GAME_LOG_COLUMNS, "LeagueGameLog")
            df = df[GAME_LOG_COLUMNS].copy()
            df.insert(0, "SEASON", season)
            df.insert(1, "SEASON_TYPE", season_type)
            logger.info("  -> %d player-game rows", len(df))
            frames.append(df)

    if not frames:
        raise RuntimeError("No player game logs were returned for any season.")

    logs = pd.concat(frames, ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.date

    # A player can only appear once per game. Duplicates mean something went
    # wrong (e.g. overlapping season types), so drop them and say so.
    duplicates = logs.duplicated(subset=["PLAYER_ID", "GAME_ID"])
    if duplicates.any():
        logger.warning("Dropping %d duplicate player-game rows", int(duplicates.sum()))
        logs = logs[~duplicates]

    return logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Schedule
# ---------------------------------------------------------------------------
# Raw camelCase name -> our UPPER_SNAKE_CASE name (matches the game logs).
# Of the ~150 raw columns we keep only these. Dropped: ~110 broadcaster
# columns, points leaders, week labels, series text, team slugs and records.
SCHEDULE_COLUMN_MAP = {
    "gameId": "GAME_ID",
    "gameDateEst": "GAME_DATE",
    "gameDateTimeEst": "TIPOFF_ET",
    "gameLabel": "GAME_LABEL",
    "homeTeam_teamId": "HOME_TEAM_ID",
    "homeTeam_teamTricode": "HOME_TEAM_ABBREVIATION",
    "awayTeam_teamId": "AWAY_TEAM_ID",
    "awayTeam_teamTricode": "AWAY_TEAM_ABBREVIATION",
    "arenaName": "ARENA_NAME",
    "arenaCity": "ARENA_CITY",
    "arenaState": "ARENA_STATE",
    "isNeutral": "IS_NEUTRAL",
    "gameStatus": "GAME_STATUS",
}

# gameStatus codes: 1 = scheduled, 2 = in progress, 3 = final.
GAME_STATUS_FINAL = 3


def fetch_schedule(seasons: list[str], delay: float) -> pd.DataFrame:
    """Pull the full league schedule (games actually played) for each season.

    Endpoint: ScheduleLeagueV2 (stats.nba.com/stats/scheduleleaguev2).
      Why this one: it's the official schedule feed and the only bulk
      endpoint with arena city/state and tip-off times. It covers every
      game type in one call. The alternative, team game logs, has dates
      but no arena location, which we want for travel features later.

    Raw data: ~1,400 rows x ~150 columns per season. That includes preseason
    and All-Star games, nested broadcaster info flattened into many columns,
    and leader stats.

    What we keep: game id, date, tip-off time, home/away team, arena
    location, neutral-site flag. We drop preseason (001) and All-Star (003)
    games and any game not marked final.

    Quirks to know:
      * Dates look like '2023-10-24T00:00:00Z'. The 'Z' suffix (normally
        UTC) is wrong here: the values are US Eastern time. We strip it and
        keep the date part for GAME_DATE and the full timestamp as TIPOFF_ET.
      * Playoff games marked "if necessary" that were never played appear
        with a non-final status, so filtering on status removes them.
      * A few regular-season games were played abroad (Paris, Mexico City)
        or at neutral sites, so arena city doesn't always match the home
        team's city. That's why we store the arena explicitly.
    """
    frames = []
    for season in seasons:
        logger.info("Fetching schedule: %s", season)
        raw = call_endpoint(scheduleleaguev2.ScheduleLeagueV2, delay=delay, season=season)[0]
        require_columns(raw, SCHEDULE_COLUMN_MAP.keys(), "ScheduleLeagueV2")

        df = raw[list(SCHEDULE_COLUMN_MAP)].rename(columns=SCHEDULE_COLUMN_MAP)
        df["GAME_TYPE"] = df["GAME_ID"].str[:3].map(GAME_TYPE_BY_PREFIX)

        # Keep only the game types we care about, and only completed games.
        keep = df["GAME_TYPE"].notna() & (df["GAME_STATUS"] == GAME_STATUS_FINAL)
        logger.info("  -> %d raw rows, keeping %d played games", len(df), int(keep.sum()))
        df = df[keep].drop(columns="GAME_STATUS")
        df.insert(1, "SEASON", season)
        frames.append(df)

    schedule = pd.concat(frames, ignore_index=True)
    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"].str.rstrip("Z")).dt.date
    schedule["TIPOFF_ET"] = pd.to_datetime(schedule["TIPOFF_ET"].str.rstrip("Z"))
    # An empty GAME_LABEL means a normal regular-season game. Make it explicit.
    label = schedule["GAME_LABEL"].str.strip().replace("", pd.NA)
    schedule["GAME_LABEL"] = label.fillna(schedule["GAME_TYPE"])

    schedule = schedule.drop_duplicates(subset="GAME_ID")
    return schedule.sort_values(["GAME_DATE", "TIPOFF_ET"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3a. Per-season physical profile
# ---------------------------------------------------------------------------
# Kept: identity + age/height/weight + games played (useful for sanity
# checks). Dropped: season-total stats like PTS, NET_RATING and USG_PCT.
# Those summarize the WHOLE season, so using them to predict an injury in
# the middle of that season would leak information from future games.
BIO_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "GP"]


def fetch_season_bio(seasons: list[str], delay: float) -> pd.DataFrame:
    """Pull age, height and weight for every player who played each season.

    Endpoint: LeagueDashPlayerBioStats (stats.nba.com/stats/leaguedashplayerbiostats).
      Why this one: one request per season returns ~570 players with
      height already in inches and weight AS LISTED THAT SEASON. Player
      weight changes over a career (often 10-20 lbs), and CommonPlayerInfo
      only returns today's value. Using today's weight for a 2022 game would
      be both inaccurate and a subtle form of look-ahead leakage.

    Raw data: 23 columns, including college, country, draft info and
    season totals. Draft/country also live in player_details.csv, and season
    totals would leak (see BIO_COLUMNS above), so both are dropped.

    Quirks to know:
      * AGE is a whole number for the season, not exact age per game. We
        get exact age from birthdates in player_details.csv. This column is
        a fallback and a cross-check.
      * Defaults to regular season. Players who appeared only in the
        playoffs (very rare) won't be listed.
    """
    frames = []
    for season in seasons:
        logger.info("Fetching player bio stats: %s", season)
        df = call_endpoint(
            leaguedashplayerbiostats.LeagueDashPlayerBioStats, delay=delay, season=season
        )[0]
        require_columns(df, BIO_COLUMNS, "LeagueDashPlayerBioStats")
        df = df[BIO_COLUMNS].copy()
        df.insert(0, "SEASON", season)
        logger.info("  -> %d players", len(df))
        frames.append(df)

    bio = pd.concat(frames, ignore_index=True)
    return bio.sort_values(["SEASON", "PLAYER_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3b. Static player details (birthdate, position)
# ---------------------------------------------------------------------------
# Raw name -> our name. Deliberately NOT kept:
#   HEIGHT, WEIGHT, SEASON_EXP, TEAM_*, ROSTERSTATUS -> these are TODAY's
#       values, not values at game time. Per-season height/weight come from
#       player_season_bio.csv. Experience is computed from FROM_YEAR.
#   Name variants, slugs, jersey, school, flags -> not useful for modeling.
PLAYER_DETAIL_COLUMN_MAP = {
    "PERSON_ID": "PLAYER_ID",
    "DISPLAY_FIRST_LAST": "PLAYER_NAME",
    "BIRTHDATE": "BIRTHDATE",
    "POSITION": "POSITION",
    "FROM_YEAR": "FIRST_SEASON_YEAR",
    "DRAFT_YEAR": "DRAFT_YEAR",
    "DRAFT_NUMBER": "DRAFT_NUMBER",
    "COUNTRY": "COUNTRY",
}


def fetch_player_details(
    player_ids: Iterable[int],
    delay: float,
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull birthdate and position for each player, one request per player.

    Endpoint: CommonPlayerInfo (stats.nba.com/stats/commonplayerinfo).
      Why this one: it's the only endpoint that gives BIRTHDATE (exact age
      on every game date, not one rounded number per season) together with
      POSITION. There's no bulk version, so this is the slow part of the
      script: ~700 players x ~0.8 s ≈ 10 minutes.

    Why these players: everyone who appears in our game logs, not just
    "currently active" players. A player who got hurt in 2023 and retired
    is exactly the kind of example the model needs. Restricting to active
    players would remove those cases (survivorship bias).

    Raw data: 1 row x ~35 columns per player.

    Caching / resume: results are checkpointed to `cache_path` every
    PLAYER_DETAILS_CHECKPOINT_EVERY players. On rerun, players already in
    the file are skipped, so a crash or block halfway through doesn't mean
    starting over. Pass refresh=True to ignore the cache.

    Quirks to know:
      * POSITION is a word ('Guard', 'Forward', 'Center', 'Guard-Forward',
        'Forward-Center', ...), not 'PG' / 'SF' etc.
      * DRAFT_YEAR / DRAFT_NUMBER are the string 'Undrafted' for undrafted
        players. We leave that as-is here and handle it in feature
        engineering.
      * If one player keeps failing, we log it and continue rather than
        abort the whole run. Missing IDs are reported in the final summary.
    """
    player_ids = sorted({int(pid) for pid in player_ids})

    cached = pd.DataFrame(columns=list(PLAYER_DETAIL_COLUMN_MAP.values()))
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
        logger.info("Loaded %d cached player details from %s", len(cached), cache_path.name)

    done_ids = set(cached["PLAYER_ID"].astype(int)) if not cached.empty else set()
    todo_ids = [pid for pid in player_ids if pid not in done_ids]
    logger.info("Player details: %d to fetch, %d already cached", len(todo_ids), len(done_ids))

    new_rows: list[pd.DataFrame] = []
    failed_ids: list[int] = []

    def checkpoint() -> pd.DataFrame:
        """Merge cached + new rows and write them to disk."""
        # Skip the cache frame when it's empty: concatenating an empty frame
        # triggers a pandas warning and can mess up column dtypes.
        parts = ([cached] if not cached.empty else []) + new_rows
        if not parts:
            return cached
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.drop_duplicates(subset="PLAYER_ID", keep="last")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False)
        return combined

    for i, player_id in enumerate(todo_ids, start=1):
        try:
            # Result set 0 is CommonPlayerInfo. Others are headline stats.
            raw = call_endpoint(commonplayerinfo.CommonPlayerInfo, delay=delay, player_id=player_id)[0]
        except RuntimeError as error:
            logger.error("Giving up on player %d: %s", player_id, error)
            failed_ids.append(player_id)
            continue

        if raw.empty:
            logger.warning("No info returned for player %d", player_id)
            failed_ids.append(player_id)
            continue

        require_columns(raw, PLAYER_DETAIL_COLUMN_MAP.keys(), "CommonPlayerInfo")
        new_rows.append(raw[list(PLAYER_DETAIL_COLUMN_MAP)].rename(columns=PLAYER_DETAIL_COLUMN_MAP))

        if i % PLAYER_DETAILS_CHECKPOINT_EVERY == 0 or i == len(todo_ids):
            checkpoint()
            logger.info("  progress: %d / %d players fetched", i, len(todo_ids))

    details = checkpoint()

    if failed_ids:
        logger.warning(
            "%d players could not be fetched. Rerun the script to retry them: %s",
            len(failed_ids), failed_ids,
        )

    # BIRTHDATE arrives as '1995-09-16T00:00:00'. Keep just the date.
    details["BIRTHDATE"] = pd.to_datetime(details["BIRTHDATE"]).dt.date
    details = details.sort_values("PLAYER_ID").reset_index(drop=True)
    details.to_csv(cache_path, index=False)
    return details


# ---------------------------------------------------------------------------
# Summary / sanity checks
# ---------------------------------------------------------------------------
def summarize(
    logs: pd.DataFrame,
    schedule: pd.DataFrame,
    bio: pd.DataFrame,
    details: pd.DataFrame | None,
) -> None:
    """Print a short report so you can eyeball whether the pull looks right.

    Reference values for a normal season: 1,230 regular-season games
    (30 teams x 82 / 2), 6 play-in games, 70-105 playoff games, roughly
    500-600 players appearing.
    """
    print("\n" + "=" * 70)
    print("DATA COLLECTION SUMMARY")
    print("=" * 70)

    print("\nPlayer game logs (rows by season and type):")
    print(logs.groupby(["SEASON", "SEASON_TYPE"]).size().unstack(fill_value=0).to_string())
    print(f"Unique players: {logs['PLAYER_ID'].nunique()}  |  "
          f"dates: {logs['GAME_DATE'].min()} to {logs['GAME_DATE'].max()}")

    print("\nSchedule (games by season and type):")
    print(schedule.groupby(["SEASON", "GAME_TYPE"]).size().unstack(fill_value=0).to_string())

    # Every game in the logs should exist in the schedule.
    missing_games = set(logs["GAME_ID"]) - set(schedule["GAME_ID"])
    print(f"Game IDs in logs but not in schedule: {len(missing_games)} (expect 0)")

    print(f"\nPlayer season bio rows: {len(bio)}")
    log_player_seasons = logs[["SEASON", "PLAYER_ID"]].drop_duplicates()
    no_bio = log_player_seasons.merge(bio, on=["SEASON", "PLAYER_ID"], how="left", indicator=True)
    print(f"Player-seasons with no bio row: {(no_bio['_merge'] == 'left_only').sum()} "
          "(a handful is fine: players who only appeared in the playoffs)")

    if details is not None:
        missing_details = set(logs["PLAYER_ID"]) - set(details["PLAYER_ID"])
        print(f"\nPlayer details rows: {len(details)}  |  players missing details: "
              f"{len(missing_details)} (expect 0)")
        print("Position counts:")
        print(details["POSITION"].value_counts(dropna=False).to_string())
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull raw NBA data for the injury risk model.")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="Seasons in 'YYYY-YY' format (default: %(default)s)",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SECONDS,
        help="Seconds to wait between API calls (default: %(default)s). Raise it if you get blocked.",
    )
    parser.add_argument(
        "--skip-player-details", action="store_true",
        help="Skip the slow per-player CommonPlayerInfo pull (~10 min).",
    )
    parser.add_argument(
        "--refresh-player-details", action="store_true",
        help="Ignore the cached player_details.csv and refetch every player.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.time()

    try:
        # Order matters: game logs first, because they define which players
        # we need details for.
        logs = fetch_player_game_logs(args.seasons, args.delay)
        save_csv(logs, "player_game_logs.csv")

        schedule = fetch_schedule(args.seasons, args.delay)
        save_csv(schedule, "schedule.csv")

        bio = fetch_season_bio(args.seasons, args.delay)
        save_csv(bio, "player_season_bio.csv")

        details = None
        if not args.skip_player_details:
            details = fetch_player_details(
                logs["PLAYER_ID"].unique(),
                delay=args.delay,
                cache_path=RAW_DATA_DIR / "player_details.csv",
                refresh=args.refresh_player_details,
            )
            logger.info("Saved data/raw/player_details.csv (%d rows)", len(details))

    except (RuntimeError, ValueError) as error:
        # Expected failures (API down, blocked, schema change): one clear line, no traceback.
        logger.error("Data collection failed: %s", error)
        return 1

    summarize(logs, schedule, bio, details)
    logger.info("Done in %.1f minutes", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
