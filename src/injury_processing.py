"""
injury_processing.py
====================

Step 2 of the NBA Injury Risk pipeline: load injury reports, match them to
NBA API players, and create the binary target variable.

Inputs (data/raw/):
    injuries.csv           Kaggle "NBA Injury Stats 1951-2023" (scraped from
                           prosportstransactions.com). Ends 2023-04-16.
    player_game_logs.csv   From src/data_collection.py
    schedule.csv           From src/data_collection.py

Outputs (data/processed/):
    labeled_game_logs.csv  One row per player-game with INJURED_NEXT_GAME (0/1).
    injury_name_matches.csv  Audit trail: every injury name and how (or
                           whether) it was matched to an NBA API player.

Target definition
-----------------
    INJURED_NEXT_GAME = 1 for player-game g if:
      (a) the player gets an injury report (not illness / COVID protocol)
          dated on or after game g and on or before the team's NEXT game, AND
      (b) the player does NOT play in that next team game.
    Otherwise 0.

Why this definition:
    * We label the game BEFORE the absence. At prediction time (after game g)
      we know everything up to and including game g. We want to know if the
      player will be out injured next game. Labeling the game after would
      mean predicting something that has already happened.
    * "Team's next game", not "player's next game": an injured player has no
      next game in the logs, but the team always does (from the schedule).
    * Condition (b) ties the label to something observable. Day-to-day
      listings where the player plays anyway are not counted as injuries.
    * One injury = one positive. If a player is listed several times during
      the same absence (e.g. day-to-day upgraded to out), all reports map to
      the same game g, so it's labeled 1 once. Games missed during the absence
      have no rows in the logs, so they never become extra positives.
    * Day-to-day and season-ending injuries get the same label (1). Severity
      is kept as separate columns (INJURY_SEVERITY, GAMES_MISSED) for analysis.

Scope: 2022-23 only, and only rows whose labeling window ends on or before
the last date in injuries.csv. Past that date "no injury report" means "no
data", not "not injured", so labeling those rows 0 would be wrong.

Usage (from the repository root):
    python src/injury_processing.py
"""

from __future__ import annotations

import difflib
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INJURIES_PATH = RAW_DIR / "injuries.csv"
GAME_LOGS_PATH = RAW_DIR / "player_game_logs.csv"
SCHEDULE_PATH = RAW_DIR / "schedule.csv"
OUTPUT_PATH = PROCESSED_DIR / "labeled_game_logs.csv"
MATCH_REPORT_PATH = PROCESSED_DIR / "injury_name_matches.csv"

# The injury file only covers 2022-23, so we label that season only (Option A).
LABEL_SEASONS = ["2022-23"]

# Minimum difflib similarity (0-1) for a fuzzy name match. 0.85 is strict
# enough to reject different people with similar names, and it only runs
# within one team's roster, which limits false matches further.
FUZZY_MATCH_CUTOFF = 0.85

# Reporting-lag tolerance. In this dataset an injury is often only logged a
# few days after the player starts missing games (median ~5 days among late
# reports). Example: Rudy Gobert played 2022-12-14, missed the next game on
# 12-16, and was "placed on IL" on 12-18. A report is linked to game g if it
# comes on or before the team's next game OR within this many days after g,
# whichever is later. The player must still have missed the next game, and
# merge_asof guarantees he played no game between g and the report.
# 7 days catches ~80% of late reports while limiting mislabels for bench
# players who sat out for other reasons and were injured later.
# Set to 0 for the strict definition.
REPORT_LAG_TOLERANCE_DAYS = 7

# The injury file uses team nicknames, the NBA API uses abbreviations.
# This mapping also acts as the "NBA only" filter: a team that isn't one of
# these 30 (G League, WNBA, defunct) is dropped.
TEAM_NICKNAME_TO_ABBREVIATION = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA",
    "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL", "Nuggets": "DEN",
    "Pistons": "DET", "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND",
    "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM", "Heat": "MIA",
    "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP", "Knicks": "NYK",
    "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI", "Suns": "PHX",
    "Blazers": "POR", "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR",
    "Jazz": "UTA", "Wizards": "WAS",
}

# Injury-report reasons that are NOT injuries. Illness and COVID protocols
# don't come from workload or physical stress, so including them would add
# noise to a workload-based model. Matched case-insensitively against Notes.
NON_INJURY_PATTERNS = {
    "illness": r"\billness\b|\bflu\b|\bsick",
    "covid_protocol": r"protocol|covid",
    "rest": r"\brest\b|load management",
    "personal": r"personal",
    "suspension": r"suspen",
    "g_league": r"g league|g-league|nbadl|d-league",
}
# Note: concussion protocol IS an injury, so it's checked before the
# exclusion patterns (see classify_note).

# Names that string matching can't solve (nicknames, legal vs common names).
# Key: normalized name as it appears in the injury file.
# Value: normalized name as it appears in the NBA API.
# Filled in by reviewing the unmatched report from a first run.
MANUAL_NAME_ALIASES: dict[str, str] = {}

logger = logging.getLogger("injury_processing")


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
NAME_SUFFIXES = r"\b(jr|sr|ii|iii|iv|v)\b"


def normalize_name(name: str) -> str:
    """Turn a player name into a canonical form for matching.

    Steps, with examples:
      * Strip accents:  "Dennis Schröder" -> "Dennis Schroder"
      * Lowercase.
      * Drop parenthetical middle names: "John Wall (Hildred)" -> "john wall"
      * Remove periods, apostrophes, hyphens: "T.J." -> "tj",
        "De'Aaron" -> "deaaron", "Gilgeous-Alexander" -> "gilgeousalexander"
      * Drop suffixes: "Jaren Jackson Jr." -> "jaren jackson"
      * Collapse whitespace.
    Both sides (injury file and NBA API) go through the same function, so
    whatever we strip is stripped consistently.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[.'’`\-]", "", text)
    text = re.sub(NAME_SUFFIXES, " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_candidates(raw_name: str) -> list[str]:
    """Return every normalized spelling of an injury-file name.

    The injury file often lists aliases separated by a slash:
    "Nicolas Claxton / Nic Claxton" -> ["nicolas claxton", "nic claxton"].
    The NBA API may use either one, so we try both.
    """
    parts = [normalize_name(part) for part in str(raw_name).split("/")]
    candidates = [p for p in parts if p]
    # Apply manual aliases on top, so a known nickname maps to the API name.
    candidates += [MANUAL_NAME_ALIASES[c] for c in candidates if c in MANUAL_NAME_ALIASES]
    return list(dict.fromkeys(candidates))  # de-duplicate, keep order


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_game_logs(path: Path, seasons: list[str]) -> pd.DataFrame:
    """Load player game logs for the seasons we label.

    GAME_ID must be read as a string: it has leading zeros ("0022200001")
    that pandas would otherwise drop by parsing it as an integer, and then
    it would no longer match the schedule.
    """
    logs = pd.read_csv(path, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    logs = logs[logs["SEASON"].isin(seasons)].copy()
    if logs.empty:
        raise ValueError(f"No game logs found for seasons {seasons} in {path}")
    logger.info("Loaded %d player-game rows for %s", len(logs), seasons)
    return logs


def build_team_schedule(path: Path, seasons: list[str]) -> pd.DataFrame:
    """Turn the schedule (one row per game) into one row per team per game,
    with each team's NEXT game date and id.

    Why: the labeling window for a player-game runs until the team's next
    game. The schedule has home and away teams in separate columns, so we
    "stack" them to get a single TEAM_ID column, sort by date within each
    team, and shift by one row to get the next game.
    """
    schedule = pd.read_csv(path, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    schedule = schedule[schedule["SEASON"].isin(seasons)]

    home = schedule[["SEASON", "GAME_ID", "GAME_DATE", "HOME_TEAM_ID"]].rename(columns={"HOME_TEAM_ID": "TEAM_ID"})
    away = schedule[["SEASON", "GAME_ID", "GAME_DATE", "AWAY_TEAM_ID"]].rename(columns={"AWAY_TEAM_ID": "TEAM_ID"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values(["TEAM_ID", "GAME_DATE"])

    # shift(-1) within each team-season = that team's next game.
    # The last game of a team's season gets NaT/NaN (no next game).
    grouped = team_games.groupby(["TEAM_ID", "SEASON"])
    team_games["NEXT_TEAM_GAME_ID"] = grouped["GAME_ID"].shift(-1)
    team_games["NEXT_TEAM_GAME_DATE"] = grouped["GAME_DATE"].shift(-1)
    return team_games.reset_index(drop=True)


def classify_note(note: str) -> str:
    """Return 'injury' or the non-injury category for an injury-report note."""
    text = str(note).lower()
    if "concussion" in text:
        return "injury"  # concussion protocol is an injury, not COVID protocol
    for category, pattern in NON_INJURY_PATTERNS.items():
        if re.search(pattern, text):
            return category
    return "injury"


def severity_from_note(note: str) -> str:
    """Rough severity tag from the free-text note (for analysis only).

    Not used in the target. Free text is inconsistent, so GAMES_MISSED
    (computed from the logs) is the more reliable severity measure.
    """
    text = str(note).lower()
    if "out for season" in text or "surgery" in text:
        return "season_ending_or_surgery"
    if "dtd" in text:
        return "day_to_day"
    if text.strip() == "placed on il":
        return "unspecified"
    return "out"


def load_injuries(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Load the Kaggle injury file and keep NBA injury placements in range.

    Raw columns: an unnamed index, Date, Team (nickname), Acquired (player
    returning), Relinquished (player going out), Notes (free text).
    Each row has exactly one of Acquired / Relinquished filled in.

    What we keep:
      * "Relinquished" rows only: that's the injury onset. "Acquired" rows
        ("activated from IL") are return dates. We don't need them because
        we measure games missed directly from the game logs, which is more
        reliable than the transaction log.
      * Dates in the labeling seasons.
      * The 30 NBA teams.
    What we drop (and log): non-injury reasons, and malformed rows where the
    name column contains injury text instead of a player name.
    """
    raw = pd.read_csv(path, index_col=0)
    expected = {"Date", "Team", "Acquired", "Relinquished", "Notes"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"injuries.csv is missing expected columns: {sorted(missing)}")

    raw["Date"] = pd.to_datetime(raw["Date"])
    logger.info("Injury file: %d rows, %s to %s", len(raw), raw["Date"].min().date(), raw["Date"].max().date())

    df = raw[(raw["Date"] >= start_date) & (raw["Date"] <= end_date) & raw["Relinquished"].notna()].copy()
    logger.info("  %d 'Relinquished' (going out) rows in %s to %s", len(df), start_date.date(), end_date.date())

    # NBA-only filter via the nickname map.
    df["TEAM_ABBREVIATION"] = df["Team"].map(TEAM_NICKNAME_TO_ABBREVIATION)
    unknown_teams = df.loc[df["TEAM_ABBREVIATION"].isna(), "Team"].unique()
    if len(unknown_teams):
        logger.info("  dropping %d rows with non-NBA teams: %s",
                    int(df["TEAM_ABBREVIATION"].isna().sum()), list(unknown_teams))
    df = df[df["TEAM_ABBREVIATION"].notna()]

    # Malformed rows: the name field holds injury text, e.g. "left knee injury (DTD)".
    malformed = df["Relinquished"].str.contains(r"\(DTD\)|injury|strained|sprained|\bsore\b", case=False)
    if malformed.any():
        logger.warning("  dropping %d malformed rows (injury text in the name field): %s",
                       int(malformed.sum()), df.loc[malformed, "Relinquished"].str.strip().tolist())
    df = df[~malformed]

    # Injury vs non-injury.
    df["NOTE_CATEGORY"] = df["Notes"].map(classify_note)
    logger.info("  note categories:\n%s", df["NOTE_CATEGORY"].value_counts().to_string())
    df = df[df["NOTE_CATEGORY"] == "injury"]

    df["INJURY_SEVERITY"] = df["Notes"].map(severity_from_note)
    df["INJURY_NAME_RAW"] = df["Relinquished"].str.replace("•", "", regex=False).str.strip()
    return df.rename(columns={"Date": "INJURY_REPORT_DATE", "Notes": "INJURY_NOTE"})[
        ["INJURY_REPORT_DATE", "TEAM_ABBREVIATION", "INJURY_NAME_RAW", "INJURY_NOTE", "INJURY_SEVERITY"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------
def match_injury_names(injuries: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Attach an NBA API PLAYER_ID to each injury row.

    Matching runs in stages, from most to least strict. A row only reaches
    the next stage if the previous one failed:
      1. team_exact:   normalized name matches a player who played for that
                       team that season.
      2. league_exact: normalized name matches exactly one player anywhere in
                       the league that season (covers trades, where the
                       injury is listed under a team he barely played for).
      3. team_fuzzy:   difflib similarity >= FUZZY_MATCH_CUTOFF against that
                       team's roster (covers small spelling differences).
    Anything left is 'unmatched' and reported for manual review.

    Why the team constraint: it removes most ambiguity. Two players named
    "Jalen Williams" or "Marcus Morris" are rarely on the same team.
    """
    # Lookup tables built from the game logs.
    roster = logs[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].drop_duplicates()
    roster = roster.assign(NAME_KEY=roster["PLAYER_NAME"].map(normalize_name))

    team_lookup: dict[tuple[str, str], int] = {
        (row.TEAM_ABBREVIATION, row.NAME_KEY): row.PLAYER_ID for row in roster.itertuples()
    }
    team_names: dict[str, list[str]] = roster.groupby("TEAM_ABBREVIATION")["NAME_KEY"].apply(list).to_dict()

    league = roster[["PLAYER_ID", "NAME_KEY"]].drop_duplicates()
    ambiguous_keys = set(league.loc[league["NAME_KEY"].duplicated(), "NAME_KEY"])
    league_lookup = {
        key: pid for key, pid in zip(league["NAME_KEY"], league["PLAYER_ID"]) if key not in ambiguous_keys
    }
    if ambiguous_keys:
        logger.warning("Names shared by different players (league-wide match disabled for them): %s",
                       sorted(ambiguous_keys))

    def match_one(raw_name: str, team: str) -> tuple[int | None, str, str]:
        candidates = name_candidates(raw_name)
        for cand in candidates:
            if (team, cand) in team_lookup:
                return team_lookup[(team, cand)], "team_exact", cand
        for cand in candidates:
            if cand in league_lookup:
                return league_lookup[cand], "league_exact", cand
        for cand in candidates:
            close = difflib.get_close_matches(cand, team_names.get(team, []), n=1, cutoff=FUZZY_MATCH_CUTOFF)
            if close:
                return team_lookup[(team, close[0])], "team_fuzzy", close[0]
        return None, "unmatched", ""

    results = [match_one(name, team) for name, team in zip(injuries["INJURY_NAME_RAW"], injuries["TEAM_ABBREVIATION"])]
    matched = injuries.copy()
    matched["PLAYER_ID"] = pd.array([r[0] for r in results], dtype="Int64")
    matched["MATCH_METHOD"] = [r[1] for r in results]
    matched["MATCHED_NAME_KEY"] = [r[2] for r in results]

    # Attach the API name so the audit file shows both spellings side by side.
    id_to_name = roster.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"]
    matched["API_PLAYER_NAME"] = matched["PLAYER_ID"].map(id_to_name)
    return matched


def report_name_matching(matched: pd.DataFrame) -> None:
    """Print match counts by method, with examples for review."""
    print("\n" + "=" * 70)
    print("NAME MATCHING REPORT")
    print("=" * 70)
    print(matched["MATCH_METHOD"].value_counts().to_string())

    # Show non-trivial matches: where the raw name differs from the API name.
    differs = matched[
        matched["API_PLAYER_NAME"].notna()
        & (matched["INJURY_NAME_RAW"] != matched["API_PLAYER_NAME"])
    ].drop_duplicates("INJURY_NAME_RAW")
    print(f"\nMatched names that differ between sources ({len(differs)} unique). Examples:")
    print(differs[["INJURY_NAME_RAW", "API_PLAYER_NAME", "TEAM_ABBREVIATION", "MATCH_METHOD"]]
          .head(25).to_string(index=False))

    unmatched = matched[matched["MATCH_METHOD"] == "unmatched"]
    print(f"\nUnmatched injury rows: {len(unmatched)} ({unmatched['INJURY_NAME_RAW'].nunique()} unique names)")
    if not unmatched.empty:
        print(unmatched[["INJURY_REPORT_DATE", "TEAM_ABBREVIATION", "INJURY_NAME_RAW", "INJURY_NOTE"]]
              .drop_duplicates("INJURY_NAME_RAW").to_string(index=False))


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
def attach_next_team_game(logs: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Add NEXT_TEAM_GAME_ID / NEXT_TEAM_GAME_DATE to every player-game.

    This defines each row's labeling window, so it has to happen before
    injuries are linked to games.
    """
    logs = logs.merge(
        team_games[["TEAM_ID", "GAME_ID", "NEXT_TEAM_GAME_ID", "NEXT_TEAM_GAME_DATE"]],
        on=["TEAM_ID", "GAME_ID"], how="left", validate="many_to_one", indicator=True,
    )
    # Every game in the logs must exist in the schedule, or the window is unknown.
    not_in_schedule = logs["_merge"] == "left_only"
    if not_in_schedule.any():
        raise ValueError(f"{int(not_in_schedule.sum())} player-games have no schedule row")
    logs = logs.drop(columns="_merge")

    # Labeling window end: the team's next game, extended to cover the
    # reporting-lag tolerance (see REPORT_LAG_TOLERANCE_DAYS). NaT stays NaT
    # for a team's last game of the season (no next game -> no label).
    lag_end = logs["GAME_DATE"] + pd.Timedelta(days=REPORT_LAG_TOLERANCE_DAYS)
    logs["LABEL_WINDOW_END"] = logs["NEXT_TEAM_GAME_DATE"].where(
        logs["NEXT_TEAM_GAME_DATE"].isna(), logs["NEXT_TEAM_GAME_DATE"].combine(lag_end, max)
    )
    return logs


def assign_injuries_to_games(logs: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Link each injury report to the player-game it should label.

    For an injury reported on date D for player P, we find P's most recent
    game played on or before D (game g). `merge_asof` does exactly this
    "last row at or before this date" lookup efficiently.

    The report only counts for game g if D falls inside g's window:
        GAME_DATE(g) <= D <= LABEL_WINDOW_END(g)
    where LABEL_WINDOW_END is the team's next game, extended by the
    reporting-lag tolerance. D == GAME_DATE(g) covers injuries suffered
    during game g and reported that same night. Reports outside any window
    are "orphans": preseason injuries (no game played yet), or reports long
    after the player's last game. They're dropped and counted.

    INJURY_REPORTED_LATE marks reports that came after the team's next game
    (within the tolerance). Filter them out to get the strict definition.

    Several reports in the same window (day-to-day, then out) collapse into
    one episode: we keep the first report as the onset.
    """
    reports = injuries[injuries["PLAYER_ID"].notna()].copy()
    reports["PLAYER_ID"] = reports["PLAYER_ID"].astype("int64")
    reports = reports.sort_values("INJURY_REPORT_DATE")

    games = logs[["PLAYER_ID", "GAME_ID", "GAME_DATE", "NEXT_TEAM_GAME_DATE", "LABEL_WINDOW_END"]]
    games = games.sort_values("GAME_DATE")

    linked = pd.merge_asof(
        reports, games,
        left_on="INJURY_REPORT_DATE", right_on="GAME_DATE",
        by="PLAYER_ID", direction="backward",
    )
    no_prior_game = linked["GAME_ID"].isna()
    in_window = ~no_prior_game & (linked["INJURY_REPORT_DATE"] <= linked["LABEL_WINDOW_END"])
    linked["INJURY_REPORTED_LATE"] = linked["INJURY_REPORT_DATE"] > linked["NEXT_TEAM_GAME_DATE"]
    logger.info(
        "Injury reports linked to a game: %d (of which %d reported late, within %d days). "
        "Orphans: %d with no prior game (preseason injuries), %d too long after the last game",
        int(in_window.sum()), int((in_window & linked["INJURY_REPORTED_LATE"]).sum()),
        REPORT_LAG_TOLERANCE_DAYS, int(no_prior_game.sum()), int((~in_window & ~no_prior_game).sum()),
    )

    episodes = (
        linked[in_window]
        .sort_values("INJURY_REPORT_DATE")
        .drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="first")
    )
    logger.info("Unique injury episodes after collapsing repeat reports: %d", len(episodes))
    return episodes[["PLAYER_ID", "GAME_ID", "INJURY_REPORT_DATE", "INJURY_REPORTED_LATE",
                     "INJURY_NOTE", "INJURY_SEVERITY"]]


def add_games_missed(logs: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """For every player-game, count the team games the player misses before
    his next appearance, and flag whether he plays the team's next game.

    PLAYED_NEXT_TEAM_GAME is condition (b) of the target.
    GAMES_MISSED is a severity measure: 0 = played the next game, 3 = sat out
    3 team games, and so on. If he never plays again that season, it counts
    every remaining team game.
    """
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    # Did the player appear in the team's next game? (Any row with that
    # GAME_ID, which also handles players who appear for the opponent after
    # a trade.)
    played = set(zip(logs["PLAYER_ID"], logs["GAME_ID"]))
    logs["PLAYED_NEXT_TEAM_GAME"] = [
        (pid, gid) in played if isinstance(gid, str) else pd.NA
        for pid, gid in zip(logs["PLAYER_ID"], logs["NEXT_TEAM_GAME_ID"])
    ]

    # Date of the player's own next appearance.
    logs["NEXT_PLAYER_GAME_DATE"] = logs.groupby(["PLAYER_ID", "SEASON"])["GAME_DATE"].shift(-1)

    # Count team games strictly between this game and his next appearance.
    # Sorted team game dates + searchsorted is fast (no row-by-row loops over the schedule).
    team_dates = {
        key: grp["GAME_DATE"].sort_values().to_numpy()
        for key, grp in team_games.groupby(["TEAM_ID", "SEASON"])
    }
    missed = []
    for team_id, season, game_date, next_date in zip(
        logs["TEAM_ID"], logs["SEASON"], logs["GAME_DATE"], logs["NEXT_PLAYER_GAME_DATE"]
    ):
        dates = team_dates.get((team_id, season))
        if dates is None:
            missed.append(pd.NA)
            continue
        after = dates.searchsorted(game_date.to_datetime64(), side="right")
        until = len(dates) if pd.isna(next_date) else dates.searchsorted(next_date.to_datetime64(), side="left")
        missed.append(max(int(until - after), 0))
    logs["GAMES_MISSED"] = pd.array(missed, dtype="Int64")
    return logs


def build_labels(
    logs: pd.DataFrame,
    team_games: pd.DataFrame,
    episodes: pd.DataFrame,
    coverage_end: pd.Timestamp,
) -> pd.DataFrame:
    """Combine everything into the final labeled table.

    Rows we DROP because their label can't be observed:
      * The team's last game of the season (no next game, so no window).
      * Games whose window (including the lag tolerance) ends after
        coverage_end, the last injury-file date. After that date a missing
        report means missing data, not "healthy".
    """
    logs = add_games_missed(logs, team_games)

    no_next_game = logs["NEXT_TEAM_GAME_DATE"].isna()
    beyond_coverage = logs["LABEL_WINDOW_END"] > coverage_end
    logger.info("Dropping %d rows with no next team game (season finale) and %d rows "
                "whose window ends after injury data coverage (%s)",
                int(no_next_game.sum()), int((beyond_coverage & ~no_next_game).sum()), coverage_end.date())
    logs = logs[~no_next_game & ~beyond_coverage].copy()

    # Attach injury episodes (at most one per player-game by construction).
    labeled = logs.merge(episodes, on=["PLAYER_ID", "GAME_ID"], how="left", validate="one_to_one")
    has_report = labeled["INJURY_REPORT_DATE"].notna()
    missed_next = labeled["PLAYED_NEXT_TEAM_GAME"].eq(False)

    # Reported but played the next game anyway (e.g. day-to-day, then cleared).
    # These are NOT positives. Keep the report columns for transparency.
    report_but_played = has_report & ~missed_next
    logger.info("Injury reports where the player still played the next game (labeled 0): %d",
                int(report_but_played.sum()))

    labeled["INJURED_NEXT_GAME"] = (has_report & missed_next).astype("int8")
    # GAMES_MISSED only describes an injury when the label is 1. For label 0
    # rows it would be rest / DNP, so blank it to avoid misuse.
    labeled.loc[labeled["INJURED_NEXT_GAME"] == 0, "INJURY_GAMES_MISSED"] = pd.NA
    labeled.loc[labeled["INJURED_NEXT_GAME"] == 1, "INJURY_GAMES_MISSED"] = labeled["GAMES_MISSED"]

    internal_cols = {"NEXT_PLAYER_GAME_DATE", "PLAYED_NEXT_TEAM_GAME", "GAMES_MISSED", "LABEL_WINDOW_END"}
    keep_cols = [c for c in logs.columns if c not in internal_cols]
    keep_cols += ["INJURY_REPORT_DATE", "INJURY_REPORTED_LATE", "INJURY_NOTE", "INJURY_SEVERITY",
                  "INJURY_GAMES_MISSED", "INJURED_NEXT_GAME"]

    # Injury columns are only meaningful for positives. Blank them on the
    # rare 0 rows that had a report but where the player played anyway.
    injury_detail_cols = ["INJURY_REPORT_DATE", "INJURY_REPORTED_LATE", "INJURY_NOTE", "INJURY_SEVERITY"]
    labeled.loc[labeled["INJURED_NEXT_GAME"] == 0, injury_detail_cols] = pd.NA
    return labeled[keep_cols].sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def report_class_balance(labeled: pd.DataFrame) -> None:
    """Print the injured / not-injured ratio and some sanity checks."""
    counts = labeled["INJURED_NEXT_GAME"].value_counts().sort_index()
    n_pos, n_neg = int(counts.get(1, 0)), int(counts.get(0, 0))
    print("\n" + "=" * 70)
    print("CLASS BALANCE")
    print("=" * 70)
    print(f"Rows:            {len(labeled):>7,}")
    print(f"Not injured (0): {n_neg:>7,}  ({n_neg / len(labeled):.2%})")
    print(f"Injured (1):     {n_pos:>7,}  ({n_pos / len(labeled):.2%})")
    if n_pos:
        print(f"Ratio:           1 injured : {n_neg / n_pos:.1f} not injured")
    print(f"Players with at least one injury: "
          f"{labeled.loc[labeled['INJURED_NEXT_GAME'] == 1, 'PLAYER_ID'].nunique()} "
          f"of {labeled['PLAYER_ID'].nunique()}")
    positives = labeled[labeled["INJURED_NEXT_GAME"] == 1]
    late = positives["INJURY_REPORTED_LATE"].astype(bool)
    print(f"Positives reported on time: {int((~late).sum())}, reported late "
          f"(within {REPORT_LAG_TOLERANCE_DAYS}-day tolerance): {int(late.sum())}")
    print("\nInjuries by severity tag:")
    print(positives["INJURY_SEVERITY"].value_counts().to_string())
    print("\nGames missed per injury (describe):")
    print(positives["INJURY_GAMES_MISSED"].astype(float).describe().round(1).to_string())
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    for path in (INJURIES_PATH, GAME_LOGS_PATH, SCHEDULE_PATH):
        if not path.exists():
            logger.error("Missing input file: %s", path.relative_to(PROJECT_ROOT))
            return 1

    try:
        logs = load_game_logs(GAME_LOGS_PATH, LABEL_SEASONS)
        team_games = build_team_schedule(SCHEDULE_PATH, LABEL_SEASONS)
        logs = attach_next_team_game(logs, team_games)

        # Injury coverage: from the season's first game to the file's last date.
        coverage_end = pd.to_datetime(pd.read_csv(INJURIES_PATH, usecols=["Date"])["Date"]).max()
        season_start = team_games["GAME_DATE"].min()
        injuries = load_injuries(INJURIES_PATH, season_start, coverage_end)

        matched = match_injury_names(injuries, logs)
        report_name_matching(matched)

        episodes = assign_injuries_to_games(logs, matched)
        labeled = build_labels(logs, team_games, episodes, coverage_end)
    except (ValueError, KeyError) as error:
        logger.error("Injury processing failed: %s", error)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_csv(MATCH_REPORT_PATH, index=False)
    labeled.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %s (%d rows x %d cols)", OUTPUT_PATH.relative_to(PROJECT_ROOT), *labeled.shape)
    logger.info("Saved %s (%d rows)", MATCH_REPORT_PATH.relative_to(PROJECT_ROOT), len(matched))

    report_class_balance(labeled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
