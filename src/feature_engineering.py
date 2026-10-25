"""
feature_engineering.py
======================

Step 3 of the NBA Injury Risk pipeline: turn labeled player-games into
model-ready features.

Inputs:
    data/processed/labeled_game_logs.csv  Rows to predict + target (Step 2)
    data/raw/player_game_logs.csv         Full game history for rolling windows
    data/raw/player_season_bio.csv        Age, height, weight per season
    data/raw/player_details.csv           Position

Output:
    data/processed/model_features.csv     One row per labeled player-game:
                                          identifiers, features, target.

Prediction framing and data leakage
-----------------------------------
Each row answers: "BEFORE game g tips off, how likely is this player to get
hurt in game g and miss the team's next game?"

Leakage means letting the model see information it wouldn't have at
prediction time. It makes offline scores look great and then fail in real
use. Here the main risk is game g's own box score: a player hurt during
game g often has unusually low minutes in that game. If those minutes are a
feature, the model "predicts" the injury by seeing its effect.

How we prevent it:
  1. Every rolling window uses closed="left": a window for game g covers the
     N calendar days BEFORE g's date and never includes game g. This is the
     time-based equivalent of "shift by 1 before rolling". A plain .shift(1)
     doesn't work well with calendar windows: "last 7 days" must be
     measured from g's date, not from the previous game's date.
  2. No box-score column from game g (MIN, PTS, ...) goes into the output.
  3. Columns derived from the target (injury note, severity, games missed)
     are not features.
  4. Schedule facts about game g itself (days of rest before it, whether
     it's a back-to-back, its number in the season) ARE allowed: the
     schedule is public weeks in advance.
  5. assert_no_leakage() checks this automatically: it inflates the minutes
     of sampled games, rebuilds the features, and confirms those games'
     features didn't change.

Also note: rolling history comes from the FULL raw game logs, not from the
labeled file. Rows dropped in Step 2 (e.g. outside injury coverage) must
not create gaps in a player's history.

Usage (from the repository root):
    python src/feature_engineering.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELED_PATH = PROCESSED_DIR / "labeled_game_logs.csv"
GAME_LOGS_PATH = RAW_DIR / "player_game_logs.csv"
BIO_PATH = RAW_DIR / "player_season_bio.csv"
DETAILS_PATH = RAW_DIR / "player_details.csv"
OUTPUT_PATH = PROCESSED_DIR / "model_features.csv"

# Rolling windows in calendar days.
#   7  = "acute" load: one week, ~3-4 NBA games. The standard acute window
#        in sports-science workload research.
#   28 = "chronic" load: four weeks, ~12-14 games. The standard chronic
#        window (Hulin et al. 2014, Gabbett 2016): long enough to show what
#        the player is conditioned for, short enough to follow changes.
#   14 = intermediate: catches a two-week build-up (e.g. a road trip
#        stretch) that 7 days is too short to see and 28 days smooths out.
# Calendar days rather than "last N games", because recovery happens over
# time: 4 games in 5 nights is very different from 4 games in 10 nights.
WORKLOAD_WINDOWS_DAYS = (7, 14, 28)
ACUTE_DAYS = 7
CHRONIC_DAYS = 28

# ACWR threshold. Gabbett (2016) describes 0.8-1.3 as a "sweet spot" and
# > 1.5 as a "danger zone" where injury risk rises sharply. Those numbers
# come from rugby, cricket and Australian football, not the NBA, and the
# metric itself has been criticized (Lolli et al. 2019; Impellizzeri et al.
# 2020). So ACWR stays continuous (the model learns its own cut-offs), and
# this flag is only an interpretable "spike" indicator.
ACWR_SPIKE_THRESHOLD = 1.5

# ACWR needs a full chronic window. In the first 28 days of the season the
# 28-day window includes the offseason (zero minutes), which makes the
# ratio look like a huge spike for every player. So ACWR is left missing
# (NaN) before day 28 instead of using a misleading value.
MIN_DAYS_INTO_SEASON_FOR_ACWR = CHRONIC_DAYS

# Position: the API gives 7 labels (Guard, Guard-Forward, Forward-Center,
# ...). We keep the PRIMARY position (first word). Hybrid labels are small
# groups and the first word is the player's main role. 3 groups keep each
# dummy well populated.
POSITION_GROUPS = ("Guard", "Forward", "Center")


# Columns that identify a row (kept in the output but NOT model inputs).
ID_COLUMNS = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]
TARGET = "INJURED_NEXT_GAME"
# Kept for sensitivity analysis (strict vs lag-tolerant label), not a feature.
METADATA_COLUMNS = ["INJURY_REPORTED_LATE"]

FEATURE_COLUMNS = [
    "MIN_LAST_7D", "MIN_LAST_14D", "MIN_LAST_28D",
    "ACWR", "ACWR_SPIKE",
    "DAYS_SINCE_LAST_GAME", "IS_BACK_TO_BACK",
    "GAMES_LAST_7D",
    "PLAYER_GAME_NUMBER",
    "MPG_TREND",
    "AGE", "HEIGHT_INCHES", "WEIGHT",
    "POS_GUARD", "POS_FORWARD", "POS_CENTER",
]

logger = logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four input files with correct dtypes.

    GAME_ID is read as a string to keep its leading zeros.
    """
    for path in (LABELED_PATH, GAME_LOGS_PATH, BIO_PATH, DETAILS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path.relative_to(PROJECT_ROOT)}")

    labeled = pd.read_csv(LABELED_PATH, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    seasons = labeled["SEASON"].unique().tolist()

    logs = pd.read_csv(GAME_LOGS_PATH, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    logs = logs[logs["SEASON"].isin(seasons)]

    bio = pd.read_csv(BIO_PATH)
    details = pd.read_csv(DETAILS_PATH)
    logger.info("Loaded %d labeled rows, %d history rows for %s", len(labeled), len(logs), seasons)
    return labeled, logs, bio, details


# ---------------------------------------------------------------------------
# Workload and schedule features
# ---------------------------------------------------------------------------
def rolling_past(logs: pd.DataFrame, column: str, days: int, how: str) -> np.ndarray:
    """Rolling `how` ('sum' or 'count') of `column` over the `days` calendar
    days BEFORE each game, per player and season.

    closed="left" excludes the current row, so game g never sees its own
    stats (see module docstring). Seasons are grouped separately so a
    window never reaches back into the previous season's games.

    `logs` must already be sorted by PLAYER_ID, SEASON, GAME_DATE. groupby
    then returns rows in that same order, which we check before returning.
    """
    rolled = (
        logs.set_index("GAME_DATE")
        .groupby(["PLAYER_ID", "SEASON"], sort=True)[column]
        .rolling(f"{days}D", closed="left")
        .agg(how)
    )
    # Alignment check: rolled rows must line up 1:1 with the input rows.
    if not np.array_equal(rolled.index.get_level_values("GAME_DATE"), logs["GAME_DATE"].to_numpy()):
        raise RuntimeError("Rolling result is misaligned with the input rows")
    # An empty window (no game in the last N days) comes back as NaN.
    # Zero minutes / zero games is the correct value there.
    return rolled.fillna(0).to_numpy()


def build_workload_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute every game-history feature for every player-game.

    Runs on the full game history (all game types in the season), so the
    windows include play-in/playoff load when it applies.
    """
    df = logs.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    by_player_season = df.groupby(["PLAYER_ID", "SEASON"])

    # -- 1. Rolling cumulative minutes (7 / 14 / 28 days) --------------------
    # Total minutes played in the N days before the game. Minutes are the
    # best public proxy for physical load. (Teams have tracking data like
    # distance and accelerations, which isn't public.)
    for days in WORKLOAD_WINDOWS_DAYS:
        df[f"MIN_LAST_{days}D"] = rolling_past(df, "MIN", days, "sum")

    # -- 5. Games in last 7 days (schedule density) --------------------------
    # Number of games the player played in the 7 days before this one.
    # 3 means this is his 4th game in 8 nights.
    df["GAMES_LAST_7D"] = rolling_past(df, "MIN", ACUTE_DAYS, "count")
    games_last_28d = rolling_past(df, "MIN", CHRONIC_DAYS, "count")

    # -- 2. ACWR (acute:chronic workload ratio) ------------------------------
    # Average daily minutes over the last 7 days divided by average daily
    # minutes over the last 28 days. 1.0 = doing what he's used to,
    # > 1.5 = a big recent spike. Coupled form (the 7 days are inside the
    # 28): it's the version used in the original papers, so results are
    # comparable.
    acute_daily = df["MIN_LAST_7D"] / ACUTE_DAYS
    chronic_daily = df["MIN_LAST_28D"] / CHRONIC_DAYS
    season_opener = df.groupby("SEASON")["GAME_DATE"].transform("min")
    days_into_season = (df["GAME_DATE"] - season_opener).dt.days
    acwr_defined = (days_into_season >= MIN_DAYS_INTO_SEASON_FOR_ACWR) & (chronic_daily > 0)
    df["ACWR"] = (acute_daily / chronic_daily).where(acwr_defined)
    # Nullable integer so "unknown" stays NA instead of being forced to 0.
    df["ACWR_SPIKE"] = (df["ACWR"] > ACWR_SPIKE_THRESHOLD).astype("Int8").where(df["ACWR"].notna())

    # -- 3. Rest days ----------------------------------------------------------
    # Calendar days since the player's previous game this season. 1 = played
    # yesterday. NaN for his first game of the season: there's no previous
    # game, and making up a number would be misleading. Very large values
    # mean he's coming back from an absence. That matters because players
    # are at higher risk right after returning.
    df["DAYS_SINCE_LAST_GAME"] = by_player_season["GAME_DATE"].diff().dt.days

    # -- 4. Back-to-back -------------------------------------------------------
    # 1 if he played the day before (second night of a back-to-back).
    # Kept as its own flag even though it's DAYS_SINCE_LAST_GAME == 1:
    # it's a well-known NBA load-management signal and easy to explain.
    df["IS_BACK_TO_BACK"] = (df["DAYS_SINCE_LAST_GAME"] == 1).astype("int8")

    # -- 6. Season fatigue -----------------------------------------------------
    # Which game of the season this is for him (1 = first). Counting games
    # doesn't use any of game g's stats, so it's not leakage.
    df["PLAYER_GAME_NUMBER"] = by_player_season.cumcount() + 1

    # -- 9. Minutes-per-game trend ---------------------------------------------
    # Average minutes PER GAME over the last 7 days minus over the last 28.
    # +8 = he's playing 8 more minutes per game than his recent norm (e.g. a
    # teammate got hurt and his role grew). Unlike ACWR, which mixes how often
    # he plays with how long, this isolates role intensity per game.
    # A difference (in minutes) rather than a ratio: easier to read, and no
    # division blow-up for low-minute players. NaN if he played no games in
    # the last 7 days (no recent per-game average exists).
    mpg_7d = (df["MIN_LAST_7D"] / df["GAMES_LAST_7D"]).where(df["GAMES_LAST_7D"] > 0)
    mpg_28d = (df["MIN_LAST_28D"] / games_last_28d).where(games_last_28d > 0)
    df["MPG_TREND"] = mpg_7d - mpg_28d

    keep = ["PLAYER_ID", "GAME_ID"] + [
        "MIN_LAST_7D", "MIN_LAST_14D", "MIN_LAST_28D", "ACWR", "ACWR_SPIKE",
        "DAYS_SINCE_LAST_GAME", "IS_BACK_TO_BACK", "GAMES_LAST_7D",
        "PLAYER_GAME_NUMBER", "MPG_TREND",
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Player profile features
# ---------------------------------------------------------------------------
def build_profile_features(bio: pd.DataFrame, details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare per-season physical profile and position dummies.

    -- 7. Age: AGE from player_season_bio.csv (whole years, per season).
       We use the season value rather than exact age from birthdate: it's
       what you asked for, it's the NBA's own figure, and over one season
       exact age adds less than one year of precision. Height and weight
       come from the same file. They're part of the physical profile in the
       project goal, so they're included too.

    -- 8. Position dummies: one 0/1 column per primary position. We keep
       all three columns (no "drop_first"): tree models don't need a
       reference category, and dropping one for logistic regression is
       done in the modeling step where it matters.
    """
    season_profile = bio.rename(columns={"PLAYER_HEIGHT_INCHES": "HEIGHT_INCHES", "PLAYER_WEIGHT": "WEIGHT"})
    season_profile = season_profile[["SEASON", "PLAYER_ID", "AGE", "HEIGHT_INCHES", "WEIGHT"]]

    primary = details["POSITION"].str.split("-").str[0]
    unknown = sorted(set(primary.dropna()) - set(POSITION_GROUPS))
    if unknown:
        logger.warning("Unrecognized position labels (left as all-zero dummies): %s", unknown)
    positions = details[["PLAYER_ID"]].copy()
    for group in POSITION_GROUPS:
        positions[f"POS_{group.upper()}"] = (primary == group).astype("int8")
    return season_profile, positions


# ---------------------------------------------------------------------------
# Assembly, checks and reports
# ---------------------------------------------------------------------------
def assemble(labeled, workload, season_profile, positions) -> pd.DataFrame:
    """Attach all features to the labeled rows.

    validate= makes pandas raise if a join unexpectedly duplicates rows,
    which would silently corrupt the dataset.
    """
    base = labeled[ID_COLUMNS + [TARGET] + METADATA_COLUMNS]
    out = (
        base.merge(workload, on=["PLAYER_ID", "GAME_ID"], how="left", validate="one_to_one")
        .merge(season_profile, on=["SEASON", "PLAYER_ID"], how="left", validate="many_to_one")
        .merge(positions, on="PLAYER_ID", how="left", validate="many_to_one")
    )
    if len(out) != len(labeled):
        raise RuntimeError("Row count changed during feature assembly")
    return out[ID_COLUMNS + FEATURE_COLUMNS + [TARGET] + METADATA_COLUMNS]


def assert_no_leakage(logs: pd.DataFrame, workload: pd.DataFrame) -> None:
    """Show that a game's features don't depend on that game's own stats.

    We add +1000 minutes to ONE random game per player, rebuild the
    features, and check that each tampered game's own features are
    unchanged. If any window accidentally included the current game, this
    raises an error.

    Why one game per player: later games are SUPPOSED to change, since the
    tampered game is in their history. If we tampered with two games from
    the same player a few days apart, the second would correctly see the
    first, and the test would fail for the wrong reason.
    """
    sample_idx = logs.groupby("PLAYER_ID").sample(n=1, random_state=42).index
    tampered = logs.copy()
    tampered.loc[sample_idx, "MIN"] = tampered.loc[sample_idx, "MIN"] + 1000

    rebuilt = build_workload_features(tampered)
    keys = logs.loc[sample_idx, ["PLAYER_ID", "GAME_ID"]]
    before = keys.merge(workload, on=["PLAYER_ID", "GAME_ID"]).sort_values(["PLAYER_ID", "GAME_ID"])
    after = keys.merge(rebuilt, on=["PLAYER_ID", "GAME_ID"]).sort_values(["PLAYER_ID", "GAME_ID"])
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    logger.info("Leakage check passed: features of %d tampered games did not change", len(sample_idx))


def report(features: pd.DataFrame) -> None:
    """Print the missing values report and the feature-target correlation table."""
    print("\n" + "=" * 78)
    print("MISSING VALUES")
    print("=" * 78)
    missing = features[FEATURE_COLUMNS].isna().sum()
    missing_report = pd.DataFrame({"missing": missing, "pct": (missing / len(features) * 100).round(2)})
    print(missing_report[missing_report["missing"] > 0].to_string() if missing.any() else "None")

    print("\n" + "=" * 78)
    print(f"FEATURE vs {TARGET}")
    print("=" * 78)
    # Pearson correlation with a 0/1 target (point-biserial correlation).
    # NaNs are skipped pairwise. With a 2.7% positive rate, correlations are
    # small even for useful features. Compare them to each other, not to 1.
    numeric = features[FEATURE_COLUMNS].astype(float)
    target = features[TARGET]
    table = pd.DataFrame({
        "corr": numeric.corrwith(target),
        "mean_injured": numeric[target == 1].mean(),
        "mean_not_injured": numeric[target == 0].mean(),
    })
    table["abs_corr"] = table["corr"].abs()
    table = table.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
    print(table.round(4).to_string())
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    try:
        labeled, logs, bio, details = load_inputs()
        workload = build_workload_features(logs)
        assert_no_leakage(logs, workload)
        season_profile, positions = build_profile_features(bio, details)
        features = assemble(labeled, workload, season_profile, positions)
    except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as error:
        logger.error("Feature engineering failed: %s", error)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %s (%d rows x %d cols, %d features)",
                OUTPUT_PATH.relative_to(PROJECT_ROOT), *features.shape, len(FEATURE_COLUMNS))
    report(features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
