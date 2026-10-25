"""
app.py: NBA Injury Risk dashboard (Step 5)
==========================================

Interactive Streamlit app over the test-period predictions.

Run from the repository root:
    streamlit run dashboard/app.py

Pages (sidebar navigation):
    1. Player Risk Monitor   risk over time for one player
    2. Team Risk Overview    a team's players ranked by current risk
    3. Feature Importance    what drives the score, can the % be trusted
    4. Model Performance     evaluation table and the scientific finding

Data (all produced by src/ scripts):
    data/processed/test_predictions.csv   predictions per player-game
    data/processed/model_features.csv     team, rest, workload per player-game
    data/processed/model_metrics.json     metrics + risk-tier cut-offs
    dashboard/feature_importance.png, dashboard/calibration.png

Which number is "risk"?
    PROB_XGB_CORRECTED: the XGBoost probability with class weighting undone
    (see src/modeling.py, correct_class_weighting). The raw PROB_XGB values
    (~0.3-0.6) are inflated by training weights and aren't real
    probabilities. Showing "45% injury risk" to a coach would be wrong.

Risk tiers (colors):
    Ranks within the displayed period: High = riskiest 5% of player-games
    (the same "top 5%" used in evaluation), Elevated = next 15%, Low = rest.
    Fixed probability cut-offs like 0.15 / 0.30 don't work here: corrected
    probabilities only range ~1-4%, so every player would get the same color.

Visual style:
    NBA-inspired dark theme. Two layers:
      * .streamlit/config.toml: native theme (colors, font, dataframe
        header). Needed because dataframes are drawn on a canvas that CSS
        can't style.
      * inject_css() below: banner, metric cards, sidebar border, alert
        box, heading typography.
    Unofficial portfolio project, not affiliated with the NBA.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"
IMPORTANCE_PNG = PROJECT_ROOT / "dashboard" / "feature_importance.png"
CALIBRATION_PNG = PROJECT_ROOT / "dashboard" / "calibration.png"

SIDEBAR_NOTE = "Predictions cover Feb–Apr 2023 test period. Model trained on 2022-23 regular season."

# NBA-inspired palette.
NBA_BLUE = "#1D428A"        # Pantone 282C
NBA_RED = "#C8102E"         # Pantone 193C
AMBER = "#F5A623"
GOLD = "#C9A84C"            # sparing highlight
BACKGROUND = "#0A0A0A"
SURFACE = "#1A1A1A"
SURFACE_BORDER = "#2A2A2A"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0A0A0"
TEXT_MUTED = "#666666"
FONT = "Barlow Condensed, Inter, sans-serif"

# Tier labels and colors. Every tier also has a text label, so color is
# never the only cue.
TIER_LABELS = {"green": "Low", "orange": "Elevated", "red": "High"}
TIER_COLORS = {"High": NBA_RED, "Elevated": AMBER, "Low": NBA_BLUE}
# Badge text color per tier: dark text on amber (white would be unreadable),
# white on red and blue.
TIER_TEXT_COLORS = {"High": TEXT_PRIMARY, "Elevated": BACKGROUND, "Low": TEXT_PRIMARY}
TIER_ORDER = ["High", "Elevated", "Low"]
BACK_TO_BACK_COLOR = NBA_RED
INJURY_MARK_COLOR = TEXT_PRIMARY
LINE_COLOR = NBA_BLUE
GUIDE_COLOR = TEXT_MUTED

# Columns joined from model_features.csv for context.
CONTEXT_COLUMNS = [
    "PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION", "IS_BACK_TO_BACK",
    "DAYS_SINCE_LAST_GAME", "MIN_LAST_7D", "MIN_LAST_28D", "AGE",
]

PAGES = ["Player Risk Monitor", "Team Risk Overview", "Feature Importance", "Model Performance"]


# ---------------------------------------------------------------------------
# Data loading (cached: runs once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading predictions...")
def load_data() -> tuple[pd.DataFrame, dict]:
    """Load predictions, attach context features and risk tiers."""
    missing = [p for p in (PREDICTIONS_PATH, FEATURES_PATH, METRICS_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing))

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["GAME_DATE"])
    features = pd.read_csv(FEATURES_PATH, usecols=CONTEXT_COLUMNS, parse_dates=["GAME_DATE"])
    metrics = json.loads(METRICS_PATH.read_text())

    # One prediction per player-game. validate= raises if the join would duplicate rows.
    df = predictions.merge(features, on=["PLAYER_ID", "GAME_DATE"], how="left", validate="one_to_one")

    df["RISK"] = df["PROB_XGB_CORRECTED"]
    cutoffs = metrics["risk_tiers"]["cutoffs"]
    df["TIER"] = "Low"
    df.loc[df["RISK"] >= cutoffs["orange"], "TIER"] = "Elevated"
    df.loc[df["RISK"] >= cutoffs["red"], "TIER"] = "High"
    # Relative risk: this player-game vs the average player-game in the period.
    df["RELATIVE_RISK"] = df["RISK"] / df["RISK"].mean()
    df["BACK_TO_BACK"] = df["IS_BACK_TO_BACK"].map({1: "Yes", 0: "No"})
    df["OUTCOME"] = df["INJURED_NEXT_GAME"].map({1: "Injured", 0: ""})
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True), metrics


def latest_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's most recent game in the period ("current" risk)."""
    return df.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)


# ---------------------------------------------------------------------------
# Styling (visual only, no data logic)
# ---------------------------------------------------------------------------
# Selectors use Streamlit's data-testid attributes, the stable hooks Streamlit
# exposes for its elements. !important is needed where Streamlit sets
# inline/emotion styles that would otherwise win.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

/* App background and base typography */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {BACKGROUND};
}}
html, body, .stApp, .stMarkdown, [data-testid="stWidgetLabel"] {{
    font-family: {FONT};
    font-weight: 400;
}}
.stMarkdown p, .stMarkdown li {{ font-size: 1.05rem; font-weight: 500; }}

/* Headings: bold, tracked, uppercase */
h1, h2, h3, h4 {{
    font-family: {FONT} !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: {TEXT_PRIMARY};
}}
[data-testid="stCaptionContainer"] {{ color: {TEXT_SECONDARY}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: #111111;
    border-right: 1px solid {NBA_BLUE};
}}

/* Header banner */
.nba-banner {{
    background: {SURFACE};
    border-left: 6px solid {NBA_BLUE};
    padding: 0.7rem 1.1rem;
    margin-bottom: 1.2rem;
    border-radius: 2px;
}}
.nba-banner span {{
    font-family: {FONT};
    font-weight: 700;
    font-size: 1.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_PRIMARY};
}}
.nba-banner small {{
    display: block;
    color: {TEXT_SECONDARY};
    font-family: {FONT};
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* Metric cards: dark card with NBA blue accent. The value stays white:
   blue text on #1A1A1A would only be 1.8:1 contrast. */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {SURFACE_BORDER};
    border-left: 4px solid {NBA_BLUE};
    border-radius: 4px;
    padding: 0.8rem 1rem;
}}
[data-testid="stMetric"]:hover {{ border-color: {SURFACE_BORDER}; border-left-color: {GOLD}; }}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    font-family: {FONT};
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}

/* Dataframes: frame and blue top rule (header color comes from config.toml) */
[data-testid="stDataFrame"] {{
    border: 1px solid {SURFACE_BORDER};
    border-top: 2px solid {NBA_BLUE};
    border-radius: 4px;
}}

/* Selectboxes and radio buttons: dark surface, blue focus border */
[data-baseweb="select"] > div {{
    background-color: {SURFACE};
    border-color: {SURFACE_BORDER};
}}
[data-baseweb="select"] > div:focus-within {{ border-color: {NBA_BLUE}; }}
[data-testid="stRadio"] label {{ font-family: {FONT}; font-size: 1.05rem; letter-spacing: 0.03em; }}

/* Info box (Page 4): dark card with blue left border */
[data-testid="stAlertContainer"] {{
    background-color: {SURFACE} !important;
    border-left: 4px solid {NBA_BLUE} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 4px;
}}

/* Tier badges */
.tier-badge {{
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 3px;
    font-family: {FONT};
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
</style>
"""

# Text wordmark (not the official logo): three letter tiles in the brand
# colors, plus the project name.
LOGO_SVG = f"""
<svg width="210" height="54" viewBox="0 0 210 54" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="NBA Injury Risk wordmark">
  <rect x="0"  y="4" width="40" height="46" rx="3" fill="{NBA_RED}"/>
  <rect x="44" y="4" width="40" height="46" rx="3" fill="{TEXT_PRIMARY}"/>
  <rect x="88" y="4" width="40" height="46" rx="3" fill="{NBA_BLUE}"/>
  <g font-family="Barlow Condensed, Inter, sans-serif" font-weight="700" font-size="36" text-anchor="middle">
    <text x="20"  y="40" fill="{TEXT_PRIMARY}">N</text>
    <text x="64"  y="40" fill="{BACKGROUND}">B</text>
    <text x="108" y="40" fill="{TEXT_PRIMARY}">A</text>
  </g>
  <g font-family="Barlow Condensed, Inter, sans-serif" font-weight="700" font-size="15" letter-spacing="1.5">
    <text x="136" y="24" fill="{TEXT_PRIMARY}">INJURY</text>
    <text x="136" y="44" fill="{TEXT_SECONDARY}">RISK</text>
  </g>
</svg>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_banner(page: str) -> None:
    """Header banner shown at the top of every page."""
    st.markdown(
        f'<div class="nba-banner"><span>NBA Injury Risk Model</span><small>{page}</small></div>',
        unsafe_allow_html=True,
    )


def tier_badge(tier: str) -> str:
    """HTML badge in the exact tier color (st.badge only has preset colors)."""
    return (f'<span class="tier-badge" style="background:{TIER_COLORS[tier]};'
            f'color:{TIER_TEXT_COLORS[tier]}">{tier}</span>')


def style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the NBA dark palette to an Altair chart.

    Charts are drawn with theme=None in st.altair_chart, so Streamlit's
    own chart theme doesn't override these settings.
    """
    return (
        chart.configure(background=SURFACE, font=FONT, padding=12)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=SURFACE_BORDER, domainColor=SURFACE_BORDER, tickColor=SURFACE_BORDER,
            labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY, labelFontSize=12, titleFontSize=12,
        )
        .configure_legend(orient="top", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY, labelFontSize=12)
    )


# ---------------------------------------------------------------------------
# Page 1: Player Risk Monitor
# ---------------------------------------------------------------------------
def page_player(df: pd.DataFrame, metrics: dict) -> None:
    st.title("Player Risk Monitor")
    st.caption("Risk = probability of picking up an injury that keeps the player out of the team's next game.")

    latest = latest_per_player(df).sort_values("PLAYER_NAME")
    # Plain "Name (TEAM)" labels mapped back to IDs. The team in the label
    # also tells apart two players with the same name.
    label_to_id = {f"{row.PLAYER_NAME} ({row.TEAM_ABBREVIATION})": row.PLAYER_ID for row in latest.itertuples()}
    options = list(label_to_id)
    # Default to the player with the highest current risk: the most useful
    # first view for staff.
    top = latest.loc[latest["RISK"].idxmax()]
    default_label = f"{top['PLAYER_NAME']} ({top['TEAM_ABBREVIATION']})"
    choice = st.selectbox("Player", options, index=options.index(default_label))
    player_id = label_to_id[choice]

    games = df[df["PLAYER_ID"] == player_id]
    current = games.iloc[-1]

    # -- Headline numbers -------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Current risk",
        f"{current['RISK']:.1%}",
        delta=f"{current['RELATIVE_RISK'] - 1:+.0%} vs average player",
        delta_color="inverse",  # higher risk = red arrow
        help=f"As of {current['GAME_DATE']:%b %d, %Y}, his most recent game in the period.",
    )
    with col2:
        st.markdown("**Risk tier**")
        st.markdown(tier_badge(current["TIER"]), unsafe_allow_html=True)
    col3.metric("Games in period", len(games))
    col4.metric("Back-to-backs played", int(games["IS_BACK_TO_BACK"].sum()))

    # -- Risk over time ---------------------------------------------------------
    st.subheader("Risk over time")
    cutoffs = metrics["risk_tiers"]["cutoffs"]
    base = alt.Chart(games).encode(x=alt.X("GAME_DATE:T", title=None))
    tooltip = [
        alt.Tooltip("GAME_DATE:T", title="Date", format="%b %d"),
        alt.Tooltip("RISK:Q", title="Risk", format=".1%"),
        alt.Tooltip("TIER:N", title="Tier"),
        alt.Tooltip("BACK_TO_BACK:N", title="Back-to-back"),
        alt.Tooltip("DAYS_SINCE_LAST_GAME:Q", title="Days since last game"),
        alt.Tooltip("MIN_LAST_7D:Q", title="Minutes, last 7 days"),
        alt.Tooltip("OUTCOME:N", title="Outcome"),
    ]
    line = base.mark_line(color=LINE_COLOR, strokeWidth=2).encode(
        y=alt.Y("RISK:Q", title="Injury risk", axis=alt.Axis(format=".1%"), scale=alt.Scale(zero=False))
    )
    # Invisible wide points: a big hover target on every game.
    hover = base.mark_point(size=150, opacity=0).encode(y="RISK:Q", tooltip=tooltip)

    # Back-to-backs (red dots) and injuries (black crosses) share one legend.
    marker_scale = alt.Scale(domain=["Back-to-back", "Injured after this game"],
                             range=[BACK_TO_BACK_COLOR, INJURY_MARK_COLOR])
    b2b = (
        alt.Chart(games[games["IS_BACK_TO_BACK"] == 1].assign(MARK="Back-to-back"))
        .mark_point(filled=True, size=80, shape="circle")
        .encode(x="GAME_DATE:T", y="RISK:Q", color=alt.Color("MARK:N", scale=marker_scale, title=None),
                tooltip=tooltip)
    )
    injuries = (
        alt.Chart(games[games["INJURED_NEXT_GAME"] == 1].assign(MARK="Injured after this game"))
        .mark_point(shape="cross", filled=True, size=140)
        .encode(x="GAME_DATE:T", y="RISK:Q", color=alt.Color("MARK:N", scale=marker_scale, title=None),
                tooltip=tooltip)
    )
    # Tier lines: gray, so they don't compete with the red back-to-back dots.
    guides = pd.DataFrame({"RISK": [cutoffs["orange"], cutoffs["red"]],
                           "LABEL": ["Elevated (top 20%)", "High (top 5%)"]})
    guide_rules = alt.Chart(guides).mark_rule(strokeDash=[4, 4], color=GUIDE_COLOR).encode(y="RISK:Q")
    guide_text = alt.Chart(guides).mark_text(align="left", dx=4, dy=-6, color=TEXT_SECONDARY, fontSize=11).encode(
        y="RISK:Q", x=alt.value(0), text="LABEL:N"
    )
    chart = (guide_rules + guide_text + line + b2b + injuries + hover).properties(height=360)
    st.altair_chart(style_chart(chart), width="stretch", theme=None)
    st.caption("Dashed lines: risk tier thresholds. Hover any game for details.")

    # -- Last 10 games ----------------------------------------------------------
    st.subheader("Last 10 games")
    last10 = games.tail(10).iloc[::-1]
    table = pd.DataFrame({
        "Date": last10["GAME_DATE"].dt.date,
        "Team": last10["TEAM_ABBREVIATION"],
        "Back-to-back": last10["BACK_TO_BACK"],
        "Rest days": last10["DAYS_SINCE_LAST_GAME"],
        "Min, last 7 days": last10["MIN_LAST_7D"],
        "Risk": last10["RISK"] * 100,
        "Tier": last10["TIER"],
        "Outcome": last10["OUTCOME"],
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Risk": st.column_config.NumberColumn(format="%.1f%%"),
            "Rest days": st.column_config.NumberColumn(format="%d"),
            "Min, last 7 days": st.column_config.NumberColumn(format="%d"),
        },
    )


# ---------------------------------------------------------------------------
# Page 2: Team Risk Overview
# ---------------------------------------------------------------------------
def page_team(df: pd.DataFrame, metrics: dict) -> None:
    st.title("Team Risk Overview")

    latest = latest_per_player(df)
    teams = sorted(latest["TEAM_ABBREVIATION"].dropna().unique())
    team = st.selectbox("Team", teams)

    # Players whose most recent game was for this team (handles trades).
    roster = latest[latest["TEAM_ABBREVIATION"] == team].sort_values("RISK", ascending=False)
    if roster.empty:
        st.warning("No players found for this team in the test period.")
        return

    counts = roster["TIER"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("High risk", int(counts.get("High", 0)))
    col2.metric("Elevated risk", int(counts.get("Elevated", 0)))
    col3.metric("Players", len(roster))

    bars = (
        alt.Chart(roster)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            x=alt.X("RISK:Q", title="Current injury risk", axis=alt.Axis(format=".1%")),
            y=alt.Y("PLAYER_NAME:N", sort="-x", title=None),
            color=alt.Color("TIER:N", title="Risk tier",
                            scale=alt.Scale(domain=TIER_ORDER, range=[TIER_COLORS[t] for t in TIER_ORDER])),
            tooltip=[
                alt.Tooltip("PLAYER_NAME:N", title="Player"),
                alt.Tooltip("RISK:Q", title="Risk", format=".1%"),
                alt.Tooltip("TIER:N", title="Tier"),
                alt.Tooltip("GAME_DATE:T", title="As of", format="%b %d"),
                alt.Tooltip("AGE:Q", title="Age"),
                alt.Tooltip("MIN_LAST_28D:Q", title="Minutes, last 28 days"),
            ],
        )
    )
    labels = bars.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text("RISK:Q", format=".1%"), color=alt.value(TEXT_SECONDARY)
    )
    height = max(220, 28 * len(roster))
    st.altair_chart(style_chart((bars + labels).properties(height=height)), width="stretch", theme=None)

    st.caption(
        "Current = each player's most recent game in the period. Dates differ: a player who got hurt "
        "or traded stops earlier. Tiers rank risk across all player-games in the period: "
        "High = riskiest 5%, Elevated = next 15%."
    )
    with st.expander("Details table"):
        st.dataframe(
            pd.DataFrame({
                "Player": roster["PLAYER_NAME"],
                "As of": roster["GAME_DATE"].dt.date,
                "Risk": roster["RISK"] * 100,
                "Tier": roster["TIER"],
                "Age": roster["AGE"],
                "Min, last 28 days": roster["MIN_LAST_28D"],
            }),
            hide_index=True,
            width="stretch",
            column_config={"Risk": st.column_config.NumberColumn(format="%.1f%%")},
        )


# ---------------------------------------------------------------------------
# Page 3: Feature Importance
# ---------------------------------------------------------------------------
def page_importance(metrics: dict) -> None:
    st.title("Feature Importance")
    st.caption("A briefing on what drives the risk score, and how much to trust the number.")

    st.subheader("What drives a player's risk score")
    if IMPORTANCE_PNG.exists():
        st.image(str(IMPORTANCE_PNG), width="stretch")
    else:
        st.warning("feature_importance.png not found. Run src/modeling.py.")
    st.markdown(
        """
**How to read it:** the longer the bar on the left, the more that factor moves a player's risk score up or down.
(The right-hand chart is the model's built-in measure. It's shown for comparison, but the left one is more reliable.)

**What raises the score the most**
- **Age.** Older players carry more risk, all else equal.
- **Minutes over the last 2–4 weeks.** Heavy recent workload is the clearest warning sign we can see.
- **How deep into the season he is.** Risk builds as the games pile up.

**What barely matters in this data**
- **Position.** Guards, forwards and centers look about the same once workload and age are known.
- **Back-to-backs.** Not because they're harmless: teams already rest their at-risk players on the second
  night, so the players who do play back-to-backs tend to be the healthier ones.
- **Sudden spikes in workload (ACWR).** A popular sports-science measure, but it showed no clear signal here.
"""
    )

    st.divider()
    st.subheader("Can you trust the percentage?")
    if CALIBRATION_PNG.exists():
        st.image(str(CALIBRATION_PNG), width="stretch")
    else:
        st.warning("calibration.png not found. Run src/modeling.py.")
    actual = metrics["test_base_rate"]
    predicted = metrics["metrics"]["XGBoost"]["MEAN_PRED_CORRECTED"]
    st.markdown(
        f"""
**How to read it:** each dot is a group of games. If the model is honest, the dots sit on the dashed line:
games it calls "3% risk" end in injury about 3% of the time.

- **Left chart:** the model's raw scores. They looked like 30–60% risk, far too high. That's a side effect of
  how the model was trained to pay extra attention to rare injuries.
- **Right chart:** after correcting for that, the numbers are realistic. The model predicts about
  **{predicted:.1%}** per game against an actual **{actual:.1%}**, so it's slightly low for this late-season stretch.

**Bottom line:** use the score to **rank** players (who needs a closer look today), not as exact odds.
"""
    )


# ---------------------------------------------------------------------------
# Page 4: Model Performance
# ---------------------------------------------------------------------------
def page_performance(metrics: dict) -> None:
    st.title("Model Performance")
    m, ci = metrics["metrics"], metrics["confidence_intervals"]
    st.caption(
        f"Test period {metrics['test_period'][0]} to {metrics['test_period'][1]}: "
        f"{metrics['n_test']:,} player-games, {metrics['n_test_injuries']} injuries "
        f"({metrics['test_base_rate']:.1%} base rate). Models never saw this period during training."
    )

    names = {"Baseline": "Baseline (minutes only)", "Logistic": "Logistic regression", "XGBoost": "XGBoost"}
    table = pd.DataFrame([
        {
            "Model": names[key],
            "ROC-AUC": m[key]["ROC_AUC"],
            "PR-AUC": m[key]["PR_AUC"],
            "PR-AUC 95% CI": ci[key]["PR_AUC_95CI"],
            "Lift @ top 5%": m[key]["LIFT_TOP5"],
            "Precision @ top 5%": m[key]["PRECISION_TOP5"] * 100,
            "Recall @ top 5%": m[key]["RECALL_TOP5"] * 100,
        }
        for key in names
    ])
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f", help="0.5 = coin flip, 1.0 = perfect"),
            "PR-AUC": st.column_config.NumberColumn(
                format="%.3f", help=f"Main metric. Random guessing scores the base rate ({metrics['test_base_rate']:.3f})."),
            "Lift @ top 5%": st.column_config.NumberColumn(
                format="%.2f×", help="How many times more injuries than random among the 5% of games flagged riskiest."),
            "Precision @ top 5%": st.column_config.NumberColumn(
                format="%.1f%%", help="Of the flagged games, share followed by an injury."),
            "Recall @ top 5%": st.column_config.NumberColumn(
                format="%.1f%%", help="Of all injuries, share that were flagged."),
        },
    )

    st.subheader("Why the lift is modest")
    st.markdown(
        """
Every model finds injuries only about **1.4 times better than chance**. Public data sees one thing about workload:
**minutes in games**. It can't see what actually drives most injuries:
- **Injury history.** A previous hamstring or ankle injury is one of the strongest known risk factors, and the
  model has no memory of it yet.
- **GPS and optical tracking.** Distance covered, sprints, jumps and decelerations, which measure load much
  better than minutes.
- **Practice and training load,** plus recovery data (sleep, soreness, strength testing), which never shows up
  in a box score.
- **Contact and bad luck.** Many injuries come from collisions and landings that no workload model can foresee.
"""
    )

    st.subheader("Scientific finding")
    gain = ci["Logistic"]["PR_AUC_GAIN_VS_BASELINE_95CI"]
    st.info(
        "**Public box-score workload data adds little beyond minutes played.** A model that only knows each "
        "player's last 28 days of minutes ranks injury risk as well as models using all 16 features "
        f"(PR-AUC gain for logistic regression: {gain}, an interval that includes zero). Heavy-minute players "
        "do get hurt more often, but schedule fatigue, back-to-backs and workload spikes don't add measurable "
        "predictive power on top of that. Better predictions need private data: injury history, tracking and "
        "practice load.",
        icon=":material/science:",
    )

    st.subheader("Do the risk tiers mean anything?")
    tiers = metrics["risk_tiers"]
    st.dataframe(
        pd.DataFrame({
            "Tier": [TIER_LABELS[t] for t in ("red", "orange", "green")],
            "Games": [tiers["n_games"][t] for t in ("red", "orange", "green")],
            "Injury rate after the game": [tiers["observed_injury_rate"][t] * 100 for t in ("red", "orange", "green")],
        }),
        hide_index=True,
        column_config={"Injury rate after the game": st.column_config.NumberColumn(format="%.1f%%")},
    )
    st.caption("High- and Elevated-tier games were followed by an injury about 1.6× as often as Low-tier games.")


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="NBA Injury Risk", page_icon=":material/monitor_heart:", layout="wide")
    inject_css()

    st.sidebar.markdown(LOGO_SVG, unsafe_allow_html=True)
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(SIDEBAR_NOTE)
    st.sidebar.caption("Risk = XGBoost probability of an injury that keeps the player out of the team's next game.")
    st.sidebar.caption("Unofficial portfolio project. Not affiliated with or endorsed by the NBA.")

    render_banner(page)

    try:
        df, metrics = load_data()
    except FileNotFoundError as error:
        st.error(f"Missing data files: {error}. Run the pipeline in src/ first (see README).")
        st.stop()

    if page == PAGES[0]:
        page_player(df, metrics)
    elif page == PAGES[1]:
        page_team(df, metrics)
    elif page == PAGES[2]:
        page_importance(metrics)
    else:
        page_performance(metrics)


main()
