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
    Light theme modeled on nba.com's design system (cotton background, white
    cards, black nav strip, navy/scarlet accents, Barlow Condensed headings,
    Roboto body). Two layers:
      * .streamlit/config.toml: native theme (colors, font, table header).
        Needed because dataframes are drawn on a canvas that CSS can't style.
      * CSS below: nav strip, page headers, cards, badges, sidebar, alert box.
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

# nba.com design tokens (names follow nba.com's CSS variables).
NAVY = "#1d428a"          # --nba-navy, primary blue
SCARLET = "#c8102e"       # --nba-scarlet, primary red
DARK_BLUE = "#0b1b2c"     # --nba-dark-blue, headings / table headers
COTTON = "#f6f6f6"        # --cotton, page background
WHITE = "#ffffff"         # --white, cards
NAV_SURFACE = "#191c23"   # --surface, dark nav surface
MINOTAUR = "#222222"      # --minotaur, body text
QUICK_SILVER = "#a4a2a2"  # --quick-silver, muted text
CONCRETE = "#e7e7e7"      # --concrete, borders and dividers
MAIZE = "#fbc64b"         # --maize-crayola, gold accent
BLACK = "#000000"
HEADING_FONT = "'Barlow Condensed', 'Roboto', arial, sans-serif"
BODY_FONT = "'Roboto', arial, sans-serif"

# Tier labels and colors. Every tier also has a text label, so color is
# never the only cue.
TIER_LABELS = {"green": "Low", "orange": "Elevated", "red": "High"}
TIER_COLORS = {"High": SCARLET, "Elevated": MAIZE, "Low": NAVY}
TIER_TEXT_COLORS = {"High": WHITE, "Elevated": DARK_BLUE, "Low": WHITE}
TIER_ORDER = ["High", "Elevated", "Low"]
BACK_TO_BACK_COLOR = SCARLET
INJURY_MARK_COLOR = BLACK
LINE_COLOR = NAVY
GUIDE_COLOR = CONCRETE

# Columns joined from model_features.csv for context.
CONTEXT_COLUMNS = [
    "PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION", "IS_BACK_TO_BACK",
    "DAYS_SINCE_LAST_GAME", "MIN_LAST_7D", "MIN_LAST_28D", "AGE",
]

PAGES = ["Player Risk Monitor", "Team Risk Overview", "Feature Importance", "Model Performance"]
# Numbers shown before each page in the sidebar index.
SECTION_NUMBERS = {page: f"{i:02d}" for i, page in enumerate(PAGES, start=1)}

# Sidebar-only palette and fonts ("The Injury Report" printed-report style).
# Dark version: same navy as the table headers (DARK_BLUE), light text.
SB_PAPER = DARK_BLUE   # #0b1b2c, matches the stats-table header row
SB_INK = "#FFFFFF"     # main text (white)
SB_MUTED = "#A9B4C2"   # labels and notes (~8:1 contrast on the navy)
SB_RULE = "#24364A"    # hairline rules, a step lighter than the background
SB_ACCENT = "#6F8FC9"  # progress line: NBA navy lightened so it shows on dark
SB_TICK = SCARLET      # masthead bar and current-page tick: the one red accent
SB_DISPLAY_FONT = "'Barlow Condensed', 'Arial Narrow', sans-serif"
SB_MONO_FONT = "'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace"
SB_SERIF_FONT = "'Source Serif 4', Georgia, serif"


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
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Barlow+Condensed:wght@600;700;800&display=swap');
/* Sidebar-only fonts: IBM Plex Mono (index) and Source Serif 4 (notes) */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

/* ---- Base ---------------------------------------------------------------- */
.stApp {{ background-color: {COTTON}; }}
html, body, .stApp, .stMarkdown, [data-testid="stWidgetLabel"] {{
    font-family: {BODY_FONT};
    color: {MINOTAUR};
}}
.stMarkdown p, .stMarkdown li {{ font-size: 14px; font-weight: 400; line-height: 1.55; }}
[data-testid="stCaptionContainer"] {{ color: {QUICK_SILVER}; font-size: 13px; }}
hr {{ border: none !important; border-top: 1px solid {CONCRETE} !important; margin: 1.5rem 0 !important; }}

/* Section titles (st.subheader): Barlow Condensed 700, uppercase, 18px */
h1, h2, h3, h4 {{
    font-family: {HEADING_FONT} !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0;
    color: {DARK_BLUE} !important;
}}
h3 {{ font-size: 18px !important; padding-top: 0.5rem !important; }}

/* Sub-labels (widget labels, metric labels, custom .sub-label) */
[data-testid="stWidgetLabel"] p, .sub-label {{
    font-family: {BODY_FONT};
    font-weight: 500 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {QUICK_SILVER} !important;
}}

/* ---- Sidebar: "The Injury Report" style ----------------------------------
   Dark navy (same as the table headers), masthead, numbered index on a
   thin progress line, serif notes. Every rule is scoped to the sidebar so the
   main page keeps its nba.com style. */
[data-testid="stSidebar"] {{
    background-color: {SB_PAPER};
    border-right: 1px solid {SB_RULE};
}}
[data-testid="stSidebar"] hr {{ border-top: 1px solid {SB_RULE} !important; margin: 2rem 0 !important; }}
[data-testid="stSidebar"] .masthead {{
    border-top: 4px solid {SB_TICK};
    border-bottom: 1px solid {SB_RULE};
    padding: 0.8rem 0 0.9rem;
    margin-bottom: 1.2rem;
}}
[data-testid="stSidebar"] .masthead .title {{
    font-family: {SB_DISPLAY_FONT};
    font-weight: 800;
    font-size: 30px;
    line-height: 1;
    text-transform: uppercase;
    color: {SB_INK};
}}
[data-testid="stSidebar"] .masthead .sub {{
    font-family: {SB_MONO_FONT};
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {SB_MUTED};
    margin-top: 0.4rem;
}}
[data-testid="stSidebar"] .index-label {{
    font-family: {SB_MONO_FONT};
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SB_MUTED};
    margin-bottom: 0.4rem;
}}
/* Each index entry sits on a thin vertical line. Entries before the current
   one turn navy (progress); the current one gets a heavier navy tick. */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    border-left: 2px solid {SB_RULE};
    padding: 0.45rem 0 0.45rem 0.75rem;
    margin: 0;
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{
    font-family: {SB_MONO_FONT};
    font-size: 12.5px;
    letter-spacing: 0.02em;
    color: {SB_MUTED};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(~ label input:checked) {{ border-left-color: {SB_ACCENT}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ border-left: 4px solid {SB_TICK}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: {SB_INK}; font-weight: 600; }}
[data-testid="stSidebar"] p.side-note {{
    font-family: {SB_SERIF_FONT};
    font-size: 13px;
    line-height: 1.5;
    color: {SB_MUTED};
}}

/* ---- Page header (nba.com STATS-section style) --------------------------- */
.page-header {{ margin: 0 0 1.4rem; }}
.page-header .title {{
    font-family: {HEADING_FONT};
    font-weight: 700;
    font-size: 32px;
    line-height: 1.1;
    text-transform: uppercase;
    color: {DARK_BLUE};
}}
.page-header .title .accent {{
    border-bottom: 3px solid {SCARLET};
    padding-bottom: 2px;
}}
.page-header .subtitle {{
    font-family: {BODY_FONT};
    font-weight: 400;
    font-size: 14px;
    color: {QUICK_SILVER};
    margin-top: 0.6rem;
}}

/* ---- Cards: st.metric and the custom risk card --------------------------- */
[data-testid="stMetric"], .risk-card {{
    background: {WHITE};
    border-top: 3px solid {NAVY};
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04);
    padding: 0.9rem 1.1rem;
    min-height: 128px;
}}
[data-testid="stMetricLabel"] p, .risk-card .sub-label {{
    font-family: {BODY_FONT};
    font-weight: 500 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {QUICK_SILVER} !important;
}}
[data-testid="stMetricValue"], .risk-card .value {{
    font-family: {HEADING_FONT};
    font-weight: 700;
    font-size: 36px;
    line-height: 1.15;
    color: {DARK_BLUE};
}}
[data-testid="stMetricDelta"] {{ font-size: 12px; }}
/* Delta colors: up (higher risk) = scarlet, down = navy */
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) * {{ color: {SCARLET} !important; fill: {SCARLET} !important; }}
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * {{ color: {NAVY} !important; fill: {NAVY} !important; }}
.risk-card .tier-line {{
    font-family: {HEADING_FONT};
    font-weight: 700;
    font-size: 15px;
    text-transform: uppercase;
    color: {DARK_BLUE};
}}
.risk-card .dot {{
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}}
.risk-card .delta {{ font-family: {BODY_FONT}; font-size: 12px; font-weight: 500; }}

/* ---- Tier chips (nba.com "NEW"/"LIVE" style: sharp corners) -------------- */
.tier-chip {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 0;
    font-family: {HEADING_FONT};
    font-weight: 700;
    font-size: 16px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}}

/* ---- Tables (header colors come from config.toml) ------------------------ */
[data-testid="stDataFrame"] {{
    background: {WHITE};
    border: 1px solid {CONCRETE};
}}

/* ---- Charts: white card with a subtle shadow ----------------------------- */
[data-testid="stVegaLiteChart"] {{
    background: {WHITE};
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04);
}}
[data-testid="stImage"] img {{ box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); }}

/* ---- Inputs: white surface, navy focus ---------------------------------- */
[data-baseweb="select"] > div {{
    background-color: {WHITE};
    border-color: {CONCRETE};
    border-radius: 0;
}}
[data-baseweb="select"] > div:focus-within {{ border-color: {NAVY}; }}

/* ---- Buttons / pills: sharp, navy fill, white uppercase text ------------- */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
    background: {NAVY};
    color: {WHITE};
    border: none;
    border-radius: 0;
    font-family: {BODY_FONT};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ---- Expander ------------------------------------------------------------ */
[data-testid="stExpander"] details {{ background: {WHITE}; border: 1px solid {CONCRETE}; border-radius: 0; }}

/* ---- Info box (Page 4): white card, navy left border --------------------- */
[data-testid="stAlertContainer"] {{
    background-color: {WHITE} !important;
    border-left: 3px solid {NAVY} !important;
    border-radius: 0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    color: {MINOTAUR} !important;
}}
[data-testid="stAlertContainer"] p {{ font-family: {BODY_FONT}; font-size: 14px; color: {MINOTAUR}; }}
/* The opening bold sentence acts as the box title */
[data-testid="stAlertContainer"] p > strong:first-child {{
    display: block;
    font-family: {HEADING_FONT};
    font-weight: 700;
    font-size: 20px;
    text-transform: uppercase;
    color: {DARK_BLUE};
    margin-bottom: 0.3rem;
}}
</style>
"""

# Sidebar masthead (text only, no logo).
MASTHEAD = (
    '<div class="masthead"><div class="title">The Injury Report</div>'
    '<div class="sub">NBA injury risk model · 2022-23</div></div>'
)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """nba.com STATS-style page header: big uppercase title with a scarlet
    underline under the first word, muted subtitle below."""
    first, _, rest = title.partition(" ")
    st.markdown(
        f'<div class="page-header"><div class="title"><span class="accent">{first}</span> {rest}</div>'
        f'<div class="subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def tier_chip(tier: str) -> str:
    """Sharp-cornered chip in the tier color (nba.com "NEW"/"LIVE" style)."""
    return (f'<span class="tier-chip" style="background:{TIER_COLORS[tier]};'
            f'color:{TIER_TEXT_COLORS[tier]}">{tier}</span>')


def risk_card(risk: float, relative: float, tier: str, as_of: pd.Timestamp) -> str:
    """Big risk number with label, tier dot and vs-average line."""
    delta_color = SCARLET if relative >= 1 else NAVY
    return f"""
<div class="risk-card" title="As of {as_of:%b %d, %Y}, his most recent game in the period.">
  <div class="sub-label">Injury risk</div>
  <div class="value">{risk:.1%}</div>
  <div class="tier-line"><span class="dot" style="background:{TIER_COLORS[tier]}"></span>{tier} tier</div>
  <div class="delta" style="color:{delta_color}">{relative - 1:+.0%} vs average player</div>
</div>
"""


def style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the nba.com light palette to an Altair chart.

    Charts are drawn with theme=None in st.altair_chart, so Streamlit's
    own chart theme doesn't override these settings.
    """
    return (
        chart.configure(background=WHITE, font="Roboto", padding=14)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=CONCRETE, domainColor=CONCRETE, tickColor=CONCRETE,
            labelColor=QUICK_SILVER, titleColor=QUICK_SILVER, labelFont="Roboto", titleFont="Roboto",
            labelFontSize=12, titleFontSize=12, titleFontWeight=500,
        )
        .configure_legend(orient="top", labelColor=MINOTAUR, labelFont="Roboto", labelFontSize=12,
                          titleColor=QUICK_SILVER)
    )


# ---------------------------------------------------------------------------
# Page 1: Player Risk Monitor
# ---------------------------------------------------------------------------
def page_player(df: pd.DataFrame, metrics: dict) -> None:
    page_header("Player Risk Monitor",
                "Risk = probability of picking up an injury that keeps the player out of the team's next game.")

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
    with col1:
        st.markdown(risk_card(current["RISK"], current["RELATIVE_RISK"], current["TIER"], current["GAME_DATE"]),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="risk-card"><div class="sub-label">Risk tier</div>{tier_chip(current["TIER"])}</div>',
            unsafe_allow_html=True,
        )
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
    line = base.mark_line(color=LINE_COLOR, strokeWidth=2.5).encode(
        y=alt.Y("RISK:Q", title="Injury risk", axis=alt.Axis(format=".1%"), scale=alt.Scale(zero=False))
    )
    # Invisible wide points: a big hover target on every game.
    hover = base.mark_point(size=150, opacity=0).encode(y="RISK:Q", tooltip=tooltip)

    # Back-to-backs (scarlet dots) and injuries (black crosses) share one legend.
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
    # Tier lines: light dashed rules with muted labels.
    guides = pd.DataFrame({"RISK": [cutoffs["orange"], cutoffs["red"]],
                           "LABEL": ["Elevated (top 20%)", "High (top 5%)"]})
    guide_rules = alt.Chart(guides).mark_rule(strokeDash=[4, 4], color=GUIDE_COLOR, strokeWidth=1.5).encode(y="RISK:Q")
    guide_text = alt.Chart(guides).mark_text(align="left", dx=4, dy=-6, color=QUICK_SILVER, fontSize=11,
                                             font="Roboto").encode(y="RISK:Q", x=alt.value(0), text="LABEL:N")
    chart = (guide_rules + guide_text + line + b2b + injuries + hover).properties(height=360)
    st.altair_chart(style_chart(chart), width="stretch", theme=None)
    st.caption("Dashed lines: risk tier thresholds. Hover any game for details.")

    # -- Last 10 games ----------------------------------------------------------
    st.subheader("Last 10 games")
    last10 = games.tail(10).iloc[::-1]
    # Uppercase column labels, like nba.com stats tables.
    table = pd.DataFrame({
        "DATE": last10["GAME_DATE"].dt.date,
        "TEAM": last10["TEAM_ABBREVIATION"],
        "B2B": last10["BACK_TO_BACK"],
        "REST DAYS": last10["DAYS_SINCE_LAST_GAME"],
        "MIN L7D": last10["MIN_LAST_7D"],
        "RISK": last10["RISK"] * 100,
        "TIER": last10["TIER"],
        "OUTCOME": last10["OUTCOME"],
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "B2B": st.column_config.TextColumn(help="Back-to-back: played the day before"),
            "RISK": st.column_config.NumberColumn(format="%.1f%%"),
            "REST DAYS": st.column_config.NumberColumn(format="%d"),
            "MIN L7D": st.column_config.NumberColumn(format="%d", help="Minutes played in the previous 7 days"),
        },
    )


# ---------------------------------------------------------------------------
# Page 2: Team Risk Overview
# ---------------------------------------------------------------------------
def page_team(df: pd.DataFrame, metrics: dict) -> None:
    page_header("Team Risk Overview", "Every player on the roster, ranked by current injury risk.")

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
        .mark_bar(height={"band": 0.7})
        .encode(
            x=alt.X("RISK:Q", title="Current injury risk", axis=alt.Axis(format=".1%")),
            y=alt.Y("PLAYER_NAME:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor=NAVY, labelFontWeight=500, labelFontSize=12)),
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
    labels = bars.mark_text(align="left", dx=4, fontSize=12, font="Roboto").encode(
        text=alt.Text("RISK:Q", format=".1%"), color=alt.value(MINOTAUR)
    )
    height = max(220, 28 * len(roster))
    st.altair_chart(style_chart((bars + labels).properties(height=height)), width="stretch", theme=None)

    st.caption(
        "Current = each player's most recent game in the period. Dates differ: a player who got hurt "
        "or traded stops earlier. Tiers rank risk across all player-games in the period: "
        "High = riskiest 5%, Elevated = next 15%."
    )
    with st.expander("Details table"):
        details = pd.DataFrame({
            "PLAYER": roster["PLAYER_NAME"],
            "AS OF": roster["GAME_DATE"].dt.date,
            "RISK": roster["RISK"] * 100,
            "TIER": roster["TIER"],
            "AGE": roster["AGE"],
            "MIN L28D": roster["MIN_LAST_28D"],
        })
        # Player names in navy, like nba.com player links. A pandas Styler is
        # the only way to color text inside Streamlit's canvas table, and it
        # formats the numbers too.
        styled = (
            details.style
            .map(lambda _: f"color: {NAVY}; font-weight: 500", subset=["PLAYER"])
            .format({"RISK": "{:.1f}%", "AGE": "{:.0f}", "MIN L28D": "{:.0f}"})
        )
        st.dataframe(styled, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page 3: Feature Importance
# ---------------------------------------------------------------------------
def page_importance(metrics: dict) -> None:
    page_header("Feature Importance", "A briefing on what drives the risk score, and how much to trust the number.")

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
    m, ci = metrics["metrics"], metrics["confidence_intervals"]
    page_header(
        "Model Performance",
        f"Test period {metrics['test_period'][0]} to {metrics['test_period'][1]}: "
        f"{metrics['n_test']:,} player-games, {metrics['n_test_injuries']} injuries "
        f"({metrics['test_base_rate']:.1%} base rate). Models never saw this period during training.",
    )

    names = {"Baseline": "Baseline (minutes only)", "Logistic": "Logistic regression", "XGBoost": "XGBoost"}
    table = pd.DataFrame([
        {
            "MODEL": names[key],
            "ROC-AUC": m[key]["ROC_AUC"],
            "PR-AUC": m[key]["PR_AUC"],
            "PR-AUC 95% CI": ci[key]["PR_AUC_95CI"],
            "LIFT @ TOP 5%": m[key]["LIFT_TOP5"],
            "PRECISION @ TOP 5%": m[key]["PRECISION_TOP5"] * 100,
            "RECALL @ TOP 5%": m[key]["RECALL_TOP5"] * 100,
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
            "LIFT @ TOP 5%": st.column_config.NumberColumn(
                format="%.2f×", help="How many times more injuries than random among the 5% of games flagged riskiest."),
            "PRECISION @ TOP 5%": st.column_config.NumberColumn(
                format="%.1f%%", help="Of the flagged games, share followed by an injury."),
            "RECALL @ TOP 5%": st.column_config.NumberColumn(
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
            "TIER": [TIER_LABELS[t] for t in ("red", "orange", "green")],
            "GAMES": [tiers["n_games"][t] for t in ("red", "orange", "green")],
            "INJURY RATE AFTER THE GAME": [tiers["observed_injury_rate"][t] * 100 for t in ("red", "orange", "green")],
        }),
        hide_index=True,
        column_config={"INJURY RATE AFTER THE GAME": st.column_config.NumberColumn(format="%.1f%%")},
    )
    st.caption("High- and Elevated-tier games were followed by an injury about 1.6× as often as Low-tier games.")


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="NBA Injury Risk", page_icon=":material/monitor_heart:", layout="wide")
    inject_css()

    with st.sidebar:
        st.markdown(MASTHEAD, unsafe_allow_html=True)
        st.markdown('<div class="index-label">Index</div>', unsafe_allow_html=True)
        page = st.radio("Page", PAGES, label_visibility="collapsed",
                        format_func=lambda p: f"{SECTION_NUMBERS[p]}  {p}")
        st.divider()
        st.markdown(
            f'<p class="side-note">{SIDEBAR_NOTE}</p>'
            '<p class="side-note">Risk = XGBoost probability of an injury that keeps the player out of the '
            "team's next game.</p>"
            '<p class="side-note">Independent project. Not affiliated with or endorsed by the NBA.</p>',
            unsafe_allow_html=True,
        )

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
