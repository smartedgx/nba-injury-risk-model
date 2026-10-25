"""
modeling.py
===========

Step 4 of the NBA Injury Risk pipeline: train and evaluate three models on
data/processed/model_features.csv.

    1. Baseline:  logistic regression on MIN_LAST_28D only
                  ("starters get hurt more")
    2. Logistic:  logistic regression on all features
    3. XGBoost:   gradient-boosted trees on all features

Outputs:
    models/baseline_model.pkl, models/logistic_model.pkl, models/xgb_model.pkl
    data/processed/test_predictions.csv
    dashboard/feature_importance.png   (XGBoost gain vs SHAP)
    dashboard/calibration.png          (reliability curves, before/after correction)
    data/processed/model_metrics.json  (metrics, CIs, risk-tier cutoffs for the dashboard)

Design decisions
----------------
Time-based split: rows are sorted by date. The first 70% go to training and
the last 30% to test. A random split would put December games in both sets:
the model would learn from games after the ones it's tested on. Rolling
features also overlap (a Dec 10 row and a Dec 12 row share most of their
28-day window), so a random test row would have a near-copy in training,
and test scores would look much better than real use. A chronological
split reproduces deployment: learn from the past, predict the future.

Purge gap: a training row's label depends on what happens up to ~7 days
after its game (team's next game + injury-report lag). Training rows within
7 days of the cutoff would have labels based on test-period events, so
they're removed ("purging", López de Prado 2018).

No tuning on the test set: hyperparameters are fixed up front, and XGBoost's
early stopping uses a validation slice taken from the END of the training
period. The test set is used exactly once, for the final numbers.

Usage (from the repository root):
    python src/modeling.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # file output only, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Single source of truth for the feature list: reuse Step 3's definition so
# the two scripts can't drift apart.
from feature_engineering import FEATURE_COLUMNS, TARGET

try:
    import xgboost as xgb
except Exception as error:  # XGBoostError when the OpenMP runtime is missing
    sys.exit(
        "Could not load XGBoost: " + str(error).strip().splitlines()[0] + "\n"
        "On macOS, XGBoost needs the OpenMP runtime (libomp). Install it with:\n"
        "    brew install libomp\n"
        "Without Homebrew, reuse the copy bundled with scikit-learn:\n"
        "    DYLD_LIBRARY_PATH=$(python3 -c 'import sklearn, os; "
        "print(os.path.join(os.path.dirname(sklearn.__file__), \".dylibs\"))') python3 src/modeling.py"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70        # first 70% of rows (by date) -> train
PURGE_DAYS = 7               # matches Step 2's reporting-lag tolerance
VALIDATION_FRACTION = 0.15   # last 15% of the training period -> XGBoost early stopping
TOP_FRACTION = 0.05          # "top 5% riskiest player-games"
N_BOOTSTRAP = 500            # resamples for confidence intervals

# Risk tiers for the dashboard, as percentiles of the corrected XGBoost
# probability over the period the dashboard displays. Top 5% = red,
# matching the "top 5%" evaluation metric; the next 15% = orange; the
# rest = green.
# Why percentiles and not fixed probabilities like 0.15 / 0.30: corrected
# probabilities only range ~1-4%, so any fixed cut-off in that style would
# put every player in the same tier.
# Why the displayed period and not training data: in-sample training scores
# run higher than out-of-sample ones (and scores drift late in the season),
# so training-based cut-offs flagged under 1% of test games. Tiers are a
# display ranking only and are NOT used in any evaluation metric. In live
# use, they'd be recomputed over a rolling window of recent predictions.
RISK_TIER_PERCENTILES = {"orange": 0.80, "red": 0.95}

BASELINE_FEATURES = ["MIN_LAST_28D"]

# Logistic regression needs one position dummy dropped (see build_logistic).
# Guard is the largest group, so it's the most stable reference category.
LOGISTIC_FEATURES = [c for c in FEATURE_COLUMNS if c != "POS_GUARD"]

# XGBoost sees every feature, missing values included.
XGB_FEATURES = list(FEATURE_COLUMNS)

# XGBoost hyperparameters. Small data (~18k rows, ~500 positives) and weak
# signals: the priority is not overfitting.
XGB_PARAMS = dict(
    # Upper limit. Early stopping picks the real number of trees.
    n_estimators=2000,
    # Small step size: each tree corrects only a little. Slower, but smoother
    # and less prone to chasing noise than the default 0.3.
    learning_rate=0.03,
    # Shallow trees: depth 3 allows up to 3-way interactions (e.g. age x
    # minutes x back-to-back). Deeper trees would memorize individual players.
    max_depth=3,
    # A split needs enough "weight" on each side. With scale_pos_weight ~35,
    # each injury weighs ~35, so this mostly stops leaves built around one
    # or two injuries.
    min_child_weight=10,
    # Each tree sees a random 80% of rows and 80% of features. The
    # randomness reduces variance, like in a random forest.
    subsample=0.8,
    colsample_bytree=0.8,
    # L2 penalty on leaf values (default 1). Keeps individual leaves from
    # making extreme predictions.
    reg_lambda=1.0,
    # Early stopping is judged on PR-AUC, our main metric, not log-loss.
    eval_metric="aucpr",
    early_stopping_rounds=100,
    # Histogram-based split finding: the fast, standard algorithm.
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Chart styling (reference palette from the dataviz guidelines).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"Baseline": "#2a78d6", "Logistic": "#eb6834", "XGBoost": "#1baf7a"}
SINGLE_SERIES = "#2a78d6"

logger = logging.getLogger("modeling")


# ---------------------------------------------------------------------------
# Data and split
# ---------------------------------------------------------------------------
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.relative_to(PROJECT_ROOT)}. Run src/feature_engineering.py first.")
    df = pd.read_csv(path, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    return df.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, train_fraction: float, purge_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split with a purge gap.

    The cutoff is a DATE (all games on one day fall on the same side), set so
    that ~train_fraction of rows come before it. Training rows within
    `purge_days` of the cutoff are dropped, because their labels depend on
    events in the test period.
    """
    cutoff = df["GAME_DATE"].iloc[int(len(df) * train_fraction)]
    test = df[df["GAME_DATE"] >= cutoff]
    train = df[df["GAME_DATE"] < cutoff - pd.Timedelta(days=purge_days)]
    purged = len(df) - len(train) - len(test)
    logger.info(
        "Split at %s | train %d rows (%s to %s, %d injuries) | purged %d | test %d rows (%s to %s, %d injuries)",
        cutoff.date(), len(train), train["GAME_DATE"].min().date(), train["GAME_DATE"].max().date(),
        int(train[TARGET].sum()), purged, len(test), test["GAME_DATE"].min().date(),
        test["GAME_DATE"].max().date(), int(test[TARGET].sum()),
    )
    return train, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_baseline() -> Pipeline:
    """Logistic regression on 28-day minutes only.

    This answers: "how far do you get just by knowing who plays a lot?"
    The full models have to beat this to show they learned more than
    "starters get hurt more" (exposure).
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])


def build_logistic() -> Pipeline:
    """Logistic regression on all features: the interpretable baseline.

    Why it comes first: it's simple, fast, hard to overfit, and every
    coefficient reads as "this feature raises/lowers risk". If XGBoost can't
    beat it, the extra complexity isn't justified.

    Preprocessing, all inside one sklearn Pipeline, so the exact same steps
    (with values learned on training data only) are applied at prediction
    time, and the saved .pkl is self-contained:

      * Missing values: logistic regression can't handle NaN. Each missing
        value gets a neutral fill plus a 0/1 "was missing" indicator, so the
        model can still learn that missingness itself means something:
          ACWR                 -> 1.0  (ratio of 1 = "normal load")
          MPG_TREND            -> 0    (no change in role)
          DAYS_SINCE_LAST_GAME -> 90   (very long rest, roughly an offseason)
          ACWR_SPIKE           -> 0    (no indicator: ACWR's covers it)
        The fill values are constants, not learned from the data, so there's
        no train/test contamination.

      * Standardization: rescale every column to mean 0, std 1 (learned on
        training data). Logistic regression applies one regularization
        penalty to all coefficients. Without scaling, a feature in big units
        (MIN_LAST_28D ~ 0-1000) needs a tiny coefficient and one in small
        units (IS_BACK_TO_BACK 0/1) a large one, so the penalty hits them
        unequally. The solver also converges faster. Bonus: coefficients
        become "effect per standard deviation", so they're comparable.
        XGBoost doesn't need this: trees split on thresholds ("MIN_LAST_28D
        > 250?"), and rescaling a feature doesn't change which threshold
        separates the data, only its number.

      * One position dummy dropped (POS_GUARD): every player is exactly one
        of Guard/Forward/Center, so the three dummies always sum to 1, which
        equals the intercept column. That's perfect multicollinearity:
        infinitely many coefficient combinations fit equally well, so the
        coefficients can't be interpreted. Dropping one makes Guard the
        reference: POS_CENTER's coefficient = "center vs guard".

      * class_weight="balanced": each injury counts ~35x in the loss, so the
        model can't win by always predicting "no injury".
    """
    imputer = ColumnTransformer(
        [
            ("acwr", SimpleImputer(strategy="constant", fill_value=1.0, add_indicator=True), ["ACWR"]),
            ("acwr_spike", SimpleImputer(strategy="constant", fill_value=0), ["ACWR_SPIKE"]),
            ("mpg", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True), ["MPG_TREND"]),
            ("rest", SimpleImputer(strategy="constant", fill_value=90, add_indicator=True), ["DAYS_SINCE_LAST_GAME"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("impute", imputer),
        ("scale", StandardScaler()),
        # C=1.0 = default L2 regularization strength. Mild shrinkage that
        # stabilizes correlated features (the three minutes windows).
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)),
    ])


def train_xgboost(train: pd.DataFrame) -> tuple[xgb.XGBClassifier, dict]:
    """Train XGBoost with early stopping, then refit on the full training set.

    Missing values: XGBoost handles NaN natively ("sparsity-aware split
    finding"). At each split it tries sending all missing rows left, then
    right, and keeps whichever direction improves the loss more. It learns
    a default direction per split, so "ACWR missing" (early season) can
    behave differently from any real ACWR value. No fill value needed.

    scale_pos_weight = negatives / positives (computed on training data, not
    hard-coded). Every injury counts that many times in the loss. Same idea
    as class_weight="balanced".

    Two-stage training:
      1. Fit on the earlier 85% of the training period and watch PR-AUC on
         the last 15% (time-ordered). Stop when it hasn't improved for 100
         trees. This picks the number of trees without touching the test set.
      2. Refit on the full training period with that number of trees, so the
         most recent training weeks aren't wasted.
    """
    pos = int(train[TARGET].sum())
    neg = len(train) - pos
    params = {**XGB_PARAMS, "scale_pos_weight": neg / pos}
    logger.info("XGBoost scale_pos_weight = %d / %d = %.1f", neg, pos, neg / pos)

    val_cutoff = train["GAME_DATE"].iloc[int(len(train) * (1 - VALIDATION_FRACTION))]
    fit_part = train[train["GAME_DATE"] < val_cutoff]
    val_part = train[train["GAME_DATE"] >= val_cutoff]

    probe = xgb.XGBClassifier(**params)
    probe.fit(fit_part[XGB_FEATURES], fit_part[TARGET],
              eval_set=[(val_part[XGB_FEATURES], val_part[TARGET])], verbose=False)
    n_trees = probe.best_iteration + 1
    logger.info("Early stopping: best validation PR-AUC %.4f at %d trees (validation from %s, %d injuries)",
                probe.best_score, n_trees, val_cutoff.date(), int(val_part[TARGET].sum()))

    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_trees
    model = xgb.XGBClassifier(**final_params)
    model.fit(train[XGB_FEATURES], train[TARGET], verbose=False)
    return model, {"n_trees": n_trees, "scale_pos_weight": neg / pos}


def correct_class_weighting(p: np.ndarray, train_base_rate: float) -> np.ndarray:
    """Turn class-weighted scores back into real probabilities ("prior
    correction", King & Zeng 2001; Elkan 2001).

    Both models counted each injury w = negatives/positives times (~39x)
    during training. That multiplies the model's odds of injury by ~w, which
    is why raw scores average ~0.45 while the real rate is ~0.03. Dividing
    the odds by w undoes it:
        corrected odds = raw odds / w,   p = odds / (1 + odds)
    It's a monotonic transform, so the ranking (ROC-AUC, PR-AUC, top-5%
    metrics) is unchanged. Only the probability values become meaningful.
    It needs no extra data, since w is known from training. It's exact for
    logistic regression's intercept and a close approximation for XGBoost.
    """
    w = (1 - train_base_rate) / train_base_rate
    odds = p / (1 - p) / w
    return odds / (1 + odds)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def top_k_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the top `fraction` of rows by score (exactly k rows)."""
    k = math.ceil(len(scores) * fraction)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between predicted probability and observed injury rate,
    over 10 equal-size bins of predictions, weighted by bin size. 0 = perfect."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """All metrics for one model.

    * ROC-AUC: probability that a random injury is ranked above a random
      non-injury. 0.5 = coin flip. With 97% negatives it can look good while
      the top of the list is still mostly false alarms, because it gives
      equal credit for correctly ranking thousands of easy negatives.
    * PR-AUC (average precision): averages precision over every recall
      level. It only looks at how injuries rank against the rows the model
      flags, so it's sensitive to exactly what matters with rare events:
      how clean the top of the risk list is. Its "random" level is the base
      rate (~0.027), not 0.5. That's why it's our main metric.
    * Precision@5%: of the 5% of player-games flagged as riskiest, the share
      that were actual injuries. "When the model raises a flag, how often
      is it right?"
    * Recall@5%: of ALL injuries, the share that fell in that top 5%. "How
      many injuries does the flag list catch?"
    * Lift@5%: Precision@5% / base rate. "How many times better than
      picking players at random?"
    * Brier score: mean squared error of the probabilities (lower = better).
    * ECE + mean predicted vs actual rate: calibration (see report).
    """
    flagged = top_k_mask(p, TOP_FRACTION)
    precision_top = y[flagged].mean()
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "PRECISION_TOP5": precision_top,
        "RECALL_TOP5": y[flagged].sum() / y.sum(),
        "LIFT_TOP5": precision_top / y.mean(),
        "BRIER": brier_score_loss(y, p),
        "ECE": expected_calibration_error(y, p),
        "MEAN_PRED": p.mean(),
    }


def bootstrap_ci(test: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """95% confidence intervals for PR-AUC and ROC-AUC, and for PR-AUC
    differences vs the baseline.

    Why: the test set has only ~200 injuries, so a 0.01 gap between models
    could be luck. We resample PLAYERS with replacement (not rows), because
    rows from the same player are correlated. Resampling rows would treat
    them as independent and give falsely narrow intervals.
    All models are scored on the same resamples, so the differences are paired.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y = test[TARGET].to_numpy()
    groups = test.groupby("PLAYER_ID").indices  # player -> row positions
    players = np.array(list(groups))
    samples = {f"{m}|{metric}": [] for m in probs for metric in ("PR_AUC", "ROC_AUC")}
    diffs = {m: [] for m in probs if m != "Baseline"}

    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(players, size=len(players), replace=True)
        idx = np.concatenate([groups[p] for p in chosen])
        if y[idx].sum() == 0:
            continue
        pr = {m: average_precision_score(y[idx], p[idx]) for m, p in probs.items()}
        for m, p in probs.items():
            samples[f"{m}|PR_AUC"].append(pr[m])
            samples[f"{m}|ROC_AUC"].append(roc_auc_score(y[idx], p[idx]))
        for m in diffs:
            diffs[m].append(pr[m] - pr["Baseline"])

    rows = {}
    for key, values in samples.items():
        model, metric = key.split("|")
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.setdefault(model, {})[f"{metric}_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
    for model, values in diffs.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows[model]["PR_AUC_GAIN_VS_BASELINE_95CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
    rows["Baseline"]["PR_AUC_GAIN_VS_BASELINE_95CI"] = "-"
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light grid, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Exact SHAP values from XGBoost's built-in TreeSHAP.

    pred_contribs=True runs the same TreeSHAP algorithm the `shap` library
    uses for tree models, so we get identical values without that extra
    dependency (which doesn't support Python 3.14 well yet).
    Output: one column per feature plus a final "bias" column (the base
    prediction). Values are in log-odds units.
    """
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)


def plot_feature_importance(model: xgb.XGBClassifier, shap_values: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Side-by-side bars: XGBoost 'gain' importance vs mean |SHAP|.

    Default importance (gain) = total loss reduction from splits on a
    feature. It's biased toward continuous, high-cardinality features
    (more possible split points), says nothing about direction, and is
    computed on training data.

    SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) comes from
    game theory: each prediction is split into fair contributions from each
    feature, and the contributions add up exactly to the prediction. Mean
    |SHAP| over the test set = "how much does this feature move predictions
    on average". It's consistent (if a model relies more on a feature, its
    SHAP importance can't go down), works per player-game (it can explain
    one specific alert), and the sign shows direction.
    """
    mean_abs_shap = shap_values.abs().mean()
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(shap_values.columns).fillna(0)
    table = pd.DataFrame({"MEAN_ABS_SHAP": mean_abs_shap, "GAIN": gain}).sort_values("MEAN_ABS_SHAP")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True, facecolor=SURFACE)
    panels = [
        ("MEAN_ABS_SHAP", "Mean |SHAP value| on test set (log-odds)", "How much each feature moves predictions"),
        ("GAIN", "Average gain per split (training)", "XGBoost built-in importance"),
    ]
    for ax, (col, xlabel, title) in zip(axes, panels):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.barh(table.index, table[col], height=0.6, color=SINGLE_SERIES)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    axes[0].tick_params(axis="y", labelcolor=TEXT_PRIMARY)
    fig.suptitle("XGBoost feature importance: SHAP vs gain", color=TEXT_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
    return table.sort_values("MEAN_ABS_SHAP", ascending=False)


def plot_calibration(
    y: np.ndarray, raw: dict[str, np.ndarray], corrected: dict[str, np.ndarray], path: Path
) -> None:
    """Reliability curves: predicted probability (x) vs observed injury rate (y),
    in 10 equal-size bins per model. A perfectly calibrated model sits on the
    dashed diagonal.

    Left: raw class-weighted scores, full 0-1 scale, showing how far off they
    are. Right: after correct_class_weighting(), on a zoomed scale where the
    diagonal is visible. The lines sit too close together for direct labels,
    so a legend identifies them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=SURFACE)
    corrected_max = max(max(p.max() for p in corrected.values()), y.mean() * 2.5)
    panels = [
        (axes[0], raw, 1.0, "Before: raw class-weighted scores"),
        (axes[1], corrected, corrected_max, "After: class weighting undone (zoomed)"),
    ]
    for ax, probs, limit, title in panels:
        style_axes(ax)
        ax.plot([0, limit], [0, limit], linestyle="--", color=TEXT_SECONDARY, linewidth=1, label="Perfect calibration")
        for name, p in probs.items():
            observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(predicted, observed, color=SERIES[name], linewidth=2, marker="o", markersize=5, label=name)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Mean predicted probability (bin)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("Observed injury rate (bin)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    axes[0].legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9, loc="upper left")
    fig.suptitle(f"Calibration on test set (actual injury rate {y.mean():.1%})",
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    try:
        df = load_features(FEATURES_PATH)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    train, test = time_split(df, TRAIN_FRACTION, PURGE_DAYS)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    # -- Train ------------------------------------------------------------------
    baseline = build_baseline().fit(train[BASELINE_FEATURES], y_train)
    logistic = build_logistic().fit(train[LOGISTIC_FEATURES], y_train)
    xgb_model, xgb_info = train_xgboost(train)

    probs = {
        "Baseline": baseline.predict_proba(test[BASELINE_FEATURES])[:, 1],
        "Logistic": logistic.predict_proba(test[LOGISTIC_FEATURES])[:, 1],
        "XGBoost": xgb_model.predict_proba(test[XGB_FEATURES])[:, 1],
    }

    # Real probabilities (class weighting undone). Ranking is unchanged.
    train_base_rate = float(y_train.mean())
    corrected = {name: correct_class_weighting(p, train_base_rate) for name, p in probs.items()}

    # -- Evaluate ---------------------------------------------------------------
    results = pd.DataFrame({name: evaluate(y_test, p) for name, p in probs.items()}).T
    # Calibration metrics after the correction (ranking metrics don't change).
    for name, p in corrected.items():
        results.loc[name, "BRIER_CORRECTED"] = brier_score_loss(y_test, p)
        results.loc[name, "ECE_CORRECTED"] = expected_calibration_error(y_test, p)
        results.loc[name, "MEAN_PRED_CORRECTED"] = p.mean()
    ci = bootstrap_ci(test, probs)

    # Risk-tier cutoffs for the dashboard (display only, see RISK_TIER_PERCENTILES).
    tier_cutoffs = {tier: float(np.quantile(corrected["XGBoost"], q)) for tier, q in RISK_TIER_PERCENTILES.items()}
    tiers = np.where(corrected["XGBoost"] >= tier_cutoffs["red"], "red",
                     np.where(corrected["XGBoost"] >= tier_cutoffs["orange"], "orange", "green"))
    tier_check = pd.Series(y_test).groupby(tiers).agg(["mean", "size"])
    logger.info("Observed injury rate by risk tier (test):\n%s", tier_check.round(4).to_string())

    # -- Save models and predictions ---------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Each .pkl bundles the model with the exact feature list it expects, so
    # the dashboard can't feed columns in the wrong order.
    for name, model, features in [
        ("baseline_model", baseline, BASELINE_FEATURES),
        ("logistic_model", logistic, LOGISTIC_FEATURES),
        ("xgb_model", xgb_model, XGB_FEATURES),
    ]:
        # train_base_rate is stored so new predictions can get the same
        # correct_class_weighting() step.
        joblib.dump({"model": model, "features": features, "trained_through": train["GAME_DATE"].max(),
                     "train_base_rate": train_base_rate},
                    MODELS_DIR / f"{name}.pkl")
    logger.info("Saved models to %s/", MODELS_DIR.relative_to(PROJECT_ROOT))

    # PROB_* = raw class-weighted scores (as specified; good for ranking).
    # PROB_*_CORRECTED = real probabilities (use these when showing a %).
    predictions = test[["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", TARGET]].assign(
        PROB_LOGISTIC=probs["Logistic"], PROB_XGB=probs["XGBoost"],
        PROB_LOGISTIC_CORRECTED=corrected["Logistic"], PROB_XGB_CORRECTED=corrected["XGBoost"],
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved %s (%d rows)", PREDICTIONS_PATH.relative_to(PROJECT_ROOT), len(predictions))

    # Everything the dashboard needs to describe the model, in one small file.
    summary = {
        "train_period": [str(train["GAME_DATE"].min().date()), str(train["GAME_DATE"].max().date())],
        "test_period": [str(test["GAME_DATE"].min().date()), str(test["GAME_DATE"].max().date())],
        "n_test": int(len(test)),
        "n_test_injuries": int(y_test.sum()),
        "train_base_rate": train_base_rate,
        "test_base_rate": float(y_test.mean()),
        "metrics": results.to_dict(orient="index"),
        "confidence_intervals": ci.to_dict(orient="index"),
        "risk_tiers": {
            "model": "XGBoost (corrected probability)",
            "basis": "percentiles of the displayed (test) period's predictions; display only",
            "percentiles": RISK_TIER_PERCENTILES,
            "cutoffs": tier_cutoffs,
            "observed_injury_rate": tier_check["mean"].to_dict(),
            "n_games": {k: int(v) for k, v in tier_check["size"].items()},
        },
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", METRICS_PATH.relative_to(PROJECT_ROOT))

    # -- Explain ----------------------------------------------------------------
    shap_values = compute_shap(xgb_model, test[XGB_FEATURES])
    importance = plot_feature_importance(xgb_model, shap_values, DASHBOARD_DIR / "feature_importance.png")
    plot_calibration(y_test, probs, corrected, DASHBOARD_DIR / "calibration.png")

    coefficients = pd.Series(
        logistic.named_steps["model"].coef_[0],
        index=logistic.named_steps["impute"].get_feature_names_out(),
    ).sort_values(key=abs, ascending=False)

    # -- Report -----------------------------------------------------------------
    base_rate = y_test.mean()
    print("\n" + "=" * 92)
    print(f"EVALUATION ON TEST SET  ({len(test):,} player-games, {int(y_test.sum())} injuries, "
          f"base rate {base_rate:.2%})")
    print("=" * 92)
    print(results.round(4).to_string())
    print("\nBootstrap 95% CIs (resampling players, 500 draws):")
    print(ci.to_string())
    print(f"\nXGBoost: {xgb_info['n_trees']} trees, scale_pos_weight {xgb_info['scale_pos_weight']:.1f}")
    print("\nXGBoost importance (mean |SHAP| on test, gain on train):")
    print(importance.round(4).to_string())
    print("\nLogistic regression coefficients (per standard deviation, + = higher risk):")
    print(coefficients.round(3).to_string())
    print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
