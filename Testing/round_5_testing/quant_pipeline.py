"""
================================================================================
 Quantitative Research Pipeline — 50-Equity Order-Book Dataset (3 days)
================================================================================
Two-phase machine-learning pipeline for next-bar return prediction.

    Phase 1  Random-Forest feature importance   →  drop bottom 50% of features
    Phase 2  XGBoost regressor                  →  evaluated with TimeSeriesSplit

Strict no-leakage rules:
  • Target = log(mid_{t+H}) − log(mid_t)  ⇒  built with .shift(-H) on the
    *price*, never on features.
  • All engineered features (rolling means/std, diffs, microprice, OFI…) are
    backward-looking by construction.
  • Random-shuffle splits are FORBIDDEN. We use sklearn TimeSeriesSplit so
    every test fold lies strictly *after* its training fold.
  • Random-Forest selection is fit on the first CV training segment only
    (not the full panel), so the held-out folds never influence which
    features are kept.

Outputs (written to OUTPUT_DIR):
    predictions.csv          out-of-fold (OOF) actual & predicted returns
    feature_importances.csv  RF importances for every initial feature
    selected_features.csv    surviving (top-50%) features used by XGBoost
    metrics.json             per-fold + aggregate evaluation metrics

Author : Lead Quant Engineer
================================================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  CONFIGURATION  ──  POINT YOUR FILE PATHS HERE
# =============================================================================
# >>>>>>>>>>>>>>>>>>>>>>  EDIT THESE FOUR LINES  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DATA_DIR   = Path(r"./data")            # folder containing the 6 csv files
OUTPUT_DIR = Path(r"./output")          # results land here
DAYS       = [2, 3, 4]                  # the three trading-day labels
ROUND_ID   = 5                          # used in the file-name template
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# File-name template — adjust if your files use a different convention.
PRICES_FILES = [DATA_DIR / f"prices_round_{ROUND_ID}_day_{d}.csv" for d in DAYS]
TRADES_FILES = [DATA_DIR / f"trades_round_{ROUND_ID}_day_{d}.csv" for d in DAYS]

# Modeling hyper-parameters
TARGET_HORIZON = 1        # predict next-bar log return (use 5/10 for longer)
N_SPLITS       = 5        # number of TimeSeriesSplit folds
TOP_FEATURE_FRAC = 0.5    # keep top 50% of features after RF importance
USE_TRADE_FEATURES = False  # user noted trades are probably less informative
RANDOM_STATE   = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1.  DATA LOADING
# =============================================================================
def load_prices(files: list[Path]) -> pd.DataFrame:
    """Read & concatenate the 3 daily order-book snapshots.

    The competition CSVs use ';' as separator. We build a single monotonically-
    increasing time key (`global_t = day * 1e7 + timestamp`) so that ordering
    across days is unambiguous.
    """
    frames = [pd.read_csv(f, sep=";") for f in files]
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp"]).reset_index(drop=True)
    prices["global_t"] = prices["day"].astype(np.int64) * 10_000_000 \
                         + prices["timestamp"].astype(np.int64)
    return prices


def load_trades(files: list[Path], days: list[int]) -> pd.DataFrame:
    """Trade prints have no `day` column → infer from file order."""
    frames = []
    for day, f in zip(days, files):
        df = pd.read_csv(f, sep=";")
        df["day"] = day
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades["global_t"] = trades["day"].astype(np.int64) * 10_000_000 \
                         + trades["timestamp"].astype(np.int64)
    return trades


# =============================================================================
# 2.  PER-PRODUCT FEATURE ENGINEERING (BACKWARD-LOOKING ONLY)
# =============================================================================
def engineer_features_for_product(df_prod: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix for a single product, sorted by time.

    EVERY feature here is computed from values at time ≤ t.  The only
    forward-looking column is `target`, created via .shift(-H) on the price
    at the very end of the function.
    """
    df = df_prod.sort_values("global_t").reset_index(drop=True).copy()

    # ---- Level-1 quote features --------------------------------------------
    df["spread"]    = df["ask_price_1"] - df["bid_price_1"]
    denom_1         = (df["bid_volume_1"] + df["ask_volume_1"]).replace(0, np.nan)
    df["microprice"] = (df["bid_price_1"] * df["ask_volume_1"]
                        + df["ask_price_1"] * df["bid_volume_1"]) / denom_1
    df["microprice_dev"] = df["microprice"] - df["mid_price"]
    df["ofi_1"] = (df["bid_volume_1"] - df["ask_volume_1"]) / denom_1

    # ---- Multi-level book depth (fill missing levels with 0) ---------------
    for col in ["bid_volume_2", "bid_volume_3",
                "ask_volume_2", "ask_volume_3",
                "bid_price_2",  "bid_price_3",
                "ask_price_2",  "ask_price_3"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["bid_depth"] = df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1)
    df["ask_depth"] = df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1)
    total_depth = (df["bid_depth"] + df["ask_depth"]).replace(0, np.nan)
    df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / total_depth

    # ---- Returns (log) -----------------------------------------------------
    df["log_mid"] = np.log(df["mid_price"].replace(0, np.nan))
    df["ret_1"]   = df["log_mid"].diff(1)
    df["ret_5"]   = df["log_mid"].diff(5)
    df["ret_20"]  = df["log_mid"].diff(20)

    # ---- Rolling statistics (strict past windows) --------------------------
    for w in (5, 10, 30, 60):
        df[f"roll_mean_ret_{w}"]  = df["ret_1"].rolling(w).mean()
        df[f"roll_std_ret_{w}"]   = df["ret_1"].rolling(w).std()
        df[f"roll_mean_spread_{w}"] = df["spread"].rolling(w).mean()
        df[f"roll_mean_ofi_{w}"]  = df["ofi_1"].rolling(w).mean()
        df[f"roll_mean_imb_{w}"]  = df["depth_imbalance"].rolling(w).mean()

    # ---- Mean-reversion z-score of mid-price -------------------------------
    rolling_mu  = df["mid_price"].rolling(60).mean()
    rolling_sig = df["mid_price"].rolling(60).std()
    df["mid_z_60"] = (df["mid_price"] - rolling_mu) / rolling_sig

    # ---- PnL deltas (already supplied by the venue) ------------------------
    if "profit_and_loss" in df.columns:
        df["pnl_delta_1"]  = df["profit_and_loss"].diff(1)
        df["pnl_roll_5"]   = df["profit_and_loss"].rolling(5).mean()

    # ---- TARGET  (the only forward-looking line in this entire file) -------
    # forward log-return:  r_{t→t+H} = log_mid_{t+H} − log_mid_t
    df["target"] = df["log_mid"].shift(-TARGET_HORIZON) - df["log_mid"]

    return df


# =============================================================================
# 3.  CROSS-SECTIONAL / PAIRS-TRADING FEATURES
# =============================================================================
def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds market-relative features that are central to pairs / stat-arb.

    All of them are computed *within the same timestamp* using only the
    contemporaneous cross-section, so there is no leakage forward in time.
    """
    # Equal-weighted market return at every timestamp
    market_ret_1   = panel.groupby("global_t")["ret_1"].transform("mean")
    market_ret_5   = panel.groupby("global_t")["ret_5"].transform("mean")
    panel["market_ret_1"] = market_ret_1
    panel["market_ret_5"] = market_ret_5

    # Idiosyncratic (residual) returns — the workhorse of pairs trading
    panel["rel_ret_1"] = panel["ret_1"] - market_ret_1
    panel["rel_ret_5"] = panel["ret_5"] - market_ret_5

    # Cross-sectional rank of recent return  (a momentum / reversal signal)
    panel["xs_rank_ret_1"] = (panel.groupby("global_t")["ret_1"]
                                   .rank(pct=True))
    panel["xs_rank_ret_5"] = (panel.groupby("global_t")["ret_5"]
                                   .rank(pct=True))
    return panel


# =============================================================================
# 4.  (OPTIONAL) TRADE-FLOW FEATURES
# =============================================================================
def add_trade_features(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades to (global_t, product) and merge.  Off by default
    because the user indicated trades are likely less informative."""
    if trades.empty:
        return panel

    agg = (trades.groupby(["global_t", "symbol"])
                  .agg(trade_count=("price", "size"),
                       trade_vol=("quantity", "sum"),
                       trade_vwap=("price", lambda x: np.average(
                           x, weights=trades.loc[x.index, "quantity"])))
                  .reset_index()
                  .rename(columns={"symbol": "product"}))
    panel = panel.merge(agg, on=["global_t", "product"], how="left")
    panel[["trade_count", "trade_vol"]] = panel[["trade_count", "trade_vol"]].fillna(0)
    panel["trade_vwap_dev"] = panel["trade_vwap"] - panel["mid_price"]
    panel["trade_vwap_dev"] = panel["trade_vwap_dev"].fillna(0)
    panel = panel.drop(columns=["trade_vwap"])
    return panel


# =============================================================================
# 5.  MAIN PIPELINE
# =============================================================================
def main() -> None:
    # ------------------------------------------------------------------ load
    print("[1/6] Loading data …")
    prices = load_prices(PRICES_FILES)
    trades = load_trades(TRADES_FILES, DAYS) if USE_TRADE_FEATURES else pd.DataFrame()
    print(f"      prices rows: {len(prices):,}  |  products: "
          f"{prices['product'].nunique()}")

    # ------------------------------------------------------ per-product feats
    print("[2/6] Engineering per-product features …")
    feature_frames = [engineer_features_for_product(g)
                      for _, g in prices.groupby("product", sort=False)]
    panel = pd.concat(feature_frames, ignore_index=True)
    panel = panel.sort_values(["global_t", "product"]).reset_index(drop=True)

    # ------------------------------------------------------ cross-sectional
    print("[3/6] Adding cross-sectional features …")
    panel = add_cross_sectional_features(panel)
    if USE_TRADE_FEATURES:
        panel = add_trade_features(panel, trades)

    # Encode product as integer id (tree models handle this fine)
    panel["product_id"] = panel["product"].astype("category").cat.codes

    # ------------------------------------------------------------- cleanup
    # Drop rows where the *target* is NaN (final H rows of each product) and
    # any rows with NaN features (the early rows where rolling windows aren't
    # yet populated).  We do NOT drop based on future information.
    panel = panel.dropna(subset=["target"]).copy()

    NON_FEATURE_COLS = {"day", "timestamp", "global_t", "product",
                        "target", "log_mid", "profit_and_loss"}
    feature_cols = [c for c in panel.columns
                    if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(panel[c])]

    # forward-fill within product, then drop any residual NaNs (start-of-day)
    panel[feature_cols] = (panel.groupby("product", sort=False)[feature_cols]
                                 .ffill())
    panel = panel.dropna(subset=feature_cols).reset_index(drop=True)

    # CRITICAL: re-sort by time once more before CV so folds are chronological
    panel = panel.sort_values("global_t", kind="mergesort").reset_index(drop=True)

    X = panel[feature_cols].to_numpy()
    y = panel["target"].to_numpy()
    print(f"      design matrix: X={X.shape},  y={y.shape}")

    # =====================================================================
    # PHASE 1 — Random-Forest feature selection
    # =====================================================================
    print("[4/6] Phase 1 — Random-Forest feature importance …")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # We deliberately fit the RF on the FIRST training segment only.  Fitting
    # on the full panel would let the eventual XGBoost test folds influence
    # which features are kept — a subtle form of look-ahead.
    first_train_idx, _ = next(iter(tscv.split(X)))
    rf = RandomForestRegressor(
        n_estimators   = 200,
        max_depth      = 8,
        min_samples_leaf = 50,
        n_jobs         = -1,
        random_state   = RANDOM_STATE,
    )
    rf.fit(X[first_train_idx], y[first_train_idx])

    importances = (pd.Series(rf.feature_importances_, index=feature_cols)
                     .sort_values(ascending=False))
    n_keep = max(1, int(np.ceil(len(importances) * TOP_FEATURE_FRAC)))
    selected_features = importances.head(n_keep).index.tolist()
    print(f"      kept top {n_keep}/{len(importances)} features "
          f"(threshold importance = {importances.iloc[n_keep-1]:.5f})")

    X_sel = panel[selected_features].to_numpy()

    # =====================================================================
    # PHASE 2 — XGBoost with TimeSeriesSplit
    # =====================================================================
    print("[5/6] Phase 2 — XGBoost with TimeSeriesSplit …")
    fold_metrics: list[dict] = []
    oof_preds = np.full(shape=y.shape, fill_value=np.nan, dtype=float)

    for fold, (tr, te) in enumerate(tscv.split(X_sel), start=1):
        model = XGBRegressor(
            n_estimators   = 400,
            max_depth      = 6,
            learning_rate  = 0.05,
            subsample      = 0.8,
            colsample_bytree = 0.8,
            objective      = "reg:squarederror",
            tree_method    = "hist",
            n_jobs         = -1,
            random_state   = RANDOM_STATE,
        )
        model.fit(X_sel[tr], y[tr],
                  eval_set=[(X_sel[te], y[te])], verbose=False)

        pred = model.predict(X_sel[te])
        oof_preds[te] = pred

        # ---- evaluation -------------------------------------------------
        mae   = mean_absolute_error(y[te], pred)
        rmse  = float(np.sqrt(mean_squared_error(y[te], pred)))
        # Information Coefficient: cross-sectional / temporal corr(pred, actual)
        ic    = float(np.corrcoef(pred, y[te])[0, 1]) if pred.std() > 0 else 0.0
        # Directional strategy: long when pred > 0, short when pred < 0
        strat = np.sign(pred) * y[te]
        ir    = float(strat.mean() / (strat.std() + 1e-12))   # per-bar IR
        # Periodic Sharpe; multiply by sqrt(bars-per-year) yourself if desired
        sharpe_per_bar = ir
        hit_rate = float(((np.sign(pred) == np.sign(y[te])) & (pred != 0)).mean())

        fold_metrics.append({
            "fold":              fold,
            "n_train":           int(len(tr)),
            "n_test":            int(len(te)),
            "mae":               float(mae),
            "rmse":              rmse,
            "information_coef":  ic,
            "information_ratio": ir,
            "sharpe_per_bar":    sharpe_per_bar,
            "hit_rate":          hit_rate,
        })
        print(f"      fold {fold}:  MAE={mae:.6f}   IC={ic:+.4f}   "
              f"IR/bar={ir:+.4f}   hit-rate={hit_rate:.3f}")

    # ----------- aggregate OOF metrics (test rows only) -----------------
    mask = ~np.isnan(oof_preds)
    oof_mae    = float(mean_absolute_error(y[mask], oof_preds[mask]))
    oof_ic     = float(np.corrcoef(oof_preds[mask], y[mask])[0, 1])
    oof_strat  = np.sign(oof_preds[mask]) * y[mask]
    oof_ir     = float(oof_strat.mean() / (oof_strat.std() + 1e-12))
    oof_hit    = float(((np.sign(oof_preds[mask]) == np.sign(y[mask]))
                        & (oof_preds[mask] != 0)).mean())

    summary = {
        "n_observations":            int(len(y)),
        "n_features_initial":        int(X.shape[1]),
        "n_features_selected":       int(len(selected_features)),
        "cv_folds":                  N_SPLITS,
        "target_horizon_bars":       TARGET_HORIZON,
        "oof_mae":                   oof_mae,
        "oof_information_coef":      oof_ic,
        "oof_information_ratio":     oof_ir,
        "oof_hit_rate":              oof_hit,
        "fold_metrics":              fold_metrics,
        "note":  ("IR / Sharpe are reported per-bar.  To annualize, multiply "
                  "by sqrt(bars_per_year).  Avoid relying on directional "
                  "Sharpe alone — IC is the more honest skill metric."),
    }

    # =====================================================================
    # 6.  EXPORT
    # =====================================================================
    print("[6/6] Writing outputs …")

    # 6a — OOF predictions
    pred_df = panel[["global_t", "day", "timestamp", "product"]].copy()
    pred_df["actual_return"]    = y
    pred_df["predicted_return"] = oof_preds
    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # 6b — Feature importance table & retained features
    (importances.to_frame("importance")
                .reset_index().rename(columns={"index": "feature"})
                .to_csv(OUTPUT_DIR / "feature_importances.csv", index=False))

    pd.Series(selected_features, name="feature").to_csv(
        OUTPUT_DIR / "selected_features.csv", index=False)

    # 6c — Metrics JSON
    with open(OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"      → {OUTPUT_DIR/'predictions.csv'}")
    print(f"      → {OUTPUT_DIR/'feature_importances.csv'}")
    print(f"      → {OUTPUT_DIR/'selected_features.csv'}")
    print(f"      → {OUTPUT_DIR/'metrics.json'}")
    print("Done.")


# =============================================================================
if __name__ == "__main__":
    main()
