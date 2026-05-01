"""
================================================================================
  Product Intelligence Report — analyse_products.py
================================================================================
Reads predictions.csv and produces:

  1. Per-product IC, hit-rate, mean predicted return, return volatility
  2. Intra-group and cross-group return correlation matrices
  3. Pairs cointegration / spread-stationarity test (ADF) for each pair
     within the same category
  4. A "trading playbook" JSON that classifies every product and recommends
     a strategy tier and any pair-trade partners
  5. Pretty console summary + 3 output CSVs + 1 playbook JSON

Run:
    python analyse_products.py

Edit the two paths below.
================================================================================
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  ← edit these
PREDICTIONS_CSV = Path(r"./output/predictions.csv")
OUTPUT_DIR      = Path(r"./output/analysis")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Product universe by category ──────────────────────────────────────────────
CATEGORIES: dict[str, list[str]] = {
    "GALAXY_SOUNDS":  ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
                       "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
                       "GALAXY_SOUNDS_SOLAR_WINDS"],
    "SLEEP_POD":      ["SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
                       "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"],
    "MICROCHIP":      ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
                       "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"],
    "PEBBLES":        ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT":          ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
                       "ROBOT_MOPPING", "ROBOT_VACUUMING"],
    "UV_VISOR":       ["UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
                       "UV_VISOR_RED", "UV_VISOR_YELLOW"],
    "TRANSLATOR":     ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
                       "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY",
                       "TRANSLATOR_VOID_BLUE"],
    "PANEL":          ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE":   ["OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
                       "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT",
                       "OXYGEN_SHAKE_MORNING_BREATH"],
    "SNACKPACK":      ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO",
                       "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY",
                       "SNACKPACK_VANILLA"],
}

# Known AR(1) coefficients from the existing bot (phi is negative → mean-rev)
KNOWN_AR1: dict[str, float] = {
    "ROBOT_DISHES":                -0.23,
    "ROBOT_IRONING":               -0.13,
    "OXYGEN_SHAKE_EVENING_BREATH": -0.12,
    "OXYGEN_SHAKE_CHOCOLATE":      -0.09,
}

# ── Load predictions ──────────────────────────────────────────────────────────
print("Loading predictions …")
df = pd.read_csv(PREDICTIONS_CSV).dropna(subset=["predicted_return", "actual_return"])
df = df.sort_values("global_t").reset_index(drop=True)

# ── 1. Per-product stats ──────────────────────────────────────────────────────
print("Computing per-product statistics …")

def product_stats(grp: pd.DataFrame) -> dict:
    act  = grp["actual_return"].to_numpy()
    pred = grp["predicted_return"].to_numpy()
    if pred.std() < 1e-12 or len(grp) < 30:
        return {}
    ic, pval = stats.pearsonr(pred, act)
    hit = float(((np.sign(pred) == np.sign(act)) & (pred != 0)).mean())
    strat = np.sign(pred) * act
    ir    = float(strat.mean() / (strat.std() + 1e-12))
    # AR(1) phi estimate on actual returns
    if len(act) > 2:
        phi, *_ = stats.linregress(act[:-1], act[1:])
    else:
        phi = 0.0
    return {
        "n":           len(grp),
        "ic":          float(ic),
        "ic_pval":     float(pval),
        "hit_rate":    hit,
        "ir_per_bar":  ir,
        "mean_actual": float(act.mean()),
        "std_actual":  float(act.std()),
        "ar1_phi":     float(phi),
        "mean_pred":   float(pred.mean()),
    }

rows = []
for prod, grp in df.groupby("product"):
    s = product_stats(grp)
    if s:
        cat = next((c for c, members in CATEGORIES.items() if prod in members), "UNKNOWN")
        rows.append({"product": prod, "category": cat, **s})

stats_df = (pd.DataFrame(rows)
              .sort_values("ic", ascending=False)
              .reset_index(drop=True))

# ── 2. Classify each product into a trading tier ──────────────────────────────
def assign_tier(row: pd.Series) -> str:
    """
    TIER 1 — STRONG SIGNAL:   IC ≥ 0.10  → aggressive mean-reversion / ML alpha
    TIER 2 — MODERATE SIGNAL: IC 0.04–0.10 → passive MM + AR1 correction
    TIER 3 — WEAK SIGNAL:     IC < 0.04  → pure passive MM, minimal skew
    TIER 0 — NOISE:           IC < 0.01  → consider skipping entirely
    """
    ic = row["ic"]
    if ic >= 0.10:
        return "TIER1_STRONG"
    elif ic >= 0.04:
        return "TIER2_MODERATE"
    elif ic >= 0.01:
        return "TIER3_WEAK"
    else:
        return "TIER0_NOISE"

stats_df["tier"] = stats_df.apply(assign_tier, axis=1)

# ── 3. Return correlation matrix (per timestamp cross-section) ────────────────
print("Building return correlation matrix …")

# Pivot to wide: rows = global_t, cols = product
wide = df.pivot_table(index="global_t", columns="product",
                      values="actual_return", aggfunc="mean")
wide = wide.ffill().dropna(axis=1, thresh=int(0.7 * len(wide)))

corr_matrix = wide.corr()

# Per-category average internal correlation
cat_corr: dict[str, float] = {}
for cat, members in CATEGORIES.items():
    present = [m for m in members if m in corr_matrix.columns]
    if len(present) < 2:
        continue
    sub = corr_matrix.loc[present, present]
    # average of off-diagonal elements
    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
    vals = sub.values[mask]
    cat_corr[cat] = float(vals.mean())

# ── 4. Cointegration test for all intra-category pairs ────────────────────────
print("Running cointegration tests (ADF + Engle-Granger) …")

# Pivot actual returns → cumulative price proxy (sum of log returns)
price_proxy = wide.cumsum()

coint_rows = []
for cat, members in CATEGORIES.items():
    present = [m for m in members if m in price_proxy.columns]
    for a, b in combinations(present, 2):
        series_a = price_proxy[a].dropna()
        series_b = price_proxy[b].dropna()
        idx = series_a.index.intersection(series_b.index)
        if len(idx) < 100:
            continue
        sa, sb = series_a[idx].values, series_b[idx].values

        # Engle-Granger cointegration test
        try:
            score, pval, _ = coint(sa, sb)
        except Exception:
            pval = 1.0

        # ADF on the spread (OLS residual)
        try:
            beta, alpha, *_ = stats.linregress(sb, sa)
            spread = sa - (beta * sb + alpha)
            adf_stat, adf_pval, *_ = adfuller(spread, maxlag=5, autolag="AIC")
        except Exception:
            adf_stat, adf_pval = 0.0, 1.0

        # Return correlation
        rc = float(corr_matrix.loc[a, b]) if (a in corr_matrix.index and
                                               b in corr_matrix.columns) else 0.0

        coint_rows.append({
            "category":      cat,
            "product_a":     a,
            "product_b":     b,
            "eg_pval":       float(pval),
            "adf_pval":      float(adf_pval),
            "adf_stat":      float(adf_stat),
            "ret_corr":      rc,
            "cointegrated":  pval < 0.05 and adf_pval < 0.05,
        })

coint_df = (pd.DataFrame(coint_rows)
              .sort_values(["cointegrated", "adf_pval"], ascending=[False, True])
              .reset_index(drop=True))

# ── 5. Build the trading playbook ─────────────────────────────────────────────
print("Building trading playbook …")

playbook: dict[str, dict] = {}

for _, row in stats_df.iterrows():
    prod  = row["product"]
    cat   = row["category"]
    tier  = row["tier"]
    ic    = row["ic"]
    phi   = row["ar1_phi"]
    sig   = row["std_actual"]

    # Find best cointegrated pair partners
    pairs = coint_df[
        ((coint_df["product_a"] == prod) | (coint_df["product_b"] == prod))
        & coint_df["cointegrated"]
    ].copy()
    pair_partners = []
    for _, pr in pairs.iterrows():
        partner = pr["product_b"] if pr["product_a"] == prod else pr["product_a"]
        pair_partners.append({
            "partner":   partner,
            "adf_pval":  pr["adf_pval"],
            "ret_corr":  pr["ret_corr"],
        })

    # Recommended strategy
    if tier == "TIER1_STRONG":
        if abs(phi) > 0.08:
            strategy = "AR1_MEAN_REVERSION"
            note     = f"Strong AR1 (phi={phi:.3f}) + high IC={ic:.3f}. Use ML alpha."
        else:
            strategy = "ML_ALPHA_ONLY"
            note     = f"High IC={ic:.3f} without strong AR1. ML signal is orthogonal — pair coupling likely."
    elif tier == "TIER2_MODERATE":
        if abs(phi) > 0.05:
            strategy = "AR1_PASSIVE_MM"
            note     = f"Moderate AR1 (phi={phi:.3f}), IC={ic:.3f}. Passive MM with AR1 fair correction."
        else:
            strategy = "PASSIVE_MM_IMBALANCE"
            note     = f"IC={ic:.3f}, weak AR1. Standard passive MM with imbalance gate."
    elif tier == "TIER3_WEAK":
        if pair_partners:
            strategy = "PAIRS_TRADE"
            note     = f"Low solo IC={ic:.3f} but cointegrated pairs found. Focus on spread."
        else:
            strategy = "PASSIVE_MM_TIGHT"
            note     = f"Weak signal. Narrow passive MM, tight inventory limits."
    else:
        strategy = "SKIP_OR_MINIMAL"
        note     = f"IC={ic:.3f} near zero. Either skip or trade minimal size."

    # Suggested AR1 phi to use in bot (blend of empirical and known)
    if prod in KNOWN_AR1:
        bot_phi = KNOWN_AR1[prod]  # trust the calibrated value
    elif abs(phi) > 0.05:
        bot_phi = round(phi * 0.8, 3)  # shrink toward zero (regularise)
    else:
        bot_phi = 0.0

    playbook[prod] = {
        "category":         cat,
        "tier":             tier,
        "ic":               round(ic, 4),
        "ar1_phi_empirical": round(phi, 4),
        "ar1_phi_for_bot":  bot_phi,
        "hit_rate":         round(row["hit_rate"], 4),
        "ir_per_bar":       round(row["ir_per_bar"], 6),
        "strategy":         strategy,
        "note":             note,
        "pair_partners":    pair_partners[:3],  # top 3 pairs
        "cat_avg_corr":     round(cat_corr.get(cat, 0.0), 4),
    }

# ── 6. Console summary ────────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  TRADING INTELLIGENCE REPORT")
print("═" * 72)

tier_order = ["TIER1_STRONG", "TIER2_MODERATE", "TIER3_WEAK", "TIER0_NOISE"]
for tier in tier_order:
    subset = stats_df[stats_df["tier"] == tier]
    if subset.empty:
        continue
    print(f"\n{'─'*72}")
    print(f"  {tier}  ({len(subset)} products)")
    print(f"{'─'*72}")
    for _, r in subset.iterrows():
        pb   = playbook[r["product"]]
        pairs_str = ", ".join(p["partner"].split("_")[-1]
                              for p in pb["pair_partners"]) or "none"
        print(f"  {r['product']:<40}  IC={r['ic']:+.4f}  φ={r['ar1_phi']:+.3f}"
              f"  {pb['strategy']:<25}  pairs:[{pairs_str}]")

print(f"\n{'─'*72}")
print("  CATEGORY INTERNAL CORRELATIONS")
print(f"{'─'*72}")
for cat, avg_c in sorted(cat_corr.items(), key=lambda x: -x[1]):
    bar = "█" * int(avg_c * 40)
    print(f"  {cat:<20}  avg_corr={avg_c:+.3f}  {bar}")

print(f"\n{'─'*72}")
print("  COINTEGRATED PAIRS  (p<0.05 on both EG and ADF)")
print(f"{'─'*72}")
coint_yes = coint_df[coint_df["cointegrated"]].head(20)
if coint_yes.empty:
    print("  (none found at 5% significance — may need longer series)")
else:
    for _, r in coint_yes.iterrows():
        print(f"  {r['product_a']:<35} ↔ {r['product_b']:<35}"
              f"  adf_p={r['adf_pval']:.4f}  corr={r['ret_corr']:+.3f}")

# ── 7. Save outputs ───────────────────────────────────────────────────────────
stats_df.to_csv(OUTPUT_DIR / "product_stats.csv", index=False)
coint_df.to_csv(OUTPUT_DIR / "cointegration_pairs.csv", index=False)
corr_matrix.to_csv(OUTPUT_DIR / "return_correlation_matrix.csv")

with open(OUTPUT_DIR / "trading_playbook.json", "w") as fh:
    json.dump(playbook, fh, indent=2)

print(f"\nOutputs written to {OUTPUT_DIR}/")
print("  product_stats.csv")
print("  cointegration_pairs.csv")
print("  return_correlation_matrix.csv")
print("  trading_playbook.json")
print("\nNext step: feed trading_playbook.json into your bot to set")
print("per-product strategy, AR1 phi, and pair-trade parameters.")
