"""
================================================================================
  Product Intelligence Report v2 — analyse_products.py
================================================================================
Upgrades over v1:
  1. Residual (idiosyncratic) correlation matrix per category
     — strips the category market factor before correlating
  2. Johansen multivariate cointegration for 3-5 product baskets
     — outputs hedge ratios (eigenvectors) for any stationary basket found
  3. Updated trading_playbook.json includes basket_strategy key

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
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  ← edit these
PREDICTIONS_CSV = Path(r"./output/predictions.csv")
OUTPUT_DIR      = Path(r"./output/analysis")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Product universe ──────────────────────────────────────────────────────────
CATEGORIES: dict[str, list[str]] = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
                      "GALAXY_SOUNDS_SOLAR_WINDS"],
    "SLEEP_POD":     ["SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
                      "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"],
    "MICROCHIP":     ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
                      "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"],
    "PEBBLES":       ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT":         ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
                      "ROBOT_MOPPING", "ROBOT_VACUUMING"],
    "UV_VISOR":      ["UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
                      "UV_VISOR_RED", "UV_VISOR_YELLOW"],
    "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
                      "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY",
                      "TRANSLATOR_VOID_BLUE"],
    "PANEL":         ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
                      "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT",
                      "OXYGEN_SHAKE_MORNING_BREATH"],
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO",
                      "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY",
                      "SNACKPACK_VANILLA"],
}

KNOWN_AR1: dict[str, float] = {
    "ROBOT_DISHES":                -0.23,
    "ROBOT_IRONING":               -0.13,
    "OXYGEN_SHAKE_EVENING_BREATH": -0.12,
    "OXYGEN_SHAKE_CHOCOLATE":      -0.09,
}

# Johansen critical value column index: 0=90%, 1=95%, 2=99%
JOHANSEN_CV_IDX = 1
MIN_COVERAGE    = 0.70


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("Loading predictions …")
df = (pd.read_csv(PREDICTIONS_CSV)
        .dropna(subset=["predicted_return", "actual_return"])
        .sort_values("global_t")
        .reset_index(drop=True))

n_timestamps = df["global_t"].nunique()
wide_ret = (df.pivot_table(index="global_t", columns="product",
                           values="actual_return", aggfunc="mean")
              .ffill()
              .dropna(axis=1, thresh=int(MIN_COVERAGE * n_timestamps)))

price_proxy = wide_ret.cumsum()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PER-PRODUCT STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("Computing per-product statistics …")

def product_stats(grp: pd.DataFrame) -> dict | None:
    act  = grp["actual_return"].to_numpy()
    pred = grp["predicted_return"].to_numpy()
    if pred.std() < 1e-12 or len(grp) < 30:
        return None
    ic, pval = stats.pearsonr(pred, act)
    hit   = float(((np.sign(pred) == np.sign(act)) & (pred != 0)).mean())
    strat = np.sign(pred) * act
    ir    = float(strat.mean() / (strat.std() + 1e-12))
    phi   = float(stats.linregress(act[:-1], act[1:])[0]) if len(act) > 2 else 0.0
    return dict(n=len(grp), ic=float(ic), ic_pval=float(pval),
                hit_rate=hit, ir_per_bar=ir,
                mean_actual=float(act.mean()), std_actual=float(act.std()),
                ar1_phi=float(phi), mean_pred=float(pred.mean()))

rows = []
for prod, grp in df.groupby("product"):
    s = product_stats(grp)
    if s:
        cat = next((c for c, m in CATEGORIES.items() if prod in m), "UNKNOWN")
        rows.append({"product": prod, "category": cat, **s})

stats_df = (pd.DataFrame(rows)
              .sort_values("ic", ascending=False)
              .reset_index(drop=True))

def assign_tier(row: pd.Series) -> str:
    ic = row["ic"]
    if ic >= 0.10: return "TIER1_STRONG"
    if ic >= 0.04: return "TIER2_MODERATE"
    if ic >= 0.01: return "TIER3_WEAK"
    return "TIER0_NOISE"

stats_df["tier"] = stats_df.apply(assign_tier, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — RAW RETURN CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("Computing raw return correlation matrix …")
corr_raw = wide_ret.corr()

cat_corr: dict[str, float] = {}
for cat, members in CATEGORIES.items():
    present = [m for m in members if m in corr_raw.columns]
    if len(present) < 2:
        continue
    sub  = corr_raw.loc[present, present]
    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
    cat_corr[cat] = float(sub.values[mask].mean())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RESIDUAL (IDIOSYNCRATIC) CORRELATION  ← NEW
#
#  Methodology:
#    1. Compute category_factor_t = mean(all product returns in category at t)
#    2. residual_i_t = return_i_t - category_factor_t
#       This strips the shared "market beta=1" component that all products
#       in a group load on identically under a position-limit-10 constraint.
#    3. Correlate the residuals within and across categories.
#
#  Why this matters:
#    Raw return correlation is dominated by the category factor.
#    Two products with raw corr=0.9 might have residual corr=0.02 (independent)
#    or residual corr=0.85 (genuinely coupled beyond the group move).
#    Only the second type is useful for pairs trading.
# ══════════════════════════════════════════════════════════════════════════════
print("Computing residual (idiosyncratic) correlation matrices …")

resid_corr_by_cat: dict[str, pd.DataFrame] = {}
resid_wide_all = pd.DataFrame(index=wide_ret.index)

for cat, members in CATEGORIES.items():
    present = [m for m in members if m in wide_ret.columns]
    if len(present) < 2:
        continue
    cat_ret    = wide_ret[present]
    cat_factor = cat_ret.mean(axis=1)              # equal-weight category factor
    resid      = cat_ret.subtract(cat_factor, axis=0)
    resid_corr_by_cat[cat] = resid.corr()
    resid.columns = [f"{c}_resid" for c in resid.columns]
    resid_wide_all = resid_wide_all.join(resid, how="outer")

resid_corr_global = resid_wide_all.corr()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PAIRWISE COINTEGRATION (EG + ADF)
# ══════════════════════════════════════════════════════════════════════════════
print("Running pairwise cointegration tests …")

coint_rows = []
for cat, members in CATEGORIES.items():
    present = [m for m in members if m in price_proxy.columns]
    for a, b in combinations(present, 2):
        sa = price_proxy[a].dropna().values
        sb = price_proxy[b].dropna().values
        n  = min(len(sa), len(sb))
        if n < 100:
            continue
        sa, sb = sa[:n], sb[:n]
        try:
            _, eg_pval, _ = coint(sa, sb)
        except Exception:
            eg_pval = 1.0
        try:
            beta, alpha, *_ = stats.linregress(sb, sa)
            spread = sa - (beta * sb + alpha)
            _, adf_pval, *_ = adfuller(spread, maxlag=5, autolag="AIC")
        except Exception:
            adf_pval = 1.0
        rc = float(corr_raw.loc[a, b]) if (a in corr_raw.index and
                                            b in corr_raw.columns) else 0.0
        ra, rb = f"{a}_resid", f"{b}_resid"
        resid_c = (float(resid_corr_global.loc[ra, rb])
                   if ra in resid_corr_global.index and rb in resid_corr_global.columns
                   else np.nan)
        coint_rows.append(dict(category=cat, product_a=a, product_b=b,
                               eg_pval=float(eg_pval), adf_pval=float(adf_pval),
                               ret_corr=rc, resid_corr=resid_c,
                               cointegrated=eg_pval < 0.05 and adf_pval < 0.05))

coint_df = (pd.DataFrame(coint_rows)
              .sort_values(["cointegrated", "adf_pval"], ascending=[False, True])
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — JOHANSEN MULTIVARIATE COINTEGRATION  ← NEW
#
#  For each category we test the full 5-product price matrix jointly.
#
#  The Johansen trace test asks: how many stationary linear combinations
#  (cointegrating vectors) exist in this system?
#    - r=0 rejected → at least 1 basket exists
#    - r≤1 rejected → at least 2 baskets exist, etc.
#
#  We extract the first eigenvector (most stationary combination) and
#  verify its stationarity with ADF.  We also compute the half-life of
#  mean reversion so the bot knows how aggressively to trade it.
#
#  det_order=0 → constant allowed in the cointegrating relation
#  k_ar_diff=1 → one lag in the VECM (standard starting point)
# ══════════════════════════════════════════════════════════════════════════════
print("Running Johansen cointegration tests …")

basket_results: dict[str, dict] = {}

for cat, members in CATEGORIES.items():
    present = [m for m in members if m in price_proxy.columns]
    if len(present) < 3:
        basket_results[cat] = {"status": "too_few_products"}
        continue
    data = price_proxy[present].ffill().dropna()
    if len(data) < 200:
        basket_results[cat] = {"status": "insufficient_data", "n_obs": len(data)}
        continue
    try:
        jres = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    except Exception as e:
        basket_results[cat] = {"status": f"test_failed:{e}"}
        continue

    trace_stat = jres.lr1
    crit_vals  = jres.cvt[:, JOHANSEN_CV_IDX]
    n_coint    = int(np.sum(trace_stat > crit_vals))

    if n_coint == 0:
        basket_results[cat] = {
            "status":      "no_cointegration",
            "n_coint":     0,
            "trace_stats": trace_stat.tolist(),
            "crit_vals":   crit_vals.tolist(),
        }
        continue

    # First eigenvector = most mean-reverting basket
    evec = jres.evec[:, 0].copy()
    if abs(evec[0]) > 1e-10:
        evec /= evec[0]

    hedge_ratios = {p: round(float(w), 6) for p, w in zip(present, evec)}

    # ADF on the basket spread
    spread_series = data.values @ evec
    _, adf_pval, *_ = adfuller(spread_series, maxlag=5, autolag="AIC")

    # Half-life via AR(1) on the spread
    phi_s, *_ = stats.linregress(spread_series[:-1], spread_series[1:])
    half_life  = float(-np.log(2) / np.log(phi_s)) if 0 < phi_s < 1 else np.nan

    basket_results[cat] = {
        "status":             "cointegrated",
        "n_coint_vecs":       n_coint,
        "hedge_ratios":       hedge_ratios,
        "adf_pval_spread":    round(float(adf_pval), 6),
        "spread_stationary":  adf_pval < 0.05,
        "half_life_bars":     round(half_life, 1) if not np.isnan(half_life) else None,
        "trace_stats":        trace_stat.tolist(),
        "crit_vals":          crit_vals.tolist(),
        "members":            present,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — TRADING PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════
print("Building trading playbook …")

playbook: dict[str, dict] = {}

for _, row in stats_df.iterrows():
    prod = row["product"]
    cat  = row["category"]
    tier = row["tier"]
    ic   = row["ic"]
    phi  = row["ar1_phi"]

    # Pairwise cointegrated partners
    mask  = (coint_df["product_a"] == prod) | (coint_df["product_b"] == prod)
    pairs = coint_df[mask & coint_df["cointegrated"]]
    pair_partners = [
        dict(partner=(pr["product_b"] if pr["product_a"] == prod else pr["product_a"]),
             adf_pval=pr["adf_pval"], ret_corr=pr["ret_corr"],
             resid_corr=pr.get("resid_corr"))
        for _, pr in pairs.iterrows()
    ]

    # Johansen basket membership
    basket_info = basket_results.get(cat, {})
    basket_strategy = None
    if basket_info.get("status") == "cointegrated":
        hr = basket_info["hedge_ratios"]
        if prod in hr:
            basket_strategy = {
                "members":           basket_info["members"],
                "hedge_ratios":      hr,
                "half_life_bars":    basket_info["half_life_bars"],
                "adf_pval_spread":   basket_info["adf_pval_spread"],
                "spread_stationary": basket_info["spread_stationary"],
            }

    # Top residual-correlated partners within category
    resid_cat = resid_corr_by_cat.get(cat, pd.DataFrame())
    resid_partners: list[dict] = []
    if prod in resid_cat.columns:
        row_resid = resid_cat[prod].drop(labels=[prod], errors="ignore")
        for partner, val in row_resid.abs().nlargest(2).items():
            resid_partners.append(dict(partner=str(partner),
                                       resid_corr=round(float(resid_cat.loc[partner, prod]), 4)))

    # Strategy assignment
    if tier == "TIER1_STRONG":
        strategy = "AR1_MEAN_REVERSION" if abs(phi) > 0.08 else "ML_ALPHA_CROSS_PRODUCT"
    elif tier == "TIER2_MODERATE":
        strategy = "AR1_PASSIVE_MM" if abs(phi) > 0.05 else "PASSIVE_MM_IMBALANCE"
    elif tier == "TIER3_WEAK":
        strategy = "PAIRS_TRADE" if pair_partners else "PASSIVE_MM_TIGHT"
    else:
        strategy = "SKIP_OR_MINIMAL"

    # Upgrade weak products that belong to a confirmed stationary basket
    if basket_strategy and basket_strategy["spread_stationary"] and \
            tier in ("TIER3_WEAK", "TIER0_NOISE"):
        strategy = "BASKET_STAT_ARB"

    bot_phi = KNOWN_AR1.get(prod,
              round(phi * 0.8, 4) if abs(phi) > 0.05 else 0.0)

    playbook[prod] = dict(
        category           = cat,
        tier               = tier,
        ic                 = round(ic, 4),
        ar1_phi_empirical  = round(phi, 4),
        ar1_phi_for_bot    = bot_phi,
        hit_rate           = round(row["hit_rate"], 4),
        ir_per_bar         = round(row["ir_per_bar"], 6),
        strategy           = strategy,
        cat_avg_corr       = round(cat_corr.get(cat, 0.0), 4),
        pair_partners      = pair_partners[:3],
        resid_partners     = resid_partners,
        basket_strategy    = basket_strategy,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print("  TRADING INTELLIGENCE REPORT  v2")
print("═" * 76)

for tier in ["TIER1_STRONG", "TIER2_MODERATE", "TIER3_WEAK", "TIER0_NOISE"]:
    sub = stats_df[stats_df["tier"] == tier]
    if sub.empty:
        continue
    print(f"\n{'─'*76}\n  {tier}  ({len(sub)} products)\n{'─'*76}")
    for _, r in sub.iterrows():
        pb   = playbook[r["product"]]
        rp   = ", ".join(p["partner"].split("_")[-1] for p in pb["resid_partners"]) or "—"
        bs   = "✓" if pb["basket_strategy"] else " "
        print(f"  {r['product']:<40}  IC={r['ic']:+.4f}  φ={r['ar1_phi']:+.4f}"
              f"  basket={bs}  resid=[{rp}]  {pb['strategy']}")

print(f"\n{'─'*76}\n  JOHANSEN BASKET RESULTS\n{'─'*76}")
for cat, br in basket_results.items():
    if br.get("status") == "cointegrated":
        hl     = br.get("half_life_bars")
        hl_str = f"{hl:.0f} bars" if hl else "N/A"
        adf_p  = br.get("adf_pval_spread", 1.0)
        hr_str = "  ".join(f"{k.split('_')[-1]}:{v:+.3f}"
                           for k, v in br["hedge_ratios"].items())
        print(f"  {cat:<20}  r={br['n_coint_vecs']}  HL={hl_str:<10}"
              f"  adf_p={adf_p:.4f}")
        print(f"    weights → {hr_str}")
    else:
        print(f"  {cat:<20}  {br.get('status','?')}")

print(f"\n{'─'*76}\n  RESIDUAL CORRELATION HIGHLIGHTS\n{'─'*76}")
for cat, rc in resid_corr_by_cat.items():
    if rc.empty:
        continue
    mask  = np.triu(np.ones(rc.shape, dtype=bool), k=1)
    pairs_list = [(rc.index[i], rc.columns[j], rc.values[i, j])
                  for i in range(len(rc))
                  for j in range(len(rc.columns)) if mask[i, j]]
    for a, b, v in sorted(pairs_list, key=lambda x: -abs(x[2]))[:2]:
        tag = "STRONG" if abs(v) > 0.5 else "moderate"
        print(f"  {cat:<16}  {a.split('_')[-1]:<15} ↔ {b.split('_')[-1]:<15}"
              f"  resid_corr={v:+.4f}  [{tag}]")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nWriting outputs to {OUTPUT_DIR} …")
stats_df.to_csv(OUTPUT_DIR / "product_stats.csv", index=False)
coint_df.to_csv(OUTPUT_DIR / "cointegration_pairs.csv", index=False)
corr_raw.to_csv(OUTPUT_DIR / "return_correlation_matrix.csv")
for cat, rc in resid_corr_by_cat.items():
    rc.to_csv(OUTPUT_DIR / f"resid_corr_{cat.lower()}.csv")
resid_corr_global.to_csv(OUTPUT_DIR / "resid_corr_global.csv")

basket_rows = []
for cat, br in basket_results.items():
    if br.get("status") == "cointegrated":
        for prod, w in br["hedge_ratios"].items():
            basket_rows.append(dict(category=cat, product=prod, hedge_weight=w,
                                    half_life_bars=br.get("half_life_bars"),
                                    adf_pval_spread=br.get("adf_pval_spread"),
                                    n_coint_vecs=br.get("n_coint_vecs")))
pd.DataFrame(basket_rows).to_csv(OUTPUT_DIR / "johansen_baskets.csv", index=False)

def _json_serialise(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError

with open(OUTPUT_DIR / "trading_playbook.json", "w") as fh:
    json.dump(playbook, fh, indent=2, default=_json_serialise)

print("  product_stats.csv")
print("  cointegration_pairs.csv")
print("  return_correlation_matrix.csv")
print("  resid_corr_<category>.csv  (one per category)")
print("  resid_corr_global.csv")
print("  johansen_baskets.csv")
print("  trading_playbook.json")
print("\nDone.")