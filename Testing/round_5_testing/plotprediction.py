"""
================================================================================
 Prediction Diagnostics — plot_predictions.py
================================================================================
Reads  predictions.csv  (output of the quant pipeline) and produces a
6-panel diagnostic figure saved to  OUTPUT_DIR/prediction_diagnostics.png

Run:
    python plot_predictions.py

Edit the two path constants below to match your setup.
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# EDIT THESE TWO PATHS
PREDICTIONS_CSV = Path(r"./output/predictions.csv")
OUTPUT_DIR      = Path(r"./output")
# ──────────────────────────────────────────────────────────────────────────────

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "font.family":       "monospace",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
})

ACCENT   = "#58a6ff"   # blue
ACCENT2  = "#3fb950"   # green
WARN     = "#f85149"   # red
MID      = "#d29922"   # amber
ALPHA_SC = 0.25        # scatter point alpha

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(PREDICTIONS_CSV)

# drop rows that were not part of any OOF test fold
df = df.dropna(subset=["predicted_return", "actual_return"]).copy()

actual = df["actual_return"].to_numpy()
pred   = df["predicted_return"].to_numpy()

# ── helper: per-product IC ────────────────────────────────────────────────────
def product_ic(group: pd.DataFrame) -> float:
    if group["predicted_return"].std() < 1e-12:
        return np.nan
    r, _ = stats.pearsonr(group["predicted_return"], group["actual_return"])
    return r

ic_by_product = (df.groupby("product")
                   .apply(product_ic)
                   .dropna()
                   .sort_values(ascending=False))

# ── strategy return (long pred>0, short pred<0) ────────────────────────────────
strat_ret = np.sign(pred) * actual
cum_strat = np.cumsum(strat_ret)
cum_bnh   = np.cumsum(actual)        # passive "buy every product" baseline

# ── rolling IC (window over rows sorted by global_t) ─────────────────────────
df_sorted = df.sort_values("global_t").reset_index(drop=True)
window = max(50, len(df_sorted) // 30)
rolling_ic = (df_sorted[["predicted_return", "actual_return"]]
              .rolling(window)
              .corr()
              .unstack()[("predicted_return", "actual_return")]
              .dropna())

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), dpi=130)
fig.suptitle("Prediction Diagnostics", fontsize=16,
             fontweight="bold", color="#e6edf3", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97,
                       top=0.93, bottom=0.06)

ax_scatter = fig.add_subplot(gs[0, 0])
ax_cum     = fig.add_subplot(gs[0, 1:])
ax_ric     = fig.add_subplot(gs[1, :2])
ax_ic_bar  = fig.add_subplot(gs[1, 2])
ax_resid   = fig.add_subplot(gs[2, :2])
ax_qq      = fig.add_subplot(gs[2, 2])

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Scatter: predicted vs actual  (global IC in title)
# ─────────────────────────────────────────────────────────────────────────────
global_ic, pval = stats.pearsonr(pred, actual)

ax_scatter.scatter(pred, actual, s=4, alpha=ALPHA_SC, color=ACCENT, linewidths=0)

# OLS regression line
m, b, *_ = stats.linregress(pred, actual)
x_line = np.linspace(pred.min(), pred.max(), 200)
ax_scatter.plot(x_line, m * x_line + b, color=ACCENT2, lw=1.5, label=f"OLS  slope={m:.2f}")

ax_scatter.axhline(0, color="#30363d", lw=0.8)
ax_scatter.axvline(0, color="#30363d", lw=0.8)
ax_scatter.set_xlabel("Predicted return")
ax_scatter.set_ylabel("Actual return")
ax_scatter.set_title(f"Predicted vs Actual\nIC = {global_ic:+.4f}  (p={pval:.3f})",
                     fontsize=9)
ax_scatter.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Cumulative strategy return vs buy-and-hold baseline
# ─────────────────────────────────────────────────────────────────────────────
ax_cum.plot(cum_strat, color=ACCENT2, lw=1.4, label="Sign strategy (OOF)")
ax_cum.plot(cum_bnh,   color=WARN,    lw=1.0, linestyle="--", alpha=0.7,
            label="Passive baseline")
ax_cum.axhline(0, color="#30363d", lw=0.8)
ax_cum.set_xlabel("OOF observation index (time-ordered)")
ax_cum.set_ylabel("Cumulative log return")
ax_cum.set_title("Cumulative PnL:  Sign Strategy vs Passive Baseline", fontsize=9)
ax_cum.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — Rolling IC over time
# ─────────────────────────────────────────────────────────────────────────────
ax_ric.plot(rolling_ic.values, color=MID, lw=1.2,
            label=f"Rolling IC (window={window} bars)")
ax_ric.axhline(0,           color="#30363d", lw=0.8)
ax_ric.axhline(global_ic,   color=ACCENT,   lw=1.0, linestyle=":",
               label=f"Global IC = {global_ic:+.4f}")
ax_ric.fill_between(range(len(rolling_ic)), rolling_ic.values, 0,
                    where=(rolling_ic.values > 0),
                    color=ACCENT2, alpha=0.15)
ax_ric.fill_between(range(len(rolling_ic)), rolling_ic.values, 0,
                    where=(rolling_ic.values < 0),
                    color=WARN, alpha=0.15)
ax_ric.set_xlabel("OOF observation index (time-ordered)")
ax_ric.set_ylabel("Information Coefficient")
ax_ric.set_title("Rolling IC — Detects Regime Changes / Model Decay", fontsize=9)
ax_ric.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Per-product IC bar chart
# ─────────────────────────────────────────────────────────────────────────────
colors_bar = [ACCENT2 if v >= 0 else WARN for v in ic_by_product.values]
y_pos = range(len(ic_by_product))
ax_ic_bar.barh(list(y_pos), ic_by_product.values,
               color=colors_bar, height=0.7, alpha=0.85)
ax_ic_bar.set_yticks(list(y_pos))
ax_ic_bar.set_yticklabels(
    [p[:18] for p in ic_by_product.index],  # truncate long names
    fontsize=5.5)
ax_ic_bar.axvline(0, color="#30363d", lw=0.8)
ax_ic_bar.set_xlabel("IC")
ax_ic_bar.set_title("Per-Product IC", fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Residual plot (actual − predicted) over time
# ─────────────────────────────────────────────────────────────────────────────
residuals = actual - pred
ax_resid.scatter(range(len(residuals)), residuals,
                 s=3, alpha=ALPHA_SC, color=ACCENT, linewidths=0)
ax_resid.axhline(0,               color=ACCENT2, lw=1.0)
ax_resid.axhline(+2 * residuals.std(), color=WARN, lw=0.8,
                 linestyle="--", label="±2σ")
ax_resid.axhline(-2 * residuals.std(), color=WARN, lw=0.8, linestyle="--")
ax_resid.set_xlabel("OOF observation index (time-ordered)")
ax_resid.set_ylabel("Residual  (actual − predicted)")
ax_resid.set_title("Residuals over Time  — Should Show No Trend", fontsize=9)
ax_resid.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 6 — Q-Q plot of residuals against Normal
# ─────────────────────────────────────────────────────────────────────────────
(osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
ax_qq.scatter(osm, osr, s=4, alpha=0.35, color=ACCENT, linewidths=0)
ax_qq.plot(osm, slope * np.array(osm) + intercept,
           color=ACCENT2, lw=1.4, label="Normal reference")
ax_qq.set_xlabel("Theoretical quantiles")
ax_qq.set_ylabel("Sample quantiles")
ax_qq.set_title("Q-Q Plot of Residuals\n(fat tails → model underestimates extremes)",
                fontsize=9)
ax_qq.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# save
# ─────────────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "prediction_diagnostics.png"
fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close(fig)