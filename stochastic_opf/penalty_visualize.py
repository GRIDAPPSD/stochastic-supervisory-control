"""
Penalty Structure Visualization for OPF Line Thermal Limit Violations.

Penalty applies ONLY to loading above 100% of line rating.
Uses a 10-segment piecewise linear approximation of x² where
x = overload fraction (excess above rating).

Last segment is unbounded — catches violations beyond x_max_fraction.

Generates four diagnostic plots:
  1. Per-line penalty curve (normalized) with segment breakdown + true x²
  2. Heatmap of total penalty vs (# overloaded lines, loading %)
  3. Stacked bar chart comparing named scenarios
  4. Iso-penalty contours — different (N, %) combos yielding equal penalty
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PWL parameters (must match the OPF formulation) ─────────────────────
N_PWL         = 10
X_MAX_FRAC    = 2.0   # Segments span [0, 200% of rating]; last seg unbounded

# Derive segment widths and slopes (x² approximation)
# First (N_PWL-1) segments are bounded, last is unbounded
_SEG_WIDTH    = X_MAX_FRAC / (N_PWL - 1)   # width of each bounded segment
_SLOPES       = [2.0 * (seg + 0.5) * _SEG_WIDTH for seg in range(N_PWL)]
_SEG_WIDTHS   = [_SEG_WIDTH] * (N_PWL - 1) + [np.inf]   # last is unbounded

# Breakpoint loading percentages for annotation (overload fraction → loading %)
_BP_FRACS     = [seg * _SEG_WIDTH for seg in range(N_PWL)]   # left edges of each seg
_BP_LOADING   = [100.0 + f * 100.0 for f in _BP_FRACS]

# Segment colors: green → red across 10 segments
_SEG_COLORS   = [cm.RdYlGn_r(i / (N_PWL - 1)) for i in range(N_PWL)]
_SEG_LABELS   = [
    f"Seg {s} ({_BP_LOADING[s]:.0f}–"
    f"{'∞' if s == N_PWL-1 else f'{_BP_LOADING[s] + _SEG_WIDTH*100:.0f}'}%)"
    f"  slope={_SLOPES[s]:.2f}"
    for s in range(N_PWL)
]


# ── Core helper ──────────────────────────────────────────────────────────
def penalty_single_line(loading_pct: float) -> dict:
    """
    Compute the normalised penalty for one line-phase at a given loading %.

    Penalty is zero for loading <= 100%. Above 100%, the excess (x) is
    distributed across N_PWL piecewise segments approximating x².

    Parameters
    ----------
    loading_pct : float
        Line loading as a percentage of its rating (e.g. 130 means 130%).

    Returns
    -------
    dict with keys:
        total       – total normalised penalty
        true_x2     – true x² value for comparison
        seg_values  – excess allocated to each segment (normalised)
        seg_costs   – penalty contribution from each segment
    """
    x = max(0.0, (loading_pct - 100.0) / 100.0)   # overload fraction
    remaining = x
    seg_values = []
    seg_costs  = []

    for slope, width in zip(_SLOPES, _SEG_WIDTHS):
        seg = min(remaining, width)
        seg_values.append(seg)
        seg_costs.append(slope * seg)
        remaining = max(0.0, remaining - (width if not np.isinf(width) else remaining))

    return dict(
        total      = sum(seg_costs),
        true_x2    = x ** 2,
        seg_values = seg_values,
        seg_costs  = seg_costs,
    )


def total_penalty(n_lines: float, loading_pct: float) -> float:
    return n_lines * penalty_single_line(loading_pct)["total"]


# ── Scenarios ────────────────────────────────────────────────────────────
SCENARIOS = [
    (100, 110, "A: 100 lines @ 110 %"),
    (20,  250, "B:  20 lines @ 250 %"),
    (50,  130, "C:  50 lines @ 130 %"),
    (10,  200, "D:  10 lines @ 200 %"),
    (5,   300, "E:   5 lines @ 300 %"),
    (200, 105, "F: 200 lines @ 105 %"),
    (1,   160, "G:   1 line  @ 160 %"),
    (50,  150, "H:  50 lines @ 150 %"),
]


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 1 — Per-line penalty curve with segment shading + true x²
# ═══════════════════════════════════════════════════════════════════════
def plot_penalty_curve(ax):
    loading  = np.linspace(80, 320, 3000)
    results  = [penalty_single_line(l) for l in loading]
    segs     = np.array([r["seg_costs"]  for r in results])   # (N_pts, N_PWL)
    totals   = np.array([r["total"]      for r in results])
    true_x2  = np.array([r["true_x2"]   for r in results])
    marginal = np.gradient(totals, loading)

    # Stacked fill per segment
    cum = np.zeros_like(loading)
    for s in range(N_PWL):
        ax.fill_between(
            loading, cum, cum + segs[:, s],
            color=_SEG_COLORS[s], alpha=0.40,
            label=_SEG_LABELS[s] if s % 2 == 0 else "_nolegend_",  # thin legend
        )
        cum += segs[:, s]

    ax.plot(loading, totals,  "k-",  lw=2.2, label="PWL total penalty")
    ax.plot(loading, true_x2, "b--", lw=1.8, label="True x² (reference)")

    # 100% boundary
    ax.axvline(100, color="black", ls="-", lw=1.5, alpha=0.8)
    ax.annotate(
        "No penalty\nbelow 100 %", xy=(97, totals.max() * 0.50),
        fontsize=9, ha="right", color="green", fontweight="bold",
    )

    # Segment boundary lines
    for edge in _BP_LOADING[1:]:
        if edge > 320:
            break
        ax.axvline(edge, color="grey", ls=":", alpha=0.5)

    # Marginal slope on secondary axis
    ax2 = ax.twinx()
    ax2.plot(loading, marginal, color="navy", ls="--", lw=1.3, alpha=0.7,
             label="Marginal slope")
    ax2.set_ylabel("Marginal slope  (dPenalty / d%)", fontsize=10, color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax2.set_ylim(bottom=0)

    ax.set_xlabel("Line loading (%)", fontsize=12)
    ax.set_ylabel("Normalised penalty  (per unit of S_max)", fontsize=12)
    ax.set_title(
        f"x² PWL Penalty Curve — {N_PWL} Segments, last unbounded\n"
        f"(x_max = {int(X_MAX_FRAC*100)}% overload, seg_width = {_SEG_WIDTH*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(80, 320)


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 2 — Heatmap (# lines × loading %)
# ═══════════════════════════════════════════════════════════════════════
def plot_heatmap(ax):
    nl_range = np.arange(1, 201)
    ld_range = np.arange(100, 321)

    norm_pen = np.array([penalty_single_line(l)["total"] for l in ld_range])
    NL, _    = np.meshgrid(nl_range, ld_range)
    Z        = NL * norm_pen[:, None]

    pcm = ax.pcolormesh(
        nl_range, ld_range, np.log10(np.clip(Z, 1e-12, None)),
        cmap="RdYlGn_r", shading="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("log₁₀( Total Raw Penalty )", fontsize=10)

    # Segment boundaries as horizontal lines
    for edge in _BP_LOADING[1:]:
        if 100 < edge <= 320:
            ax.axhline(edge, color="white", ls=":", lw=0.8, alpha=0.6)

    for nl, ld, lbl in SCENARIOS:
        if ld < 100 or nl > 200 or ld > 320:
            continue
        letter = lbl.split(":")[0]
        ax.plot(nl, ld, "k*", ms=12, zorder=5)
        ax.annotate(letter, (nl, ld), textcoords="offset points",
                    xytext=(6, 4), fontsize=10, fontweight="bold", color="white")

    ax.set_xlabel("Number of overloaded line-phases", fontsize=12)
    ax.set_ylabel("Line loading (%)", fontsize=12)
    ax.set_title("Total Raw Penalty Heatmap  (loading > 100 %)", fontsize=12,
                 fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 3 — Horizontal stacked bar chart of scenarios
# ═══════════════════════════════════════════════════════════════════════
def plot_scenario_bars(ax):
    labels, totals, breakdowns = [], [], []
    for nl, ld, lbl in SCENARIOS:
        res = penalty_single_line(ld)
        tot = nl * res["total"]
        bd  = [nl * c for c in res["seg_costs"]]
        labels.append(lbl)
        totals.append(tot)
        breakdowns.append(bd)

    breakdowns = np.array(breakdowns)
    y = np.arange(len(labels))

    left = np.zeros(len(labels))
    for s in range(N_PWL):
        ax.barh(
            y, breakdowns[:, s], left=left,
            color=_SEG_COLORS[s], edgecolor="white", lw=0.5,
            label=f"Seg {s}" if s % 2 == 0 else "_nolegend_",
        )
        left += breakdowns[:, s]

    for i, tot in enumerate(totals):
        ax.text(tot * 1.15, i, f"{tot:.2f}", va="center", fontsize=8.5,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("Total raw penalty (no weight multiplier)", fontsize=12)
    ax.set_title("Scenario Comparison — Segment Breakdown", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="x", which="both")
    ax.set_xlim(right=max(totals) * 8)


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 4 — Iso-penalty contours
# ═══════════════════════════════════════════════════════════════════════
def plot_iso_penalty(ax):
    thresholds = [1, 10, 50, 100, 500, 1000, 5000]
    cmap_iso   = plt.cm.RdYlGn_r
    norm_iso   = plt.Normalize(vmin=0, vmax=len(thresholds) - 1)

    for idx, thresh in enumerate(thresholds):
        nls, lds = [], []
        for ld in np.linspace(100.01, 320, 3000):
            p = penalty_single_line(ld)["total"]
            if p <= 0:
                continue
            nl = thresh / p
            if 0 < nl <= 300:
                nls.append(nl)
                lds.append(ld)
        if nls:
            ax.plot(nls, lds, color=cmap_iso(norm_iso(idx)), lw=2.2,
                    label=f"Penalty = {thresh:,.0f}")

    for nl, ld, lbl in SCENARIOS[:4]:
        letter = lbl.split(":")[0]
        ax.plot(nl, ld, "k*", ms=12, zorder=5)
        ax.annotate(letter, (nl, ld), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of overloaded line-phases", fontsize=12)
    ax.set_ylabel("Line loading (%)", fontsize=12)
    ax.set_title(
        "Iso-Penalty Contours\n(equal total raw penalty combinations)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 300)
    ax.set_ylim(100, 320)


# ═══════════════════════════════════════════════════════════════════════
#  BONUS PLOT — PWL vs true x² error
# ═══════════════════════════════════════════════════════════════════════
def plot_pwl_vs_true_x2():
    """Shows the PWL approximation quality vs true x²."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    loading  = np.linspace(100, 320, 2000)
    results  = [penalty_single_line(l) for l in loading]
    pwl_vals = np.array([r["total"]   for r in results])
    x2_vals  = np.array([r["true_x2"] for r in results])
    error    = pwl_vals - x2_vals
    rel_err  = np.where(x2_vals > 1e-12, np.abs(error) / x2_vals * 100, 0.0)

    # Left: overlay
    ax = axes[0]
    ax.plot(loading, x2_vals,  "b-",  lw=2.5, label="True x²")
    ax.plot(loading, pwl_vals, "r--", lw=2.0, label=f"PWL ({N_PWL} segs)")
    ax.fill_between(loading, x2_vals, pwl_vals,
                    alpha=0.25, color="orange", label="Approximation error")

    # Segment boundary verticals
    for edge in _BP_LOADING[1:]:
        if 100 < edge <= 320:
            ax.axvline(edge, color="grey", ls=":", lw=0.7, alpha=0.6)

    # Annotate each segment's slope
    for s in range(N_PWL - 1):
        x_mid = _BP_LOADING[s] + _SEG_WIDTH * 50
        if x_mid > 320:
            break
        y_mid = penalty_single_line(x_mid)["total"] * 0.5
        ax.text(x_mid, y_mid, f"m={_SLOPES[s]:.2f}",
                fontsize=7, ha="center", color="darkred", rotation=70)

    ax.set_xlabel("Line loading (%)", fontsize=12)
    ax.set_ylabel("Penalty value", fontsize=12)
    ax.set_title(f"PWL vs True x²  ({N_PWL} segments, last unbounded)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 320)

    # Right: relative error
    ax2 = axes[1]
    ax2.plot(loading, rel_err, "m-", lw=2)
    ax2.fill_between(loading, 0, rel_err, alpha=0.2, color="purple")
    for edge in _BP_LOADING[1:]:
        if 100 < edge <= 320:
            ax2.axvline(edge, color="grey", ls=":", lw=0.7, alpha=0.6)
    ax2.set_xlabel("Line loading (%)", fontsize=12)
    ax2.set_ylabel("Relative error  |PWL − x²| / x²  (%)", fontsize=12)
    ax2.set_title("PWL Approximation Relative Error", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(100, 320)
    ax2.set_ylim(bottom=0)

    plt.suptitle(
        f"x² Approximation Quality  —  seg_width = {_SEG_WIDTH*100:.1f}%,  "
        f"x_max = {int(X_MAX_FRAC*100)}%,  last segment unbounded",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIR / "pwl_vs_true_x2.png", dpi=150, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  Console summary table
# ═══════════════════════════════════════════════════════════════════════
def print_summary_table():
    print(f"\n{'='*90}")
    print(f"  PWL x² Penalty — {N_PWL} segments, seg_width={_SEG_WIDTH*100:.2f}%, "
          f"x_max={int(X_MAX_FRAC*100)}%, last segment unbounded")
    print(f"  Slopes: {[f'{s:.3f}' for s in _SLOPES]}")
    print(f"{'='*90}")
    hdr = (f"{'Scenario':<28} {'#Lines':>7} {'Loading':>9} "
           f"{'PWL Total':>12} {'True x²·N':>12} {'Error %':>9}")
    print(hdr)
    print("-" * 90)
    for nl, ld, lbl in SCENARIOS:
        res     = penalty_single_line(ld)
        pwl_tot = nl * res["total"]
        x2_tot  = nl * res["true_x2"]
        err_pct = abs(pwl_tot - x2_tot) / max(x2_tot, 1e-12) * 100
        print(f"{lbl:<28} {nl:>7} {ld:>8.0f}% "
              f"{pwl_tot:>12.4f} {x2_tot:>12.4f} {err_pct:>8.2f}%")
    print("=" * 90)

    print(f"\n── Segment slopes and boundaries ──")
    print(f"  {'Seg':>4} {'Loading range':>20} {'Slope':>10} {'Width (frac)':>14}")
    for s in range(N_PWL):
        lo   = _BP_LOADING[s]
        hi   = ("∞" if s == N_PWL - 1
                else f"{_BP_LOADING[s] + _SEG_WIDTH * 100:.1f}%")
        print(f"  {s:>4}   {lo:>6.1f}% → {hi:>8}   {_SLOPES[s]:>10.4f}   "
              f"{'unbounded' if s == N_PWL-1 else f'{_SEG_WIDTH:.4f}'}")


# ═══════════════════════════════════════════════════════════════════════
#  Threshold explorer
# ═══════════════════════════════════════════════════════════════════════
def plot_threshold_explorer():
    fig, ax = plt.subplots(figsize=(10, 6))
    budgets  = [5, 50, 500, 5000, 50000]
    cmap_t   = plt.cm.viridis
    norm_t   = plt.Normalize(0, len(budgets) - 1)

    for idx, budget in enumerate(budgets):
        n_lines    = np.arange(1, 301)
        max_loading = []
        for nl in n_lines:
            lo, hi = 100.0, 600.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if total_penalty(nl, mid) <= budget:
                    lo = mid
                else:
                    hi = mid
            max_loading.append(lo)
        ax.plot(n_lines, max_loading, color=cmap_t(norm_t(idx)), lw=2,
                label=f"Budget = {budget:,.0f}")

    ax.axhline(100, color="black", ls=":", lw=1, alpha=0.5)
    ax.text(5, 101, "100 % = at rating (no penalty)", fontsize=8, color="grey")

    ax.set_xlabel("Number of overloaded line-phases", fontsize=12)
    ax.set_ylabel("Max tolerable loading (%)", fontsize=12)
    ax.set_title(
        "Penalty Budget Explorer — x² PWL formulation\n"
        "Maximum uniform loading % before exceeding budget",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 300)
    ax.set_ylim(100, 400)
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIR / "penalty_threshold_explorer.png", dpi=150, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_penalty_curve(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_heatmap(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    plot_scenario_bars(ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    plot_iso_penalty(ax4)

    fig.suptitle(
        f"Line Thermal-Limit Penalty — x² PWL Approximation  "
        f"({N_PWL} segments, last unbounded)\n"
        f"Slopes: {[round(s, 3) for s in _SLOPES]}   |   "
        f"Seg width: {_SEG_WIDTH*100:.1f}%   |   "
        f"x_max: {int(X_MAX_FRAC*100)}%",
        fontsize=13, fontweight="bold", y=0.999,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(_OUTPUT_DIR / "penalty_structure_overview.png", dpi=150, bbox_inches="tight")
    plt.show()

    print_summary_table()
    plot_threshold_explorer()
    plot_pwl_vs_true_x2()


if __name__ == "__main__":
    main()