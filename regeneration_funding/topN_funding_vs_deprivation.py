"""
Regeneration funding per head vs household deprivation, English local authorities
(2024 boundaries, as in the funding model). Islands/Highlands outliers are excluded
(the spreadsheet's "exc. outliers" column is blank for them, so load_funding drops
them). The top 10 and the next 10 (ranks 11-20) by funding per head are highlighted;
all top 10 are labelled.

x = household deprivation (Census 2021, mean dimensions 0-4; higher = more deprived)
y = regeneration funding per head (2025 prices)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent          # regeneration_funding/
MID = HERE / "midlands"                          # loaders + data live here
C_TOP10, C_TOP20, C_REST = "#B3330A", "#F3B089", "#dcdcdc"   # dark -> light -> grey

# per-authority label placement: (dx, dy in points, h-align, draw short leader line)
# Newark sits on the LESS-deprived side, so its label leans left; all others lean right.
LABELS = {
    "Newark and Sherwood": (-9, 0, "right", False),
    "Darlington":          (9, -1, "left", False),
    "Lancaster":           (9, 8, "left", True),
    "Cumberland":          (9, -9, "left", True),
    "Lincoln":             (9, 0, "left", False),
    "Boston":              (9, -10, "left", True),
    "Ashfield":            (9, 10, "left", True),
    "Hartlepool":          (9, 0, "left", False),
    "Great Yarmouth":      (9, 0, "left", False),
    "Blackpool":           (9, 0, "left", False),
}
# bottom-5 (least funded) labels, stacked above the £0 dots
BOTTOM_LABELS = {
    "City of London":         (0, 13, "center"),
    "Richmond upon Thames":   (0, 34, "center"),
    "Kingston upon Thames":   (0, 13, "center"),
    "Hammersmith and Fulham": (0, 13, "center"),
    "Kensington and Chelsea": (0, 34, "center"),
}


def main():
    spec = importlib.util.spec_from_file_location("rm", str(MID / "recreate_core_analysis.py"))
    rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
    f = rm.load_funding()
    dep = rm.load_deprivation()[["lad", "deprivation_mean"]]
    e = (f[f.lad.str.startswith("E")].merge(dep, on="lad", how="left")
         .dropna(subset=["funding_pc", "deprivation_mean"])
         .sort_values("funding_pc", ascending=False).reset_index(drop=True))
    e["fund_rank"] = np.arange(1, len(e) + 1)
    med = e.deprivation_mean.median()
    print("Top 20 by funding per head (England, 2024 LADs, outliers excluded):")
    print(e.head(20)[["fund_rank", "lad_nm", "funding_pc", "deprivation_mean"]]
          .to_string(index=False, formatters={"funding_pc": lambda v: f"£{v:,.0f}",
                                               "deprivation_mean": lambda v: f"{v:.3f}"}))

    top10, top20 = e.head(10), e.iloc[10:20]
    rest = e.iloc[20:]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter(rest.deprivation_mean, rest.funding_pc, s=14, color=C_REST, zorder=1)
    ax.scatter(top20.deprivation_mean, top20.funding_pc, s=60, color=C_TOP20,
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.scatter(top10.deprivation_mean, top10.funding_pc, s=85, color=C_TOP10,
               edgecolor="white", linewidth=0.5, zorder=3)
    bottom5 = e.nsmallest(5, "funding_pc")
    ax.scatter(bottom5.deprivation_mean, bottom5.funding_pc, s=60, color="#111111",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(med, color="#555", ls=(0, (6, 4)), lw=1.8, zorder=1.5)
    ax.text(med, ax.get_ylim()[1], "  England median deprivation", va="top", ha="left",
            fontsize=9, color="#555", fontweight="bold")

    # label every top-10 authority, close to its dot (short leader line only if nudged)
    for _, r in top10.iterrows():
        dx, dy, ha, line = LABELS.get(r.lad_nm, (9, 0, "left", False))
        ax.annotate(r.lad_nm, (r.deprivation_mean, r.funding_pc), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8,
                    color="#7a2d12", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#c8c8c8", lw=0.6) if line else None)
    for _, r in bottom5.iterrows():
        dx, dy, ha = BOTTOM_LABELS.get(r.lad_nm, (0, 13, "center"))
        ax.annotate(r.lad_nm, (r.deprivation_mean, r.funding_pc), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=7.5,
                    color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6))
    ax.set_xlim(right=e.deprivation_mean.max() + 0.05)

    ax.set_xlabel("Household deprivation (Census 2021, mean dimensions 0–4; higher = more deprived)")
    ax.set_ylabel("Regeneration funding per head (£, 2025 prices)")
    ax.set_title("Regeneration funding per head versus household deprivation",
                 fontweight="bold", fontsize=13, pad=24)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TOP10, markeredgecolor="#7a2306", markersize=10, label="Top 10 funded"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TOP20, markeredgecolor="#c98a63", markersize=10, label="Ranks 11–20"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=C_REST, markeredgecolor="#b0b0b0", markersize=9, label="Other England LADs"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markersize=9, label="Bottom 5 (least funded)")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "topN_funding_vs_deprivation.png", dpi=170, bbox_inches="tight")
    print("saved -> regeneration_funding/topN_funding_vs_deprivation.png")


if __name__ == "__main__":
    main()
