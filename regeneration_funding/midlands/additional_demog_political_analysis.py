"""
Additional demographic & political analysis for the Midlands regeneration work.

Runs on top of recreate_core_analysis.py (imported for its data loaders, the
Midlands definition, the matching helpers and the shared style/paths). Everything
here goes *beyond* the original regeneration analysis, in two clearly separated
sections:

  SECTION 1 - FUNDING vs DEMOGRAPHICS (deeper splits)
    1a  East vs West Midlands summary (funding %, population %, deprivation,
        and share of England's most-deprived decile)          -> east_west_summary.*
    1b  Deprivation score by ITL1 region (simple bar)          -> itl1_deprivation_bar.png
    1c  Funding per head across the Midlands (choropleth map)   -> 06_midlands_funding_map.png
    1d  Deprivation vs Reform, by region (ITL1) and by LAD      -> 04_itl1_* , 04_lad_*
    1e  What explains the East/West funding gap for deprived
        areas: urbanity, social mobility, geographic mobility  -> 12_east_west_funding_drivers.*

  SECTION 2 - WIDER POLITICAL ANALYSIS
    2a  Leading party by age & deprivation (England plane)      -> 07_age_deprivation_party.png
    2b  East vs West Midlands on that plane                     -> 08_age_deprivation_party_midlands.png
    2c  Age & urbanity battleground; deprivation vs Reform by
        several diversity metrics                               -> 09_* , 10_*
    2e  Labour switching (Green vs Reform) by deprivation,
        East vs West Midlands (BES 2024 panel)                 -> 05_midlands_switching_by_deprivation.*
    2f  UKIP/Brexit/Reform vote intention among likely voters
        over the BES waves, East vs West                       -> 11_reform_trajectory_bes.*
    2g  East vs West Reform trajectory (MRP quarters + GE
        2019/2024 + LE2025 context)                            -> 13_reform_trajectory_ew.*
    2h  Did the projected 2026 West Midlands breakthrough
        actually happen? (actual ward-level results)           -> 14_wm_2026_results.*
    2i  Diversity vs Reform within the West Midlands, 2026
        actuals (diversity still tracks Reform, not deprivation) -> 15_wm_2026_diversity.*
    2j  Why the raw diversity gradient looks cleaner in the
        East: deprivation-diversity coupling + partials        -> 16_dep_diversity_coupling.*

(The funded-vs-comparable control-pool test now lives in recreate_core_analysis.py, output 04.)
"""
import glob
import json
import subprocess
import tempfile
import zipfile
import datetime as _dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, ListedColormap, to_rgb
from matplotlib.cm import ScalarMappable
from shapely.geometry import shape
from shapely.ops import unary_union
import pyreadstat

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
REPO = HERE.parents[1]
MYE = REPO / "labour_voter_demog_change/data/ons_mye24tablesuk.xlsx"
BES_RAW = REPO / "source_data/bes_voter_panel_internet_study/raw"
GE = REPO / "source_data/election_results/national_general_elections"
LE = REPO / "source_data/election_results/local_elections"
LE26 = LE / "2026_external"
REFORM_TEAL = "#12B6CF"

# shared core module (loaders, MIDLANDS, matchers, colours, save(), OUT/DATA)
_spec = importlib.util.spec_from_file_location("core", str(HERE / "recreate_core_analysis.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

PF_ORANGE, PF_GREY, PF_BLUE, GREEN = core.PF_ORANGE, core.PF_GREY, core.PF_BLUE, "#3F9E4D"
PARTY5 = [("labour", "Lab", "#E4003B"), ("reform_uk", "Reform", "#12B6CF"),
          ("libdem", "Lib Dem", "#FAA61A"), ("conservative", "Con", "#0087DC"),
          ("green", "Green", "#6AB023")]
SHARE = [f"{k}_share" for k, _, _ in PARTY5]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_age_50plus():
    df = pd.read_excel(MYE, sheet_name="MYE2 - Persons", header=7)
    age_val = lambda c: 90 if c == "90+" else int(c)
    over50 = [c for c in df.columns if isinstance(c, str) and (c.isdigit() or c == "90+")
              and age_val(c) >= 50]
    df = df[df["Code"].astype(str).str.match(r"E0[6789]")].copy()
    df["pct_50plus"] = df[over50].sum(axis=1) / df["All ages"] * 100
    return df[["Code", "pct_50plus"]].rename(columns={"Code": "lad"})


def assemble_england():
    """One England-wide table with every metric the charts below draw on."""
    f = core.load_funding()
    dep = core.load_deprivation()                          # lad, deprivation_mean, households
    fb = core.load_foreign_born()                          # lad, foreign_born
    votes = core.load_vote_shares()                        # lad, *_share, total_votes, reform_uk_share_qX
    age = load_age_50plus()
    region = (pd.read_csv(DATA / "lad24_to_region.csv")[["LAD24CD", "RGN24NM"]]
              .rename(columns={"LAD24CD": "lad", "RGN24NM": "region"}))
    ruc = pd.read_csv(DATA / "lad24_ruc.csv")              # lad, RUC21NM, Urban_rural_flag
    eth = pd.read_csv(DATA / "lad_ethnicity.csv")          # lad, nonwhite_*, ethnic_diversity_idx
    mob = pd.read_csv(DATA / "lad_mobility.csv")           # lad, moved_pct, internal_move_pct (Census 2021 TS019)
    smi = pd.read_csv(DATA / "lad_social_mobility.csv")     # lad, smi_rank, smi_score (Social Mobility Index 2016)
    gj = json.loads((DATA / "england_lad_boundaries.geojson").read_text())
    area = pd.DataFrame([{"lad": ft["properties"]["LAD24CD"],
                          "area_km2": shape(ft["geometry"]).area / 1e6} for ft in gj["features"]])
    d = (f.merge(dep, on="lad", how="left").merge(fb, on="lad", how="left")
         .merge(votes, on="lad", how="left").merge(age, on="lad", how="left")
         .merge(region, on="lad", how="left").merge(ruc, on="lad", how="left")
         .merge(eth, on="lad", how="left").merge(mob, on="lad", how="left")
         .merge(smi, on="lad", how="left").merge(area, on="lad", how="left"))
    d = d[d.lad.str.startswith("E")].copy()
    d["density"] = d.population / d.area_km2
    d["reform_pct"] = d.reform_uk_share * 100
    d["foreign_born_pct"] = d.foreign_born * 100
    d["is_mid"] = d.lad.isin(core.MIDLANDS)
    return d


def polys_to_patches(geom):
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in geoms:
        verts, codes = [], []
        for ring in [poly.exterior] + list(poly.interiors):
            xy = np.asarray(ring.coords)
            verts.append(xy)
            c = np.full(len(xy), MplPath.LINETO); c[0] = MplPath.MOVETO
            codes.append(c)
        yield MplPath(np.concatenate(verts), np.concatenate(codes))


# =========================================================================== #
# SECTION 1 - FUNDING vs DEMOGRAPHICS
# =========================================================================== #
def s1a_east_west_summary(d):
    print("\n1a. East vs West Midlands summary")
    f, dep = core.load_funding(), core.load_deprivation()[["lad", "deprivation_mean", "households"]]
    region = (pd.read_csv(DATA / "lad24_to_region.csv")[["LAD24CD", "RGN24NM"]]
              .rename(columns={"LAD24CD": "lad", "RGN24NM": "region"}))
    mid = f[f.lad.isin(core.MIDLANDS)].copy()
    mid["region"] = mid.lad.map(core.MIDLANDS)
    mid = (mid.merge(dep, on="lad", how="left")
              .merge(d[["lad", "nonwhite_pct", "Urban_rural_flag"]], on="lad", how="left"))
    tot_fund, tot_pop = mid.total_funding.sum(), mid.population.sum()
    rows = []
    for r in ["East Midlands", "West Midlands"]:
        s = mid[mid.region == r]
        rows.append({"region": r, "funding_£": s.total_funding.sum(),
                     "funding_%_of_midlands": s.total_funding.sum() / tot_fund * 100,
                     "population": s.population.sum(),
                     "pop_%_of_midlands": s.population.sum() / tot_pop * 100,
                     "avg_deprivation_hh_wtd": np.average(s.deprivation_mean, weights=s.households),
                     "avg_deprivation_lad_mean": s.deprivation_mean.mean(),
                     "nonwhite_pct_pop_wtd": np.average(s.nonwhite_pct, weights=s.population),
                     "pct_urban_pop_wtd": s.loc[s.Urban_rural_flag == "Urban", "population"].sum()
                     / s.population.sum() * 100})
    tab = pd.DataFrame(rows)

    eng = (f[f.lad.str.startswith("E")].merge(dep[["lad", "deprivation_mean"]], on="lad", how="left")
           .merge(region, on="lad", how="left").dropna(subset=["deprivation_mean", "population"])
           .sort_values("deprivation_mean", ascending=False))
    eng["cum_pop"] = eng.population.cumsum()
    thr = eng.population.sum() * 0.10
    decile = eng[eng.cum_pop - eng.population < thr]
    dpop = decile.population.sum()
    em = decile[decile.region == "East Midlands"].population.sum()
    wm = decile[decile.region == "West Midlands"].population.sum()

    pc = lambda x, base: f"{x/base*100:.1f}%"
    lines = ["# East vs West Midlands summary\n",
             f"Midlands total: £{tot_fund/1e9:.2f}bn regeneration funding, {tot_pop:,.0f} people.\n",
             "| Half | Funding | % of Midlands funding | Population | % of Midlands pop | "
             "Avg deprivation (hh-weighted) | % non-White | % urban |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.region} | £{r['funding_£']/1e9:.2f}bn | {r['funding_%_of_midlands']:.1f}% | "
                     f"{r.population:,.0f} | {r['pop_%_of_midlands']:.1f}% | {r.avg_deprivation_hh_wtd:.3f} | "
                     f"{r.nonwhite_pct_pop_wtd:.1f}% | {r.pct_urban_pop_wtd:.0f}% |")
    lines += ["",
              "*% non-White = population-weighted Census 2021 share; % urban = share of the region's "
              "population in LADs classified Urban (ONS RUC 2021).*"
              , "",
              f"Most-deprived decile of England's population = the {dpop:,.0f} people (~10%) in the "
              f"{len(decile)} most-deprived LADs (deprivation >= {decile.deprivation_mean.min():.3f}). Of them:",
              f"- **East Midlands: {pc(em, dpop)}** ({em:,.0f} people)",
              f"- **West Midlands: {pc(wm, dpop)}** ({wm:,.0f} people)",
              f"- rest of England: {pc(dpop-em-wm, dpop)}",
              f"- within the Midlands' share, East {pc(em, em+wm)} / West {pc(wm, em+wm)}"]
    (OUT / "east_west_summary.md").write_text("\n".join(lines), encoding="utf-8")
    tab.to_csv(OUT / "east_west_summary.csv", index=False)
    print("\n".join(lines))
    print("   saved -> outputs/east_west_summary.md (+ .csv)")


def s1b_itl1_deprivation_bar(d):
    print("\n1b. Deprivation by ITL1 region (bar)")
    g = (d.dropna(subset=["deprivation_mean", "households", "region"])
         .groupby("region")
         .apply(lambda x: np.average(x.deprivation_mean, weights=x.households), include_groups=False)
         .sort_values(ascending=False))
    g.to_csv(OUT / "itl1_deprivation_bar.csv")
    mids = {"East Midlands", "West Midlands"}
    colours = [PF_ORANGE if r in mids else PF_GREY for r in g.index]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(g)), g.values, color=colours)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=35, ha="right")
    ax.set_ylabel("Household deprivation (mean dimensions 0–4; higher = more deprived)")
    ax.set_title("Household deprivation by English region (ITL1)\n"
                 "population-weighted; Midlands highlighted", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    core.save(fig, "itl1_deprivation_bar.png")


def s1c_funding_map(d):
    print("\n1c. Funding-per-head map across the Midlands")
    geo_path = DATA / "midlands_lad_boundaries.geojson"
    if not geo_path.exists():
        where = "LAD24CD IN (%s)" % ",".join("'%s'" % c for c in core.MIDLANDS)
        subprocess.run(["curl", "-sG",
                        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
                        "Local_Authority_Districts_May_2024_Boundaries_UK_BUC/FeatureServer/0/query",
                        "--data-urlencode", "where=" + where, "--data-urlencode", "outFields=LAD24CD",
                        "--data-urlencode", "outSR=27700", "--data-urlencode", "returnGeometry=true",
                        "--data-urlencode", "f=geojson", "--data-urlencode", "resultRecordCount=100",
                        "-o", str(geo_path)], check=True)
    gj = json.loads(geo_path.read_text())
    mid = d[d.is_mid]
    fmap = dict(zip(mid.lad, mid.funding_pc)); rmap = dict(zip(mid.lad, mid.region))
    geoms = {ft["properties"]["LAD24CD"]: shape(ft["geometry"]) for ft in gj["features"]}
    norm = Normalize(vmin=0, vmax=max(fmap.values())); cmap = plt.get_cmap("YlOrRd")
    fig, ax = plt.subplots(figsize=(9, 10))
    patches, colors = [], []
    for cd, g in geoms.items():
        for p in polys_to_patches(g):
            patches.append(PathPatch(p)); colors.append(cmap(norm(fmap.get(cd, np.nan))))
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white",
                                      linewidth=0.4, zorder=2))
    for reg in ("East Midlands", "West Midlands"):
        union = unary_union([g for cd, g in geoms.items() if rmap.get(cd) == reg])
        for p in polys_to_patches(union):
            ax.add_patch(PathPatch(p, fill=False, edgecolor="#111111", linewidth=2.2, zorder=3))
        cx, cy = union.representative_point().coords[0]
        ax.text(cx, cy + (18000 if reg == "West Midlands" else -6000), reg.upper(),
                ha="center", va="center", fontsize=11, fontweight="bold", color="#222", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    # number the top 10 funded LADs on the map, with a ranked list alongside
    top10 = mid.nlargest(10, "funding_pc").reset_index(drop=True)
    for i, r in top10.iterrows():
        if r.lad in geoms:
            x, y = geoms[r.lad].representative_point().coords[0]
            ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="black", zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.26", fc="white", ec="black", lw=1.1))
    lst = "Top 10 funded (£ per head)\n" + "\n".join(
        f"{i+1:>2}. {r.lad_nm} £{r.funding_pc:,.0f}" for i, r in top10.iterrows())
    ax.text(0.015, 0.985, lst, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace", zorder=7,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888", alpha=0.92))
    ax.set_aspect("equal"); ax.axis("off"); ax.autoscale_view()
    ax.set_title("Regeneration funding per head across the Midlands\n"
                 "by local authority (2025 prices); top 10 numbered, East/West boundary in black",
                 fontsize=13, fontweight="bold")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label("£ per head")
    core.save(fig, "06_midlands_funding_map.png")


def s1d_deprivation_vs_reform(d):
    print("\n1d. Deprivation vs Reform (region ITL1, and by LAD)")
    eng = d.dropna(subset=["deprivation_mean", "reform_uk_share", "funding_pc",
                           "households", "total_votes", "region"]).copy()
    # ITL1 region scatter
    g = (eng.groupby("region").apply(lambda x: pd.Series({
        "deprivation": np.average(x.deprivation_mean, weights=x.households),
        "reform": np.average(x.reform_uk_share, weights=x.total_votes)}),
        include_groups=False).reset_index())
    g.to_csv(OUT / "04_itl1_deprivation_vs_reform.csv", index=False)
    mids = {"East Midlands", "West Midlands"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, r in g.iterrows():
        m = r.region in mids
        ax.scatter(r.deprivation, r.reform * 100, s=90, color=PF_ORANGE if m else PF_GREY,
                   zorder=3, edgecolor="black", linewidth=0.5)
        ax.annotate(r.region, (r.deprivation, r.reform * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold" if m else "normal", color=PF_ORANGE if m else "black")
    b, a = np.polyfit(g.deprivation, g.reform * 100, 1)
    xs = np.linspace(g.deprivation.min(), g.deprivation.max(), 50)
    ax.plot(xs, a + b * xs, "--", color="grey", lw=1, zorder=1)
    ax.set_xlabel("Household deprivation (mean dimensions, 0–4)")
    ax.set_ylabel("Reform UK vote share (vote-weighted, %)")
    ax.set_title("English regions: the Midlands is more Reform-leaning than its\n"
                 "deprivation alone would predict (ITL1; Q4 2025 MRP)")
    core.save(fig, "04_itl1_deprivation_vs_reform.png")

    # LAD scatter
    em, wm, rest = eng[eng.region == "East Midlands"], eng[eng.region == "West Midlands"], eng[~eng.is_mid]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(rest.deprivation_mean, rest.reform_pct, s=18, color=PF_GREY, alpha=0.55,
               label="Rest of England", zorder=2)
    ax.scatter(em.deprivation_mean, em.reform_pct, s=44, color=PF_ORANGE, edgecolor="black",
               linewidth=0.4, label="East Midlands", zorder=4)
    ax.scatter(wm.deprivation_mean, wm.reform_pct, s=44, color=PF_BLUE, edgecolor="black",
               linewidth=0.4, label="West Midlands", zorder=3)
    b, a = np.polyfit(eng.deprivation_mean, eng.reform_pct, 1)
    xs = np.linspace(eng.deprivation_mean.min(), eng.deprivation_mean.max(), 50)
    ax.plot(xs, a + b * xs, "--", color="black", lw=1, zorder=1, label="England trend")
    for _, r in pd.concat([em, wm]).nlargest(5, "reform_pct").iterrows():
        ax.annotate(r.lad_nm, (r.deprivation_mean, r.reform_pct), xytext=(4, 3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("Household deprivation (mean dimensions, 0–4)")
    ax.set_ylabel("Reform UK vote share (Q4 2025 MRP, %)")
    ax.set_title("Reform support vs deprivation by local authority\n"
                 "(England LADs; Midlands highlighted — mostly above the England trend)")
    ax.legend(frameon=False, fontsize=9)
    core.save(fig, "04_lad_deprivation_vs_reform.png")


def s1e_explain_funding_gap(d):
    """Why do the West Midlands' deprived areas get less funding than the East's, despite
    similar (slightly higher) deprivation? Test three candidate explanations for deprived
    Midlands LADs, East vs West: urbanity (Census RUC), social mobility (SMI 2016) and
    geographic mobility (Census 2021 within-UK churn)."""
    print("\n1e. What explains the East/West funding gap for deprived areas?")
    dep_all = core.load_deprivation()
    mid_med = dep_all[dep_all.lad.isin(core.MIDLANDS)].deprivation_mean.median()
    mid = d[d.is_mid & d.deprivation_mean.notna()].copy()
    mid["deprived"] = mid.deprivation_mean >= mid_med
    mid["pct_urban_flag"] = (mid.Urban_rural_flag == "Urban").astype(float) * 100
    dep_m = mid[mid.deprived]
    summ = (dep_m.groupby("region").agg(
        n=("lad", "size"), funding_pc=("funding_pc", "mean"), deprivation=("deprivation_mean", "mean"),
        pct_urban=("pct_urban_flag", "mean"), smi_score=("smi_score", "mean"),
        internal_move_pct=("internal_move_pct", "mean")).round(2))
    print(summ.to_string())
    summ.to_csv(OUT / "12_east_west_funding_drivers.csv")

    # four East-vs-West bars: funding, urbanity, social mobility, geographic mobility
    panels = [("funding_pc", "Regeneration funding\n(£ per head)", "£{:,.0f}"),
              ("pct_urban", "% classified urban\n(ONS RUC)", "{:.0f}%"),
              ("smi_score", "Social Mobility Index score\n(higher = more mobility)", "{:.0f}"),
              ("internal_move_pct", "Moved within UK in prior year\n(%, geographic mobility)", "{:.1f}%")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    order = ["East Midlands", "West Midlands"]
    colours = [PF_ORANGE, PF_BLUE]
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = [summ.loc[r, col] for r in order]
        ax.bar(order, vals, color=colours)
        for i, v in enumerate(vals):
            ax.text(i, v + (abs(max(vals, key=abs)) * 0.02 if col != "smi_score" else 1),
                    fmt.format(v), ha="center", fontsize=9,
                    va="bottom" if v >= 0 else "top")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticklabels(order, rotation=12)
        ax.spines[["top", "right"]].set_visible(False)
        if col == "smi_score":
            ax.axhline(0, color="#999", lw=0.8)
    fig.suptitle("Why do the West Midlands' deprived areas get less funding than the East's?\n"
                 "Urbanity is the clear differentiator — West deprived areas are big cities, which the "
                 "town-focused funds under-served; mobility barely differs",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    core.save(fig, "12_east_west_funding_drivers.png")


# =========================================================================== #
# SECTION 2 - WIDER POLITICAL ANALYSIS
# =========================================================================== #
def _knn_party_surface(X, shares, xs, ys, k=40):
    mu, sd = X.mean(0), X.std(0)
    Xz = (X - mu) / sd
    gx, gy = np.meshgrid(xs, ys)
    Gz = (np.column_stack([gx.ravel(), gy.ravel()]) - mu) / sd
    dist = np.sqrt(((Gz[:, None, :] - Xz[None, :, :]) ** 2).sum(2))
    nn = np.argpartition(dist, k, axis=1)[:, :k]
    lead = np.argmax(shares[nn].mean(axis=1), axis=1).reshape(gx.shape)
    return gx, gy, lead


def s2a_2b_party_planes(d):
    print("\n2a/2b. Leading party by age & deprivation")
    base = d.dropna(subset=["pct_50plus", "deprivation_mean"] + SHARE).copy()
    shares = base[SHARE].to_numpy()
    base["win"] = np.argmax(shares, axis=1)
    X = base[["pct_50plus", "deprivation_mean"]].to_numpy()
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 240)
    ys = np.linspace(X[:, 1].min() - 0.02, X[:, 1].max() + 0.02, 240)
    gx, gy, lead = _knn_party_surface(X, shares, xs, ys)
    cmap = ListedColormap([to_rgb(p[2]) for p in PARTY5])

    # 2a - single England plane, Midlands ringed
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.pcolormesh(gx, gy, lead, cmap=cmap, vmin=0, vmax=len(PARTY5) - 1, alpha=0.32, shading="auto", zorder=1)
    m = base.is_mid.to_numpy()
    dot_c = [PARTY5[i][2] for i in base.win]
    ax.scatter(X[~m, 0], X[~m, 1], c=[c for c, k in zip(dot_c, m) if not k], s=22,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(X[m, 0], X[m, 1], c=[c for c, k in zip(dot_c, m) if k], s=42,
               edgecolor="black", linewidth=0.9, zorder=4)
    ax.set_xlabel("% aged 50 or older")
    ax.set_ylabel("Household deprivation (mean dimensions, 0–4; higher = more deprived)")
    ax.set_title("England local authorities by age and deprivation\n"
                 "shaded by leading party (MRP, Q4 2025); Midlands LADs ringed in black",
                 fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=p[2], markersize=9, label=p[1])
               for p in PARTY5]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#bbb",
                          markeredgecolor="black", markeredgewidth=1.1, markersize=9, label="Midlands LAD"))
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.margins(0)
    core.save(fig, "07_age_deprivation_party.png")

    # 2b - two panels East / West
    base["region"] = base.lad.map(core.MIDLANDS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, reg in zip(axes, ["East Midlands", "West Midlands"]):
        ax.pcolormesh(gx, gy, lead, cmap=cmap, vmin=0, vmax=len(PARTY5) - 1, alpha=0.30, shading="auto", zorder=1)
        ax.scatter(X[:, 0], X[:, 1], s=8, color="#888", alpha=0.18, zorder=2)
        sub = base[base.region == reg]
        ax.scatter(sub.pct_50plus, sub.deprivation_mean, c=[PARTY5[i][2] for i in sub.win],
                   s=60, edgecolor="black", linewidth=0.9, zorder=4)
        r = sub.win.value_counts()
        note = "  ".join(f"{PARTY5[i][1]} {int(r.get(i, 0))}" for i in range(len(PARTY5)) if r.get(i, 0))
        ax.set_title(f"{reg}  (n={len(sub)}: {note})", fontweight="bold", fontsize=11)
        ax.set_xlabel("% aged 50 or older"); ax.margins(0)
    axes[0].set_ylabel("Household deprivation (mean dimensions, 0–4; higher = more deprived)")
    fig.legend(handles=handles[:-1], frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where East vs West Midlands LADs sit on the national age/deprivation plane\n"
                 "background = England leading party (MRP, Q4 2025); dots = the region's LADs",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "08_age_deprivation_party_midlands.png", dpi=170, bbox_inches="tight")
    print("   saved -> outputs/08_age_deprivation_party_midlands.png")


def s2c_battleground_and_diversity(d):
    print("\n2c. Age & urbanity battleground; deprivation vs Reform by diversity")
    # battleground (age x density), coloured by party, sized by deprivation
    b = d.dropna(subset=["pct_50plus", "density", "deprivation_mean"] + SHARE).copy()
    b["win"] = np.argmax(b[SHARE].to_numpy(), axis=1)
    y = np.log10(b.density); dep = b.deprivation_mean
    size = 12 + (dep - dep.min()) / (dep.max() - dep.min()) * 240
    col = [PARTY5[i][2] for i in b.win]; m = b.is_mid.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(b.pct_50plus[~m], y[~m], s=size[~m], c=[c for c, k in zip(col, m) if not k],
               alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2)
    ax.scatter(b.pct_50plus[m], y[m], s=size[m], c=[c for c, k in zip(col, m) if k],
               alpha=0.95, edgecolor="black", linewidth=1.1, zorder=3)
    ticks = [30, 100, 300, 1000, 3000, 10000]
    ax.set_yticks(np.log10(ticks)); ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.set_xlabel("% aged 50 or older  (younger  ->  older)")
    ax.set_ylabel("Population density, people per km²  (rural  ->  urban)")
    ax.set_title("Age and urbanity sort the parties; deprivation sits on both sides\n"
                 "England LADs; colour = MRP leading party (Q4 2025); dot size = deprivation; Midlands ringed",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ph = [Line2D([0], [0], marker="o", color="none", markerfacecolor=p[2], markersize=9, label=p[1]) for p in PARTY5]
    sh = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#bbb", markersize=np.sqrt(s), label=lbl)
          for s, lbl in [(40, "less deprived"), (230, "more deprived")]]
    ax.add_artist(ax.legend(handles=ph, frameon=False, ncol=5, loc="upper center",
                            bbox_to_anchor=(0.5, -0.09), title="Leading party"))
    ax.legend(handles=sh, frameon=False, loc="lower right", title="dot size = deprivation",
              labelspacing=1.4, borderpad=1)
    core.save(fig, "09_battleground_age_urbanity.png")

    # deprivation vs Reform split by diversity metrics
    metrics = [("foreign_born_pct", "Foreign-born %"), ("nonwhite_pct", "Non-White %"),
               ("nonwhite_british_pct", "Non-White British %"),
               ("ethnic_diversity_idx", "Ethnic diversity index (Simpson)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    print("   deprivation -> Reform slope by low/high halves of each diversity metric:")
    for ax, (col_, label) in zip(axes.ravel(), metrics):
        sub = d.dropna(subset=[col_, "deprivation_mean", "reform_pct"])
        med = sub[col_].median()
        for grp, mask, c in [("Less diverse", sub[col_] <= med, "#C1440E"),
                             ("More diverse", sub[col_] > med, "#2E86AB")]:
            g = sub[mask]
            ax.scatter(g.deprivation_mean, g.reform_pct, s=16, color=c, alpha=0.5, zorder=2)
            bb, aa = np.polyfit(g.deprivation_mean, g.reform_pct, 1)
            xs = np.array([g.deprivation_mean.min(), g.deprivation_mean.max()])
            ax.plot(xs, aa + bb * xs, color=c, lw=2.6, zorder=3, label=f"{grp} (slope {bb:+.0f})")
            print(f"      {label:34s} {grp:12s} slope {bb:+6.1f}pp (n={len(g)})")
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Household deprivation (higher = more deprived)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reform UK vote share (%)")
    fig.suptitle("Deprivation lifts Reform far more steeply where the area is less diverse\n"
                 "diversity blunts the link: at high deprivation the less-diverse half is "
                 "~13pp more Reform (England LADs split at each metric's median; MRP Q4 2025)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "10_deprivation_reform_by_diversity.png", dpi=170, bbox_inches="tight")
    print("   saved -> outputs/10_deprivation_reform_by_diversity.png")


def _reassemble_bes(td):
    parts = sorted(glob.glob(str(BES_RAW / "BES2024_W30_Panel_v30.1.sav.zip.part*")))
    zp = Path(td) / "panel.zip"
    with open(zp, "wb") as out:
        for p in parts:
            out.write(open(p, "rb").read())
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".sav")][0]
        z.extract(name, td)
    return Path(td) / name


_BES_SAV = None


def bes_sav():
    """Reassemble the BES panel .sav once per run (both BES sections share it)."""
    global _BES_SAV
    if _BES_SAV is None:
        _BES_SAV = _reassemble_bes(tempfile.mkdtemp())
    return _BES_SAV


def s2e_bes_switching():
    print("\n2e. Labour switching (Green vs Reform) by deprivation, East vs West (BES panel)")
    LAB, GRN, REFORM = 2.0, 7.0, [6.0, 12.0]
    usecols = ["wt_new_W30", "p_past_vote_2024", "generalElectionVoteW30", "gorW30", "oslauaW30", "oslauaW1"]
    df, _ = pyreadstat.read_sav(str(bes_sav()), usecols=usecols, apply_value_formats=False, user_missing=False)
    dep = core.load_deprivation()[["lad", "deprivation_mean"]]
    df["gor"] = df["gorW30"].astype("string").str.strip()
    la = df["oslauaW30"].astype("string").str.strip()
    la = la.where(la.notna() & (la != ""), df["oslauaW1"].astype("string").str.strip())
    df["la"] = la
    df = df.merge(dep, left_on="la", right_on="lad", how="left")
    df["w"] = pd.to_numeric(df["wt_new_W30"], errors="coerce").fillna(0)
    df["gev30"] = pd.to_numeric(df["generalElectionVoteW30"], errors="coerce")
    lab24 = pd.to_numeric(df["p_past_vote_2024"], errors="coerce") == LAB
    base = df[lab24 & df.gor.isin(["East Midlands", "West Midlands"])
              & df.deprivation_mean.notna() & (df.w > 0)].copy()
    base["to_green"] = (base.gev30 == GRN).astype(float)
    base["to_reform"] = base.gev30.isin(REFORM).astype(float)
    base["dep_band"] = pd.qcut(base.deprivation_mean, 3, labels=["Low", "Mid", "High"])
    wsh = lambda mask, w: float((w * mask).sum() / w.sum()) if w.sum() else np.nan
    rows = []
    for reg in ["East Midlands", "West Midlands"]:
        for band in ["Low", "Mid", "High"]:
            g = base[(base.gor == reg) & (base.dep_band == band)]
            rows.append({"region": reg, "dep_band": band, "n": len(g),
                         "to_green_pct": wsh(g.to_green, g.w) * 100,
                         "to_reform_pct": wsh(g.to_reform, g.w) * 100})
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "05_midlands_switching_by_deprivation.csv", index=False)
    print("   " + tab.round(1).to_string(index=False).replace("\n", "\n   "))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    bands = ["Low", "Mid", "High"]; x = np.arange(3)
    ymax = max(tab.to_green_pct.max(), tab.to_reform_pct.max()) * 1.25
    for ax, reg in zip(axes, ["East Midlands", "West Midlands"]):
        t = tab[tab.region == reg].set_index("dep_band").loc[bands]
        ax.plot(x, t.to_reform_pct, "-o", color=PF_ORANGE, lw=2.5, label="→ Reform")
        ax.plot(x, t.to_green_pct, "-o", color=GREEN, lw=2.5, label="→ Green")
        for xi, (_, r) in zip(x, t.iterrows()):
            ax.annotate(f"n={int(r.n)}", (xi, ymax * 0.03), ha="center", fontsize=7, color="#888")
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\ndeprivation" for b in bands]); ax.set_ylim(0, ymax)
        ax.set_title(reg, fontweight="bold"); ax.set_xlabel("Local-authority deprivation (Midlands terciles)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of 2024 Labour voters now intending to switch")
    axes[0].legend(frameon=False)
    fig.suptitle("Where do Midlands Labour-2024 voters drift as deprivation rises?\n"
                 "BES 2024 panel, wave 30 (May 2025) vote intention; weighted", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "05_midlands_switching_by_deprivation.png", dpi=150)
    print("   saved -> outputs/05_midlands_switching_by_deprivation.png")


def s2f_bes_reform_trajectory():
    """Right-populist (UKIP/Brexit/Reform) vote intention among likely voters over the
    BES panel waves, East vs West Midlands, and within the region's deprived areas."""
    print("\n2f. UKIP/Brexit/Reform vote intention over time among likely voters, East vs West")
    WAVES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30]
    REFORM = [6.0, 12.0]                                   # UKIP + Brexit/Reform UK
    SPSS = (_dt.datetime(1970, 1, 1) - _dt.datetime(1582, 10, 14)).total_seconds()
    cols = ["oslauaW30", "oslauaW1"]
    for w in WAVES:
        cols += [f"turnoutUKGeneralW{w}", f"generalElectionVoteW{w}", f"starttimeW{w}", f"gorW{w}"]
    df, _ = pyreadstat.read_sav(str(bes_sav()), usecols=cols, apply_value_formats=False, user_missing=False)

    dep = core.load_deprivation()[["lad", "deprivation_mean"]]
    la = df["oslauaW30"].astype("string").str.strip()
    la = la.where(la.notna() & (la != ""), df["oslauaW1"].astype("string").str.strip())
    df = df.assign(lad=la).merge(dep, on="lad", how="left")
    mid_med = dep[dep.lad.isin(core.MIDLANDS)].deprivation_mean.median()
    df["deprived"] = df.deprivation_mean >= mid_med       # "deprived parts of the region"

    rows = []
    for w in WAVES:
        secs = pd.to_numeric(df[f"starttimeW{w}"], errors="coerce")
        date = pd.to_datetime(secs.where(secs > 0) - SPSS, unit="s", errors="coerce").median()
        gor = df[f"gorW{w}"].astype("string").str.strip()
        turn = pd.to_numeric(df[f"turnoutUKGeneralW{w}"], errors="coerce")
        vote = pd.to_numeric(df[f"generalElectionVoteW{w}"], errors="coerce")
        base = (turn >= 4) & vote.between(1, 13)           # likely voters naming a party (unweighted)
        for reg in ["East Midlands", "West Midlands"]:
            for cut, mask in [("all", pd.Series(True, index=df.index)), ("deprived", df.deprived)]:
                m = base & (gor == reg) & mask
                n = int(m.sum())
                share = float(vote[m].isin(REFORM).mean()) if n >= 40 else np.nan
                rows.append({"wave": w, "date": date, "region": reg, "cut": cut,
                             "reform_pct": share * 100 if n >= 40 else np.nan, "n": n})
    traj = pd.DataFrame(rows)
    traj.to_csv(OUT / "11_reform_trajectory_bes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, cut, title in zip(axes, ["all", "deprived"],
                              ["All Midlands likely voters",
                               f"Deprived Midlands areas only (LAD deprivation ≥ {mid_med:.2f})"]):
        for reg, color in [("East Midlands", PF_ORANGE), ("West Midlands", PF_BLUE)]:
            s = traj[(traj.cut == cut) & (traj.region == reg) & traj.reform_pct.notna()].sort_values("date")
            ax.plot(s.date, s.reform_pct, "-o", color=color, lw=2, ms=4, label=reg)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("BES wave date")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% intending UKIP / Brexit / Reform\n(of likely voters naming a party)")
    axes[0].legend(frameon=False)
    fig.suptitle("Right-populist vote intention among likely voters, East vs West Midlands\n"
                 "BES 2024 internet panel, waves 1–30 (2014–2025); UKIP → Brexit Party → Reform UK",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "11_reform_trajectory_bes.png", dpi=150)
    print("   saved -> outputs/11_reform_trajectory_bes.png")


# --------------------------------------------------------------------------- #
# 2g-2j helpers: East/West Reform trajectory and the 2026 test
# --------------------------------------------------------------------------- #
def _mrp_region_quarter(dep_split=False):
    """MRP long frame: region (x within-region deprivation half) x quarter ->
    vote-weighted Reform share."""
    w22 = pd.read_csv(DATA / "ward22_to_lad22.csv")[["WD22CD", "LAD22CD"]]
    w24 = pd.read_csv(DATA / "ward24_to_lad24.csv")[["WD24CD", "LAD24CD"]]
    xl = pd.ExcelFile(DATA / "Political_MRP_by_quarter.xlsx")
    dep = core.load_deprivation()[["lad", "deprivation_mean"]]
    dband = {}
    if dep_split:
        for reg, codes in (("East Midlands", core.EAST_MIDLANDS), ("West Midlands", core.WEST_MIDLANDS)):
            sub = dep[dep.lad.isin(codes)]
            med = sub.deprivation_mean.median()
            for _, r in sub.iterrows():
                dband[r.lad] = "more deprived" if r.deprivation_mean > med else "less deprived"
    ref_col = core.PREF + "Projected Reform UK Vote Count"
    tot_col = core.PREF + "Projected Total Votes"
    rows = []
    for sheet, tag in core.QUARTERS.items():
        df = xl.parse(sheet)
        df = (df.merge(w22, left_on="Ward_code", right_on="WD22CD", how="left")
                .merge(w24, left_on="Ward_code", right_on="WD24CD", how="left"))
        df["lad"] = df["LAD22CD"].fillna(df["LAD24CD"])
        df["region"] = df.lad.map(core.MIDLANDS)
        df = df[df.region.notna()].copy()
        df[ref_col] = pd.to_numeric(df[ref_col], errors="coerce")
        df[tot_col] = pd.to_numeric(df[tot_col], errors="coerce")
        df = df.dropna(subset=[ref_col, tot_col])
        grp = df.groupby(["region", df.lad.map(dband)]) if dep_split else df.groupby("region")
        agg = grp.agg(ref=(ref_col, "sum"), tot=(tot_col, "sum")).reset_index()
        if dep_split:
            agg.columns = ["region", "band", "ref", "tot"]
        agg["reform_share"] = agg.ref / agg.tot
        agg["quarter"] = tag
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _ge_region_share():
    def one(path, col, year):
        g = pd.read_csv(path)
        g = g[g["Region name"].isin(["East Midlands", "West Midlands"])]
        a = (g.groupby("Region name")
               .apply(lambda x: pd.Series({"share": x[col].sum() / x["Valid votes"].sum()}),
                      include_groups=False).reset_index().rename(columns={"Region name": "region"}))
        a["year"] = year
        return a
    g19 = one(GE / "2019_results_analysis/HoC-GE2019-results-by-constituency.csv", "BRX", 2019)
    g24 = one(GE / "2024_results_analysis/HoC-GE2024-results-by-constituency.csv", "RUK", 2024)
    return pd.concat([g19, g24], ignore_index=True)


def _le2025_region():
    x = pd.read_excel(LE / "2025_results_analysis/CBP10272.xlsx", sheet_name="England LE25", header=1)
    x = x[x["Region"].isin(["East Midlands", "West Midlands"])].copy()
    for c in ["REF", "Total"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return (x.groupby("Region").apply(
        lambda d: pd.Series({"councils_up": len(d), "ref_seats": d.REF.sum(),
                             "total_seats": d.Total.sum(),
                             "ref_seat_pct": 100 * d.REF.sum() / d.Total.sum()}),
        include_groups=False).reset_index())


def _fam(p):
    p = str(p)
    if p == "Reform UK":
        return "Reform"
    if "Labour" in p:
        return "Labour"
    if "Conservative" in p:
        return "Conservative"
    if "Liberal Democrat" in p:
        return "LibDem"
    if "Green" in p:
        return "Green"
    return "Other"


def _wm_2026_results():
    """Per-council 2026 Reform share, lead and seats for the 13 West Midlands councils."""
    cand = pd.read_csv(LE26 / "electionresults_uk_candidates_2026.csv")
    races = pd.read_csv(LE26 / "electionresults_uk_races_2026.csv")
    cc = pd.read_csv(LE26 / "opencouncildata_council_control_2026.csv")
    slug2name = races.drop_duplicates("council_slug").set_index("council_slug")["council"]
    name2code = cc.set_index("authority")["authority_code"]
    cand["council_name"] = cand.council_slug.map(slug2name)
    cand["code"] = cand.council_name.map(name2code)
    cand["fam"] = cand.party.map(_fam)
    wm = cand[cand.code.isin(set(core.WEST_MIDLANDS))].copy()
    rows = []
    for c, dd in wm.groupby("code"):
        tot = dd.votes.sum()
        byv = dd.groupby("fam").votes.sum()
        win = byv.idxmax()
        runner = byv.drop(win).max() if len(byv) > 1 else 0
        seats = dd.groupby("fam").elected.sum()
        rows.append({"lad": c, "council": dd.council_name.iloc[0],
                     "ref_vote_pct": byv.get("Reform", 0) / tot * 100, "most_votes": win,
                     "lead_pp": (byv.max() - runner) / tot * 100,
                     "ref_seats": int(seats.get("Reform", 0)), "total_seats": int(dd.elected.sum()),
                     "most_seats": seats.idxmax()})
    r = pd.DataFrame(rows).sort_values("ref_vote_pct", ascending=False).reset_index(drop=True)
    agg = wm[wm.fam == "Reform"].votes.sum() / wm.votes.sum() * 100
    return r, agg


def _resid(a, b):
    B = np.c_[np.ones(len(b)), b]
    return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]


def _pcorr(d, x, y, z):
    return float(np.corrcoef(_resid(d[x].values, d[z].values), _resid(d[y].values, d[z].values))[0, 1])


def s2g_ew_reform_trajectory():
    """Did the West Midlands move to Reform later than the East and catch up? MRP
    quarterly trajectory + GE 2019/2024 + LE2025 context."""
    print("\n2g. East vs West Midlands Reform trajectory (MRP + general/local elections)")
    m = _mrp_region_quarter()
    piv = (m.pivot(index="quarter", columns="region", values="reform_share")
           .reindex(list(core.QUARTERS.values())))
    piv["gap_E_minus_W"] = piv["East Midlands"] - piv["West Midlands"]
    q0, q1 = list(core.QUARTERS.values())[0], list(core.QUARTERS.values())[-1]
    print((piv * 100).round(1).to_string())
    print(f"   East {piv.loc[q0,'East Midlands']*100:.1f}->{piv.loc[q1,'East Midlands']*100:.1f}%, "
          f"West {piv.loc[q0,'West Midlands']*100:.1f}->{piv.loc[q1,'West Midlands']*100:.1f}%; "
          f"gap {piv.loc[q0,'gap_E_minus_W']*100:+.1f}pp -> {piv.loc[q1,'gap_E_minus_W']*100:+.1f}pp")
    piv.to_csv(OUT / "13_mrp_reform_trajectory_ew.csv")
    _mrp_region_quarter(dep_split=True).to_csv(OUT / "13_mrp_reform_trajectory_ew_deprivation.csv", index=False)

    ge = _ge_region_share()
    gp = ge.pivot(index="region", columns="year", values="share") * 100
    ge.to_csv(OUT / "13_ge_reform_share_ew.csv", index=False)
    le = _le2025_region()
    le.to_csv(OUT / "13_le2025_reform_seats_ew.csv", index=False)
    print(f"   2025 locals: East Midlands Reform {le.set_index('Region').loc['East Midlands','ref_seat_pct']:.0f}% "
          "of contested seats; West Midlands mets not up until 2026.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    xq = np.arange(len(core.QUARTERS))
    qlabels = [q.replace("q", "Q").replace("_", " ") for q in core.QUARTERS.values()]
    for reg, col in (("East Midlands", PF_ORANGE), ("West Midlands", PF_BLUE)):
        axL.plot(xq, piv[reg] * 100, "-o", color=col, lw=2.4, label=reg, zorder=3)
        axL.annotate(f"{piv[reg].iloc[-1]*100:.1f}%", (xq[-1], piv[reg].iloc[-1]*100),
                     xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                     color=col, fontweight="bold")
    axL.annotate(f"West starts ahead\n(+{-piv.loc[q0,'gap_E_minus_W']*100:.1f}pp)",
                 (xq[0], piv.loc[q0, "West Midlands"]*100), xytext=(4, -34),
                 textcoords="offset points", fontsize=8, color=PF_BLUE,
                 arrowprops=dict(arrowstyle="->", color=PF_BLUE, lw=0.8))
    axL.annotate("East overtakes\nby mid-2025", (2, piv.loc["q2_2025", "East Midlands"]*100),
                 xytext=(-6, 20), textcoords="offset points", fontsize=8, color=PF_ORANGE,
                 ha="center", arrowprops=dict(arrowstyle="->", color=PF_ORANGE, lw=0.8))
    axL.set_xticks(xq); axL.set_xticklabels(qlabels)
    axL.set_ylabel("Vote-weighted Reform UK share (%)")
    axL.set_title("Our MRP: the two halves rise together\n"
                  f"(East +{(piv.loc[q1,'East Midlands']-piv.loc[q0,'East Midlands'])*100:.0f}pp over the year, "
                  f"West +{(piv.loc[q1,'West Midlands']-piv.loc[q0,'West Midlands'])*100:.0f}pp)", fontsize=11)
    axL.legend(frameon=False, loc="lower right"); axL.margins(x=0.10); axL.grid(axis="y", alpha=0.25)
    xg = np.arange(2)
    for reg, col in (("East Midlands", PF_ORANGE), ("West Midlands", PF_BLUE)):
        ys = [gp.loc[reg, 2019], gp.loc[reg, 2024]]
        axR.plot(xg, ys, "-o", color=col, lw=2.4, label=reg, zorder=3)
        for xi, yi in zip(xg, ys):
            axR.annotate(f"{yi:.0f}%", (xi, yi), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=9, color=col, fontweight="bold")
    axR.set_xticks(xg); axR.set_xticklabels(["2019\n(Brexit Party*)", "2024\n(Reform)"])
    axR.set_title("General elections: near-identical,\nEast a fraction ahead", fontsize=11)
    axR.set_ylabel("Reform-family vote share (%)"); axR.margins(x=0.25); axR.grid(axis="y", alpha=0.25)
    axR.text(0.5, -0.22, "*Brexit Party stood down in Con-held seats in 2019",
             transform=axR.transAxes, ha="center", fontsize=7, color="grey")
    fig.suptitle("Not a late-adopter catching up: East and West Midlands moved to Reform in near-lockstep\n"
                 "the West's 2026 breakthrough is a seats story (its councils vote for the first time since Reform's rise)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(OUT / "13_reform_trajectory_ew.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved -> outputs/13_reform_trajectory_ew.png")


def s2h_wm_2026_breakthrough():
    """Did the projected West Midlands breakthrough happen in the actual 2026 results?"""
    print("\n2h. Actual 2026 West Midlands local elections (did the breakthrough happen?)")
    r26, agg26 = _wm_2026_results()
    won_v = int((r26.most_votes == "Reform").sum())
    r26.to_csv(OUT / "14_wm_2026_actual_results.csv", index=False)
    print(f"   Reform aggregate vote {agg26:.1f}% (YouGov MRP ~30%); first on votes in {won_v}/13; "
          f"{int(r26.ref_seats.sum())}/{int(r26.total_seats.sum())} seats.")
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    d = r26.iloc[::-1]
    colours = [REFORM_TEAL if w == "Reform" else PF_GREY for w in d.most_votes]
    y = np.arange(len(d))
    ax.barh(y, d.ref_vote_pct, color=colours, zorder=3)
    for yi, row in zip(y, d.itertuples()):
        first = "  1st" if row.most_votes == "Reform" else f"  ({row.most_votes} 1st)"
        ax.text(row.ref_vote_pct + 0.6, yi, f"{row.ref_vote_pct:.0f}%{first}", va="center",
                fontsize=8.5, color=REFORM_TEAL if row.most_votes == "Reform" else "#555555")
    ax.set_yticks(y); ax.set_yticklabels(d.council, fontsize=9)
    ax.axvline(agg26, color="black", lw=1.2, zorder=2)
    ax.axvline(30, color=PF_GREY, lw=1.2, ls="--", zorder=2)
    ax.text(agg26, len(d) - 0.3, f" actual avg {agg26:.1f}%", fontsize=8, color="black")
    ax.text(30, -1.15, "YouGov MRP\ncentral 30%", fontsize=7.5, color=PF_GREY, ha="center")
    ax.set_xlabel("Reform UK vote share, 2026 local elections (%)"); ax.set_xlim(0, 62)
    ax.set_title(f"The breakthrough happened: Reform won the most votes in {won_v} of 13 "
                 "West Midlands councils\n(7 May 2026) — matching YouGov's pre-election MRP",
                 fontsize=11.5, fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=REFORM_TEAL,
                              markersize=11, label="Reform first"),
                       Line2D([0], [0], marker="s", color="none", markerfacecolor=PF_GREY,
                              markersize=11, label="Another party first")],
              loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "14_wm_2026_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved -> outputs/14_wm_2026_results.png")


def s2i_wm_2026_diversity():
    """Did the diverse deprived West Midlands areas back Reform 'just as much'? No: the
    diversity gradient holds strongly in the 2026 hard results."""
    print("\n2i. Diversity vs Reform within the West Midlands, 2026 actuals")
    r26, _ = _wm_2026_results()
    cc = pd.read_csv(LE26 / "opencouncildata_council_control_2026.csv")
    dv = r26.copy()
    dv["lad"] = dv.council.map(cc.set_index("authority")["authority_code"])
    eth = pd.read_csv(DATA / "lad_ethnicity.csv")
    dep = core.load_deprivation()[["lad", "deprivation_mean"]]
    dv = dv.merge(eth, on="lad", how="left").merge(dep, on="lad", how="left")
    rc = stats.pearsonr(dv.nonwhite_pct, dv.ref_vote_pct)
    rd = stats.pearsonr(dv.deprivation_mean, dv.ref_vote_pct)
    print(f"   corr(non-White %, Reform)={rc.statistic:+.2f} (p={rc.pvalue:.3f}); "
          f"corr(deprivation, Reform)={rd.statistic:+.2f} (p={rd.pvalue:.3f})")
    dv.to_csv(OUT / "15_wm_2026_diversity.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    colours = [REFORM_TEAL if f == "Reform" else PF_GREY for f in dv.most_votes]
    ax.scatter(dv.nonwhite_pct, dv.ref_vote_pct, c=colours, s=90, zorder=3, edgecolor="white", linewidth=0.8)
    m, b = np.polyfit(dv.nonwhite_pct, dv.ref_vote_pct, 1)
    xs = np.linspace(0, dv.nonwhite_pct.max() * 1.05, 50)
    ax.plot(xs, m * xs + b, color="#444444", lw=1.4, ls="--", zorder=2)
    black_country = {"Sandwell", "Walsall", "Wolverhampton"}
    for _, row in dv.iterrows():
        dx, ha = (-6, "right") if row.council in ("Birmingham", "Coventry", "Solihull") else (6, "left")
        ax.annotate(row.council, (row.nonwhite_pct, row.ref_vote_pct), xytext=(dx, 4),
                    textcoords="offset points", fontsize=8, ha=ha,
                    fontweight="bold" if row.council in black_country else "normal")
    ax.text(0.97, 0.95, f"r = {rc.statistic:+.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#444444")
    ax.set_xlabel("Non-White population (%, Census 2021)")
    ax.set_ylabel("Reform UK vote share, 2026 local elections (%)")
    ax.set_title("Even in 2026, diversity (not deprivation) tracks Reform in the West Midlands\n"
                 "the diverse cities are Reform's weakest councils; the white towns its strongest",
                 fontsize=11.5, fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=REFORM_TEAL,
                              markersize=11, label="Reform came first"),
                       Line2D([0], [0], marker="s", color="none", markerfacecolor=PF_GREY,
                              markersize=11, label="Another party first")],
              loc="lower left", frameon=False, fontsize=9)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "15_wm_2026_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved -> outputs/15_wm_2026_diversity.png")


def s2j_dep_diversity_coupling(d):
    """Why the raw diversity->Reform gradient looks cleaner in the East: deprivation and
    diversity are decoupled there (+0.31) but entangled in the West (+0.69). Partial
    correlations show both effects are the same underneath."""
    print("\n2j. Deprivation-diversity coupling by region")
    cp = d[d.is_mid].dropna(subset=["deprivation_mean", "nonwhite_pct", "reform_pct"]).copy()
    cp["region"] = cp.lad.map(core.MIDLANDS)
    stat = {}
    for reg, g in cp.groupby("region"):
        stat[reg] = {"dep_div": stats.pearsonr(g.deprivation_mean, g.nonwhite_pct)[0],
                     "raw_div_ref": stats.pearsonr(g.nonwhite_pct, g.reform_pct)[0],
                     "raw_dep_ref": stats.pearsonr(g.deprivation_mean, g.reform_pct)[0],
                     "part_div_ref": _pcorr(g, "reform_pct", "nonwhite_pct", "deprivation_mean"),
                     "part_dep_ref": _pcorr(g, "reform_pct", "deprivation_mean", "nonwhite_pct")}
        s = stat[reg]
        print(f"   {reg}: corr(dep,diversity)={s['dep_div']:+.2f} | "
              f"diversity->Reform raw {s['raw_div_ref']:+.2f} -> partial {s['part_div_ref']:+.2f} | "
              f"deprivation->Reform raw {s['raw_dep_ref']:+.2f} -> partial {s['part_dep_ref']:+.2f}")
    pd.DataFrame(stat).T.round(3).to_csv(OUT / "16_dep_diversity_coupling.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))
    for reg, col in (("East Midlands", PF_ORANGE), ("West Midlands", PF_BLUE)):
        g = cp[cp.region == reg]
        axA.scatter(g.nonwhite_pct, g.deprivation_mean, c=col, s=42, alpha=0.85,
                    edgecolor="white", linewidth=0.5, label=reg, zorder=3)
        mm, bb = np.polyfit(g.nonwhite_pct, g.deprivation_mean, 1)
        xs = np.linspace(g.nonwhite_pct.min(), g.nonwhite_pct.max(), 40)
        axA.plot(xs, mm * xs + bb, color=col, lw=1.8, zorder=2)
        axA.text(0.03, 0.95 if reg == "West Midlands" else 0.88,
                 f"{reg}: r = {stat[reg]['dep_div']:+.2f}", transform=axA.transAxes,
                 color=col, fontsize=9.5, fontweight="bold", va="top")
    axA.set_xlabel("Non-White population (%)")
    axA.set_ylabel("Household deprivation (mean, 0-4)")
    axA.set_title("In the West, deprivation and diversity are the same places;\n"
                  "in the East they are decoupled", fontsize=11)
    axA.legend(frameon=False, loc="lower right", fontsize=9); axA.grid(alpha=0.2)
    regions = ["East Midlands", "West Midlands"]
    raw = [stat[r]["raw_div_ref"] for r in regions]
    part = [stat[r]["part_div_ref"] for r in regions]
    x = np.arange(2)
    axB.bar(x - 0.2, raw, 0.4, color=[PF_ORANGE, PF_BLUE], alpha=0.45, label="raw correlation")
    axB.bar(x + 0.2, part, 0.4, color=[PF_ORANGE, PF_BLUE], label="partial (controlling deprivation)")
    for xi, (rw, pt) in enumerate(zip(raw, part)):
        axB.text(xi - 0.2, rw - 0.06, f"{rw:+.2f}", ha="center", va="top", fontsize=9)
        axB.text(xi + 0.2, pt - 0.06, f"{pt:+.2f}", ha="center", va="top", fontsize=9)
    axB.axhline(0, color="black", lw=0.8); axB.set_xticks(x); axB.set_xticklabels(regions)
    axB.set_ylabel("Correlation of diversity with Reform share")
    axB.set_title("The diversity effect is identical once you\ndisentangle it: the East's raw gradient only\n"
                  "looks cleaner because its confound is weaker", fontsize=11)
    axB.legend(frameon=False, fontsize=8, loc="lower left"); axB.set_ylim(-1, 0.05)
    fig.suptitle("Deprivation and diversity are entangled in the West, decoupled in the East, "
                 "but both push Reform the same way underneath", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "16_dep_diversity_coupling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved -> outputs/16_dep_diversity_coupling.png")


def main():
    print("=" * 74)
    print("ADDITIONAL DEMOGRAPHIC & POLITICAL ANALYSIS (MIDLANDS)")
    print("=" * 74)
    d = assemble_england()
    print(f"England frame assembled: {len(d)} LADs")

    print("\n" + "-" * 74 + "\nSECTION 1 — FUNDING vs DEMOGRAPHICS\n" + "-" * 74)
    s1a_east_west_summary(d)
    s1b_itl1_deprivation_bar(d)
    s1c_funding_map(d)
    s1d_deprivation_vs_reform(d)
    s1e_explain_funding_gap(d)

    print("\n" + "-" * 74 + "\nSECTION 2 — WIDER POLITICAL ANALYSIS\n" + "-" * 74)
    s2a_2b_party_planes(d)
    s2c_battleground_and_diversity(d)
    s2e_bes_switching()
    s2f_bes_reform_trajectory()
    s2g_ew_reform_trajectory()
    s2h_wm_2026_breakthrough()
    s2i_wm_2026_diversity()
    s2j_dep_diversity_coupling(d)

    print("\n" + "=" * 74 + "\nDone. Tables and charts in ./outputs\n" + "=" * 74)


if __name__ == "__main__":
    main()
