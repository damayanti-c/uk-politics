"""
Regeneration funding in the Midlands: deprivation, vote choice, and Reform support
=================================================================================

Midlands rebuild of the UK-wide "Regeneration analysis.R" / Public First report
("What became of the likely lads?"). It answers three questions for the Midlands
(East Midlands + West Midlands = 65 local authority districts):

  1. Which Midlands LADs have received the most regeneration funding per head
     (top 10)?
  2. How does regeneration funding relate to deprivation and to vote choice
     (Reform support, seat marginality)?
  3. Using matching (treatment = high funding, matched on deprivation), are
     heavily funded Midlands areas more supportive of Reform than similarly
     deprived but less-funded ones? Robustness: mean vs median treatment
     thresholds, several matching estimators, and dropping high-immigration LADs.

Data (pulled from the project Google Drive into ./data):
  - funding_per_lad.csv   Regeneration funding per LAD (LAD24, per capita, £2025)
  - deprivation_ew.csv    ONS Census 2021 household deprivation (6 categories)
  - nov_mrp_ward.csv       Nov 2025 MRP ward-level projected vote (single snapshot)
  - ward22_to_lad22.csv    Ward (2022) -> LAD (2022) lookup
  - ward24_to_lad24.csv    Ward (2024) -> LAD (2024) lookup (fallback)
  - cob_ew.csv            ONS Census 2021 country of birth (UK/non-UK)

NOTE ON POLITICAL DATA. The original R work used a five-quarter MRP time series
(Q4 2024 -> Q4 2025) from Political_MRP_by_quarter.xlsx to also measure the
*change* in Reform support. That 7.3 MB workbook is over the Drive connector's
transfer limit, so this script uses the single latest snapshot (Nov 2025 MRP).
The cross-sectional relationships and the matching test of Reform *levels* are
fully reproduced; only the over-time dynamics are omitted. Drop the xlsx into
./data and the vote-share loader can be extended to the full series.

Run:  python regeneration_midlands.py
Outputs: tables + charts written to ./outputs, key numbers printed to stdout.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and cannot print '£'; force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

PF_ORANGE = "#E8622D"
PF_GREY = "#9AA0A6"
PF_BLUE = "#2E5A87"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def num(x):
    """Parse '£1,234' / '1,234' / '£108  ' -> float; NaN-safe."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^0-9.\-]", "", str(x))
    return float(s) if s not in ("", "-", ".") else np.nan


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved chart -> outputs/{name}")


# --------------------------------------------------------------------------- #
# 1. Define the Midlands (ONS regions East Midlands E12000004 + West Midlands
#    E12000005). 35 + 30 = 65 LADs (2024). Codes are validated against the
#    funding file below, and the LAD names are printed for review.
#    Source: ONS standard region -> LAD membership.
# --------------------------------------------------------------------------- #
EAST_MIDLANDS = [
    # Unitaries / upper tier
    "E06000015",  # Derby
    "E06000016",  # Leicester
    "E06000017",  # Rutland
    "E06000018",  # Nottingham
    "E06000061",  # North Northamptonshire
    "E06000062",  # West Northamptonshire
    # Derbyshire districts
    "E07000032", "E07000033", "E07000034", "E07000035",
    "E07000036", "E07000037", "E07000038", "E07000039",
    # Leicestershire districts
    "E07000129", "E07000130", "E07000131", "E07000132",
    "E07000133", "E07000134", "E07000135",
    # Lincolnshire districts (county Lincolnshire only; N & NE Lincs are Yorks & Humber)
    "E07000136", "E07000137", "E07000138", "E07000139",
    "E07000140", "E07000141", "E07000142",
    # Nottinghamshire districts
    "E07000170", "E07000171", "E07000172", "E07000173",
    "E07000174", "E07000175", "E07000176",
]
WEST_MIDLANDS = [
    # Metropolitan boroughs
    "E08000025", "E08000026", "E08000027", "E08000028",
    "E08000029", "E08000030", "E08000031",
    # Unitaries
    "E06000019",  # Herefordshire
    "E06000020",  # Telford and Wrekin
    "E06000021",  # Stoke-on-Trent
    "E06000051",  # Shropshire
    # Staffordshire districts
    "E07000192", "E07000193", "E07000194", "E07000195",
    "E07000196", "E07000197", "E07000198", "E07000199",
    # Warwickshire districts
    "E07000218", "E07000219", "E07000220", "E07000221", "E07000222",
    # Worcestershire districts
    "E07000234", "E07000235", "E07000236", "E07000237",
    "E07000238", "E07000239",
]
MIDLANDS = {c: "East Midlands" for c in EAST_MIDLANDS}
MIDLANDS.update({c: "West Midlands" for c in WEST_MIDLANDS})


# --------------------------------------------------------------------------- #
# 2. Funding
# --------------------------------------------------------------------------- #
def load_funding():
    f = pd.read_csv(DATA / "funding_per_lad.csv")
    ren = {
        "LAD24CD": "lad", "LAD24NM": "lad_nm",
        "Population (mid-2024 estimates)": "population",
        "Total funding (current prices)": "total_funding",
        "Per capita funding (exc. outliers - islands and the Highlands)": "funding_pc",
        "Per capita funding": "funding_pc_raw",
        "Per capita funding from Conservative governments per year in power": "con_pc_year",
        "Per capita funding from Labour governments per year in power": "lab_pc_year",
    }
    f = f.rename(columns=ren)
    for c in ["population", "total_funding", "funding_pc", "funding_pc_raw",
              "con_pc_year", "lab_pc_year"]:
        f[c] = f[c].map(num)
    # For all Midlands LADs the 'exc. outliers' column equals the raw per-capita
    # figure (no islands/Highlands), but fall back just in case.
    # Keep the spreadsheet's outlier exclusion: the "exc. outliers" column is blank
    # for the islands and the Highlands, so those stay NaN (dropped downstream).
    # No Midlands LAD is an outlier, so the Midlands analysis is unaffected.
    return f


# April-2023 unitaries -> their pre-2023 component districts, so census/MRP data keyed
# to the old geography can be bridged up to the 2024 unitary codes (used by both the
# deprivation loader and the MRP vote-share aggregator).
UNITARY_2023 = {
    "E06000063": ["E07000026", "E07000028", "E07000029"],                       # Cumberland
    "E06000064": ["E07000027", "E07000030", "E07000031"],                       # Westmorland & Furness
    "E06000065": ["E07000163", "E07000164", "E07000165", "E07000166",
                  "E07000167", "E07000168", "E07000169"],                       # North Yorkshire
    "E06000066": ["E07000187", "E07000188", "E07000189", "E07000246"],          # Somerset
}


# --------------------------------------------------------------------------- #
# 3. Deprivation (ONS Census 2021 household deprivation, 6 categories)
#    Mean number of deprivation dimensions per household (0-4); higher = more
#    deprived. Category code 1..5 -> 0..4 dimensions; -8 "Does not apply" dropped.
# --------------------------------------------------------------------------- #
def load_deprivation():
    d = pd.read_csv(DATA / "deprivation_ew.csv")
    d.columns = ["lad", "lad_nm", "cat_code", "cat_label", "obs"]
    d = d[d.cat_code >= 1].copy()          # drop -8 "Does not apply"
    d["dims"] = d.cat_code - 1             # 1->0 dims ... 5->4 dims
    g = d.groupby("lad").apply(
        lambda x: pd.Series({
            "deprivation_mean": np.average(x.dims, weights=x.obs),
            "households": x.obs.sum(),
        }),
        include_groups=False,
    ).reset_index()
    # The 2021 census predates the April-2023 unitaries, so build them from their
    # component districts (household-weighted) to keep England-wide rankings complete.
    extra = []
    for new, olds in UNITARY_2023.items():
        sub = g[g.lad.isin(olds)]
        if len(sub):
            extra.append({"lad": new,
                          "deprivation_mean": np.average(sub.deprivation_mean, weights=sub.households),
                          "households": sub.households.sum()})
    return pd.concat([g, pd.DataFrame(extra)], ignore_index=True)


# --------------------------------------------------------------------------- #
# 4. Country of birth -> share foreign-born (for the immigration robustness cut)
# --------------------------------------------------------------------------- #
def load_foreign_born():
    c = pd.read_csv(DATA / "cob_ew.csv")
    c.columns = ["lad", "lad_nm", "cat_code", "cat_label", "obs"]
    c = c[c.cat_code >= 1].copy()          # drop -8
    tot = c.groupby("lad").obs.sum().rename("cob_total")
    outside = c[c.cat_code == 2].groupby("lad").obs.sum().rename("born_outside")
    fb = pd.concat([tot, outside], axis=1).fillna(0)
    fb["foreign_born"] = fb.born_outside / fb.cob_total
    return fb.reset_index()[["lad", "foreign_born"]]


# --------------------------------------------------------------------------- #
# 5. MRP -> LAD-level vote shares & seat safeness
#    Five-quarter series Q4 2024 -> Q4 2025 (Political_MRP_by_quarter.xlsx),
#    using the "Projected w/ Tactical" projections (as in the original R script).
#    Ward codes are 2022 vintage; map ward -> LAD (2022 == 2024 for the Midlands).
#    Returns, per LAD:
#      - latest-quarter (Q4 2025) party shares + seat_safeness  (cross-sectional)
#      - reform_uk_share_<quarter> for each quarter               (time series)
#      - reform_change = Q4 2025 - Q4 2024
# --------------------------------------------------------------------------- #
PREF = "Projected w/ Tactical"
PARTIES = {
    "Projected Conservative Vote Count": "conservative",
    "Projected Labour Vote Count": "labour",
    "Projected Liberal Democrats Vote Count": "libdem",
    "Projected Green Vote Count": "green",
    "Projected Reform UK Vote Count": "reform_uk",
    "Projected Other Parties Vote Count": "other",
}
QUARTERS = {
    "Q4 2024": "q4_2024", "Q1 2025": "q1_2025", "Q2 2025": "q2_2025",
    "Q3 2025": "q3_2025", "Q4 2025": "q4_2025",
}
LATEST = "Q4 2025"


def _agg_quarter(df, w22, w24):
    """Aggregate one MRP sheet to LAD: returns party shares + seat_safeness."""
    df = (df.merge(w22, left_on="Ward_code", right_on="WD22CD", how="left")
            .merge(w24, left_on="Ward_code", right_on="WD24CD", how="left"))
    df["lad"] = df["LAD22CD"].fillna(df["LAD24CD"])
    tot, mar = PREF + "Projected Total Votes", PREF + "Win Margin"
    df[tot] = pd.to_numeric(df[tot], errors="coerce")
    df[mar] = pd.to_numeric(df[mar], errors="coerce")
    cols = {}
    for suf in PARTIES:
        df[PREF + suf] = pd.to_numeric(df[PREF + suf], errors="coerce")
        cols[PREF + suf] = "sum"
    df = df.dropna(subset=["lad", tot])
    lad = df.groupby("lad").agg({tot: "sum", mar: "sum", **cols}).reset_index()
    out = pd.DataFrame({"lad": lad["lad"]})
    for suf, name in PARTIES.items():
        out[f"{name}_share"] = lad[PREF + suf] / lad[tot]
    out["seat_safeness"] = lad[mar] / lad[tot]
    out["total_votes"] = lad[tot]
    # Bridge the April-2023 unitaries from their component districts (vote-weighted),
    # since the MRP wards are keyed to the old (pre-2023) district geography.
    extra = []
    for new, olds in UNITARY_2023.items():
        sub = out[out.lad.isin(olds)]
        if len(sub):
            w = sub.total_votes
            row = {"lad": new, "total_votes": float(w.sum())}
            for c in out.columns:
                if c.endswith("_share") or c == "seat_safeness":
                    row[c] = float(np.average(sub[c], weights=w))
            extra.append(row)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    return out


def load_vote_shares():
    xl = pd.ExcelFile(DATA / "Political_MRP_by_quarter.xlsx")
    w22 = pd.read_csv(DATA / "ward22_to_lad22.csv")[["WD22CD", "LAD22CD"]]
    w24 = pd.read_csv(DATA / "ward24_to_lad24.csv")[["WD24CD", "LAD24CD"]]

    per_q = {tag: _agg_quarter(xl.parse(sheet), w22, w24)
             for sheet, tag in QUARTERS.items()}

    # cross-sectional base = latest quarter (all party shares + seat safeness)
    out = per_q[QUARTERS[LATEST]].copy()
    # time series of Reform share
    for tag, q in per_q.items():
        out = out.merge(q[["lad", "reform_uk_share"]].rename(
            columns={"reform_uk_share": f"reform_uk_share_{tag}"}), on="lad", how="outer")
    out["reform_change"] = out["reform_uk_share_q4_2025"] - out["reform_uk_share_q4_2024"]
    return out


REFORM_Q = [f"reform_uk_share_{t}" for t in QUARTERS.values()]


# --------------------------------------------------------------------------- #
# 6. Matching toolkit (no MatchIt in Python; several estimators for robustness)
#    Treatment = high funding; single covariate = deprivation_mean; outcome =
#    Reform (or other) vote share. Each returns (treated_mean, control_mean,
#    diff, p_value) where diff > 0 => treated (heavily funded) more pro-outcome.
# --------------------------------------------------------------------------- #
def _ttest(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(np.nanmean(a)), float(np.nanmean(b)), float(np.nanmean(a) - np.nanmean(b)), float(p)


def m_nn_covariate(df, treat, outcome, cov="deprivation_mean", caliper=None):
    """Greedy 1:1 nearest-neighbour match on the covariate, without replacement."""
    t = df[df[treat] == 1]
    c = df[df[treat] == 0].copy()
    sd = df[cov].std(ddof=0)
    cap = None if caliper is None else caliper * sd
    used, tt, cc = set(), [], []
    for _, row in t.sort_values(cov).iterrows():
        avail = c[~c.index.isin(used)]
        if avail.empty:
            break
        d = (avail[cov] - row[cov]).abs()
        j = d.idxmin()
        if cap is not None and d.loc[j] > cap:
            continue
        used.add(j)
        tt.append(row[outcome])
        cc.append(c.loc[j, outcome])
    if len(tt) < 3:
        return (np.nan, np.nan, np.nan, np.nan, len(tt))
    res = _ttest(tt, cc)
    return (*res, len(tt))


def m_nn_propensity(df, treat, outcome, cov="deprivation_mean", caliper=0.2):
    """1:1 NN match on the logit propensity score, with a caliper."""
    X = sm.add_constant(df[[cov]])
    try:
        ps = sm.Logit(df[treat], X).fit(disp=0).predict(X)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    lp = np.log(ps / (1 - ps)).replace([np.inf, -np.inf], np.nan)
    work = df.assign(_lp=lp).dropna(subset=["_lp"])
    return m_nn_covariate(work, treat, outcome, cov="_lp", caliper=caliper)


def m_subclass(df, treat, outcome, cov="deprivation_mean", q=5):
    """Subclassification: OLS outcome ~ treat + C(deprivation quintile)."""
    d = df.copy()
    d["blk"] = pd.qcut(d[cov], q=min(q, d[cov].nunique()), duplicates="drop")
    X = pd.get_dummies(d["blk"], drop_first=True).astype(float)
    X["treat"] = d[treat].values
    X = sm.add_constant(X)
    r = sm.OLS(d[outcome], X).fit()
    tm = d.loc[d[treat] == 1, outcome].mean()
    cm = d.loc[d[treat] == 0, outcome].mean()
    return (tm, cm, float(r.params["treat"]), float(r.pvalues["treat"]),
            int((d[treat] == 1).sum()))


def m_ipw(df, treat, outcome, cov="deprivation_mean"):
    """Inverse-propensity weighted difference (WLS outcome ~ treat)."""
    X = sm.add_constant(df[[cov]])
    try:
        ps = sm.Logit(df[treat], X).fit(disp=0).predict(X).clip(0.02, 0.98)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    w = np.where(df[treat] == 1, 1 / ps, 1 / (1 - ps))
    Xt = sm.add_constant(df[[treat]].astype(float))
    r = sm.WLS(df[outcome], Xt, weights=w).fit()
    tm = df.loc[df[treat] == 1, outcome].mean()
    cm = df.loc[df[treat] == 0, outcome].mean()
    return (tm, cm, float(r.params[treat]), float(r.pvalues[treat]),
            int((df[treat] == 1).sum()))


def m_ancova(df, treat, outcome, cov="deprivation_mean"):
    """Regression adjustment: OLS outcome ~ treat + deprivation."""
    X = sm.add_constant(df[[treat, cov]].astype(float))
    r = sm.OLS(df[outcome], X).fit()
    tm = df.loc[df[treat] == 1, outcome].mean()
    cm = df.loc[df[treat] == 0, outcome].mean()
    return (tm, cm, float(r.params[treat]), float(r.pvalues[treat]),
            int((df[treat] == 1).sum()))


MATCHERS = {
    "nn_deprivation": lambda d, t, o: m_nn_covariate(d, t, o),
    "nn_deprivation_caliper": lambda d, t, o: m_nn_covariate(d, t, o, caliper=0.25),
    "nn_propensity_caliper": lambda d, t, o: m_nn_propensity(d, t, o),
    "subclass_quintile": lambda d, t, o: m_subclass(d, t, o),
    "ipw": lambda d, t, o: m_ipw(d, t, o),
    "ancova": lambda d, t, o: m_ancova(d, t, o),
}


# For NN specs the diff is the matched-group mean difference; for the others it
# is a covariate-adjusted coefficient (so treated/control means are raw, and the
# reported gap is the adjusted estimate).
SPEC_TYPE = {
    "nn_deprivation": "matched", "nn_deprivation_caliper": "matched",
    "nn_propensity_caliper": "matched", "subclass_quintile": "adjusted",
    "ipw": "adjusted", "ancova": "adjusted",
}


def run_matching(df, outcome="reform_uk_share", label=""):
    rows = []
    for treat in ["treat_mean", "treat_median"]:
        for spec, fn in MATCHERS.items():
            tm, cm, diff, p, n = fn(df, treat, outcome)
            rows.append({
                "sample": label, "outcome": outcome, "treatment": treat,
                "spec": spec, "estimate_type": SPEC_TYPE[spec],
                "treated_mean": tm, "control_mean": cm,
                "diff_pp": diff * 100, "p_value": p, "n_treated": n,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 74)
    print("REGENERATION FUNDING IN THE MIDLANDS")
    print("=" * 74)

    funding = load_funding()
    dep = load_deprivation()
    fb = load_foreign_born()
    votes = load_vote_shares()

    # ---- assemble Midlands frame ----
    mid = funding[funding.lad.isin(MIDLANDS)].copy()
    mid["region"] = mid.lad.map(MIDLANDS)
    # Validation: every defined Midlands code must be present in the funding file.
    missing = sorted(set(MIDLANDS) - set(funding.lad))
    print(f"\nMidlands LADs defined: {len(MIDLANDS)} "
          f"(East {len(EAST_MIDLANDS)}, West {len(WEST_MIDLANDS)})")
    print(f"Matched in funding data: {len(mid)}   Missing codes: {missing}")
    assert not missing, f"Unknown LAD codes in Midlands definition: {missing}"

    mid = (mid.merge(dep[["lad", "deprivation_mean"]], on="lad", how="left")
              .merge(fb, on="lad", how="left")
              .merge(votes, on="lad", how="left"))
    mid.to_csv(OUT / "midlands_master.csv", index=False)
    print(f"Master table -> outputs/midlands_master.csv  "
          f"(deprivation matched {mid.deprivation_mean.notna().sum()}, "
          f"vote shares matched {mid.reform_uk_share.notna().sum()})")

    # ======================================================================= #
    # ANALYSIS 0 — Headline: the Midlands' share of the ~£19bn
    # ======================================================================= #
    tot = funding.total_funding.sum()
    midf = funding[funding.lad.isin(MIDLANDS)].total_funding.sum()
    uk_pop = funding.population.sum()
    eng_pop = funding[funding.lad.str.startswith("E")].population.sum()
    mid_pop = funding[funding.lad.isin(MIDLANDS)].population.sum()
    sentence = (
        f"Over the last decade, Westminster governments spent £{tot/1e9:.0f} billion on "
        f"funding regeneration, of which £{midf/1e9:.2f} billion went to regions in the "
        f"Midlands - {midf/tot*100:.1f}% of the total, despite the fact the Midlands "
        f"constitutes {mid_pop/uk_pop*100:.1f}% of the UK and {mid_pop/eng_pop*100:.1f}% "
        f"of the English population."
    )
    # East vs West split
    reg = {r: funding[funding.lad.map(MIDLANDS) == r] for r in ("East Midlands", "West Midlands")}
    rf = {r: d.total_funding.sum() for r, d in reg.items()}
    rp = {r: d.population.sum() for r, d in reg.items()}
    ew_sentence = (
        f"Within the Midlands, the West Midlands took £{rf['West Midlands']/1e9:.2f} billion "
        f"({rf['West Midlands']/tot*100:.1f}% of the UK total, {rf['West Midlands']/midf*100:.1f}% "
        f"of Midlands funding) and the East Midlands £{rf['East Midlands']/1e9:.2f} billion "
        f"({rf['East Midlands']/tot*100:.1f}% of the UK total, {rf['East Midlands']/midf*100:.1f}% "
        f"of Midlands funding). The East Midlands holds {rp['East Midlands']/mid_pop*100:.1f}% of "
        f"the Midlands population and the West Midlands {rp['West Midlands']/mid_pop*100:.1f}%, so "
        f"per head the East Midlands is the slightly more heavily funded half."
    )
    with open(OUT / "00_headline_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# Midlands regeneration funding: headline\n\n")
        fh.write(sentence + "\n\n")
        fh.write(ew_sentence + "\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Total regeneration funding (all UK LADs) | £{tot:,.0f} |\n")
        fh.write(f"| Funding to the Midlands (East + West) | £{midf:,.0f} |\n")
        fh.write(f"| Midlands share of funding | {midf/tot*100:.1f}% |\n")
        fh.write(f"| Midlands share of UK population | {mid_pop/uk_pop*100:.1f}% |\n")
        fh.write(f"| Midlands share of English population | {mid_pop/eng_pop*100:.1f}% |\n")
        fh.write("\n| Region | Funding | % of UK total | % of Midlands funding | % of Midlands population |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in ("East Midlands", "West Midlands"):
            fh.write(f"| {r} | £{rf[r]:,.0f} | {rf[r]/tot*100:.1f}% | {rf[r]/midf*100:.1f}% | "
                     f"{rp[r]/mid_pop*100:.1f}% |\n")
    print("\n" + "-" * 74)
    print("0. HEADLINE  (-> outputs/00_headline_summary.md)")
    print("-" * 74)
    print(sentence)
    print(ew_sentence)

    # ======================================================================= #
    # ANALYSIS 1 — Top 10 by regeneration funding per capita
    # ======================================================================= #
    print("\n" + "-" * 74)
    print("1. TOP 10 MIDLANDS LADs BY REGENERATION FUNDING PER CAPITA")
    print("-" * 74)
    top10 = mid.sort_values("funding_pc", ascending=False).head(10)
    print(top10[["lad_nm", "region", "funding_pc", "population"]]
          .to_string(index=False,
                     formatters={"funding_pc": lambda v: f"£{v:,.0f}",
                                 "population": lambda v: f"{v:,.0f}"}))
    top10.to_csv(OUT / "midlands_top10_funding.csv", index=False)
    print(f"\nMidlands mean per-capita funding:   £{mid.funding_pc.mean():,.0f}")
    print(f"Midlands median per-capita funding: £{mid.funding_pc.median():,.0f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = top10.iloc[::-1]
    colours = [PF_ORANGE if r == "East Midlands" else PF_BLUE for r in t.region]
    ax.barh(t.lad_nm, t.funding_pc, color=colours)
    for y, v in enumerate(t.funding_pc):
        ax.text(v + 6, y, f"£{v:,.0f}", va="center", fontsize=9)
    ax.set_xlabel("Regeneration funding per capita (£, 2025 prices)")
    ax.set_title("Top 10 Midlands local authorities by regeneration funding per head")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PF_ORANGE, label="East Midlands"),
                       Patch(color=PF_BLUE, label="West Midlands")],
              loc="lower right", frameon=False)
    ax.margins(x=0.12)
    save(fig, "01_top10_funding.png")

    # ======================================================================= #
    # ANALYSIS 2 — Funding vs deprivation and vs vote choice
    # ======================================================================= #
    print("\n" + "-" * 74)
    print("2. FUNDING vs DEPRIVATION AND VOTE CHOICE (Midlands)")
    print("-" * 74)
    a = mid.dropna(subset=["deprivation_mean", "funding_pc"])
    r_dep = stats.pearsonr(a.deprivation_mean, a.funding_pc)
    print(f"corr(funding, deprivation)      r={r_dep.statistic:+.3f}  p={r_dep.pvalue:.4f}  n={len(a)}")

    b = mid.dropna(subset=["reform_uk_share", "funding_pc"])
    r_ref = stats.pearsonr(b.reform_uk_share, b.funding_pc)
    print(f"corr(funding, Reform share)     r={r_ref.statistic:+.3f}  p={r_ref.pvalue:.4f}  n={len(b)}")

    c = mid.dropna(subset=["seat_safeness", "funding_pc"])
    r_saf = stats.pearsonr(c.seat_safeness, c.funding_pc)
    print(f"corr(funding, seat safeness)    r={r_saf.statistic:+.3f}  p={r_saf.pvalue:.4f}  n={len(c)}")

    def scatter(x, y, xlab, ylab, title, fname, xfmt=None):
        d = mid.dropna(subset=[x, y])
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for reg, col in [("East Midlands", PF_ORANGE), ("West Midlands", PF_BLUE)]:
            s = d[d.region == reg]
            ax.scatter(s[x], s[y], c=col, alpha=0.8, s=42, label=reg, edgecolor="white", linewidth=0.5)
        # OLS fit line
        m, q = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xs, m * xs + q, color=PF_GREY, lw=1.5, ls="--")
        # label a few notable points
        for _, row in d.sort_values(y, ascending=False).head(5).iterrows():
            ax.annotate(row.lad_nm, (row[x], row[y]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        if xfmt:
            ax.xaxis.set_major_formatter(xfmt)
        save(fig, fname)

    from matplotlib.ticker import FuncFormatter, PercentFormatter
    gbp = FuncFormatter(lambda v, _: f"£{v:,.0f}")
    scatter("deprivation_mean", "funding_pc",
            "Household deprivation (mean dimensions, 0-4)",
            "Funding per capita (£)",
            "Regeneration funding vs deprivation (Midlands)",
            "02a_funding_vs_deprivation.png")
    scatter("reform_uk_share", "funding_pc",
            "Reform UK vote share (Nov 2025 MRP)", "Funding per capita (£)",
            "Regeneration funding vs Reform support (Midlands)",
            "02b_funding_vs_reform.png")
    scatter("seat_safeness", "funding_pc",
            "Seat safeness (winning margin / total votes)", "Funding per capita (£)",
            "Regeneration funding vs marginality (Midlands)",
            "02c_funding_vs_marginality.png")

    # ---- OLS: which correlates more strongly with funding? ----
    reg_df = mid.dropna(subset=["funding_pc", "deprivation_mean",
                                "reform_uk_share", "seat_safeness"]).copy()
    # standardise predictors so coefficients are comparable
    for col in ["deprivation_mean", "reform_uk_share", "seat_safeness"]:
        reg_df[col + "_z"] = (reg_df[col] - reg_df[col].mean()) / reg_df[col].std()
    X = sm.add_constant(reg_df[["deprivation_mean_z", "reform_uk_share_z", "seat_safeness_z"]])
    ols = sm.OLS(reg_df["funding_pc"], X).fit()
    print("\nOLS: funding_pc ~ deprivation + Reform share + seat safeness "
          "(standardised predictors)")
    print(ols.summary().tables[1])
    (OUT / "02_ols_funding_drivers.txt").write_text(str(ols.summary()), encoding="utf-8")

    # ======================================================================= #
    # ANALYSIS 3 — Matching: is Reform support higher in funded Midlands areas?
    # ======================================================================= #
    print("\n" + "-" * 74)
    print("3. MATCHING: REFORM SUPPORT IN FUNDED vs COMPARABLE MIDLANDS AREAS")
    print("-" * 74)
    mm = mid.dropna(subset=["deprivation_mean", "reform_uk_share", "funding_pc"]).copy()
    mm["treat_mean"] = (mm.funding_pc > mm.funding_pc.mean()).astype(int)
    mm["treat_median"] = (mm.funding_pc > mm.funding_pc.median()).astype(int)
    print(f"Analysis sample: {len(mm)} Midlands LADs "
          f"(treated>mean: {mm.treat_mean.sum()}, treated>median: {mm.treat_median.sum()})")

    res_main = run_matching(mm, "reform_uk_share", label="all_midlands")

    # ---- immigration robustness: drop LADs > 13% foreign-born ----
    mm_imm = mm[mm.foreign_born <= 0.13].copy()
    mm_imm["treat_mean"] = (mm_imm.funding_pc > mm_imm.funding_pc.mean()).astype(int)
    mm_imm["treat_median"] = (mm_imm.funding_pc > mm_imm.funding_pc.median()).astype(int)
    print(f"Immigration-controlled sample (<=13% foreign-born): {len(mm_imm)} LADs")
    res_imm = run_matching(mm_imm, "reform_uk_share", label="low_immigration")

    # matching on the CHANGE in Reform support over the year (Q4'24 -> Q4'25)
    res_chg = run_matching(mm, "reform_change", label="all_midlands")
    res_chg_imm = run_matching(mm_imm, "reform_change", label="low_immigration")

    res = pd.concat([res_main, res_imm, res_chg, res_chg_imm], ignore_index=True)
    res.to_csv(OUT / "03_matching_results.csv", index=False)

    def show(block, title):
        print(f"\n{title}")
        print(block[["treatment", "spec", "estimate_type", "treated_mean",
                     "control_mean", "diff_pp", "p_value", "n_treated"]]
              .to_string(index=False,
                         formatters={"treated_mean": lambda v: f"{v:.3f}",
                                     "control_mean": lambda v: f"{v:.3f}",
                                     "diff_pp": lambda v: f"{v:+.1f}pp",
                                     "p_value": lambda v: f"{v:.3f}"}))
    show(res_main, "All Midlands LADs (treated = heavily funded):")
    show(res_imm, "Excluding LADs with >13% foreign-born:")

    # summary of the headline (Reform gap), main sample
    d = res_main.diff_pp
    print(f"\nHEADLINE (all Midlands): median Reform gap across specs = "
          f"{np.median(d):+.1f}pp; positive in {int((d > 0).sum())}/{len(d)} specs; "
          f"significant (p<0.05) in {int((res_main.p_value < 0.05).sum())}/{len(res_main)}.")
    di = res_imm.diff_pp
    print(f"HEADLINE (low-immigration): median Reform gap = {np.median(di):+.1f}pp; "
          f"positive in {int((di > 0).sum())}/{len(di)} specs.")

    # change over the year: is the RISE in Reform bigger in funded areas?
    dc = res_chg.diff_pp
    print(f"\nReform CHANGE Q4'24->Q4'25 (all Midlands): treated rose "
          f"{res_chg.treated_mean.mean()*100:+.1f}pp vs control "
          f"{res_chg.control_mean.mean()*100:+.1f}pp; matched gap median "
          f"{np.median(dc):+.1f}pp, significant in "
          f"{int((res_chg.p_value < 0.05).sum())}/{len(res_chg)} specs.")

    # ---- headline chart: matched treated vs control Reform share (subclass spec) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    specs = res_main[res_main.treatment == "treat_mean"].copy()
    # anchor both bars so the visual gap == each spec's estimated gap
    specs["comp"] = specs.treated_mean - specs.diff_pp / 100
    x = np.arange(len(specs))
    ax.bar(x - 0.2, specs.comp * 100, width=0.4, color=PF_GREY,
           label="Comparable (less-funded) areas")
    ax.bar(x + 0.2, specs.treated_mean * 100, width=0.4, color=PF_ORANGE,
           label="Heavily funded areas")
    for xi, (_, r) in zip(x, specs.iterrows()):
        star = "*" if r.p_value < 0.05 else ""
        ax.text(xi, max(r.treated_mean, r.comp) * 100 + 0.4,
                f"{r.diff_pp:+.1f}{star}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(specs.spec, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Reform UK vote share (%)")
    ax.set_title("Reform support: heavily funded vs comparable Midlands areas\n"
                 "(matched on deprivation; treatment = above-mean funding)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "03_matching_reform_gap.png")

    # ---- over-time chart: matched treated vs control Reform share by quarter ----
    # (mirrors the R "average Reform support over time" plot: mean across the
    #  matched specs of the treated/control group means, for each quarter)
    def over_time(sample):
        rows = []
        for tag in QUARTERS.values():
            rr = run_matching(sample, f"reform_uk_share_{tag}")
            rr = rr[rr.estimate_type == "matched"]
            rows.append({"quarter": tag,
                         "treated": rr.treated_mean.mean(),
                         "control": rr.control_mean.mean()})
        return pd.DataFrame(rows)

    ot = over_time(mm)
    ot.to_csv(OUT / "03_reform_over_time.csv", index=False)
    labels = list(QUARTERS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.plot(x, ot.treated * 100, "-o", color=PF_ORANGE, lw=2, label="Heavily funded areas")
    ax.plot(x, ot.control * 100, "-o", color=PF_GREY, lw=2, label="Comparable (less-funded) areas")
    ax.set_xticks(x); ax.set_xticklabels([q.replace("_", " ").upper() for q in labels])
    ax.set_ylabel("Mean Reform UK vote share (%)")
    ax.set_title("Reform support over time: funded vs comparable Midlands areas\n"
                 "(matched on deprivation; average of matched specs)")
    ax.legend(frameon=False)
    save(fig, "03_reform_over_time.png")

    # ======================================================================= #
    # ANALYSIS 4 — Funded vs comparable areas (control-pool), incl. the same
    #              immigration adjustment as the national analysis
    # ======================================================================= #
    print("\n" + "-" * 74)
    print("4. FUNDED vs COMPARABLE AREAS (control-pool), with immigration adjustment")
    print("-" * 74)
    region = (pd.read_csv(DATA / "lad24_to_region.csv")[["LAD24CD", "RGN24NM"]]
              .rename(columns={"LAD24CD": "lad", "RGN24NM": "region"}))
    eng = (funding.merge(dep[["lad", "deprivation_mean"]], on="lad", how="inner")
                  .merge(votes[["lad", "reform_uk_share"]], on="lad", how="inner")
                  .merge(fb, on="lad", how="left").merge(region, on="lad", how="inner")
                  .dropna(subset=["deprivation_mean", "reform_uk_share", "funding_pc"]))
    eng["is_mid"] = eng.lad.isin(MIDLANDS)
    thr = eng.funding_pc.mean()

    def control_bars(frame):
        treated = frame[frame.is_mid & (frame.funding_pc > thr)]
        pools = {"Comparable areas\nin the Midlands": frame[frame.is_mid & (frame.funding_pc <= thr)],
                 "Comparable areas in\nthe rest of England": frame[~frame.is_mid & (frame.funding_pc <= thr)]}
        rows = []
        for name, ctrl in pools.items():
            fr = pd.concat([treated.assign(t=1), ctrl.assign(t=0)])
            tm, cm, diff, p, n = m_nn_covariate(fr, "t", "reform_uk_share")
            rows.append({"pool": name, "treated": tm, "control": cm, "gap": diff, "p": p, "n": n})
        return pd.DataFrame(rows), len(treated)

    raw, n_all = control_bars(eng)
    adj, n_adj = control_bars(eng[eng.foreign_born <= 0.13])   # national analysis: drop >13% foreign-born
    raw["adjustment"], adj["adjustment"] = "all areas", "<=13% foreign-born"
    pd.concat([raw, adj], ignore_index=True).to_csv(OUT / "04_control_pool_comparison.csv", index=False)
    print(f"treated (funded Midlands > £{thr:,.0f} mean): all={n_all}, <=13% foreign-born={n_adj}")
    for lbl, sub in [("all areas", raw), ("<=13% foreign-born", adj)]:
        for _, r in sub.iterrows():
            print(f"   [{lbl:18s}] {r.pool.replace(chr(10),' '):40s} gap {r.gap*100:+.1f}pp (p={r.p:.3f}, n={r.n})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, (title, sub) in zip(axes, [("Deprivation-matched (all areas)", raw),
                                       ("Also adjusting for immigration (LADs ≤13% foreign-born)", adj)]):
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub.treated * 100, 0.4, color=PF_ORANGE, label="Funded Midlands areas")
        ax.bar(x + 0.2, sub.control * 100, 0.4, color=PF_GREY, label="Comparable (matched) areas")
        for xi, r in zip(x, sub.itertuples()):
            ax.text(xi, max(r.treated, r.control) * 100 + 0.8, f"gap {r.gap*100:+.1f}pp",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 44); ax.set_xticks(x); ax.set_xticklabels(sub.pool)
        ax.set_title(title, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Mean Reform UK vote share (%)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Reform in funded Midlands areas vs comparable areas (matched on deprivation)\n"
                 "left = all areas; right = adjusting for immigration as in the national analysis",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "04_control_pool_comparison.png")

    print("\n" + "=" * 74)
    print("Done. Tables and charts in ./outputs")
    print("=" * 74)


if __name__ == "__main__":
    main()
