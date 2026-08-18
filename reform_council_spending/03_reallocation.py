"""
Is there DOGE-style reallocation in the first Reform budget?

Signature we test: in 2026-27 (Reform's first own budget), the discretionary "target" lines
(cultural, environmental/climate, planning, central/back-office) grow less / shrink and lose
budget share, relative BOTH to the councils' own 2022-25 pre-trend AND to comparable non-Reform
councils. We difference treated-minus-control so the common finance settlement and inflation
cancel out.

Cleanest like-for-like: 6 Reform-majority shire counties vs 6 non-Reform shire counties that
also voted May 2025. Minority-Reform counties (3) shown as a weaker-treatment contrast.

Descriptive only (n=6 per group). We report the treated-vs-control difference AND a consistency
count (how many treated councils moved the same way), which is more honest than a t-test at n=6.

Run:  python 03_reallocation.py   ->  outputs/reallocation_*.csv  + stdout tables
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# genuine spending services that sum to TOTAL service expenditure (police/fire/other = 0 for
# counties; the rest are aggregates, not services).
SERVICES = ["Education", "Highways & transport", "Children's social care", "Adult social care",
            "Public health", "Housing (non-HRA)", "Cultural & related",
            "Environmental & regulatory", "Planning & development", "Central services"]
# DOGE would target these; social care / education are demand- and grant-driven (statutory).
DISCRETIONARY = ["Cultural & related", "Environmental & regulatory",
                 "Planning & development", "Central services", "Highways & transport"]
TOTAL = "TOTAL service expenditure"
PRE, POST = "2025-26", "2026-27"   # predecessor budget -> first Reform budget


def council_year_matrix(long, frame):
    """council x service, per year, plus each service's share of total that year.

    Pivot on ons/la/year only: putting the (often-empty) group labels in the index would let
    pivot_table drop treated rows whose control_quality is NaN. Merge the labels back after.
    """
    m = long[long.service.isin(SERVICES + [TOTAL])].pivot_table(
        index=["ons", "la", "year"], columns="service", values="amount_000").reset_index()
    for s in SERVICES:
        m[s + " %"] = 100 * m[s] / m[TOTAL]
    return m.merge(frame[["ons_code", "treatment_group", "control_quality"]],
                   left_on="ons", right_on="ons_code", how="left")


def group_slice(m, mask, year):
    return m[mask & (m.year == year)].set_index("ons")


def main():
    long = pd.read_csv(DATA / "panel_ra_long.csv")
    frame = pd.read_csv(DATA / "treated_control_frame.csv")
    m = council_year_matrix(long, frame)

    # explicit council sets (shire counties only, like-for-like)
    tmaj = m[(m.treatment_group == "treated_majority")]
    counties_t = tmaj[tmaj.la.isin(["Derbyshire", "Kent", "Lancashire", "Lincolnshire",
                                    "Nottinghamshire", "Staffordshire"])].ons.unique()
    counties_c = m[m.control_quality == "clean_2025_shire_county"].ons.unique()
    counties_min = m[(m.treatment_group == "treated_minority") &
                     m.la.isin(["Leicestershire", "Warwickshire", "Worcestershire"])].ons.unique()

    def deltas(ons_set):
        """per-service: mean share-change (pp) and mean real-ish growth 2025-26->2026-27."""
        pre = m[(m.ons.isin(ons_set)) & (m.year == PRE)].set_index("ons")
        post = m[(m.ons.isin(ons_set)) & (m.year == POST)].set_index("ons")
        rows = {}
        for s in SERVICES:
            dshare = (post[s + " %"] - pre[s + " %"])            # percentage points
            growth = 100 * (post[s] / pre[s] - 1)                # nominal % (common inflation cancels in diff)
            rows[s] = dict(share_chg_pp=dshare.mean(), growth_pct=growth.mean(),
                           n_cut_share=int((dshare < 0).sum()), n=len(dshare))
        return pd.DataFrame(rows).T

    dt = deltas(counties_t)
    dc = deltas(counties_c)
    dm = deltas(counties_min)

    # treated-minus-control difference (the diff-in-diff)
    comp = pd.DataFrame({
        "treated_share_chg_pp": dt.share_chg_pp,
        "control_share_chg_pp": dc.share_chg_pp,
        "DiD_share_pp": dt.share_chg_pp - dc.share_chg_pp,
        "treated_growth_%": dt.growth_pct,
        "control_growth_%": dc.growth_pct,
        "DiD_growth_pp": dt.growth_pct - dc.growth_pct,
        "treated_n_cut": dt.n_cut_share.astype(int),
    }).round(2)
    comp["target"] = comp.index.isin(DISCRETIONARY)
    comp = comp.sort_values("DiD_growth_pp")

    comp.to_csv(OUT / "reallocation_service_did.csv")
    print(f"Reform-majority counties (n={len(counties_t)}) vs non-Reform counties (n={len(counties_c)})")
    print(f"budget window {PRE} (predecessor) -> {POST} (first Reform budget)\n")
    print("service-level diff-in-diff, sorted by relative growth (negative = Reform grew it less):\n")
    show = comp[["treated_growth_%", "control_growth_%", "DiD_growth_pp",
                 "DiD_share_pp", "treated_n_cut", "target"]]
    print(show.to_string())

    # discretionary bundle: event-study across all years to check pre-trend parallelism
    print("\ndiscretionary bundle (culture+env+planning+central+transport) as % of service spend:")
    disc = long[long.service.isin(DISCRETIONARY)].groupby(["ons", "year"]).amount_000.sum()
    tot = long[long.service == TOTAL].set_index(["ons", "year"]).amount_000
    dshare = (100 * disc / tot).rename("disc_pct").reset_index()
    for label, oset in [("treated_majority_counties", counties_t),
                        ("control_counties", counties_c),
                        ("minority_counties", counties_min)]:
        row = dshare[dshare.ons.isin(oset)].groupby("year").disc_pct.mean().round(2)
        print(f"  {label:<28} " + "  ".join(f"{y}:{row[y]}" for y in sorted(row.index)))
    dshare.to_csv(OUT / "reallocation_disc_share_by_year.csv", index=False)

    # headline verdict
    tg = comp.loc[DISCRETIONARY, "DiD_growth_pp"]
    print(f"\ndiscretionary lines DiD growth (pp, treated minus control):")
    print("  " + "  ".join(f"{s.split()[0]}:{v:+.1f}" for s, v in tg.items()))


if __name__ == "__main__":
    main()
