"""
Build the treated / control frame for the Reform-council spending study.

Treated  = councils Reform won at the May 2025 locals (their first own budget is 2026/27).
Control   = comparable non-Reform authorities of the same type, to net out the annual
            finance settlement and general austerity from any "Reform effect".

The treated set is curated by hand (small, and needs judgement on majority vs minority,
election wave and LGR status). ONS/class codes are pulled from the RA 2026-27 file so
they are guaranteed to match the spending panel. The control pool is derived from the
same file (all shire counties / unitaries / met boroughs that are NOT Reform-run).

Sources for the treated list and postponements:
  - 2025 UK local elections (Wikipedia; House of Commons Library CBP-10272)
  - LGR postponements (Wikipedia "Upcoming structural changes to local government")

Run:  python 01_build_frame.py   ->  data/treated_control_frame.csv
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
RA = DATA / "RA_2026-27_Part_1.ods"

# --- treated set (May 2025), curated -----------------------------------------
# group: majority = outright control (clean treatment, full budget control 2026/27)
#        minority = largest party / minority administration (budget needs cross-party support)
TREATED = {
    # shire counties, outright majority -- all two-tier, inside the LGR programme
    "Derbyshire":       ("majority", "shire_county", "two_tier_reorganising"),
    "Kent":             ("majority", "shire_county", "two_tier_reorganising"),
    "Lancashire":       ("majority", "shire_county", "two_tier_reorganising"),
    "Lincolnshire":     ("majority", "shire_county", "two_tier_reorganising"),
    "Nottinghamshire":  ("majority", "shire_county", "two_tier_reorganising"),
    "Staffordshire":    ("majority", "shire_county", "two_tier_reorganising"),
    # unitaries / met, outright majority -- structurally stable
    "North Northamptonshire": ("majority", "unitary", "stable_unitary"),   # created 2021
    "West Northamptonshire":  ("majority", "unitary", "stable_unitary"),   # created 2021
    "Doncaster":              ("majority", "met_borough", "stable"),
    # minority / largest-party administrations (May 2025)
    "Durham":         ("minority", "unitary", "stable_unitary"),           # created 2009
    "Leicestershire": ("minority", "shire_county", "two_tier_reorganising"),
    "Warwickshire":   ("minority", "shire_county", "two_tier_reorganising"),
    "Worcestershire": ("minority", "shire_county", "two_tier_reorganising"),
}

# Shire counties whose May-2025 election was POSTPONED to 2026 (LGR). Not clean controls:
# they are a possible staggered second treatment cohort (first Reform budget 2027/28 where won).
POSTPONED_2026 = {
    "East Sussex", "Essex", "Hampshire", "Norfolk", "Suffolk", "Surrey", "West Sussex",
}


def load_ids() -> pd.DataFrame:
    raw = pd.read_excel(RA, engine="odf", sheet_name="RA_LA_Data_2026-27", header=None)
    ids = raw.iloc[10:, 0:5].copy()
    ids.columns = ["ecode", "ons", "la", "class", "subclass"]
    ids = ids.dropna(subset=["la"]).reset_index(drop=True)
    ids["la"] = ids["la"].astype(str).str.strip()
    return ids[["ons", "la", "class"]]


def main() -> None:
    ids = load_ids()
    rows = []
    for _, r in ids.iterrows():
        name, cls, ons = r["la"], r["class"], r["ons"]
        if name in TREATED:
            grp, gtype, lgr = TREATED[name]
            rows.append(dict(
                ons_code=ons, la_name=name, ra_class=cls,
                treatment_group=f"treated_{grp}",
                reform_control=grp, wave="2025", first_reform_budget="2026-27",
                govt_type=gtype, lgr_status=lgr, control_quality="",
            ))
        elif cls == "SC" and name in POSTPONED_2026:
            rows.append(dict(
                ons_code=ons, la_name=name, ra_class=cls,
                treatment_group="second_wave_or_postponed",
                reform_control="varies_2026", wave="2026", first_reform_budget="2027-28",
                govt_type="shire_county", lgr_status="two_tier_reorganising",
                control_quality="postponed_2026",
            ))
        elif cls == "SC":
            # remaining shire counties that voted May-2025 and stayed non-Reform = gold controls
            rows.append(dict(
                ons_code=ons, la_name=name, ra_class=cls,
                treatment_group="control_candidate",
                reform_control="none", wave="", first_reform_budget="",
                govt_type="shire_county", lgr_status="two_tier_reorganising",
                control_quality="clean_2025_shire_county",
            ))
        elif cls in ("UA", "MD") and name not in TREATED:
            rows.append(dict(
                ons_code=ons, la_name=name, ra_class=cls,
                treatment_group="control_candidate",
                reform_control="none", wave="", first_reform_budget="",
                govt_type="unitary" if cls == "UA" else "met_borough",
                lgr_status="stable_unitary" if cls == "UA" else "stable",
                control_quality="same_type_non_reform",
            ))

    frame = pd.DataFrame(rows).sort_values(
        ["treatment_group", "govt_type", "la_name"]
    ).reset_index(drop=True)

    out = DATA / "treated_control_frame.csv"
    frame.to_csv(out, index=False)

    # --- report ---
    print(f"wrote {out}  ({len(frame)} authorities)")
    print("\ntreated (May 2025):")
    t = frame[frame.treatment_group.str.startswith("treated")]
    for _, r in t.iterrows():
        print(f"  {r.reform_control:<9} {r.govt_type:<12} {r.la_name:<24} {r.ons_code}  [{r.lgr_status}]")
    print("\ncontrol pool by quality:")
    print(frame[frame.treatment_group == "control_candidate"]
          .control_quality.value_counts().to_string())
    print("\nsecond-wave / postponed (not clean controls):")
    print("  " + ", ".join(frame[frame.treatment_group == "second_wave_or_postponed"].la_name))


if __name__ == "__main__":
    main()
