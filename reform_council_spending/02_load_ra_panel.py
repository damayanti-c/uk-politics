"""
Load MHCLG Revenue Account (RA) budget data into a tidy per-authority, per-service panel.

Covers the 2022-23 -> 2026-27 budget files (Reform's first own budget is 2026-27). Service
lines are picked by their MHCLG data-asset CODE (e.g. 'asctot'), not by column position, and
the code row / header row / identifier columns are all DETECTED per file, so the loader
survives the layout drift between years (2022-23 puts codes on row 2, later years on row 6).

The 2021-22 file is intentionally excluded: it uses full-text column headers instead of short
codes, a different identifier layout, and is COVID-distorted. Add a bespoke parser only if a
longer pre-trend is needed.

Money is nominal £000s as published. Real-terms deflation and per-capita scaling happen in the
analysis step (join a GDP deflator and ONS population), not here.

Run:  python 02_load_ra_panel.py
  ->  data/panel_ra_long.csv   (tidy national panel: authority x service x year)
  ->  data/panel_ra_wide.csv   (analysis frame: treated + controls, service columns)
"""
from pathlib import Path
import re
import pandas as pd

GSS = re.compile(r"^[ESWN]\d{8}$")  # 9-char ONS/GSS code, e.g. E10000016

DATA = Path(__file__).parent / "data"

# financial year -> RA Part-1 (or combined) file in ./data. Add prior years to extend.
YEAR_FILES = {
    "2022-23": "RA_2022-23.ods",
    "2023-24": "RA_2023-24_Part_1.ods",
    "2024-25": "RA_2024-25_Part_1.ods",
    "2025-26": "RA_2025-26_Part_1.ods",
    "2026-27": "RA_2026-27_Part_1.ods",
}

# MHCLG asset code -> friendly service name (reporting order).
SERVICE_CODES = {
    "edutot": "Education", "transtot": "Highways & transport",
    "csctot": "Children's social care", "asctot": "Adult social care",
    "phtot": "Public health", "housgfcftot": "Housing (non-HRA)",
    "cultot": "Cultural & related", "envtot": "Environmental & regulatory",
    "plantot": "Planning & development", "poltot": "Police", "frstot": "Fire & rescue",
    "centot": "Central services", "othtot": "Other services",
    "servicetot": "TOTAL service expenditure", "netcurrtot": "Net current expenditure",
    "revenuetot": "Revenue expenditure", "ctrtot": "Council tax requirement",
}
DISCRETIONARY = ["Cultural & related", "Environmental & regulatory",
                 "Planning & development", "Central services", "Highways & transport"]
STATUTORY = ["Adult social care", "Children's social care", "Public health", "Education"]


def _find_row(raw, predicate, limit=15):
    for r in range(min(limit, raw.shape[0])):
        vals = [str(raw.iat[r, c]).strip() for c in range(raw.shape[1])]
        if predicate(vals):
            return r
    return None


def load_ra(filename: str, year: str) -> pd.DataFrame:
    """Tidy long frame for one year: ons, la, ra_class, year, service, amount_000."""
    path = DATA / filename
    xl = pd.ExcelFile(path, engine="odf")
    sheets = [s for s in xl.sheet_names if s.startswith("RA_LA_Data")]
    if not sheets:
        raise ValueError(f"{year}: no 'RA_LA_Data' sheet in {filename}")
    raw = pd.read_excel(path, engine="odf", sheet_name=sheets[0], header=None)

    coderow = _find_row(raw, lambda v: "edutot" in v)
    hdrrow = _find_row(raw, lambda v: "ONS Code" in v)
    if coderow is None or hdrrow is None:
        raise ValueError(f"{year}: short-code layout not found (pre-2022 text-label format?)")

    codes = {str(raw.iat[coderow, c]): c for c in range(raw.shape[1])}
    missing = [k for k in SERVICE_CODES if k not in codes]
    if missing:
        raise ValueError(f"{year}: service codes absent: {missing}")

    hdr = {str(raw.iat[hdrrow, c]).strip(): c for c in range(raw.shape[1])}
    la_c, cls_c = hdr["Local authority"], hdr["Class"]

    body = raw.iloc[hdrrow + 1:, :]
    # Detect the GSS-code column by CONTENT, not header label: the 2022-23 file mislabels the
    # E-code / ONS Code headers (they are swapped relative to the data), so trusting the label
    # extracts the wrong column. Pick the id column with the most 9-char GSS codes.
    ons_c = max(range(cls_c),
                key=lambda c: body.iloc[:, c].astype(str).str.strip().str.match(GSS).sum())
    wide = pd.DataFrame({
        "ons": body.iloc[:, ons_c].astype(str).str.strip(),
        "la": body.iloc[:, la_c].astype(str).str.strip(),
        "ra_class": body.iloc[:, cls_c].astype(str).str.strip(),
    })
    wide = wide[wide["la"].ne("nan") & wide["la"].ne("")]
    for code, name in SERVICE_CODES.items():
        wide[name] = pd.to_numeric(body.iloc[:, codes[code]].reindex(wide.index),
                                   errors="coerce")
    wide["year"] = year
    return wide.melt(id_vars=["ons", "la", "ra_class", "year"],
                     var_name="service", value_name="amount_000")


def main() -> None:
    frame = pd.read_csv(DATA / "treated_control_frame.csv")
    long = pd.concat([load_ra(fn, yr) for yr, fn in YEAR_FILES.items()], ignore_index=True)

    long = long.merge(frame[["ons_code", "la_name", "treatment_group", "reform_control",
                             "govt_type", "control_quality"]],
                      left_on="ons", right_on="ons_code", how="left")
    # keep one canonical name per ONS code (files rename e.g. "Durham" <-> "Durham UA" by year)
    long["la"] = long["la_name"].fillna(long["la"])
    long = long.drop(columns=["la_name"])
    long["treatment_group"] = long["treatment_group"].fillna("other")
    for col in ["reform_control", "govt_type", "control_quality"]:
        long[col] = long[col].fillna("")

    long.drop(columns=["ons_code"]).to_csv(DATA / "panel_ra_long.csv", index=False)

    # wide = analysis frame (treated + controls) for the latest year, one col per service
    latest = max(YEAR_FILES)
    wframe = long[(long.treatment_group != "other") & (long.year == latest)]
    wide = wframe.pivot_table(index=["ons", "la", "ra_class", "treatment_group",
                                     "reform_control", "govt_type", "control_quality"],
                              columns="service", values="amount_000").reset_index()
    wide.to_csv(DATA / "panel_ra_wide.csv", index=False)

    # --- report ---
    yrs = sorted(long.year.unique())
    print(f"panel: {long.ons.nunique()} authorities x {len(yrs)} years {yrs}  ({len(long):,} rows)")
    dropped = sorted(set(wframe.la) - set(wide.la))
    if dropped:
        print(f"no {latest} return for: {', '.join(dropped)}")

    # treated total service spend over time (£m), wide across years -> pre-trend eyeball
    t = long[long.treatment_group.str.startswith("treated") &
             (long.service == "TOTAL service expenditure")]
    piv = (t.pivot_table(index="la", columns="year", values="amount_000")
           / 1000).round(0)
    print(f"\ntreated councils, TOTAL service budget by year (£m):")
    print(piv.sort_values(latest, ascending=False).to_string())

    def disc_share(df):
        return (100 * df[DISCRETIONARY].sum(axis=1) / df["TOTAL service expenditure"]).mean()
    tc = wide[(wide.treatment_group == "treated_majority") & (wide.ra_class == "SC")]
    cc = wide[wide.control_quality == "clean_2025_shire_county"]
    print(f"\ndiscretionary share of service spend, shire counties, {latest} (mean %):")
    print(f"  treated (Reform majority): {disc_share(tc):.1f}%  (n={len(tc)})")
    print(f"  control (non-Reform 2025): {disc_share(cc):.1f}%  (n={len(cc)})")


if __name__ == "__main__":
    main()
