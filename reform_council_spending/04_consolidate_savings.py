"""
Consolidate the per-council savings extractions (data/savings_raw/*.json) into one coded
dataset, then run the savings-claim WATERFALL and the DOGE tests.

Waterfall logic (transparent, each filter is a subtotal, not a black box):
  headline           = sum of all extracted 2026-27 savings lines
  - carried_forward  = savings inherited from the pre-May-2025 MTFS  -> NEW savings
  - not attributable = new but technical/corporate (debt, pensions, MRP, grant income) the
                       administration did not choose  -> NEW & REFORM-ATTRIBUTABLE
  - income/grant     = income generation & grant substitution (not a spending cut) -> CASHABLE cut
  split statutory vs discretionary -> GENUINE NEW DISCRETIONARY CASHABLE CUTS
DOGE test: how much of the cuts hit the discretionary "target" areas, and are any lines
explicitly EDI / climate / comms / consultancy.

Run:  python 04_consolidate_savings.py
  ->  data/savings_by_council.csv  + stdout waterfall & DOGE tables
"""
from pathlib import Path
import json
import pandas as pd

DATA = Path(__file__).parent / "data"
RAW = DATA / "savings_raw"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

DISCRETIONARY = {"Cultural & related", "Environmental & regulatory", "Planning & development",
                 "Central services", "Highways & transport"}
ATTRIB = {"yes", "partial"}  # reform-attributable (partial counted in; also reported separately)
DOGE_KEYWORDS = ["edi", "diversity", "equalit", "climate", "net zero", "net-zero",
                 "consultan", "communications", "comms", " pr ", "public relations"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, meta = [], []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        meta.append({k: d.get(k) for k in ("council", "source_doc", "source_url",
                     "headline_total_savings_2026_27_000", "council_tax_decision",
                     "coverage_note")})
        # Kent's Appendix F uses negative = saving; flip so all councils share
        # "positive = saving". Flagged per-file with sign_is_negative_saving.
        flip = -1 if d.get("sign_is_negative_saving") else 1
        for s in d.get("savings", []):
            s = dict(s)
            s["council"] = d["council"]
            s["service_area"] = str(s.get("service_area", "")).replace("&amp;", "&").strip()
            s["f4_shows_in_ra"] = s.get("f4_shows_in_ra", "tbd")
            for a in ("amount_000_2026_27", "amount_000_full_mtfs"):
                if s.get(a) is not None:
                    s[a] = flip * s[a]
            recs.append(s)
    return pd.DataFrame(recs), pd.DataFrame(meta)


def waterfall(df: pd.DataFrame) -> pd.DataFrame:
    v = df.amount_000_2026_27.fillna(0)
    new = df.f1_new_vs_carried.eq("new")
    attr = df.reform_attributable.isin(ATTRIB)
    income = df.f2_type.eq("income")
    costav = df.f2_type.eq("cost_avoidance")
    disc = df.service_area.isin(DISCRETIONARY)
    cut = new & attr & ~income & ~costav          # genuine new, attributable, cashable cut
    steps = [
        ("Headline (all extracted lines)", v.sum()),
        ("  less carried-forward / inherited", -v[~new].sum()),
        ("= New savings", v[new].sum()),
        ("  less not reform-attributable (technical/corporate)", -v[new & ~attr].sum()),
        ("= New & reform-attributable", v[new & attr].sum()),
        ("  less income generation & grant substitution", -v[new & attr & income].sum()),
        ("  less cost-avoidance (demand mitigation, not a cut)", -v[new & attr & ~income & costav].sum()),
        ("= New, attributable, CASHABLE spending cuts", v[cut].sum()),
        ("     of which STATUTORY (social care/education/PH)", v[cut & ~disc].sum()),
        ("     of which DISCRETIONARY (the DOGE target)", v[cut & disc].sum()),
    ]
    return pd.DataFrame(steps, columns=["step", "gbp_000"])


def main():
    df, meta = load()
    if df.empty:
        print("no extractions in", RAW); return
    cols = ["council", "saving_ref", "description", "service_area", "amount_000_2026_27",
            "amount_000_full_mtfs", "f1_new_vs_carried", "f2_type",
            "f3_statutory_vs_discretionary", "f4_shows_in_ra", "f5_delivery_rag",
            "reform_attributable", "notes"]
    df = df.reindex(columns=cols)
    df.to_csv(DATA / "savings_by_council.csv", index=False)

    loaded = sorted(df.council.unique())
    print(f"loaded {len(loaded)} councils: {', '.join(loaded)}  ({len(df)} savings lines)\n")

    # headline reconciliation vs council-stated totals
    print("headline totals (£m):")
    for _, r in meta.iterrows():
        extracted = df[df.council == r.council].amount_000_2026_27.fillna(0).sum() / 1000
        stated = (r.headline_total_savings_2026_27_000 or 0) / 1000
        print(f"  {r.council:<16} stated {stated:6.1f}   extracted-lines {extracted:6.1f}")

    print("\n=== SAVINGS WATERFALL (all loaded councils, £m) ===")
    wf = waterfall(df)
    wf["gbp_m"] = (wf.gbp_000 / 1000).round(1)
    print(wf[["step", "gbp_m"]].to_string(index=False))

    # partial-attribution sensitivity (genuine cashable discretionary cuts)
    v = df.amount_000_2026_27.fillna(0)
    new = df.f1_new_vs_carried.eq("new")
    disc = df.service_area.isin(DISCRETIONARY)
    cashcut = ~df.f2_type.isin(["income", "cost_avoidance"])
    yes_only = v[new & df.reform_attributable.eq("yes") & cashcut & disc].sum() / 1000
    yes_part = v[new & df.reform_attributable.isin(ATTRIB) & cashcut & disc].sum() / 1000
    print(f"\ndiscretionary CASHABLE cuts: {yes_only:.1f}m (attributable=yes only) "
          f"to {yes_part:.1f}m (incl. partial)")

    # by service area (new, attributable, cashable cut)
    core = df[new & df.reform_attributable.isin(ATTRIB) & cashcut]
    by_svc = (core.groupby("service_area").amount_000_2026_27.sum() / 1000).round(1).sort_values(ascending=False)
    print("\nnew attributable cashable cuts by service (£m):")
    print(by_svc.to_string())

    # DOGE keyword scan across description + notes. Use word boundaries for short/ambiguous
    # tokens so "edi" does not match "cr-edi-t" and "comms" does not match "commissioning".
    text = (df.description.fillna("") + " | " + df.notes.fillna("")).str.lower()
    patterns = {"edi": r"\bedi\b", "diversity": r"diversity", "equalit": r"equalit",
                "climate": r"climate", "net zero": r"net.?zero", "consultan": r"consultan",
                "communications": r"communications", "comms": r"\bcomms\b"}
    print("\nDOGE keyword scan (explicit lines):")
    for kw, pat in patterns.items():
        hits = df[text.str.contains(pat, regex=True)]
        if len(hits):
            amt = hits.amount_000_2026_27.fillna(0).sum() / 1000
            egs = "; ".join(hits.council + ": " + hits.description.str.slice(0, 40))
            print(f"  {kw:<14} {len(hits)} line(s), £{amt:.1f}m  [{egs[:120]}]")
        else:
            print(f"  {kw:<14} 0 lines")

    # --- Filter 4: do the coded discretionary cuts show up as falls in the RA budget? ---
    try:
        long = pd.read_csv(DATA / "panel_ra_long.csv")
    except FileNotFoundError:
        long = None
    if long is not None:
        disc_ra = long[long.service.isin(DISCRETIONARY)]
        pv = disc_ra.pivot_table(index="la", columns="year", values="amount_000", aggfunc="sum")
        if {"2025-26", "2026-27"}.issubset(pv.columns):
            coded = (df[new & df.reform_attributable.isin(ATTRIB) & cashcut & disc]
                     .groupby("council").amount_000_2026_27.sum())
            print("\n=== Filter 4: coded discretionary cuts vs actual RA discretionary change (£m) ===")
            print(f"{'council':<16}{'coded cut':>10}{'RA 25/26':>10}{'RA 26/27':>10}{'RA change':>10}")
            for c in sorted(coded.index):
                if c in pv.index:
                    a, b = pv.loc[c, "2025-26"] / 1000, pv.loc[c, "2026-27"] / 1000
                    print(f"{c:<16}{coded[c]/1000:>10.1f}{a:>10.0f}{b:>10.0f}{b-a:>+10.0f}")
            print("(RA discretionary bundle keeps rising in every council: the coded cuts offset "
                  "growth, they are not absolute line reductions.)")

    print(f"\nwrote {DATA/'savings_by_council.csv'}")


if __name__ == "__main__":
    main()
