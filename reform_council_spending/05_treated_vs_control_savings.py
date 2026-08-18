"""
Treated (Reform) vs control (non-Reform) comparison of genuinely-new discretionary savings.

Runs the IDENTICAL waterfall metric on both groups so the £30-44m Reform figure has a
like-for-like benchmark. The headline metric is "new, administration-attributable, cashable,
discretionary cuts" as a share of net budget (Revenue expenditure), computed per council and
per group. Treated schedules live in data/savings_raw/, controls in data/savings_raw_control/.

Field note: the JSON field `reform_attributable` means "attributable to the current
administration's deliberate choice" for BOTH groups (for controls it is the non-Reform
administration). Same definition, so the comparison is valid.

Run:  python 05_treated_vs_control_savings.py  ->  outputs/treated_vs_control_savings.csv + stdout
"""
from pathlib import Path
import json
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

DISCRETIONARY = {"Cultural & related", "Environmental & regulatory", "Planning & development",
                 "Central services", "Highways & transport"}
ATTRIB = {"yes", "partial"}


def load_dir(d: Path, group: str):
    recs, meta = [], []
    for f in sorted(d.glob("*.json")):
        dd = json.loads(f.read_text(encoding="utf-8"))
        flip = -1 if dd.get("sign_is_negative_saving") else 1
        meta.append(dict(council=dd["council"], group=group,
                         headline=dd.get("headline_total_savings_2026_27_000"),
                         council_tax=dd.get("council_tax_decision", ""),
                         coverage=dd.get("coverage_note", "")))
        for s in dd.get("savings", []):
            s = dict(s)
            s["council"], s["group"] = dd["council"], group
            s["service_area"] = str(s.get("service_area", "")).replace("&amp;", "&").strip()
            a = s.get("amount_000_2026_27")
            s["amount_000_2026_27"] = flip * a if a is not None else 0
            recs.append(s)
    return pd.DataFrame(recs), pd.DataFrame(meta)


def genuine_disc_cut(df, strict=True):
    """per-council sum (£m) of new, attributable, cashable, discretionary cuts.

    strict=True counts only f1=='new'. strict=False also counts f1=='unclear' (excludes only
    explicit carried_forward), correcting for councils whose documents don't tag new vs prior
    (e.g. Devon's directorate tables). Reported both ways so the comparison is tagging-robust.
    """
    newmask = df.f1_new_vs_carried.eq("new") if strict else df.f1_new_vs_carried.ne("carried_forward")
    m = (newmask & df.reform_attributable.isin(ATTRIB)
         & ~df.f2_type.isin(["income", "cost_avoidance"]) & df.service_area.isin(DISCRETIONARY))
    return df[m].groupby("council").amount_000_2026_27.sum() / 1000


def main():
    dt, mt = load_dir(DATA / "savings_raw", "treated")
    dc, mc = load_dir(DATA / "savings_raw_control", "control")
    if dc.empty:
        print("No control extractions yet in data/savings_raw_control/. Run once agents land.")
    df = pd.concat([dt, dc], ignore_index=True)
    meta = pd.concat([mt, mc], ignore_index=True)

    budget = pd.read_csv(DATA / "panel_ra_wide.csv").set_index("la")["Revenue expenditure"] / 1000
    cut_s = genuine_disc_cut(df, strict=True)
    cut_i = genuine_disc_cut(df, strict=False)

    rows = []
    for _, m in meta.iterrows():
        c = m.council
        b = budget.get(c, float("nan"))
        xs, xi = cut_s.get(c, 0.0), cut_i.get(c, 0.0)
        rows.append(dict(council=c, group=m.group, headline_m=round((m.headline or 0) / 1000, 1),
                         cut_strict_m=round(xs, 1), cut_incl_m=round(xi, 1), budget_m=round(b, 0),
                         pct_strict=round(100 * xs / b, 3) if b == b else None,
                         pct_incl=round(100 * xi / b, 3) if b == b else None))
    tab = pd.DataFrame(rows).sort_values(["group", "pct_incl"], ascending=[True, False])
    tab.to_csv(OUT / "treated_vs_control_savings.csv", index=False)

    print("Genuinely-new discretionary cashable cuts, 2026/27 (treated vs control)")
    print("(strict = f1==new; incl = new + untagged, excluding only explicit carried-forward)\n")
    print(tab.to_string(index=False))

    print("\ngroup comparison (discretionary cut as % of net budget):")
    for g in ["treated", "control"]:
        sub = tab[tab.group == g]
        if len(sub):
            ps = 100 * sub.cut_strict_m.sum() / sub.budget_m.sum()
            pi = 100 * sub.cut_incl_m.sum() / sub.budget_m.sum()
            print(f"  {g:<8} n={len(sub)}  headline £{sub.headline_m.sum():.0f}m | "
                  f"strict pooled {ps:.2f}% (mean {sub.pct_strict.mean():.2f}%) | "
                  f"incl pooled {pi:.2f}% (mean {sub.pct_incl.mean():.2f}%)")

    # DOGE keyword scan on controls, for symmetry with the treated finding
    if not dc.empty:
        text = (dc.description.fillna("") + " | " + dc.notes.fillna("")).str.lower()
        print("\ncontrol DOGE keyword scan:")
        import re
        for kw, pat in {"edi": r"\bedi\b", "diversity": r"\bdiversity", "equalit": "equalit",
                        "climate": "climate", "net zero": r"net.?zero", "consultan": "consultan",
                        "comms/communications": r"comms|communications"}.items():
            hits = dc[text.str.contains(pat, regex=True)]
            amt = hits.amount_000_2026_27.fillna(0).sum() / 1000
            print(f"  {kw:<22} {len(hits)} line(s), £{amt:.1f}m")


if __name__ == "__main__":
    main()
