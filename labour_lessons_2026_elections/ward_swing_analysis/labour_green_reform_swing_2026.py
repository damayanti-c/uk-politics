"""Labour's 2026 collapse vs the Green and Reform advance, by ward.

Question
--------
In wards last contested in 2022 (the all-out cohort — London boroughs and the
four-year districts — so 2022 -> 2026 is a clean like-for-like comparison), did
Labour's vote fall further where the *Greens* surged, or where *Reform* surged?

Method
------
For each ward we compute the change in each party's vote share from its last
pre-2026 contest to 2026, then bin wards by how far the Greens (resp. Reform)
advanced and report the mean Labour change in each band.

  * Vote share = party votes / all valid votes in the ward's contest (summed
    across seats), from the electionresults.uk candidate feeds staged in
    ``source_data/election_results/local_elections/2026_external``.
  * Wards are identified by ONS code. Pre-2026 contests carry the code in their
    ``ward_slug``; the 2026 feed (Democracy Club slugs) is bridged to a code via
    a (council, ward-name) lookup built from historical races.
  * Cohort = wards whose most recent pre-2026 contest was in --base-year (2022).

Outputs (written next to this script):
  * ``labour_green_reform_swing_2026.png`` — grouped bar chart,
  * ``labour_green_reform_swing_bands_2026.csv`` — the band table,
  * ``labour_green_reform_swing_wards_2026.csv`` — per-ward changes.

Run:  python labour_lessons_2026_elections/labour_green_reform_swing_2026.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EXT = BASE_DIR.parents[1] / "source_data" / "election_results" / "local_elections" / "2026_external"
RACES_CSV = EXT / "electionresults_uk_races.csv"
CANDIDATES_CSV = EXT / "electionresults_uk_candidates.csv"
CANDIDATES_2026_CSV = EXT / "electionresults_uk_candidates_2026.csv"
EC_SUFFIX = re.compile(r"([EWS]\d{8})$", re.IGNORECASE)

GREEN, REFORM_CYAN = "#02A95B", "#12B6CF"
BANDS = [(-np.inf, 10, "≤10"), (10, 20, "10–20"), (20, 30, "20–30"), (30, np.inf, "≥30")]


def norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def classify(party: str) -> str:
    p = party.lower()
    if "labour" in p:
        return "Lab"
    if "green" in p:
        return "Grn"
    if "reform" in p:
        return "Ref"
    return "Oth"


def shares(votes: dict[str, float]) -> dict[str, float] | None:
    total = votes.get("__total__", 0.0)
    if total <= 0:
        return None
    return {p: 100 * votes.get(p, 0.0) / total for p in ("Lab", "Grn", "Ref")}


def build_bridge() -> dict[tuple[str, str], str]:
    """(council_slug, normalised ward_name) -> ONS code, from historical races."""
    bridge: dict[tuple[str, str], str] = {}
    with RACES_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("ec_code") or "").strip()
            if re.match(r"^E05\d{6}$", code):
                bridge[(row["council_slug"], norm_name(row["ward_name"]))] = code
    return bridge


def historical_shares() -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, set]]:
    """Per (ONS code, year) vote tallies for pre-2026 contests, plus the set of years."""
    tally: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    years: dict[str, set] = defaultdict(set)
    with CANDIDATES_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            m = EC_SUFFIX.search(row.get("ward_slug") or "")
            if not m:
                continue
            try:
                year = int(float(row["year"]))
                v = float(row["votes"] or 0)
            except ValueError:
                continue
            if year >= 2026:
                continue
            code = m.group(1).upper()
            tally[(code, year)][classify(row["party"])] += v
            tally[(code, year)]["__total__"] += v
            years[code].add(year)
    return tally, years


def shares_2026(bridge: dict[tuple[str, str], str]) -> dict[str, dict[str, float]]:
    tally: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with CANDIDATES_2026_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = bridge.get((row["council_slug"], norm_name(row["ward_name"])))
            if not code:
                continue
            try:
                v = float(row["votes"] or 0)
            except ValueError:
                continue
            tally[code][classify(row["party"])] += v
            tally[code]["__total__"] += v
    return tally


def band_label(value: float) -> str:
    for lo, hi, label in BANDS:
        if lo < value <= hi:
            return label
    return BANDS[0][2]


def mean_labour_by_band(rows: list[dict], advance_key: str) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for _, _, label in BANDS:
        vals = [r["dLab"] for r in rows if band_label(r[advance_key]) == label]
        out[label] = (float(np.mean(vals)) if vals else float("nan"), len(vals))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-year", type=int, default=2022,
                        help="Cohort = wards whose last pre-2026 contest was this year.")
    args = parser.parse_args()

    bridge = build_bridge()
    hist, years = historical_shares()
    v2026 = shares_2026(bridge)

    rows = []
    for code, v26 in v2026.items():
        prior_years = [y for y in years.get(code, ()) if y < 2026]
        if not prior_years or max(prior_years) != args.base_year:
            continue
        base = shares(hist[(code, args.base_year)])
        now = shares(v26)
        if not base or not now:
            continue
        rows.append({
            "ward_code": code,
            "dLab": now["Lab"] - base["Lab"],
            "dGrn": now["Grn"] - base["Grn"],
            "dRef": now["Ref"] - base["Ref"],
            "lab_base": base["Lab"], "lab_2026": now["Lab"],
            "grn_base": base["Grn"], "grn_2026": now["Grn"],
            "ref_base": base["Ref"], "ref_2026": now["Ref"],
        })

    overall = float(np.mean([r["dLab"] for r in rows]))
    print(f"Cohort: {len(rows)} wards last contested in {args.base_year}. "
          f"Mean Labour change {args.base_year}->2026: {overall:+.1f}pp\n")

    green = mean_labour_by_band(rows, "dGrn")
    reform = mean_labour_by_band(rows, "dRef")
    labels = [b[2] for b in BANDS]
    print(f"{'advance band':14}{'Labour Δ where GREEN up':>26}{'Labour Δ where REFORM up':>28}")
    for lab in labels:
        g, gn = green[lab]; r, rn = reform[lab]
        print(f"  {lab:12}{g:+8.1f}pp (n={gn:>4}){'':4}{r:+8.1f}pp (n={rn:>4})")

    # ---- grouped bar chart ----
    x = np.arange(len(labels))
    w = 0.4
    gvals = [green[l][0] for l in labels]
    rvals = [reform[l][0] for l in labels]
    gns = [green[l][1] for l in labels]
    rns = [reform[l][1] for l in labels]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("Where the Greens surged, Labour collapsed — where Reform surged, Labour fell about the same",
                 fontsize=13, fontweight="bold")
    bg = ax.bar(x - w / 2, gvals, w, color=GREEN, label="Wards where the Greens advanced")
    br = ax.bar(x + w / 2, rvals, w, color=REFORM_CYAN, label="Wards where Reform advanced")
    ax.axhline(overall, color="#666666", ls="--", lw=1,
               label=f"cohort mean Labour change ({overall:+.0f}pp)")

    def annotate(bars, ns):
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ax.annotate(f"{h:+.1f}\n(n={n})", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="top", fontsize=9, xytext=(0, -2),
                        textcoords="offset points")
    annotate(bg, gns); annotate(br, rns)

    ax.set_xticks(x); ax.set_xticklabels([f"{l} pts" for l in labels])
    ax.set_xlabel(f"How far that party's vote share rose, {args.base_year} → 2026")
    ax.set_ylabel(f"Mean change in Labour vote share, {args.base_year} → 2026 (pp)")
    ax.set_title("English wards last contested in 2022", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = BASE_DIR / "labour_green_reform_swing_2026.png"
    fig.savefig(png, dpi=130)
    print(f"\nWrote {png.name}")

    # ---- band-summary CSV ----
    bands_csv = BASE_DIR / "labour_green_reform_swing_bands_2026.csv"
    with bands_csv.open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["advance_band", "rival_party", "mean_labour_change_pp", "n_wards"])
        for lab in labels:
            w_.writerow([lab, "Green", round(green[lab][0], 2), green[lab][1]])
            w_.writerow([lab, "Reform", round(reform[lab][0], 2), reform[lab][1]])
    print(f"Wrote {bands_csv.name}")

    # ---- per-ward CSV (biggest Labour losses first) ----
    wards_csv = BASE_DIR / "labour_green_reform_swing_wards_2026.csv"
    fields = ["ward_code", "dLab", "dGrn", "dRef", "lab_base", "lab_2026",
              "grn_base", "grn_2026", "ref_base", "ref_2026"]
    with wards_csv.open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w_.writeheader()
        for r in sorted(rows, key=lambda d: d["dLab"]):
            w_.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {wards_csv.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
