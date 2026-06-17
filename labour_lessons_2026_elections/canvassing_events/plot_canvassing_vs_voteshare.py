"""Did Labour canvass where it was fighting the Greens and/or Reform?

Plots Labour canvassing intensity per ward against each party's vote share in the
2026 local elections, for English local-election wards.

Population
----------
All English wards that held a 2026 local-election contest and that we can map to
an ONS ward code (~90%; the rest are councils whose 2026 boundaries are brand new
and have no historical code to bridge on). Wards with **zero** canvassing are kept
(plotted at y=0) — that is the whole point: it shows the high-Green / high-Reform
wards Labour did *not* visit as well as the ones it did.

Vote shares
-----------
* Labour: each ward's most recent PRE-2026 local election. Labour's own share is
  taken from before the canvassing so it cannot have been inflated by that
  canvassing (the 2026 share would be endogenous to Labour's campaign effort).
* Green & Reform: the 2026 contest. Reform barely stood in pre-2026 locals, so the
  2026 result is the only basis on which its ward-level strength is observable.
Share = party votes / all valid votes in that ward's contest (summed across seats).
Source: electionresults.uk candidate feeds.

Canvassing
----------
Canvassing events (27 Mar – 7 May 2026) mapped to wards by postcode via the cached
postcodes.io lookup built by ``profile_canvassing_events.py``.

Join key
--------
ONS ward code (E05…). Canvassing -> code comes from postcodes.io; 2026 contest ->
code via a (council, ward-name) bridge built from historical races that carry both
the code and the name.

Outputs: ``canvassing_vs_voteshare_2026.png`` and ``canvassing_vs_voteshare_2026.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EVENTS_JSON = BASE_DIR / "canvassing_events_apr_may_2026.json"
CACHE_PATH = BASE_DIR / "postcode_ward_lookup.json"
EXT = BASE_DIR.parent.parent / "source_data" / "election_results" / "local_elections" / "2026_external"
RACES_CSV = EXT / "electionresults_uk_races.csv"
CANDIDATES_CSV = EXT / "electionresults_uk_candidates.csv"            # all years (pre-2026 carry ONS codes)
CANDIDATES_2026_CSV = EXT / "electionresults_uk_candidates_2026.csv"  # 2026-only (Democracy Club slugs)
EC_SUFFIX = re.compile(r"(E05\d{6})$", re.IGNORECASE)

START_DATE, END_DATE = "2026-03-27", "2026-05-07"
LABOUR_RED, GREEN, REFORM_CYAN = "#E4003B", "#02A95B", "#12B6CF"


def norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def classify_party(party: str) -> str | None:
    p = party.lower()
    if "labour" in p:
        return "Labour"
    if "green" in p:
        return "Green"
    if "reform" in p:
        return "Reform"
    return None


def canvassing_by_ward_code() -> Counter:
    """Count in-window canvassing events per ONS ward code (England) via the cache."""
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    events = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for e in events:
        if e.get("category_name") != "Canvassing":
            continue
        if not (START_DATE <= (e.get("start_time") or "")[:10] <= END_DATE):
            continue
        info = cache.get((e.get("postcode") or "").strip())
        if info and info.get("ward_code", "").startswith("E05"):
            counts[info["ward_code"]] += 1
    return counts


def build_name_to_code_bridge() -> dict[tuple[str, str], str]:
    """(council_slug, normalised ward_name) -> ONS code, from historical races."""
    bridge: dict[tuple[str, str], str] = {}
    with RACES_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("ec_code") or "").strip()
            if re.match(r"^E05\d{6}$", code):
                bridge[(row["council_slug"], norm_name(row["ward_name"]))] = code
    return bridge


def votes_2026_by_code(bridge: dict[tuple[str, str], str]) -> dict[str, dict]:
    """Per-ward 2026 party shares keyed by ONS code (English contests only)."""
    # ward (council_slug, ward_name) -> party -> votes
    tally: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    names: dict[tuple[str, str], tuple[str, str]] = {}
    with CANDIDATES_2026_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["council_slug"], norm_name(row["ward_name"]))
            try:
                tally[key]["__total__"] += float(row["votes"] or 0)
            except ValueError:
                continue
            party = classify_party(row["party"])
            if party:
                tally[key][party] += float(row["votes"] or 0)
            names[key] = (row["council_slug"], row["ward_name"])

    out: dict[str, dict] = {}
    unbridged = 0
    for key, votes in tally.items():
        code = bridge.get(key)
        total = votes.get("__total__", 0.0)
        if total <= 0:
            continue
        if not code:
            unbridged += 1
            continue
        out[code] = {
            "council": names[key][0],
            "ward_name": names[key][1],
            "labour_share": 100 * votes.get("Labour", 0.0) / total,
            "green_share": 100 * votes.get("Green", 0.0) / total,
            "reform_share": 100 * votes.get("Reform", 0.0) / total,
        }
    print(f"2026 English contests with shares: {len(out)} mapped to ONS codes "
          f"({unbridged} unbridged — new-boundary wards).")
    return out


def last_election_shares_by_code() -> tuple[dict[str, dict], Counter]:
    """Per-ward party shares at each ward's most recent pre-2026 election.

    Keyed by ONS code, which the historical candidate feed carries as the suffix
    of ``ward_slug`` (e.g. ``buckingham-e05007562``). Pre-determined w.r.t. 2026
    canvassing, so free of the reverse-causality that contaminates 2026 shares.
    """
    # code -> year -> party -> votes
    tally: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with CANDIDATES_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            m = EC_SUFFIX.search(row.get("ward_slug") or "")
            if not m:
                continue
            try:
                year = int(float(row["year"]))
            except (ValueError, KeyError):
                continue
            if year >= 2026:
                continue
            code = m.group(1).upper()
            try:
                v = float(row["votes"] or 0)
            except ValueError:
                continue
            tally[code][year]["__total__"] += v
            party = classify_party(row["party"])
            if party:
                tally[code][year][party] += v

    out: dict[str, dict] = {}
    last_year = Counter()
    for code, by_year in tally.items():
        year = max(by_year)  # most recent pre-2026 contest
        votes = by_year[year]
        total = votes.get("__total__", 0.0)
        if total <= 0:
            continue
        out[code] = {
            "last_election_year": year,
            "labour_share": 100 * votes.get("Labour", 0.0) / total,
            "green_share": 100 * votes.get("Green", 0.0) / total,
            "reform_share": 100 * votes.get("Reform", 0.0) / total,
        }
        last_year[year] += 1
    return out, last_year


def scatter(ax, x, y, color, xlabel, title):
    ax.scatter(x, y, s=18, alpha=0.45, color=color, edgecolors="none")
    if len(x) > 2 and np.std(x) > 0:
        r = np.corrcoef(x, y)[0, 1]
        a, b = np.polyfit(x, y, 1)
        xs = np.array([min(x), max(x)])
        ax.plot(xs, a * xs + b, color="#333333", lw=1.4, ls="--")
        ax.text(0.04, 0.95, f"r = {r:+.2f}", transform=ax.transAxes,
                va="top", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Labour canvassing events in ward")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()  # --help only; design is fixed

    canv = canvassing_by_ward_code()
    # 2026 contest defines the population (English wards in play) + labels + Green/Reform shares.
    contests_2026 = votes_2026_by_code(build_name_to_code_bridge())
    # Labour's own share is taken from each ward's LAST pre-2026 election, which is
    # pre-determined and so cannot have been inflated by the 2026 canvassing itself.
    last, last_year = last_election_shares_by_code()
    print("Last-election year for the in-play wards: "
          + ", ".join(f"{y}:{last_year[y]}" for y in sorted(last_year)))

    rows = []
    for code, meta in contests_2026.items():
        rows.append({
            "ward_code": code,
            "council": meta["council"],
            "ward_name": meta["ward_name"],
            "canvassing_events": canv.get(code, 0),
            "labour_share_last": last[code]["labour_share"] if code in last else None,
            "last_election_year": last[code]["last_election_year"] if code in last else None,
            "green_share_2026": meta["green_share"],
            "reform_share_2026": meta["reform_share"],
        })
    print(f"{len(rows)} in-play 2026 English wards; "
          f"{sum(1 for r in rows if r['canvassing_events'] > 0)} canvassed, "
          f"{sum(1 for r in rows if r['canvassing_events'] == 0)} not. "
          f"{sum(1 for r in rows if r['labour_share_last'] is not None)} have a prior Labour share.")

    cv = np.array([r["canvassing_events"] for r in rows])
    grn = np.array([r["green_share_2026"] for r in rows])
    ref = np.array([r["reform_share_2026"] for r in rows])
    # Labour panel uses only wards with a prior result.
    lab_rows = [r for r in rows if r["labour_share_last"] is not None]
    lab = np.array([r["labour_share_last"] for r in lab_rows])
    lab_cv = np.array([r["canvassing_events"] for r in lab_rows])

    # ---- Main figure: (a) Labour [last election] | (b) Green [2026] | (c) Reform [2026] ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Labour canvassing vs party vote share — English wards in play in 2026",
                 fontsize=14, fontweight="bold")
    scatter(axes[0], lab, lab_cv, LABOUR_RED, "Labour vote share, last election (%)",
            "(a) Labour share (last election)")
    scatter(axes[1], grn, cv, GREEN, "Green vote share, 2026 (%)",
            "(b) Green share (2026 contest)")
    scatter(axes[2], ref, cv, REFORM_CYAN, "Reform vote share, 2026 (%)",
            "(c) Reform share (2026 contest)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = BASE_DIR / "canvassing_vs_voteshare.png"
    fig.savefig(png, dpi=130)
    print(f"Wrote {png.name}")

    # ---- Figure (d): Green vs Reform share (2026); size = canvassing,
    #      colour = Labour's LAST pre-2026 vote share (pre-determined, not after-the-fact) ----
    lab_all = np.array([r["labour_share_last"] if r["labour_share_last"] is not None else np.nan
                        for r in rows])
    vmax = float(np.nanpercentile(lab_all, 95))
    # Lighter red ramp: use only the pale-to-medium part of "Reds" (skip dark maroon).
    light_reds = LinearSegmentedColormap.from_list("light_reds", plt.cm.Reds(np.linspace(0.03, 0.55, 256)))
    figd, axd = plt.subplots(figsize=(9.5, 7.5))
    sc = axd.scatter(grn, ref, s=12 + cv * 6, c=lab_all, cmap=light_reds,
                     vmin=0, vmax=vmax, alpha=0.85, edgecolors="#555555", linewidths=0.5)
    axd.set_xlabel("Green vote share, 2026 (%)")
    axd.set_ylabel("Reform vote share, 2026 (%)")
    axd.set_title("(d) Green vs Reform share, 2026\nsize = Labour canvassing events; "
                  "colour = Labour share at last election",
                  fontsize=11, fontweight="bold")
    axd.grid(True, alpha=0.25)
    figd.colorbar(sc, ax=axd, label=f"Labour vote share, last election (%; capped at {vmax:.0f})")
    figd.tight_layout()
    pngd = BASE_DIR / "canvassing_green_vs_reform_2026.png"
    figd.savefig(pngd, dpi=130)
    print(f"Wrote {pngd.name}")

    # ---- data behind figure (d), one row per plotted ward, most-canvassed first ----
    csvd = BASE_DIR / "canvassing_green_vs_reform_2026.csv"
    dfields = ["ward_code", "council", "ward_name", "canvassing_events",
               "green_share_2026", "reform_share_2026", "labour_share_last", "last_election_year"]
    with csvd.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=dfields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda d: -d["canvassing_events"]):
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {csvd.name}")

    # ---- data CSV ----
    out_csv = BASE_DIR / "canvassing_vs_voteshare.csv"
    fields = ["ward_code", "council", "ward_name", "canvassing_events",
              "labour_share_last", "last_election_year", "green_share_2026", "reform_share_2026"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda d: -d["canvassing_events"]):
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {out_csv.name}")

    print("\nCorrelations with canvassing intensity:")
    print(f"  Labour  share (last election) vs canvassing: r = {np.corrcoef(lab, lab_cv)[0, 1]:+.3f}")
    print(f"  Green   share (2026)          vs canvassing: r = {np.corrcoef(grn, cv)[0, 1]:+.3f}")
    print(f"  Reform  share (2026)          vs canvassing: r = {np.corrcoef(ref, cv)[0, 1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
