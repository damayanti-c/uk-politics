"""Profile Labour canvassing frequency by ward for the 2026 local elections.

Goal
----
Rank the wards Labour canvassed in most, and express each ward's canvassing
volume relative to what the *average* ward got, in the context of the 2026
local elections.

Method
------
1. Read canvassing events (category "Canvassing", within the requested date
   window) from the extractor output ``canvassing_events_apr_may_2026.json``.
2. Map each event's ``postcode`` to an ONS ward via the free postcodes.io API
   (bulk, cached locally to ``postcode_ward_lookup.json`` so re-runs are instant
   and the API is only hit for postcodes not seen before).
3. Join wards to the 2026 local-election universe
   (``../contextual_analysis/contesting_wards_2026_by_last_result_matched.csv``) on the ONS ward
   code (postcodes.io ``codes.admin_ward`` == the file's ``boundary_code``).
   This universe is the set of wards up for election in 2026 (England), so it is
   the natural denominator for "the average ward".
4. Count canvassing events per ward and compare each to the average.

"Average ward" is reported two ways, because both are meaningful:
  * avg over **all** contested 2026 wards (wards that got zero canvassing count
    as zero) — "the average ward up for election in 2026", and
  * avg over **canvassed** contested wards only (wards with >=1 event).

Outputs (written into this folder):
  * console table of the top-N wards,
  * ``ward_canvassing_frequency_2026.csv`` — every contested ward ranked,
  * ``profile_canvassing_events_summary.json`` — headline stats + the top-N.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from curl_cffi import requests as cffi_requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # ward names can contain non-ASCII

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVENTS_JSON = BASE_DIR / "canvassing_events_apr_may_2026.json"
CONTESTED_CSV = BASE_DIR.parent / "contextual_analysis" / "contesting_wards_2026_by_last_result_matched.csv"
CACHE_PATH = BASE_DIR / "postcode_ward_lookup.json"

POSTCODES_IO_BULK = "https://api.postcodes.io/postcodes"
POSTCODES_IO_TERMINATED = "https://api.postcodes.io/terminated_postcodes/{pc}"

# 2026 in-scope ward universe (ward-level, comparable units across GB).
#   * England:  wards actually holding 2026 local elections (the contested CSV).
#   * Wales:    ALL principal-council wards (the 2026 Senedd election was national).
#   * Scotland: ALL council wards (the 2026 Holyrood election was national).
# Welsh/Scottish totals are the count of current ONS wards (W05*/S13*) from the
# ONS "Wards (December 2024) Boundaries UK BFC" register (Open Geography Portal),
# used as the denominator so that wards which got zero canvassing still count.
WELSH_WARD_TOTAL = 762
SCOTTISH_WARD_TOTAL = 355
ELECTION_BY_NATION = {
    "England": "Local (2026)",
    "Wales": "Senedd (2026)",
    "Scotland": "Holyrood (2026)",
}


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_canvassing_events(
    json_path: Path, category: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """Return events of the given category whose start date is within the window."""
    records = json.loads(json_path.read_text(encoding="utf-8"))
    events = []
    for rec in records:
        if rec.get("category_name") != category:
            continue
        day = (rec.get("start_time") or "")[:10]
        if not day or day < start_date or day > end_date:
            continue
        events.append(rec)
    return events


def load_contested_wards(csv_path: Path) -> dict[str, dict[str, str]]:
    """Map ONS ward code (``boundary_code``) -> ward metadata for 2026 contests."""
    contested: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("boundary_code") or "").strip()
            if not code:
                continue
            contested[code] = {
                "ward_name": (row.get("ward_name") or "").strip(),
                "council": (row.get("council") or "").strip(),
                "seats": (row.get("seats") or "").strip(),
                "control_group": (row.get("control_group") or "").strip(),
                "majority": (row.get("majority") or "").strip(),
            }
    return contested


# --------------------------------------------------------------------------- #
# Postcode -> ward lookup (cached)                                            #
# --------------------------------------------------------------------------- #
def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _ward_from_result(result: dict[str, Any]) -> dict[str, str]:
    codes = result.get("codes") or {}
    return {
        "ward_code": codes.get("admin_ward") or "",
        "ward_name": result.get("admin_ward") or "",
        "district": result.get("admin_district") or "",
        "country": result.get("country") or "",
    }


def lookup_wards(postcodes: list[str], cache: dict[str, Any]) -> dict[str, Any]:
    """Resolve postcodes to wards, using and updating the on-disk cache."""
    missing = sorted({pc for pc in postcodes if pc and pc not in cache})
    if missing:
        print(f"Looking up {len(missing)} new postcodes via postcodes.io "
              f"({len(postcodes) - len(missing)} already cached)...")
    session = cffi_requests.Session(impersonate="chrome131")

    # Bulk-resolve in batches of 100 (postcodes.io limit).
    for batch_start in range(0, len(missing), 100):
        batch = missing[batch_start : batch_start + 100]
        try:
            resp = session.post(POSTCODES_IO_BULK, json={"postcodes": batch}, timeout=30)
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! bulk lookup failed for a batch ({exc!r}); will retry items singly")
            payload = {"result": [{"query": pc, "result": None} for pc in batch]}

        for item in payload.get("result", []):
            query = item.get("query", "")
            result = item.get("result")
            cache[query] = _ward_from_result(result) if result else None
        save_cache(cache)
        time.sleep(0.3)  # be polite to a free public API

    # Fallback: try the terminated-postcodes endpoint for anything still unresolved.
    still_missing = [pc for pc in missing if cache.get(pc) is None]
    if still_missing:
        print(f"  retrying {len(still_missing)} unresolved postcodes against terminated-postcode records...")
    for pc in still_missing:
        try:
            resp = session.get(POSTCODES_IO_TERMINATED.format(pc=pc), timeout=20)
            if resp.status_code == 200 and resp.json().get("result"):
                cache[pc] = _ward_from_result(resp.json()["result"])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
    save_cache(cache)
    return cache


# --------------------------------------------------------------------------- #
# Profiling                                                                   #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-json", default=str(DEFAULT_EVENTS_JSON))
    parser.add_argument("--category", default="Canvassing")
    parser.add_argument("--start-date", default="2026-03-27")
    parser.add_argument("--end-date", default="2026-05-07")
    parser.add_argument("--top", type=int, default=20, help="How many wards to show.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_canvassing_events(Path(args.events_json), args.category, args.start_date, args.end_date)
    contested = load_contested_wards(CONTESTED_CSV)
    cache = lookup_wards([e.get("postcode", "").strip() for e in events], load_cache())

    # In-scope ward universe sizes (denominator for "the average ward").
    universe_size = {
        "England": len(contested),       # English wards actually up in 2026
        "Wales": WELSH_WARD_TOTAL,        # whole nation up (Senedd)
        "Scotland": SCOTTISH_WARD_TOTAL,  # whole nation up (Holyrood)
    }
    n_universe = sum(universe_size.values())
    print(f"{len(events)} {args.category} events in {args.start_date}..{args.end_date}.")
    print(f"2026 in-scope ward universe: {n_universe} wards "
          f"(England {universe_size['England']} contested + Wales {universe_size['Wales']} "
          f"+ Scotland {universe_size['Scotland']}).\n")

    # Tally events per ward. A ward is in-scope if it had a 2026 election:
    #   England -> only wards in the contested CSV; Wales/Scotland -> any ward.
    per_ward: Counter[str] = Counter()
    ward_meta: dict[str, dict[str, str]] = {}
    unmapped = 0
    out_of_scope: Counter[str] = Counter()   # mapped but no 2026 election (e.g. English ward not up)
    for event in events:
        info = cache.get(event.get("postcode", "").strip())
        if not info or not info.get("ward_code"):
            unmapped += 1
            continue
        code = info["ward_code"]
        country = info.get("country") or ""
        if country == "England":
            if code not in contested:
                out_of_scope["England (ward not up in 2026)"] += 1
                continue
            meta = contested[code]
            ward_meta[code] = {"ward_name": meta["ward_name"], "council": meta["council"],
                               "control_group": meta["control_group"], "nation": "England"}
        elif country in ("Wales", "Scotland"):
            ward_meta[code] = {"ward_name": info.get("ward_name", ""), "council": info.get("district", ""),
                               "control_group": "", "nation": country}
        else:
            out_of_scope[country or "Unknown"] += 1
            continue
        per_ward[code] += 1

    in_scope_events = sum(per_ward.values())
    n_canvassed = len(per_ward)
    avg_all = in_scope_events / n_universe if n_universe else 0.0
    avg_canvassed = in_scope_events / n_canvassed if n_canvassed else 0.0

    ranked = sorted(per_ward.items(), key=lambda kv: (-kv[1], ward_meta[kv[0]]["ward_name"]))

    def row_for(code: str, count: int, rank: int) -> dict[str, Any]:
        meta = ward_meta[code]
        return {
            "rank": rank,
            "ward_name": meta["ward_name"],
            "council": meta["council"],
            "nation": meta["nation"],
            "election": ELECTION_BY_NATION.get(meta["nation"], ""),
            "control_group": meta["control_group"],
            "ward_code": code,
            "canvassing_events": count,
            "vs_avg_ward": round(count / avg_all, 1) if avg_all else 0.0,
            "vs_avg_canvassed_ward": round(count / avg_canvassed, 1) if avg_canvassed else 0.0,
        }

    table = [row_for(code, count, i) for i, (code, count) in enumerate(ranked, start=1)]

    # ----- console report -----
    print("=" * 102)
    print(f"TOP {args.top} WARDS BY LABOUR CANVASSING EVENTS — GB, 2026 ELECTIONS (ward-level)")
    print("=" * 102)
    print(f"Avg events per in-scope ward (all {n_universe} GB wards up in 2026): {avg_all:6.2f}")
    print(f"Avg events per *canvassed* in-scope ward ({n_canvassed} wards):          {avg_canvassed:6.2f}")
    print("-" * 102)
    print(f"{'#':>3}  {'Ward':24s} {'Council/Area':20s} {'Election':14s} {'Events':>6s} {'xAvg':>6s} {'xCanv':>6s}")
    print("-" * 102)
    for r in table[: args.top]:
        print(f"{r['rank']:>3}  {r['ward_name'][:24]:24s} {r['council'][:20]:20s} "
              f"{r['election'][:14]:14s} {r['canvassing_events']:>6} "
              f"{r['vs_avg_ward']:>5.1f}x {r['vs_avg_canvassed_ward']:>5.1f}x")
    print("=" * 102)

    # ----- coverage / context -----
    total = len(events)
    mapped = total - unmapped
    by_nation_events = Counter(ward_meta[c]["nation"] for c in per_ward.elements())
    canvassed_by_nation = Counter(ward_meta[c]["nation"] for c in per_ward)
    print("\nCoverage:")
    print(f"  canvassing events:                  {total}")
    print(f"  mapped to a ward:                   {mapped} ({mapped / total:.1%})")
    print(f"  in a 2026-election ward (in scope): {in_scope_events} ({in_scope_events / total:.1%})  "
          f"by nation: {dict(by_nation_events.most_common())}")
    print(f"  mapped but no 2026 election:        {sum(out_of_scope.values())} "
          f"({dict(out_of_scope.most_common())})")
    print(f"  unmapped (bad/terminated postcode): {unmapped}")
    print(f"  in-scope wards canvassed >=1:       {n_canvassed} of {n_universe} ({n_canvassed / n_universe:.1%})  "
          f"by nation: {dict(canvassed_by_nation.most_common())}")

    # ----- write outputs -----
    out_csv = BASE_DIR / "ward_canvassing_frequency_2026.csv"
    fields = ["rank", "ward_name", "council", "nation", "election", "control_group",
              "ward_code", "canvassing_events", "vs_avg_ward", "vs_avg_canvassed_ward"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    summary = {
        "category": args.category,
        "date_window": [args.start_date, args.end_date],
        "approach": "ward-level across GB; in-scope = wards that had a 2026 election "
                    "(England: contested local wards; Wales: all wards/Senedd; Scotland: all wards/Holyrood)",
        "in_scope_ward_universe": universe_size,
        "in_scope_ward_universe_total": n_universe,
        "total_events": total,
        "events_mapped_to_ward": mapped,
        "events_in_scope": in_scope_events,
        "events_in_scope_by_nation": dict(by_nation_events.most_common()),
        "events_out_of_scope": dict(out_of_scope.most_common()),
        "events_unmapped": unmapped,
        "in_scope_wards_canvassed": n_canvassed,
        "in_scope_wards_canvassed_by_nation": dict(canvassed_by_nation.most_common()),
        "avg_events_per_in_scope_ward": round(avg_all, 3),
        "avg_events_per_canvassed_in_scope_ward": round(avg_canvassed, 3),
        "top_wards": table[: args.top],
    }
    summary_path = BASE_DIR / "profile_canvassing_events_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {out_csv.name} ({len(table)} wards) and {summary_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
