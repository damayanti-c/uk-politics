"""Pull Labour campaign events from the public events.labour.org.uk JSON API.

This is a gentle, resumable extractor that reads each event from the site's
public ``/api/v2/event/{id}`` endpoint (clean JSON) rather than scraping the
JavaScript single-page-app HTML shell, which no longer embeds event data.

Why this approach:
  * The site's search/listing API (``/api/v2/results``) only returns *future*
    events, so historical windows can only be recovered by event ID.
  * Event IDs do not track start dates tightly: events for a single campaign
    window are scattered across a wide ID band, so the band must be swept.
  * The endpoint is fronted by Cloudflare, which blocks plain ``requests`` and
    rate-limits aggressive scans. We therefore impersonate a real browser TLS
    fingerprint (curl_cffi), scan in modest concurrent chunks with jitter, back
    off on blocks, and stop cleanly after repeated blocks.

The run is append-only and deduplicated by ``event_id``:
  * already-captured events (present in the output JSON) are skipped,
  * IDs previously confirmed not to exist (the scan log) are skipped,
so re-running after a block, an IP switch, or a widened range only fetches what
is genuinely missing and merges non-duplicates into the existing data.

Example
-------
    python pull_canvassing_events.py \
        --start-id 518000 --end-id 538500 \
        --start-date 2026-03-27 --end-date 2026-05-07
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curl_cffi import requests as cffi_requests

BASE_DIR = Path(__file__).resolve().parent
API_EVENT_URL = "https://events.labour.org.uk/api/v2/event/{event_id}"
PAGE_EVENT_URL = "https://events.labour.org.uk/event/{event_id}"
IMPERSONATE = "chrome131"
DEFAULT_TIMEOUT_SECONDS = 25

# Browser-like headers for the XHR JSON endpoint.
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://events.labour.org.uk/",
}
BLOCKED_MARKERS = (
    "Attention Required! | Cloudflare",
    "Sorry, you have been blocked",
    "You are unable to access labour.org.uk",
    "cf-error-details",
)

# Output schema — identical to the previously generated CSV/JSON exports.
OUTPUT_COLUMNS = [
    "event_id",
    "title",
    "category_name",
    "start_time",
    "end_time",
    "location",
    "postcode",
    "constituency",
    "constituency_code",
    "is_online_event",
    "is_private",
    "is_unlisted",
    "rsvp_is_allowed",
    "capacity",
    "attendees",
    "rsvps",
    "priority",
    "meeting_point",
    "location_is_tbc",
    "contact_number",
    "description",
    "event_url",
]


@dataclass(frozen=True)
class FetchResult:
    event_id: int
    status: str  # "ok" | "not_found" | "blocked" | "error"
    event: dict[str, Any] | None = None
    detail: str = ""


def _text(value: Any) -> str:
    """Collapse whitespace and coerce nullish values to an empty string."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int(value: Any, default: int = 0) -> int:
    """Coerce a value to int, tolerating None and non-numeric junk (e.g. '*')."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default


def map_api_event(event_id: int, api_event: dict[str, Any]) -> dict[str, Any]:
    """Normalise an ``/api/v2/event`` payload into the existing output schema."""
    return {
        "event_id": int(api_event.get("event_id", event_id)),
        "title": _text(api_event.get("title")),
        "category_name": _text(api_event.get("category_name")),
        "start_time": _text(api_event.get("start_time")),
        "end_time": _text(api_event.get("end_time")),
        "location": _text(api_event.get("location")),
        "postcode": _text(api_event.get("postcode")).replace(" ", ""),
        "constituency": _text(api_event.get("constituency")),
        "constituency_code": _text(api_event.get("constituency_code")),
        "is_online_event": bool(api_event.get("is_online_event", False)),
        "is_private": bool(api_event.get("is_private", False)),
        "is_unlisted": bool(api_event.get("is_unlisted", False)),
        "rsvp_is_allowed": bool(api_event.get("rsvp_is_allowed", True)),
        "capacity": _int(api_event.get("capacity")),
        "attendees": _int(api_event.get("attendees")),
        "rsvps": _int(api_event.get("rsvps")),
        "priority": _int(api_event.get("priority")),
        "meeting_point": _text(api_event.get("meeting_point")),
        "location_is_tbc": bool(api_event.get("location_is_tbc", False)),
        "contact_number": _text(api_event.get("contact_number")),
        "description": _text(api_event.get("description")),
        "event_url": PAGE_EVENT_URL.format(event_id=event_id),
    }


def fetch_event(
    session: cffi_requests.Session,
    event_id: int,
    timeout_seconds: int,
    min_sleep: float,
    max_sleep: float,
) -> FetchResult:
    """Fetch and normalise a single event from the JSON API."""
    # Per-request jitter keeps the scan from arriving in a detectable burst.
    if max_sleep > 0:
        time.sleep(random.uniform(min_sleep, max_sleep))

    url = API_EVENT_URL.format(event_id=event_id)
    try:
        response = session.get(url, timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - network errors are reported, not raised
        return FetchResult(event_id, "error", detail=repr(exc)[:160])

    if response.status_code == 404:
        return FetchResult(event_id, "not_found", detail="404")
    if response.status_code in (403, 429, 503) or any(
        marker in response.text for marker in BLOCKED_MARKERS
    ):
        return FetchResult(event_id, "blocked", detail=f"status={response.status_code}")
    if response.status_code != 200:
        return FetchResult(event_id, "error", detail=f"status={response.status_code}")

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return FetchResult(event_id, "error", detail=f"bad_json:{repr(exc)[:80]}")

    api_event = payload.get("event") if isinstance(payload, dict) else None
    if not api_event:
        return FetchResult(event_id, "not_found", detail="no_event_in_payload")

    try:
        return FetchResult(event_id, "ok", event=map_api_event(event_id, api_event))
    except Exception as exc:  # noqa: BLE001 - a malformed event must not kill the run
        return FetchResult(event_id, "error", detail=f"map_error:{repr(exc)[:80]}")


def in_window(record: dict[str, Any], start_date: str | None, end_date: str | None) -> bool:
    """Inclusive YYYY-MM-DD window test on the event's start date."""
    day = (record.get("start_time") or "")[:10]
    if not day:
        return False
    if start_date and day < start_date:
        return False
    if end_date and day > end_date:
        return False
    return True


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(
    records_by_id: dict[int, dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    start_date: str | None,
    end_date: str | None,
    preserve_ids: set[int],
) -> list[dict[str, Any]]:
    """Persist id-sorted records to JSON and CSV. Returns kept records.

    A record is kept if it falls in the requested date window OR it was already
    present in the existing output before this run (``preserve_ids``). This makes
    the run strictly additive: newly fetched events are constrained to the window,
    but pre-existing data is never dropped.
    """
    kept = [
        records_by_id[eid]
        for eid in sorted(records_by_id)
        if eid in preserve_ids or in_window(records_by_id[eid], start_date, end_date)
    ]
    json_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in kept:
            writer.writerow({col: record.get(col, "") for col in OUTPUT_COLUMNS})
    return kept


def build_summary(
    kept: list[dict[str, Any]],
    start_id: int,
    end_id: int,
    status_counts: Counter[str],
    blocked: bool,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    days = sorted({r["start_time"][:10] for r in kept if r.get("start_time")})
    months = Counter(r["start_time"][:7] for r in kept if r.get("start_time"))
    categories = Counter(r.get("category_name", "") for r in kept)
    return {
        "id_range": [start_id, end_id],
        "requested_date_window": [start_date, end_date],
        "date_range": [days[0], days[-1]] if days else [],
        "complete": not blocked,
        "status": "partial_blocked" if blocked else "complete_api_extraction",
        "event_count": len(kept),
        "categories": dict(categories.most_common()),
        "months": dict(sorted(months.items())),
        "scan_status_counts": dict(status_counts),
        "source": "https://events.labour.org.uk/api/v2/event/{id}",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": [
            "Extracted from the public events.labour.org.uk JSON API by event ID.",
            "Append-only and deduplicated by event_id; re-running merges only new events.",
        ]
        + (
            ["Run stopped early after repeated Cloudflare blocks; treat as partial and resume."]
            if blocked
            else []
        ),
    }


def chunked(items: list[int], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--start-date", help="Inclusive YYYY-MM-DD lower bound on start date.")
    parser.add_argument("--end-date", help="Inclusive YYYY-MM-DD upper bound on start date.")
    parser.add_argument("--output-stem", default="canvassing_events_apr_may_2026")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests per chunk.")
    parser.add_argument("--chunk-size", type=int, default=150, help="IDs processed between checkpoints.")
    parser.add_argument("--min-sleep", type=float, default=0.15, help="Min per-request jitter (s).")
    parser.add_argument("--max-sleep", type=float, default=0.40, help="Max per-request jitter (s).")
    parser.add_argument("--chunk-pause", type=float, default=0.8, help="Pause between chunks (s).")
    parser.add_argument("--max-block-chunks", type=int, default=4, help="Consecutive blocked chunks before stopping.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing outputs and start fresh.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    json_path = BASE_DIR / f"{args.output_stem}.json"
    csv_path = BASE_DIR / f"{args.output_stem}.csv"
    summary_path = BASE_DIR / f"{args.output_stem}_summary.json"
    scanlog_path = BASE_DIR / f"{args.output_stem}_scanlog.json"

    if args.overwrite:
        records_by_id: dict[int, dict[str, Any]] = {}
        scan_log: dict[str, str] = {}
    else:
        existing = load_json(json_path, [])
        records_by_id = {int(rec["event_id"]): rec for rec in existing}
        scan_log = load_json(scanlog_path, {})
    preserve_ids = set(records_by_id)  # pre-existing records are never dropped

    # Build the work list: skip ids already captured or confirmed non-existent.
    to_scan = [
        eid
        for eid in range(args.start_id, args.end_id + 1)
        if eid not in records_by_id and scan_log.get(str(eid)) != "not_found"
    ]
    print(
        f"Range {args.start_id}-{args.end_id}: {len(to_scan)} ids to fetch "
        f"({len(records_by_id)} already captured, {sum(v == 'not_found' for v in scan_log.values())} known-missing)."
    )

    session = cffi_requests.Session(impersonate=IMPERSONATE)
    session.headers.update(DEFAULT_HEADERS)

    status_counts: Counter[str] = Counter()
    consecutive_block_chunks = 0
    blocked_stop = False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for chunk_index, chunk in enumerate(chunked(to_scan, args.chunk_size), start=1):
            results = list(
                pool.map(
                    lambda eid: fetch_event(
                        session, eid, DEFAULT_TIMEOUT_SECONDS, args.min_sleep, args.max_sleep
                    ),
                    chunk,
                )
            )

            chunk_blocks = 0
            chunk_new = 0
            for res in results:
                status_counts[res.status] += 1
                if res.status == "ok" and res.event is not None:
                    if res.event_id not in records_by_id:
                        chunk_new += 1
                    records_by_id[res.event_id] = res.event
                    scan_log[str(res.event_id)] = "ok"
                elif res.status == "not_found":
                    scan_log[str(res.event_id)] = "not_found"
                elif res.status == "blocked":
                    chunk_blocks += 1
                # transient "error" ids are intentionally left unlogged so they retry on resume

            kept = write_outputs(records_by_id, json_path, csv_path, args.start_date, args.end_date, preserve_ids)
            scanlog_path.write_text(json.dumps(scan_log, ensure_ascii=False), encoding="utf-8")

            last_id = chunk[-1]
            print(
                f"chunk {chunk_index}: ids..{last_id} | +{chunk_new} new | "
                f"blocked={chunk_blocks} | kept_in_window={len(kept)} | total_captured={len(records_by_id)}",
                flush=True,
            )

            # Block handling: a chunk is "blocked" if a meaningful share was blocked.
            if chunk_blocks >= max(1, len(chunk) // 4):
                consecutive_block_chunks += 1
                backoff = min(60.0, 5.0 * (2 ** (consecutive_block_chunks - 1)))
                print(
                    f"  ! {chunk_blocks} blocks in chunk — backing off {backoff:.0f}s "
                    f"(consecutive blocked chunks: {consecutive_block_chunks}/{args.max_block_chunks})",
                    flush=True,
                )
                if consecutive_block_chunks >= args.max_block_chunks:
                    print("  ! Repeated Cloudflare blocks — stopping. Switch IP and re-run to resume.", flush=True)
                    blocked_stop = True
                    break
                time.sleep(backoff)
            else:
                consecutive_block_chunks = 0
                if args.chunk_pause > 0:
                    time.sleep(args.chunk_pause)

    kept = write_outputs(records_by_id, json_path, csv_path, args.start_date, args.end_date, preserve_ids)
    summary = build_summary(
        kept, args.start_id, args.end_id, status_counts, blocked_stop, args.start_date, args.end_date
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scanlog_path.write_text(json.dumps(scan_log, ensure_ascii=False), encoding="utf-8")

    print(
        f"\nDone. {len(kept)} events in window written to {json_path.name} / {csv_path.name}. "
        f"Status counts: {dict(status_counts)}"
    )
    return 1 if blocked_stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
