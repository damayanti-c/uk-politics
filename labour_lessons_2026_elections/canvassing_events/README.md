# Canvassing Events

This folder contains Labour public-event extracts and the script used to recreate them.

- Script: `pull_canvassing_events.py`
- Output schema: one row per public event (see `OUTPUT_COLUMNS` in the script)
- Formats: `.json`, `.csv`, `_summary.json`, plus a `_scanlog.json` resume checkpoint

## How it works

- The extractor reads each event from the site's public **JSON API**:
  `https://events.labour.org.uk/api/v2/event/{id}`. The older approach of
  scraping `https://events.labour.org.uk/event/{id}` no longer works — those
  pages are now an empty JavaScript app shell with no event data embedded.
- The site's search/listing API (`/api/v2/results`) only returns *future*
  events, so a past window can only be recovered by sweeping event IDs.
- Event IDs do **not** track start dates: events for a single campaign window
  are scattered across a wide ID band, so the whole band must be scanned and
  then filtered to the requested date window.

## Cloudflare / gentleness

- The site is behind Cloudflare. Plain `requests`, `cloudscraper`, and even a
  real headless/headful Chrome (Selenium) all get a 403 "you have been blocked"
  page once the IP is flagged; aggressive scans (high request volume) trip it.
- This script therefore:
  - impersonates a real browser TLS fingerprint via `curl_cffi`,
  - scans in small concurrent chunks with per-request jitter,
  - backs off exponentially on blocks and stops cleanly after repeated blocks,
  - checkpoints after every chunk and is **resumable** (re-running skips
    already-captured and known-missing IDs), so a block just means "switch
    network/IP and re-run to continue".
- If the IP does get blocked, switching network (VPN / mobile hotspot / other
  Wi-Fi) clears it; the block is IP-reputation based.

## Append-only merge

Runs are **deduplicated by `event_id`** and additive: newly fetched events are
constrained to the requested date window, while any records already present in
the output JSON are preserved. Re-running with a wider ID range or a later date
simply merges the new non-duplicate events into the existing files.

## Example

The current `canvassing_events_apr_may_2026.*` files were built to cover the
27 March – 7 May 2026 window with:

```bash
python canvassing_events/pull_canvassing_events.py ^
  --start-id 518000 ^
  --end-id 538500 ^
  --start-date 2026-03-27 ^
  --end-date 2026-05-07 ^
  --output-stem canvassing_events_apr_may_2026
```

Dependencies: `curl_cffi` (TLS impersonation).

## Analysis scripts

- `profile_canvassing_events.py` — ranks wards by Labour canvassing volume in the
  2026 elections. Maps each event's postcode to an ONS ward via postcodes.io
  (cached in `postcode_ward_lookup.json`). Ward-level across GB: in-scope wards
  are England's contested local wards plus *all* Welsh and Scottish wards (the
  2026 Senedd/Holyrood elections were national). Reports each ward's events vs the
  average in-scope ward. Outputs `ward_canvassing_frequency_2026.csv` and
  `profile_canvassing_events_summary.json`.
- `plot_canvassing_vs_voteshare.py` — scatters canvassing intensity per English
  ward against party vote share, to show which parties Labour was fighting where.
  Labour's share is taken from each ward's **last pre-2026 election** (pre-determined,
  so canvassing can't have inflated it); Green and Reform are from the **2026 contest**
  (Reform barely stood pre-2026). Outputs: `canvassing_vs_voteshare.png` (3 panels:
  a=Labour/last, b=Green/2026, c=Reform/2026) — **binned scatters** (wards sorted by
  x, grouped into ~100-ward bins, one dot per bin = bin mean; trend lines and r use
  the full ward-level data). `canvassing_green_vs_reform_2026.png` — ward-level
  scatter on Green (x) × Reform (y) 2026 shares, point size = canvassing events,
  colour = Labour share at the last election. Also `canvassing_vs_voteshare.csv`
  (now carries `labour_share_2026` alongside `labour_share_last`).
  The script also writes `canvassing_vs_voteshare_correlations.json`:
    - **Ward level** — Pearson + Spearman of canvassing volume vs Labour-last,
      Labour-2026 (flagged endogenous — effort may have lifted it), Green-2026 and
      Reform-2026 share.
    - **Partials** — Green/Reform-2026 effects net of Labour's own prior strength.
      Labour canvassed more where Greens were strong even controlling for its own
      strength (green r≈+0.23 partial), and *less* where Reform was strong (≈−0.20).
    - **Constituency level** — Labour GE2024 share vs canvassing volume, joined on
      Westminster ONS code (GB and England-only), from the GE2024 results file.

Analysis dependencies: `matplotlib`, `numpy` (and `curl_cffi` for the postcode lookup).
