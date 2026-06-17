# House of Commons Library election results

This folder contains House of Commons Library files downloaded for the 2016-2025 calendar years, plus a separate external staging folder for 2026 local-election data.

- Scope used here: Westminster general election results pages and local election results/handbook pages published by the Commons Library.
- Additional 2026 staging: `local_elections/2026_external/` contains external local-election result files because no Commons Library 2026 local-election handbook/results-analysis release was located as of 31 May 2026.
- Storage layout: `national_general_elections/` and `local_elections/`, grouped into page-specific subfolders.
- Manifest: see `manifest.csv` for the Commons Library source page, download URL and saved path for every Commons file. The `2026_external/` folder has its own manifest.
- Refresh command: `python scripts/download_hoc_election_results.py`.

Downloaded files: 44
Source page groups: 16

Notes:
- A separate 2024 Commons Library local-election results-analysis page was not located during source collection, so only the 2024 handbook/dataset page is included for that year.
- A Commons Library 2026 local-election handbook page or results-analysis briefing was not located as of 31 May 2026, so the `2026_external/` folder uses non-Commons sources for that year.
