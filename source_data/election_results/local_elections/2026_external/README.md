# 2026 Local Election Results: external staging

This folder stages 2026 local-election result files from external sources.

- `electionresults_uk_races.csv`: cached full `electionresults.uk` ward or division race feed.
- `electionresults_uk_candidates.csv`: cached full `electionresults.uk` candidate feed.
- `electionresults_uk_races_2026.csv`: 2026-only subset of the race feed.
- `electionresults_uk_candidates_2026.csv`: 2026-only subset of the candidate feed.
- `opencouncildata_history2016_26.csv`: cached Open Council Data annual composition archive.
- `opencouncildata_council_control_2026.csv`: 2026-only subset of the composition archive.
- `manifest.csv`: source URLs and local output paths for the staged files.

Why external?

- I could not locate a published Commons Library 2026 local-election handbook page or 2026 local-election results-analysis briefing as of 31 May 2026.
- To avoid leaving a gap, this folder uses `electionresults.uk` for contest and candidate results, and Open Council Data for 2026 council-control context.

Refresh command:

- `python scripts/download_local_elections_2026_external.py`
