# BES voter panel internet study

This folder stores the BES internet panel (combined file, v30.1 release) inputs and cleaned CSV exports.

Note: this is the *combined* longitudinal panel covering all waves 1-30 (2014-2024), not a single wave-30 cross-section. Each survey item is repeated per wave with a `W1`..`W30` suffix (e.g. `generalElectionVoteW30`); `wave1`..`wave30` flag which waves each respondent took, and `waves_taken` counts them. The `W30` in the source filename (`BES2024_W30_Panel_v30.1.sav`) is the release version, not the wave coverage.

- Raw sources are staged in `raw/`.
- Any staged raw artifact larger than 95 MB is stored as `.partNNN` chunks to keep every file under 100 MB.
- Cleaned outputs are UTF-8 CSV files split into `*_partNNN.csv` files, each below 95 MB.
- Metadata is flattened into `*_codebook.csv` and part-level row ranges are tracked in `*_parts_manifest.csv`.
- Refresh command: `python source_data/bes_voter_panel_internet_study/convert_bes_voter_panel_internet_study.py`.

Datasets:
- `bes_voter_panel_2024_combined_w1_w30_panel`: 122,382 rows, 12,910 columns, 80 CSV parts, max part size 39.0 MB, encoding `UTF-8`.
- `bes_voter_panel_2024_combined_w1_w30_strings`: 122,382 rows, 422 columns, 3 CSV parts, max part size 61.0 MB, encoding `latin1`.
