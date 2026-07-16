# Regeneration funding

Work built on Public First's 2026 analysis of a decade of UK regeneration funding
(*"What became of the likely lads?"*). This folder holds the original national
analysis and a Python rebuild focused on the Midlands (in [`midlands/`](midlands/)).

## Published analysis (source material)

Everything here keys off one published model and its accompanying write-ups.

| Resource | What it is |
|---|---|
| [Public report: *What became of the likely lads?* (PDF)](https://assets.nationbuilder.com/stonehaven/pages/5044/attachments/original/1782726485/What_became_of_the_likely_lads_%E2%80%94_Public_First_report_compressed.pdf?1782726485) | The headline report: ~£19bn of regeneration funding mapped to every UK local authority since the Brexit referendum, compared with deprivation, voting intention and marginality, plus Blackpool/Clacton focus groups. |
| [Google Sheets model: *Regeneration funding initatives*](https://docs.google.com/spreadsheets/d/1KL0VPL_AwtQXPtU70bUxR-BBLHEWbl3xL0vw9leM5sY/edit) | **The canonical model.** Every qualifying fund mapped to LAD, harmonised to 2024 boundaries and 2025 prices, giving total and per-capita funding, the top-10 lists, etc. The `Funding per LAD per capita` tab is the single source of the funding figures used downstream. |
| [Methodology: towns analysis](https://docs.google.com/document/d/1zDH7W0__gS0ua1O6dRoqBLDaxHWQBAkxTe5-fquftHI/edit) | How funds were selected (the three inclusion tests) and how the per-head top-10 recipients were derived. |
| [Location to Standard Geography Matcher (Colab)](https://colab.research.google.com/drive/1Vb69uYgMmEPpI2pjgdp_KCWov1bnM2at) | Matches funding recipients (e.g. Levelling Up Round 2 bids) to standard LAD geography; an upstream input to the Google Sheet. A copy is in this folder. |
| [Follow-on blog](https://docs.google.com/document/d/1kdKFIgFF9-6BpW8itbrLNlFxgha-ifOH1oZJfaALE58/edit) | Short "deliverism" write-up building on the report. |
| [Data folder (Google Drive)](https://drive.google.com/drive/folders/1EcgKv6Uoymmw7VOS1y3pR72JwtpB6yMF) | Raw inputs: ONS Census 2021, MRP by quarter, geography lookups, boundaries. |
| [FT coverage](https://www.ft.com/content/a1085f61-2089-4a80-9bd3-4af7737fef4b) | Financial Times write-up of the findings. |
| Public First research: [a decade of post-Brexit regeneration funding](https://www.publicfirst.co.uk/a_decade_of_post-brexit_regeneration_funding_reform_and_resentment) · [when reality and perception match](https://www.publicfirst.co.uk/when-reality-perception-match) | Related PF research referenced by the report. |

## Files in this folder

| File | What it does | Uses the published model? |
|---|---|---|
| [`Regeneration analysis.R`](Regeneration%20analysis.R) | The original **national** analysis in R: choropleth maps, funding vs deprivation, funding vs vote choice / marginality, and `MatchIt` matching of Reform support in funded vs comparable LADs. | Yes: reads the Google Sheet's `Funding per LAD per capita` CSV export, plus census/MRP. |
| [`Location to Standard Geography Matcher`](Location%20to%20Standard%20Geography%20Matcher) | The geography-matching notebook (same as the Colab above), mapping fund recipients to LADs. | Upstream: feeds the Sheet. |
| [`topN_funding_vs_deprivation.py`](topN_funding_vs_deprivation.py) → `topN_funding_vs_deprivation.png` | England scatter of funding per head vs household deprivation, top 10 and ranks 11-20 highlighted, least-funded 5 marked. | Yes: funding per head comes from the model (via the `midlands/` loaders). |
| [`midlands/`](midlands/) | Python **rebuild of the analysis for the Midlands** (top 10, deprivation/vote relationships, matching, plus BES switching, maps and demographic charts). See [`midlands/README.md`](midlands/README.md). | Yes: see below. |

## Which scripts pull from the published analysis

The dependency chain is: **Google Sheets model → `midlands/data/funding_per_lad.csv` (an export of its `Funding per LAD per capita` tab) → the scripts.**

- `midlands/regeneration_midlands.py` reads that export in `load_funding()`. Everything that imports it therefore inherits the published funding figures:
  - `topN_funding_vs_deprivation.py`
  - `midlands/bes_midlands_switching.py`
  - `midlands/map_midlands_funding.py`
  - `midlands/age_deprivation_party.py`
  - `midlands/age_deprivation_party_regions.py`
  - `midlands/structure_charts.py`
- `Regeneration analysis.R` reads the same Sheet exports directly (national version).

The two **methodology notes** describe the method these scripts implement; the **matcher notebook** produced the geography matching that built the Sheet in the first place. Census/MRP/boundary inputs come from the Drive data folder (and, for the Midlands rebuild, are cached under `midlands/data/`).
</content>
