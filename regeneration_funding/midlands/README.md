# Regeneration funding in the Midlands

A Midlands-specific rebuild of the UK-wide regeneration analysis (`Regeneration analysis.R`
and the Public First report *"What became of the likely lads?"*), reimplemented in Python.

**Midlands = East Midlands (E12000004) + West Midlands (E12000005) = 65 local authority
districts (35 + 30).** The LAD set is defined explicitly in the script from ONS region
membership and validated against the funding data at runtime.

## Two scripts

### `recreate_core_analysis.py` — the original analysis, rebuilt (outputs `00`–`03`)
Also holds the shared data loaders, the Midlands definition and the matching helpers
that the second script imports.
1. **Headline** (`00`) — the Midlands' share of the ~£19bn, and the East/West split.
2. **Top 10** (`01`) Midlands LADs by regeneration funding per head.
3. **Relationships** (`02`) between funding and (a) deprivation, (b) Reform UK vote share,
   (c) seat marginality — scatters, correlations, and a standardised OLS.
4. **Matching** (`03`) — treatment = heavily funded (above the Midlands mean / median),
   matched on deprivation, testing whether funded areas are more pro-Reform than comparable
   less-funded ones. Six estimators × two thresholds, an immigration-robustness cut, and
   both the **level** (latest quarter) and **change** (Q4 2024 → Q4 2025) of Reform support.
5. **Control-pool** (`04`) — the same funded Midlands areas matched (on deprivation) to
   comparable areas in the rest of the Midlands vs the rest of England, shown both raw and
   with the national analysis's immigration adjustment (LADs ≤13% foreign-born).

### `additional_demog_political_analysis.py` — everything beyond the original (imports the core module)
**Section 1 — funding vs demographics:** East/West summary table (`east_west_summary.*`),
household deprivation by ITL1 region (`itl1_deprivation_bar.png`), funding-per-head
choropleth with the East/West boundary (`06`), deprivation vs Reform by region and by LAD
(`04_itl1_*`, `04_lad_*`), and a test of **what explains the East/West funding gap for
deprived areas** — urbanity, social mobility (SMI) and geographic mobility
(`12_east_west_funding_drivers.png`).
**Section 2 — wider political:** leading party by age & deprivation (England plane `07`;
East/West panels `08`), the age × urbanity battleground (`09`) and deprivation-vs-Reform by
several diversity metrics (`10`), Labour→Green/Reform switching by deprivation from the BES
panel (`05`), and the UKIP/Brexit/Reform vote-intention trajectory among likely voters over
the BES waves, East vs West (`11_reform_trajectory_bes.png`). It then tests the East/West
Reform **trajectory** against hard election data: the MRP quarters plus GE 2019/2024 and the
2025 locals (`13`), whether the projected 2026 West Midlands breakthrough actually happened
(`14`, actual ward results vs the YouGov MRP), whether diverse deprived areas backed Reform
"just as much" (`15`, they did not — diversity still tracks Reform), and why the raw
diversity gradient looks cleaner in the East (`16`, deprivation-diversity coupling +
partials).
(The control-pool Reform test now lives in the core script, `04`.)

## Run

```bash
python recreate_core_analysis.py                 # outputs 00-03
python additional_demog_political_analysis.py    # outputs 04-10, maps, tables (reassembles the BES panel; ~2 min)
```

Outputs (tables + charts) are written to `./outputs`; key numbers print to stdout.

## Data (`./data`, pulled from the project Google Drive)

| File | Source | Use |
|---|---|---|
| `funding_per_lad.csv` | Regeneration funding model (Google Sheet) | Per-capita funding by LAD24 (2025 prices) |
| `deprivation_ew.csv` | ONS Census 2021, household deprivation (6 categories) | Deprivation score (mean dimensions 0–4) |
| `Political_MRP_by_quarter.xlsx` | MRP, five quarters Q4 2024 → Q4 2025 | Projected vote shares + seat safeness (tactical projection) |
| `ward22_to_lad22.csv` / `ward24_to_lad24.csv` | ONS lookups | Ward → LAD aggregation of the MRP |
| `cob_ew.csv` | ONS Census 2021, country of birth | Share foreign-born (immigration robustness) |
| `lad24_to_region.csv` | ONS Open Geography (`LAD24_RGN24_EN_LU`) | LAD24 → region, for the Midlands definition and the ITL1 charts |
| `lad24_ruc.csv` | ONS Open Geography (`LAD24_RUC21_EW_LU`) | Urban/rural classification (additional analysis) |
| `lad_ethnicity.csv` | Census 2021 TS021 via NOMIS | Non-White %, non-White-British %, ethnic diversity index (additional) |
| `lad_mobility.csv` | Census 2021 TS019 (Migrant Indicator) via NOMIS | % who moved address / moved within the UK in the year before the census (geographic mobility) |
| `lad_social_mobility.csv` (+ `social_mobility_index.xlsx`) | Social Mobility Commission, Social Mobility Index 2016 | LAD social-mobility score & rank (old Northamptonshire districts aggregated to the 2024 unitaries) |
| `england_lad_boundaries.geojson`, `midlands_lad_boundaries.geojson` | ONS Open Geography (BUC) | Boundaries for the funding map and LAD density (additional) |

The additional analysis also reads several inputs from elsewhere in the repo: the **BES 2024
internet panel** (`source_data/bes_voter_panel_internet_study/`) for the Labour-switching and
BES-waves charts, **ONS mid-2024 population estimates**
(`labour_voter_demog_change/data/ons_mye24tablesuk.xlsx`) for the age (% 50+) axis, and
**election results** from `source_data/election_results/` for the trajectory / 2026 charts:
GE 2019 & 2024 by constituency, the 2025 local-election analysis (`CBP10272.xlsx`), and the
actual 2026 local results (`local_elections/2026_external/`, ward-level candidate votes).

The MRP workbook provides the full five-quarter series, using the "Projected w/ Tactical"
projection as in the original R script. Cross-sectional analyses use the latest quarter
(Q4 2025); the matching also tests the *change* in Reform support over the year.

Midlands deprivation, vote choice and funding are all England-coded on stable LAD24 codes.
For the England-wide charts in the additional script, `load_deprivation()` rebuilds the four
April-2023 unitaries (Cumberland, Westmorland & Furness, North Yorkshire, Somerset) from
their component 2021 districts (household-weighted), since the Census predates them; none of
these are in the Midlands, so the core Midlands analysis is unaffected.

## Headline findings (Q4 2025 MRP; five-quarter series for dynamics)

- **Top 10** is dominated by the **East Midlands** (7 of 10): Ashfield (£930), Boston
  (£929), Lincoln (£872), Newark & Sherwood (£846), Chesterfield (£803), East Lindsey
  (£768), Nuneaton & Bedworth (£760), Wyre Forest (£686), South Derbyshire (£683),
  Worcester (£666). Midlands mean per-capita funding ≈ £359.
- Funding is positively correlated with **deprivation** (r ≈ 0.50), **Reform share**
  (r ≈ 0.40) and **seat safeness** (r ≈ 0.51). In the standardised OLS, **deprivation is
  the strongest correlate** of funding; Reform share adds little once deprivation and
  marginality are included — echoing the national finding that funding broadly followed
  deprivation, not the electoral map.
- **Matching (level):** across all Midlands LADs the Reform gap between heavily funded and
  comparable areas is **small and positive (median ≈ +1.5pp)**, positive in all 12 specs
  but significant in only the uncapped nearest-neighbour specs (which permit poor
  matches). Under the immigration-controlled cut the gap **disappears / turns slightly
  negative** (median ≈ −1.7pp). So the strong national result (a 5–7pp Reform premium in
  funded areas) **does not robustly replicate within the Midlands**.
- **Why (national context):** this is not the method failing — run England-wide the same
  matching reproduces the national premium (**+6.7pp**, significant in 12/12). The
  within-Midlands gap is small because the region is **already Reform-leaning relative to
  its deprivation** (36.3% vs England's 31.1% at the same ≈0.73 deprivation; the East
  Midlands is the highest-Reform English region despite middling deprivation). Holding the
  funded Midlands areas fixed, the Reform gap is **+2.6pp vs comparable Midlands areas** but
  **+11.3pp vs comparable areas in the rest of England**.
- **Matching (change over time):** Reform support surged in *both* groups over the year
  (≈ +11pp), and the rise was **no larger in funded areas** (matched gap ≈ +0.4pp, not
  significant in any spec). The over-time chart shows two near-parallel lines: funded
  areas sit a little above comparable ones throughout, but funding neither accelerated nor
  damped the Reform surge.

Caveats: N = 65 (46 after the immigration filter) limits statistical power. Treat the
matching estimates as descriptive.
