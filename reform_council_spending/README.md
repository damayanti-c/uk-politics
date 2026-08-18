# Reform councils and spending decisions

Does electing Reform UK change how a local council spends, in level and in pattern
(e.g. more on core/community-safety services, less on culture, climate and EDI)?

This folder holds the completed **interim study**: an audit of Reform's first budgets and its
"DOGE" savings claims, using data that is public **now**, benchmarked against matched non-Reform
councils. The full difference-in-differences study on *actual* spending cannot run until the
first Reform-controlled outturn data releases (~Sept 2027); the release calendar and the
mayoral/reorganisation caveats are documented in `Reform Local Spending Analysis.md` (section 6).

## Why the study splits in two: the budget cycle

Reform won ~10 councils outright at the **May 2025** locals. Councils set budgets each
February/March for the year starting in April. So:

| Financial year | Budget set by | Status for us |
|---|---|---|
| 2024/25 and earlier | predecessor administrations | clean "before" |
| **2025/26** | predecessors (Feb 2025, before Reform arrived) | transition year, Reform inherits it |
| **2026/27** | **Reform (Feb 2026), their first own budget** | clean "after", **budget data out now** |
| 2026/27 *outturn* (actual money spent) | Reform | releases **~Sept 2027** |

So *intent* (budgets, savings plans) is measurable now; *actuals* (outturn) are not.
We ship the intent study now and refresh with actuals in 2027.

## The interim deliverable: a savings-claim "waterfall"

Reform's DOGE claims ~£700m of savings "identified" and ~£400m in 2026/27 proposals;
[LGC independently counted ~£330m](https://www.lgcplus.com/finance/behind-reform-uks-700m-savings-claim-20-04-2026/)
in the budget papers. The audit runs each claimed saving through five filters that
separate a real, distinctive cut from normal budgeting. What survives is defensible.

| Filter | Question | Source |
|---|---|---|
| 1 New vs carried | newly proposed by Reform, or inherited from the predecessor MTFS? | budget book vs prior MTFS |
| 2 Cashable vs avoidance | an actual cut, or "growth we chose not to add"? | savings schedule (RAG-rated) |
| 3 Discretionary vs statutory | does it hit services Reform controls, or assume cuts to social care / SEND that officers say cannot be delivered? | schedule + officer commentary |
| 4 Shows in the RA? | did the budgeted service line actually fall, or is it a reserves / accounting shift? | MHCLG RA panel |
| 5 Distinctive vs normal | unusual, or the same scale matched non-Reform counties booked under the same settlement? | control councils |

Filter 5 is what makes it more than churnalism: it converts "Reform claims £X" into
"Reform cut £X, of which £Y is more than politically-comparable councils did anyway".

**Two framing numbers already computed** (`02_load_ra_panel.py`):

1. The **discretionary** services Reform can actually move (culture, environmental, planning,
   central, highways) are only **~9-10% of total service spend** in the treated counties. The
   other ~90% is the statutory adult/children's social care, public health and education floor.
   Any large savings claim must come from a small discretionary base, from the statutory floor
   (contested by officers), or be cost-avoidance / rebadging.
2. That discretionary share is **9.4% in the Reform-majority counties vs 9.8% in comparable
   non-Reform counties** (2026/27 budgets): near-identical. So there is **no dramatic
   compositional shift in the top-line service budgets**. The real signal, if any, is in the
   within-line savings decisions, which is precisely why the study codes budget-book savings
   schedules rather than resting on the aggregate RA totals.

Together these are the spine of the story: the aggregate budget looks normal, so the claim
has to be tested at the level of the individual savings line.

## "More policing, less diversity", made measurable

- **Policing is not a county-council budget line** (`poltot` = 0 for every county: it is
  funded via Police & Crime Commissioners / the precept). The policing angle belongs to the
  **Reform mayors** (Greater Lincolnshire, Hull & East Yorkshire), a separate study, not the
  council panel.
- **EDI / "diversity" is not a standard MHCLG line** either. It lives in the £500
  transparency data (EDI consultants, training, grants), budget books and FOI, not the RA.
  It is a targeted sub-study (Phase 2), not something the clean panel returns.

What the panel *does* measure well: total spend, and the shares going to the ~11%
discretionary bundle, plus council tax level and reserves drawdown.

## Data

| Source | What | Status |
|---|---|---|
| MHCLG Revenue Account (RA) budget | per-authority, per-service **budgeted** spend | `data/RA_2026-27_Part_1.ods` (+ Part 2 = reserves, council tax support). Backbone. |
| MHCLG Revenue Outturn (RO) | per-authority, per-service **actual** spend | releases ~Sept after year end; first Reform year = Sept 2027 |
| Council budget books / MTFS / DOGE reports | line-by-line savings schedules | manual, one PDF per council (Feb 2026) |
| £500 transparency data | supplier-level payments | per-council CSVs, messy; Phase 2 only |

RA files: [2026-27 budget release](https://www.gov.uk/government/statistics/local-authority-revenue-expenditure-and-financing-england-2026-to-2027-budget)
and the [full collection](https://www.gov.uk/government/collections/local-authority-revenue-expenditure-and-financing) (all prior years for the panel).

## The sample (`data/treated_control_frame.csv`)

- **Treated, majority (9):** Derbyshire, Kent, Lancashire, Lincolnshire, Nottinghamshire,
  Staffordshire (counties); North & West Northamptonshire (unitaries); Doncaster (met).
- **Treated, minority (4):** Durham, Leicestershire, Warwickshire, Worcestershire (Reform
  largest party / minority administration; budget control shared, flag separately).
- **Gold controls (6):** the shire counties that voted May 2025 and stayed non-Reform
  (Cambridgeshire, Devon, Gloucestershire, Hertfordshire, Oxfordshire, plus one more).
- **Wider control pool:** same-type non-Reform unitaries / met boroughs (matching in analysis).
- **Excluded (second wave):** East Sussex, Essex, Hampshire, Norfolk, Suffolk, Surrey,
  West Sussex. Elections postponed to May 2026 by LGR; possible staggered second cohort later.

## Files

```
01_build_frame.py     -> data/treated_control_frame.csv   (curated sample + control pool)
02_load_ra_panel.py   -> data/panel_ra_{long,wide}.csv     (tidy RA panel; add prior years to backfill)
03_reallocation.py    -> outputs/reallocation_*            (aggregate composition diff-in-diff)
04_consolidate_savings.py -> data/savings_by_council.csv   (6 Reform councils' 375 coded lines + waterfall)
05_treated_vs_control_savings.py -> outputs/treated_vs_control_savings.csv (Reform vs 5 non-Reform controls)
06_council_tax_did.py -> council tax difference-in-differences (Reform vs controls, MHCLG Band D data)
data/savings_raw/*.json          (per-council coded savings; Reform)
data/savings_raw_control/*.json  (per-council coded savings; non-Reform controls)
data/savings_extraction_template.csv                        (the 5-filter coding schema for budget PDFs)
Reform Local Spending Analysis.md   (comprehensive write-up)   BRIEFING.md (one-pager)
outputs/savings_audit_takeaways.md  (savings + control-comparison findings)
```

Run:
```bash
pip install odfpy pandas
python 01_build_frame.py
python 02_load_ra_panel.py
```

## Caveats

- **Budget, not actuals.** The interim study measures intent. Reform can miss its own
  savings targets, and predecessors' budgets can overspend. Definitive answer needs outturn.
- **Local Government Reorganisation.** All six Reform-run shire counties are two-tier and in
  the LGR programme; their county councils may be abolished into new unitaries around
  2027/28, breaking panel continuity exactly in the treated group. Confirm per-council status.
- **Statutory floor** (~89% of spend) limits how far total spend can move regardless of politics.
- **Small N, short post-period.** Treat interim estimates as descriptive; power grows with
  each outturn release.

## Status

Phase 0 done: sample fixed; **RA budget panel backfilled 2022-23 -> 2026-27** (5 years,
453 authorities); savings template ready. 2021-22 is excluded (COVID-distorted and uses a
different, text-label file layout); add it only if a longer pre-trend is needed.

Preliminary read from the panel: treated councils' **total budgets rose ~9-12% in the first
Reform budget (2026-27), in line with prior-year growth**, so overall spending did not fall
in cash terms. Any DOGE savings are reallocations inside a still-growing budget, not net cuts.
Combined with the near-identical discretionary share vs controls, the story is compositional
and lives in the savings lines.

Savings audit done: 375 savings lines coded across the six Reform counties, plus five matched
non-Reform controls coded the same way. Result: genuine new discretionary cuts are only
~£30-44m of a ~£254m headline (~0.4% of budget), **statistically the same as the non-Reform
controls (~0.3-0.4%)**; zero EDI/diversity lines in either group; and on council tax a
difference-in-differences (Reform vs controls, `06_council_tax_did.py`) finds Reform cut its
rise by ~0.7pp (concentrated in Lancs/Kent/Staffs/Notts; Lincs was already low pre-Reform, so
the raw 3.9%-vs-5% level gap overstates it). See `Reform Local Spending Analysis.md`, `BRIEFING.md`,
`outputs/savings_audit_takeaways.md`.

Next (open): (a) deflate the panel to real per-capita terms and formally test pre-trends;
(b) capital programmes and the spending-additions side (the "add" half of reallocation);
(c) refresh with 2025/26 then 2026/27 outturn as it releases (~Sept 2026 / ~Sept 2027).
