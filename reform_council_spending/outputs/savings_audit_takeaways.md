# DOGE savings audit: six Reform counties' first budgets (2026-27)

Line-by-line coding of the savings schedules in the six Reform-majority counties' own
Feb-2026 budget documents (Derbyshire, Kent, Lancashire, Lincolnshire, Nottinghamshire,
Staffordshire). 375 savings lines extracted and coded through the 5-filter waterfall.
Sources: each council's primary budget/MTFS PDF (cited in `data/savings_raw/*.json`);
consolidated in `data/savings_by_council.csv`; method `04_consolidate_savings.py`.

## Headline: the claim is real in name, small in substance

Councils' own stated 2026-27 savings across the six total **~£254m** (Derbyshire £55.3m,
Kent £62m, Lancashire £62m, Lincolnshire £35.5m, Nottinghamshire £18m, Staffordshire £21.5m).
That is the bulk of the [£330m LGC counted](https://www.lgcplus.com/finance/behind-reform-uks-700m-savings-claim-20-04-2026/)
across all Reform councils. But most of it is not new, not chosen by Reform, or not a cut:

| Waterfall step | £m |
|---|---|
| Headline (extracted lines) | 226.5 |
| less carried-forward / inherited (pre-May-2025 MTFS) | -48.5 |
| **= New savings** | **178.0** |
| less not reform-attributable (debt, pensions, MRP, energy, grant income) | -33.6 |
| **= New & reform-attributable** | **144.4** |
| less income generation & grant substitution | -18.9 |
| less cost-avoidance (demand mitigation, not a cut) | -34.5 |
| **= New, attributable, CASHABLE spending cuts** | **91.0** |
| &nbsp;&nbsp;of which STATUTORY (social care / SEND / education) | 47.5 |
| &nbsp;&nbsp;of which DISCRETIONARY (the DOGE target) | **43.5** (30.1 on strict attribution) |

So of a ~£254m headline, genuine **new discretionary cashable cuts are ~£30-44m** across six
large county councils. Roughly half of the real cuts fall on **statutory** social care / SEND
(assessment assurance, placement cost control, fee-rate reviews), which is demand-driven
pressure management, not ideology.

## The DOGE test: the culture-war targets are essentially absent

Keyword scan across all 375 lines:

- **EDI / diversity / equalities: 0 lines.** Not one named saving in any of the six councils.
  Nottinghamshire's budget explicitly *reaffirms* its equality duty and climate declaration.
- **Net zero: 0 lines. Climate: ~£0.3m** (Staffordshire only: removing a tree-planting
  investment programme £250k + sustainability-team efficiency £45k).
- **Consultancy: £1.6m** (Lancashire, one line). **Communications/PR: £1.2m** (Lancashire
  restructure + a carried-forward Staffordshire line).

Where the real discretionary cuts sit: **back-office and corporate** (£27.6m, the largest
category), property/estate rationalisation, IT/legacy systems, **senior-management delayering**
(Kent's £1.5m "spans and layers", £1m senior restructures), plus operational trims to street
lighting, gritting, waste centres, libraries, and subsidies to tourism bodies / the abolished
LEP. Even the tiniest, most symbolic-looking savings, Kent cutting **office plants (£40k) and
drinking-water provision (£30k)**, are not distinctive: they are the same routine facilities
trims non-Reform councils make (Cambridgeshire, Lib Dem, ended non-statutory first-class post
for £30k), and no council frames them as "DOGE", that label was our shorthand, not theirs.

## Council tax: Reform cut its rise, but by less than the raw gap suggests

In 2026/27 the six Reform councils' own precept rises averaged 3.93% against 4.99% for the five
controls (a 1.06pp gap). But a level comparison conflates a Reform effect with pre-existing
council character, so we ran a difference-in-differences on each council's change from its
predecessor's 2025/26 rise (`06_council_tax_did.py`, MHCLG Band D data):

| Council | 25/26 (predecessor) | 26/27 (Reform) | change |
|---|---|---|---|
| Lancashire | 4.99 | 3.80 | -1.19 |
| Kent | 4.99 | 3.99 | -1.00 |
| Staffordshire | 4.99 | 3.99 | -1.00 |
| Nottinghamshire | 4.84 | 3.99 | -0.85 |
| Derbyshire | 4.99 | 4.90 | -0.09 |
| Lincolnshire | 2.99 | 2.90 | -0.09 |
| Reform mean | 4.63 | 3.93 | **-0.70** |
| Control mean | 4.99 | 4.99 | 0.00 |

**Difference-in-differences = -0.70pp.** The genuine Reform effect is about 0.7 points, not the
1.06pp level gap. The controls all held flat at the 4.99% maximum through their own 2025
flips to Lib Dem / no overall control, while Reform councils cut their rise; so councils that
flipped to Reform reduced their tax rise and councils that flipped to the Lib Dems did not.
The effect is **concentrated in four councils** (Lancashire, Kent, Staffordshire, Nottinghamshire,
all cutting from the cap); Derbyshire and Lincolnshire barely moved. Crucially, **Lincolnshire's
low 2.90% is not a Reform achievement**, it was already a 2.99% council under the Conservatives,
which is exactly why the raw level gap overstates Reform restraint by about a third. The tax
rate is a genuine annual Reform decision (unlike the largely inherited spending base), but there
is no uniform Reform low-tax policy, and the "cut waste to cut tax" promise did not materialise
(savings fund pressures; Derbyshire set 4.90% amid a broken-pledge row, having rejected a
Conservative 3.90% amendment).

## Compared with non-Reform councils: no different on cuts

Coding five comparable non-Reform shire counties (Cambridgeshire, Devon, Gloucestershire,
Hertfordshire, Oxfordshire) through the identical waterfall gives a like-for-like benchmark
(`05_treated_vs_control_savings.py`). Genuine new discretionary cuts run at **~0.4% of budget
for the six Reform councils and ~0.3-0.4% for the five controls**, statistically
indistinguishable, both spanning roughly 0.1% to 1.0%. The **single largest discretionary
cutter of all eleven councils is a Liberal Democrat council** (Hertfordshire, ~1.0%, cutting
libraries, urban grass-cutting, its communications team and a climate programme). The control
councils also make **zero** EDI/equalities cuts and only trivial climate cuts, exactly as the
Reform councils do. On this evidence Reform is not cutting differently, or more, than
comparable non-Reform councils.

## Filter 4: the cuts don't show as budget-line reductions

Comparing coded discretionary cuts to the actual RA discretionary-budget change 2025-26 ->
2026-27: in **four of six councils the discretionary budget rose** despite the cuts (Kent
+£40m, Lancashire +£12m, Lincolnshire +£11m, Staffordshire +£7m). The savings offset growth
and demand; they are not absolute line reductions. This is why the aggregate RA panel (script
03) showed no compositional shift: the cuts are real but small and swallowed by pressures.

## Bottom line

The line-by-line evidence confirms the aggregate finding. **There is no DOGE-style
reallocation of substance.** The headline savings are mostly inherited programmes, technical
and actuarial items, income, and demand-avoidance in statutory social care. Genuine new
discretionary cuts are a small slice, concentrated in conventional back-office efficiency and
senior-management/property, with essentially nothing on EDI/diversity and only trivial climate
cuts. The rhetoric substantially outruns the budgets, consistent with
[Kent conceding its DOGE review found little](https://www.gbnews.com/money/reform-elon-musk-doge-failed-savings)
and Staffordshire's efficiency review yielding £2.7m.

Caveat: budgeted intent, not outturn. Whether even these modest cuts are delivered is testable
only when 2026-27 outturn releases (~Sept 2027). Extraction quality varies by council (Kent and
Lincolnshire publish full line schedules; Nottinghamshire publishes only pressure-mitigation
totals); coverage notes per council in `data/savings_raw/`.
