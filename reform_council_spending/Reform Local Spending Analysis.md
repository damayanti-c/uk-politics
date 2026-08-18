# Reform Local Spending Analysis

*How Reform-run councils are spending compared with others: what the data shows, what it cannot yet show, and when a fuller answer becomes possible. As of August 2026.*

---

## Executive summary

Reform UK won majority control of ten English councils at the May 2025 local elections, including six large shire counties. This analysis asks whether those councils spend differently from comparable non-Reform councils, in both level and pattern, and in particular whether they have delivered the "DOGE"-style efficiency drive and cuts to "waste", diversity and climate spending that the party has publicised.

The central constraint is timing. Councils set budgets in February/March for the financial year starting in April. Reform took control in May 2025, so the entire 2025/26 budget was set by the previous administrations, and Reform's first own budget is 2026/27. Actual spending figures (outturn) for that first Reform-set year are not published until around September 2027. Today we can therefore measure Reform councils' budgeted **intent**, not their **actual** spending.

On the intent data available now, the finding is consistent across every test: there is **no reallocation of substance**, and almost none of the current spending is Reform's own choice. Total budgets rose in line with comparable councils; the discretionary share of spending is near-identical to non-Reform counties; and a line-by-line audit of the six majority-run counties' savings schedules finds that, of roughly £254m in stated savings, only about £30-44m are genuinely new discretionary cuts, with **zero** diversity or EDI savings lines and only trivial climate cuts. Coding five comparable non-Reform counties' budgets the same way confirms this is normal rather than distinctive: their genuine discretionary cuts run at a near-identical share of budget, and they too make no EDI cuts. On council tax Reform did tax residents somewhat **less**: a difference-in-differences test finds Reform councils cut their council tax rise by about 0.7 points relative to comparable councils, which all held at the maximum, though the effect is concentrated in four of the six and the raw gap overstates it. The rhetoric substantially outruns the budgets. Whether even these modest changes are delivered is testable only from around September 2027.

---

## 1. The question, and why timing shapes the answer

The question has two parts: has **overall** spending changed under Reform, and has the **pattern** changed (for example more on core services and less on diversity, climate or back-office)?

Answering it depends entirely on the local budget cycle:

| Financial year | Set by | Status for this analysis |
|---|---|---|
| 2024/25 and earlier | Predecessor administrations | Clean "before" |
| 2025/26 | Predecessors (Feb 2025, before the election) | Transition year; Reform inherited it whole |
| **2026/27** | **Reform (Feb 2026), first own budget** | Clean "after", **available now as budget** |
| 2026/27 outturn (actual money spent) | Reform | Publishes around **September 2027** |

So "before versus after" cleanly means the years to 2024/25 versus 2026/27 onwards, with 2025/26 as a transition year that Reform inherited but could adjust in-year.

---

## 2. Data sources and availability

The comparison is well served at the aggregate level and thin at the granular level.

- **MHCLG Revenue Account (budget) and Revenue Outturn (actuals) returns.** The authoritative backbone: standardised, per-authority, per-service revenue spending for every council in England, comparable across authorities and years. Budget data is published around June each year; outturn around September of the following year. We built a budget panel covering 2022/23 to 2026/27 for all Reform-run councils plus matched non-Reform counties.
- **Councils' own budget books, Medium Term Financial Strategies and savings schedules.** These carry the line-by-line detail of planned changes: savings by service, council tax decisions, reserves. We coded the full 2026/27 savings schedules of the six Reform-majority counties from their primary committee papers.
- **The £500 transaction data.** Under the transparency code every council publishes each payment over £500. This is the only source granular enough to see supplier-level shifts (specific consultants, grants, roles), but it is inconsistent across councils and has not yet been analysed here.

**The comparison design.** Treated group: the ten councils Reform won in May 2025, of which the six shire counties with outright majorities (Derbyshire, Kent, Lancashire, Lincolnshire, Nottinghamshire, Staffordshire) are the cleanest cases. Control group: comparable non-Reform shire counties that also went to the polls in May 2025 (Cambridgeshire, Devon, Gloucestershire, Hertfordshire, Oxfordshire and others), matched on type and deprivation. Note two structural facts: police and fire are not funded from county council budgets (they sit with Police and Crime Commissioners and fire authorities), so the "more on policing" question does not belong to the council panel; and diversity or EDI spending is not a standard reporting line, so it is visible only in budget-book savings, the £500 data or FOI.

---

## 3. What we did

1. **Built a service-level budget panel** for 2022/23 to 2026/27 from the MHCLG Revenue Account returns, covering every treated and control authority, with each year's data validated against the published totals.
2. **Tested the aggregate level and composition** using a difference-in-differences design: the change in each service's spending and budget share from the predecessor's last budget (2025/26) to Reform's first (2026/27), for the Reform counties versus matched non-Reform counties, so the common finance settlement and inflation cancel out.
3. **Coded the savings schedules line by line.** For each of the six majority-run counties we located the primary budget document (the January or February 2026 Cabinet or Full Council budget report and its savings appendix) and extracted every savings line: 375 lines in total. Each was coded through a five-filter "waterfall" designed to separate a real, distinctive cut from normal budgeting: new versus carried-forward; cashable versus cost-avoidance versus income; statutory versus discretionary; whether it shows up in the aggregate return; and whether it is attributable to the Reform administration or inherited.
4. **Measured attribution**, quantifying how much of the current budget was in fact set before Reform took power.
5. **Coded a matched control group the same way.** We repeated the full line-by-line savings coding for five comparable non-Reform shire counties (Cambridgeshire, Devon, Gloucestershire, Hertfordshire, Oxfordshire), so the Reform figure has a like-for-like benchmark rather than standing alone.

---

## 4. Findings so far

Every test points the same way.

**Overall level: no cash cut.** Reform councils' total service budgets rose about 9 to 12% into their first budget (2026/27), in line with prior-year growth and with comparable councils. Any savings are reallocations inside a still-growing budget, not net reductions.

**Composition: no visible shift.** The discretionary services a council can actually move (culture, environment, planning, central and highways) are only about 9-10% of service spend, the rest being the statutory social care, SEND and education floor. That discretionary share is **9.4% in the Reform-majority counties versus 9.8% in comparable non-Reform counties**, near-identical. At the aggregate level there is no compositional reallocation.

**The savings audit: the claim is real in name, small in substance.** The six councils' own stated 2026/27 savings total about £254m. After the waterfall strips out what is inherited, technical, income or demand-avoidance, little remains:

| Waterfall step | £m |
|---|---|
| Headline (extracted lines) | 226 |
| less carried-forward / inherited (pre-May-2025 plans) | -48 |
| **= New savings** | **178** |
| less not reform-attributable (debt, pensions, MRP, energy, grant income) | -34 |
| **= New and reform-attributable** | **144** |
| less income generation and grant substitution | -19 |
| less cost-avoidance (demand mitigation, not a cut) | -34 |
| **= New, attributable, cashable spending cuts** | **91** |
| &nbsp;&nbsp;of which statutory (social care / SEND / education) | 48 |
| &nbsp;&nbsp;of which discretionary (the DOGE target) | **43** (30 on strict attribution) |

So genuinely new discretionary cuts are about **£30-44m across six £1-3bn budgets**, and roughly half of the real cuts fall on statutory social care demand management, not ideology.

**The DOGE test: the culture-war targets are essentially absent.** Across all 375 lines there are **zero** EDI, diversity or equalities savings; **zero** net-zero lines; and only about £0.3m of climate cuts (Staffordshire alone, removing a tree-planting programme). Consultancy appears once (£1.6m, Lancashire) and communications twice (£1.2m). The real discretionary cuts are conventional back-office and corporate efficiency, property and IT rationalisation, senior-management delayering (Kent's £1.5m "spans and layers"), and minor operational trims to street lighting, gritting, waste centres and libraries. Even the smallest, most symbolic-looking savings, such as Kent cutting office plants (£40k) and drinking-water provision (£30k), are not distinctively Reform: these are the same routine facilities trims non-Reform councils make (Cambridgeshire, a Liberal Democrat council, ended non-statutory first-class post for £30k), and neither the councils' own documents nor our coding frame them as part of a "DOGE" drive; the label was our shorthand, not theirs. Nottinghamshire's budget explicitly reaffirms its equality duty and climate declaration.

**Council tax: Reform cut its rise, but by less than the raw gap suggests.** In 2026/27 the six Reform councils' own precept rises averaged 3.93%, against 4.99% for the five controls, a 1.06-point gap. Because a level comparison cannot separate a Reform decision from the character of the councils Reform happened to win, we ran the same difference-in-differences we used for spending: each council's change from its predecessor's 2025/26 rise, treated versus control (source `06_council_tax_did.py`, MHCLG Band D data). That puts the genuine Reform effect at about **0.7 points**, not the full gap. The controls all held flat at the 4.99% maximum straight through their own 2025 changes of administration (mostly Conservative to Liberal Democrat or no overall control), while the Reform councils cut their rise by 0.70 points on average: councils that flipped to Reform reduced their tax rise, councils that flipped to the Lib Dems did not. The effect is real but concentrated: Lancashire (-1.19), Kent (-1.00), Staffordshire (-1.00) and Nottinghamshire (-0.85) genuinely cut their rise from the cap, whereas Derbyshire (-0.09) and Lincolnshire (-0.09) barely moved. Lincolnshire's headline-grabbing 2.90% is **not** a Reform achievement: it was already a 2.99% council under the Conservatives, which is exactly why the raw level gap overstates Reform restraint by about a third. The tax rate is a genuine annual Reform decision (unlike the largely inherited spending base), but there is no uniform Reform low-tax policy, and the "cut waste to cut tax" promise did not materialise: the savings fund cost pressures, not tax cuts, and Derbyshire set 4.90% amid opposition accusations of a broken pledge, having voted down a Conservative amendment for a lower 3.90%. The same result holds against a broader baseline: the 12 non-Reform shire counties held at 4.90% (up 0.02 points), so the difference-in-differences is -0.72 points, and the -0.70-point Reform effect is statistically significant despite the small sample (p ≈ 0.02, 95% confidence interval about -1.2 to -0.2). The clearest, assumption-free version: of the nine English shire counties that cut their rise this year, eight are Reform-led. (For context, the national average rise in the total Band D bill across all councils was 4.9%.)

**Compared with non-Reform councils, no different on cuts.** Coding the five control counties through the identical waterfall gives a like-for-like benchmark for the genuine-new-discretionary-cut figure. The two groups are statistically indistinguishable: Reform councils cut about **0.4% of budget** in genuine new discretionary savings, the non-Reform controls about **0.3-0.4%**, with both spanning roughly 0.1% to 1.0%. The **single largest discretionary cutter of all eleven councils is a Liberal Democrat council** (Hertfordshire, ~1.0% of budget, cutting libraries, urban grass-cutting, its communications team and a climate programme). The control councils also make **zero** EDI or equalities savings and only trivial climate cuts, exactly as the Reform councils do. On this evidence Reform councils are not cutting differently, or more, than comparable non-Reform councils.

**Attribution: almost none of the current spending is Reform's own choice.** The whole of 2025/26 was the predecessors'. Reform's 2026/27 budget is roughly 95% rolled-forward base (staff, contracts, statutory demand), with the savings package itself only about 1.5-4% of revenue spend. Even within that thin layer, about 16 to 17% of the savings were explicitly carried over from the previous administrations' plans (and much more once continuations of pre-existing programmes are counted). The inherited share varies widely: Lancashire about 58% of its savings were inherited, Nottinghamshire 29%, Staffordshire 25%, Derbyshire 22%, Lincolnshire 16%. Kent is the informative exception, writing off about £28m of the predecessor's savings as undeliverable and having to replace them.

**Cross-check against the returns.** Comparing the coded discretionary cuts to the actual budgeted change in discretionary lines, in four of the six councils the discretionary budget rose despite the cuts. The savings offset growth and demand rather than reducing lines, which is why the aggregate panel shows no compositional shift.

---

## 5. What this analysis does not cover

The findings above are a thorough look at one side of the revenue budget for six of the treated councils. The main gaps:

- **Revenue only, not capital.** Capital programmes (roads, buildings, regeneration, active travel) are a separate and large budget, untouched here. A priority shift, for example away from cycling and net-zero capital toward highways, would live there.
- **The cut side, not the additions side.** We coded savings in detail but not new spending and investments. "Reallocation" means cut X and add Y; we have measured X well and Y not at all.
- **Six of thirteen councils.** The four minority-run counties (Durham, Leicestershire, Warwickshire, Worcestershire) and the unitaries and metropolitan borough (North and West Northamptonshire, Doncaster) are not yet coded.
- **The £500 transaction data and reserves** are not yet analysed, so supplier-level EDI or consultancy shifts below the level of a named savings line remain unexamined (though the absence of any named line makes a large hidden cut unlikely).
- **Budgeted intent, not actual outturn.** Whether these plans are delivered is not yet observable.

---

## 6. What data is not yet available, and when

The definitive question (did actual spending change, in level and pattern) needs outturn data, which follows a fixed release calendar:

| When | Data that releases | What it unlocks |
|---|---|---|
| Now (2026) | 2026/27 budgets (all published) | The intent analysis above |
| ~Sept-Dec 2026 | 2025/26 outturn | First actuals touching Reform, but on a predecessor budget; shows any in-year changes |
| ~June 2027 | 2027/28 budgets | Reform's second budget; strengthens the intent picture |
| **~Sept-Dec 2027** | **2026/27 outturn** | **First clean Reform-controlled actuals: the trigger for the definitive study** |
| ~Sept-Dec 2028 | 2027/28 outturn | Second Reform year, enabling a proper before-and-after |

**The reorganisation clock.** All six Reform-run shire counties are two-tier authorities inside the local government reorganisation programme, and their county councils are expected to be abolished into new unitary authorities around 2027/28. This narrows the clean before-and-after window on the counties to essentially one or two years of outturn. The structurally stable Reform councils (Durham and the two Northamptonshire unitaries, and Doncaster) provide the longer, cleaner panel.

**The mayoral question is separate and currently blocked.** Reform also won two mayoralties (Greater Lincolnshire and Hull and East Yorkshire). These bodies were created in 2025, so there is no pre-2025 spending to compare against, and mayors do not take on Police and Crime Commissioner powers until around 2028 (and, for Hull and East Yorkshire, not at all, because of a police-force boundary mismatch). A before-and-after study of Reform mayors is therefore not viable now; the earliest a policing angle becomes testable is around 2028, and cleanly only for Greater Lincolnshire.

---

## 7. Bottom line

On the budget and savings data available now, the evidence is consistent and one-directional: Reform councils are not spending materially differently from comparable councils, they have not reallocated the budget in any substantive way, and the specific "DOGE" targets of diversity and climate spending barely feature in their savings plans. The control comparison makes this concrete: five non-Reform counties coded the same way show the same small scale of genuine discretionary cuts and the same complete absence of EDI cuts, and every one of them raised council tax by more than the Reform councils did. Almost none of what these councils are currently spending was chosen by Reform: the whole of 2025/26 was inherited, and 2026/27 is overwhelmingly rolled-forward base with a small, largely conventional savings layer on top. The public narrative substantially outruns what the budgets show, consistent with Kent and Staffordshire both conceding that their efficiency reviews found little.

This is the intent picture. The definitive answer on actual spending arrives with the 2026/27 outturn around September 2027, with a narrow window before reorganisation reshapes the treated councils.

---

## Appendix: sources and method

**Primary budget documents coded (2026/27):**

- Derbyshire: Revenue Budget Report 2026-27, Appendix 7 Budget Savings Proposals (Cabinet 22 Jan / Full Council 11 Feb 2026).
- Kent: Final Draft Revenue Budget 2026-27, Appendix F Spending, Savings and Reserves (County Council 12 Feb 2026).
- Lancashire: Draft Budget 2026/27, Appendix A MTFS, Annex 3 Savings (Cabinet 5 Feb / Full Council 26 Feb 2026).
- Lincolnshire: 2026/27 Budget Setting Report, Appendix B Budget Book Detail (Executive 6 Jan / Full Council 20 Feb 2026).
- Nottinghamshire: Annual Budget Report 2026-27, Appendix B Budget pressure mitigations (Full Council 26 Feb 2026).
- Staffordshire: MTFS 2026-2031 and 2026/27 Budget, Appendices 2 and 3 (Full Council 12 Feb 2026).

**Control (non-Reform) budget documents coded (2026/27):** Cambridgeshire (Business Plan 2026-31, Section 3), Devon (Budget Book 2026/27), Gloucestershire (Budget and MTFS 2026/27-2029/30 efficiencies plan), Hertfordshire (Integrated Plan Part I portfolio summaries), Oxfordshire (Service and Resource Planning Annex 1). Full source titles and URLs are held with the coded data.

**Council tax:** councils' own budget reports; national average from MHCLG "Council Tax Levels in England 2026/27".

**Aggregate spending data:** MHCLG "Local authority revenue expenditure and financing, England" Revenue Account (budget) and Revenue Outturn series.

**Waterfall filter definitions:** (1) new versus carried-forward, whether newly proposed by the current administration or already in the previous plan; (2) type, cashable saving versus cost-avoidance (growth not added) versus income generation; (3) statutory versus discretionary by service; (4) whether the saving is visible as a fall in the aggregate return; (5) whether the saving is attributable to the Reform administration or inherited/technical.

**Coverage note:** extraction quality varies by council. Kent and Lincolnshire publish full line-level schedules; Nottinghamshire publishes only pressure-mitigation totals, so its coverage is thinner. All amounts are in the councils' own published terms and reconcile to their stated subtotals.
