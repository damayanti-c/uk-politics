# Six Midlands takeaways

Midlands (East + West Midlands, 65 LADs) versions of the national report's headline
findings. The three data-driven national findings are mirrored directly; the three
focus-group findings are replaced with further Midlands data findings. Figures are from
`recreate_core_analysis.py` and `additional_demog_political_analysis.py` (funding in 2025
prices; vote data = MRP, tactical projection, Q4 2025 for levels and Q4 2024 → Q4 2025 for
change). Findings A–D below were added after the trajectory / 2026-election work.

**1. Governments of every party funded the same kind of Midlands place: deprived and
politically alienated.**
The biggest per-head recipients are deprived ex-industrial and coastal towns — Ashfield
(£930), Boston (£929), Lincoln (£872), Newark & Sherwood (£846), Chesterfield (£803) —
several among the most heavily funded districts anywhere in England. Per-head funding
rises with deprivation (r = 0.50) and with Reform support (the ten best-funded Midlands
districts average ~40% Reform vs ~36% across the rest). The most-funded towns are also the
most Reform-leaning: Boston and Ashfield top both lists (each ~51% Reform).

**2. Funding broadly followed deprivation, not the electoral map.**
When deprivation, Reform support and seat marginality compete to explain funding,
deprivation is the only one that matters (standardised OLS: deprivation strongest; Reform
share and marginality not significant). If anything, more money reached *safer* seats, not
marginal ones (funding–seat-safeness r = +0.51) — targeting tracked need, not electoral
vulnerability.

**3. But *within* the Midlands, heavily funded areas are only marginally more pro-Reform
than comparable ones — the national premium does not replicate locally.**
Matching funded Midlands districts to similarly deprived but less-funded Midlands ones, the
Reform gap is about **+1.5pp** (median across specs; ≈ +2.6pp on a simple nearest-neighbour
match), positive in all twelve specifications but statistically robust only in the loosest,
and it vanishes once immigration is controlled (≈ −1.7pp). This is *not* the method
failing: the same matching run England-wide reproduces the national result (**+6.7pp**,
significant in 12/12 specs). The gap is small *inside* the Midlands specifically — see
takeaway 6 (chart: `04_control_pool_comparison.png`).

**4. Regeneration money did not bend the Reform trajectory.**
Over the past year Reform support surged by roughly **+11 points in funded and comparable
Midlands areas alike**; the matched difference in that rise is +0.4pp and insignificant in
every specification. The two groups move in near-parallel — funding neither slowed nor
accelerated the shift to Reform.

**5. The money is largely an East Midlands story, concentrated in smaller ex-industrial and
coastal towns, not the big cities.**
Seven of the ten best-funded Midlands districts are in the East Midlands — the
Nottinghamshire/Derbyshire coalfield (Ashfield, Newark & Sherwood, Chesterfield,
Mansfield) and the Lincolnshire coast (Boston, East Lindsey, Lincoln). The region's big
urban centres received far less per head — Birmingham £284, Coventry £115, Leicester £382 —
against a Midlands average of £359. The Black Country boroughs are the West Midlands
exception: both well-funded (Wolverhampton £543, Sandwell £521) and strongly pro-Reform.

**6. The reason (3) looks weak locally: the whole Midlands is already Reform-leaning — more
than its deprivation would predict — so the *comparison* areas are elevated too.**
The Midlands is more pro-Reform than England (36.3% vs 31.1% at the LAD mean) despite being
no more deprived (0.73 vs 0.72). Among English regions (chart:
`04_itl1_deprivation_vs_reform.png`), the **East Midlands has the highest Reform share
(36%) despite among the lowest deprivation** of the high-Reform regions, and the West Midlands
also sits above the England deprivation→Reform line; London is the mirror image (deprived
but only ~22% Reform). Because the region's *unfunded, comparably deprived* areas are
themselves already ~7pp more Reform than equivalent English areas, funded towns don't stand
out from their neighbours. The decisive test: holding the funded Midlands areas fixed and
only changing who they're matched against, the Reform gap is **+2.6pp vs comparable
Midlands areas** but **+11.3pp vs comparable areas in the rest of England**. So the funding
did land in strongly pro-Reform places — the Midlands just starts from a high Reform floor
that funding status doesn't add much to locally.

---

## Further findings: the East/West Reform trajectory and the 2026 test

**A. The West Midlands did not move to Reform *later* than the East — they moved in
near-lockstep.** On vote share the two halves track within a couple of points throughout
(our MRP: West actually a touch ahead in late 2024, the East rising faster to edge ahead by
Q4 2025, 36.0% vs 33.7%). General elections agree they are near-identical, the East a
fraction ahead at both 2019 (Brexit 1.5% vs 1.4%) and 2024 (Reform 18.9% vs 18.1%). The
West's much-discussed 2026 "breakthrough" is a **seats/electoral-calendar** story: its
councils simply had their first test since Reform's rise in 2026, exactly as the East
Midlands did in 2025 (Reform took 59% of contested county-council seats). Chart:
`13_reform_trajectory_ew.png`.

**B. The projected 2026 West Midlands breakthrough happened, almost exactly as YouGov's
pre-election MRP forecast.** Across the 13 West Midlands councils up on 7 May 2026, Reform
won the most votes in **11** (the projection said 11/13), averaged **31.5%** (projected
~30%), had double-digit leads in **7** councils, and took **250 of 493 seats**. Its ceiling
was the white Staffordshire towns (Cannock Chase 54%, Tamworth 52%); its floor the diverse
cities (Birmingham 19%, Greens first; Coventry 29%). Chart: `14_wm_2026_results.png`.

**C. Even in 2026, diversity — not deprivation — governs Reform support.** Within the 13
West Midlands councils, Reform's vote share falls steeply with diversity (r = −0.74) while
deprivation on its own does nothing (r = −0.10). Diverse deprived cities did **not** back
Reform "just as much"; the diverse metros are Reform's weakest councils. The one partial
exception is the Black Country (Sandwell/Walsall/Wolverhampton): diverse *and* deprived, yet
Reform won them at 34–37%, a few points above the diversity trend. Chart:
`15_wm_2026_diversity.png`.

**D. Deprivation and diversity are entangled in the West but decoupled in the East, which is
why the raw diversity gradient looks different by region.** corr(deprivation, diversity) is
just **+0.31** in the East (its deprived areas span white coalfield/coastal towns *and*
diverse cities) but **+0.69** in the West (its deprived areas essentially *are* the diverse
metros). That confound flattens the West's raw diversity→Reform correlation (−0.20 vs the
East's −0.60), but once you partial out deprivation the diversity effect is identical in both
(−0.82 West, −0.80 East), and deprivation lifts Reform equally in both (+0.85 / +0.71).
Underneath, both regions behave the same way. Chart: `16_dep_diversity_coupling.png`.

---
*Caveat: N = 65 Midlands districts (46 after the immigration filter) limits statistical
power; treat the matching estimates as descriptive. The 2026 findings (B–C) cover only the
13 West Midlands councils contested that year; there is no East Midlands 2026 comparator.*
