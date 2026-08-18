# DOGE-style reallocation: evidence from the first Reform budget

**Question:** in 2026-27 (Reform's first own budget), did Reform-majority counties cut the
discretionary "target" lines (culture, environmental/climate, planning, central/back-office)
and protect core, more than comparable non-Reform councils did?

**Design:** 6 Reform-majority shire counties vs 6 non-Reform shire counties that also voted
May 2025. Difference-in-differences of the 2025-26 (predecessor) -> 2026-27 (Reform) budget
change, so the finance settlement and inflation cancel. Budgeted, not actual. n=6 per group,
descriptive. Source: `03_reallocation.py`, tables in `reallocation_service_did.csv`.

## Verdict: little to no evidence at the aggregate-budget level

The DOGE signature is largely **absent** from the top-line service budgets. Read the budget
**shares** (the honest reallocation metric), not the eye-catching growth rates on tiny lines.

- **Discretionary bundle share fell in both groups on a shared downward trend**: treated
  9.97% -> 9.39%, control 10.06% -> 9.82%. Treated squeezed it ~0.34pp more. Real but small,
  and social care is crowding out discretionary spend everywhere, not just under Reform.
- **Culture was NOT hit harder.** Reform counties cut cultural spend *less* than controls
  (-1.2% vs -2.8%). No targeted libraries/arts raid visible in the aggregate.
- **Central services rose, not fell.** The classic DOGE target grew faster in treated than
  control. On its face this is *anti*-DOGE; more likely it reflects restructuring / a one-off
  or costs recoded in from elsewhere. Needs the budget book to resolve (this is exactly the
  Filter-4 "shows in the RA?" problem).
- **The dominant shift is toward adult social care** (+0.84pp share vs control), i.e. statutory
  demand pressure, the same force acting on every council.
- **Genuine treated-specific squeeze** shows only in **highways** (-0.39pp share, but just 3 of
  6 councils) and mildly **environmental** (-0.05pp, 4 of 6). Subset-driven, not systematic.

The two dramatic growth numbers (Planning -35pp DiD, Central +34pp DiD) are on small lines and
net to <0.4pp of budget share each. They read as reclassification between planning/economic-
development and corporate/central, not a real cut. Do not headline them.

## Implication

The aggregate RA cannot detect DOGE reallocation because, if it is happening, it is **within**
service lines (specific EDI/policy posts, consultancy contracts, climate programmes) that the
service totals net out. The null result at this level *sharpens* the case for coding the
councils' line-by-line savings schedules from the Feb-2026 budget books (the savings-waterfall
in the README). It also matches the sector reporting: Kent's DOGE review found little; officers
warned the statutory ~90% cannot be cut.

Consistent with the level finding: total budgets rose 9-12% into the first Reform budget, so
any savings are reallocations inside a growing budget, not net cuts.
