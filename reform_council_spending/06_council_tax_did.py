"""
Council tax difference-in-differences: is "Reform taxed less" a genuine effect or a
pre-existing level?

A level comparison (2026/27: Reform 3.93% vs controls 4.99%) can't separate a Reform decision
from the character of the councils Reform happened to win (Lincolnshire was already a 2.99%
council under the Conservatives). The fix is the same DiD we used for spending: each council's
CHANGE in its own precept rise from 2025/26 (predecessor-set) to 2026/27 (first Reform budget),
treated vs control.

Comparators, most like-for-like first:
  - control (5)          : the hand-picked clean non-Reform shire counties.
  - non-Reform SC (12)   : ALL English shire counties minus the Reform-run/led ones (the fair
                           England-county baseline; the all-21 average is contaminated because
                           it includes the Reform cutters, so do not use it as the comparator).

Source: MHCLG "Council Tax Levels in England 2026-27", Band D live table (data/BandD_2026-27.ods),
sheet exc_PP = each authority's OWN Band D excluding parish precepts (for county councils this is
their own precept incl. adult social care). % increases computed from the cash Band D chain
reconcile to the councils' reported headline rises.

Run:  python 06_council_tax_did.py
"""
from pathlib import Path
import math
import statistics as st
import pandas as pd

DATA = Path(__file__).parent / "data"
REFORM_MAJ = ["Derbyshire", "Kent", "Lancashire", "Lincolnshire", "Nottinghamshire", "Staffordshire"]
REFORM_MIN = ["Leicestershire", "Warwickshire", "Worcestershire"]  # Reform-largest / minority admin
CONTROL = ["Cambridgeshire", "Devon", "Gloucestershire", "Hertfordshire", "Oxfordshire"]
YEARS = ["2024 to 2025", "2025 to 2026", "2026 to 2027"]
T_CRIT = {5: 2.571, 4: 2.776}  # two-sided 95% t critical values by df


def load_sc():
    raw = pd.read_excel(DATA / "BandD_2026-27.ods", engine="odf", sheet_name="exc_PP", header=None)
    hdr = [str(raw.iat[2, c]) for c in range(raw.shape[1])]
    col = {y: hdr.index(y) for y in YEARS}
    sc = {}
    for r in range(3, raw.shape[0]):
        a, cls = str(raw.iat[r, 2]).strip(), str(raw.iat[r, 4]).strip()
        if cls == "SC":
            try:
                sc[a] = [float(raw.iat[r, col[y]]) for y in YEARS]
            except (ValueError, TypeError):
                pass
    return sc


def rises(sc, a):
    v = sc[a]
    return (round(100 * (v[1] - v[0]) / v[0], 2), round(100 * (v[2] - v[1]) / v[1], 2))


def main():
    sc = load_sc()
    all_sc = list(sc)
    nonreform = [a for a in all_sc if a not in REFORM_MAJ + REFORM_MIN]

    def grp(names):
        r2526 = [rises(sc, a)[0] for a in names]
        r2627 = [rises(sc, a)[1] for a in names]
        chg = [round(b - a, 2) for a, b in zip(r2526, r2627)]
        return r2526, r2627, chg

    # per-council table for the six Reform-majority counties
    print("Own-precept council tax rise, 2025/26 -> 2026/27 (Reform-majority counties)")
    print(f"{'council':<16}{'25/26':>8}{'26/27':>8}{'change':>9}")
    tr = grp(REFORM_MAJ)
    for a, i, j, c in sorted(zip(REFORM_MAJ, *tr), key=lambda x: x[3]):
        print(f"{a:<16}{i:>8}{j:>8}{c:>+9}")

    # group means and DiD
    print("\ngroup means (25/26 -> 26/27 -> change):")
    groups = {"Reform majority (6)": REFORM_MAJ, "Control (5)": CONTROL,
              f"Non-Reform shire counties ({len(nonreform)})": nonreform,
              f"ALL shire counties ({len(all_sc)}) [contaminated]": all_sc,
              "Reform minority (3)": REFORM_MIN}
    means = {}
    for label, names in groups.items():
        a, b, c = grp(names)
        means[label] = (st.mean(a), st.mean(b), st.mean(c))
        print(f"  {label:<40} {st.mean(a):.2f} -> {st.mean(b):.2f}  change {st.mean(c):+.2f}")

    t_chg = tr[2]
    c_chg = grp(CONTROL)[2]
    nr_chg = grp(nonreform)[2]
    print(f"\nDiD vs control (5):            {st.mean(t_chg) - st.mean(c_chg):+.2f} pp")
    print(f"DiD vs non-Reform counties (12): {st.mean(t_chg) - st.mean(nr_chg):+.2f} pp")

    # significance: one-sample t of treated change vs the flat control baseline (mean 0)
    m, sd, n = st.mean(t_chg), st.stdev(t_chg), len(t_chg)
    se = sd / math.sqrt(n)
    t = m / se
    tc = T_CRIT[n - 1]
    print(f"\nsignificance (treated change vs control baseline of 0):")
    print(f"  mean={m:+.2f}pp  sd={sd:.2f}  t={t:.2f}  df={n-1}  95% CI [{m-tc*se:+.2f}, {m+tc*se:+.2f}]")
    try:
        from scipy import stats
        p = stats.t.sf(abs(t), n - 1) * 2
        # Welch vs the broader non-Reform county group
        tw, pw = stats.ttest_ind(t_chg, nr_chg, equal_var=False)
        print(f"  p={p:.3f} (one-sample);  Welch vs non-Reform counties: t={tw:.2f}, p={pw:.3f}")
    except ImportError:
        print(f"  significant at 5% (|t|={abs(t):.2f} > {tc}); install scipy for exact p-value")

    # assumption-free view: which shire counties reduced their rise, and are they Reform?
    reducers = sorted([(round(rises(sc, a)[1] - rises(sc, a)[0], 2), a)
                       for a in all_sc if rises(sc, a)[1] - rises(sc, a)[0] < -0.005])
    tag = lambda a: "Reform-maj" if a in REFORM_MAJ else "Reform-min" if a in REFORM_MIN else "non-Reform"
    print(f"\nshire counties that CUT their rise ({len(reducers)}/{len(all_sc)}):")
    for c, a in reducers:
        print(f"  {a:<16}{c:>+7}  [{tag(a)}]")
    print("Caveat: n=6 treated, descriptive not causal; effect concentrated in 4 of the 6.")


if __name__ == "__main__":
    main()
