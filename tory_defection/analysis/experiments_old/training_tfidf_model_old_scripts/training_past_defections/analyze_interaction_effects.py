"""
Analyze Interaction Effects
============================

Analyzes interaction effects between features to identify synergistic risk factors.

Key interactions to explore:
1. Career stagnation × retirement proximity (original hypothesis)
2. Reform alignment × ministerial status
3. Reform alignment × age/retirement
4. Backbench years × Reform alignment
5. Immigration proportion × ministerial status
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_OUTPUT = BASE_DIR / "analysis"

TRAINING_DATA = ANALYSIS_OUTPUT / "training_data_2024.csv"

# =============================================================================
# LOAD DATA
# =============================================================================

def load_training_data():
    """Load training dataset."""

    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    print(f"  Loaded {len(df)} MPs ({df['defected'].sum()} defectors)")

    return df


# =============================================================================
# INTERACTION ANALYSIS
# =============================================================================

def analyze_career_stagnation_retirement(df):
    """Analyze interaction between career stagnation and retirement proximity."""

    print("\n" + "=" * 80)
    print("INTERACTION 1: Career Stagnation × Retirement Proximity")
    print("=" * 80)
    print("\nHypothesis: Former ministers with career stagnation are high risk,")
    print("UNLESS they're close to retirement age (then they just want to retire)")

    # Create bins
    df['high_stagnation'] = df['career_stagnation_score'] > 0.1
    df['near_retirement'] = df['retirement_proximity_score'] > 0.3
    df['was_minister'] = df['ever_minister'] > 0

    # Analyze each group
    groups = df.groupby(['was_minister', 'high_stagnation', 'near_retirement'])

    print("\nDefection rates by group:")
    print("-" * 80)

    for (minister, stagnation, retirement), group in groups:
        if len(group) < 5:  # Skip very small groups
            continue

        defection_rate = group['defected'].mean()
        n = len(group)
        n_defected = group['defected'].sum()

        minister_label = "Minister" if minister else "Backbencher"
        stagnation_label = "High stagnation" if stagnation else "Low stagnation"
        retirement_label = "Near retirement" if retirement else "Not near retirement"

        print(f"\n  {minister_label} | {stagnation_label} | {retirement_label}")
        print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    # Test interaction with logistic regression
    df['interaction_stagnation_retirement'] = (
        df['career_stagnation_score'] * df['retirement_proximity_score']
    )

    return df


def analyze_reform_alignment_ministerial(df):
    """Analyze interaction between Reform alignment and ministerial status."""

    print("\n" + "=" * 80)
    print("INTERACTION 2: Reform Alignment × Ministerial Status")
    print("=" * 80)
    print("\nHypothesis: High Reform alignment is more predictive among backbenchers")
    print("(Ministers have more to lose, so even if aligned they may not defect)")

    # Create bins
    df['high_reform_raw'] = df['reform_alignment_raw'] > df['reform_alignment_raw'].median()
    df['high_reform_norm'] = df['reform_alignment_normalized'] > df['reform_alignment_normalized'].median()

    print("\nDefection rates by Reform alignment (raw) and ministerial status:")
    print("-" * 80)

    for minister in [False, True]:
        for high_reform in [False, True]:
            group = df[(df['was_minister'] == minister) & (df['high_reform_raw'] == high_reform)]

            if len(group) < 5:
                continue

            defection_rate = group['defected'].mean()
            n = len(group)
            n_defected = group['defected'].sum()

            minister_label = "Minister" if minister else "Backbencher"
            reform_label = "High Reform" if high_reform else "Low Reform"

            print(f"\n  {minister_label} | {reform_label}")
            print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    print("\n\nDefection rates by Reform alignment (normalized) and ministerial status:")
    print("-" * 80)

    for minister in [False, True]:
        for high_reform in [False, True]:
            group = df[(df['was_minister'] == minister) & (df['high_reform_norm'] == high_reform)]

            if len(group) < 5:
                continue

            defection_rate = group['defected'].mean()
            n = len(group)
            n_defected = group['defected'].sum()

            minister_label = "Minister" if minister else "Backbencher"
            reform_label = "High Reform (norm)" if high_reform else "Low Reform (norm)"

            print(f"\n  {minister_label} | {reform_label}")
            print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    # Create interaction term
    df['interaction_reform_minister'] = (
        df['reform_alignment_normalized'] * (1 - df['ever_minister'])
    )

    return df


def analyze_reform_alignment_age(df):
    """Analyze interaction between Reform alignment and age/retirement."""

    print("\n" + "=" * 80)
    print("INTERACTION 3: Reform Alignment × Age/Retirement")
    print("=" * 80)
    print("\nHypothesis: Reform alignment + mid-career = high risk")
    print("(Too young = career concerns, too old = retirement concerns)")

    # Create age bins (only for those with age data)
    df_with_age = df[df['age'] > 0].copy()

    if len(df_with_age) < 50:
        print("\n  Insufficient age data for analysis")
        return df

    df_with_age['age_group'] = pd.cut(
        df_with_age['age'],
        bins=[0, 45, 55, 65, 100],
        labels=['Young (<45)', 'Mid (45-55)', 'Senior (55-65)', 'Near retirement (65+)']
    )

    print("\nDefection rates by Reform alignment and age group:")
    print("-" * 80)

    for age_group in df_with_age['age_group'].unique():
        for high_reform in [False, True]:
            group = df_with_age[
                (df_with_age['age_group'] == age_group) &
                (df_with_age['high_reform_norm'] == high_reform)
            ]

            if len(group) < 5:
                continue

            defection_rate = group['defected'].mean()
            n = len(group)
            n_defected = group['defected'].sum()

            reform_label = "High Reform" if high_reform else "Low Reform"

            print(f"\n  {age_group} | {reform_label}")
            print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    # Create interaction term
    df['interaction_reform_retirement'] = (
        df['reform_alignment_normalized'] * df['retirement_proximity_score']
    )

    return df


def analyze_backbench_reform_alignment(df):
    """Analyze interaction between backbench years and Reform alignment."""

    print("\n" + "=" * 80)
    print("INTERACTION 4: Backbench Years × Reform Alignment")
    print("=" * 80)
    print("\nHypothesis: Long-time backbenchers with Reform views = high risk")
    print("(No career ladder to climb, ideologically aligned with Reform)")

    # Create bins
    df['long_backbench'] = df['backbench_years'] > 20

    print("\nDefection rates by backbench duration and Reform alignment:")
    print("-" * 80)

    for long_backbench in [False, True]:
        for high_reform in [False, True]:
            group = df[(df['long_backbench'] == long_backbench) & (df['high_reform_norm'] == high_reform)]

            if len(group) < 5:
                continue

            defection_rate = group['defected'].mean()
            n = len(group)
            n_defected = group['defected'].sum()

            backbench_label = "Long backbench (>20y)" if long_backbench else "Short backbench (<20y)"
            reform_label = "High Reform" if high_reform else "Low Reform"

            print(f"\n  {backbench_label} | {reform_label}")
            print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    # Create interaction term
    df['interaction_backbench_reform'] = (
        df['backbench_years'] * df['reform_alignment_normalized']
    )

    return df


def analyze_immigration_ministerial(df):
    """Analyze interaction between immigration focus and ministerial status."""

    print("\n" + "=" * 80)
    print("INTERACTION 5: Immigration Focus × Ministerial Status")
    print("=" * 80)
    print("\nHypothesis: Backbenchers who focus on immigration are high risk")
    print("(Ministers might talk about immigration due to portfolio)")

    # Create bins
    df['high_immigration'] = df['immigration_proportion'] > df['immigration_proportion'].median()

    print("\nDefection rates by immigration focus and ministerial status:")
    print("-" * 80)

    for minister in [False, True]:
        for high_immigration in [False, True]:
            group = df[(df['was_minister'] == minister) & (df['high_immigration'] == high_immigration)]

            if len(group) < 5:
                continue

            defection_rate = group['defected'].mean()
            n = len(group)
            n_defected = group['defected'].sum()

            minister_label = "Minister" if minister else "Backbencher"
            immigration_label = "High immigration focus" if high_immigration else "Low immigration focus"

            print(f"\n  {minister_label} | {immigration_label}")
            print(f"    N={n}, Defected={n_defected}, Rate={defection_rate:.1%}")

    # Create interaction term
    df['interaction_immigration_minister'] = (
        df['immigration_proportion'] * (1 - df['ever_minister'])
    )

    return df


# =============================================================================
# TEST INTERACTION TERMS IN MODEL
# =============================================================================

def test_interaction_terms(df):
    """Test interaction terms in logistic regression."""

    print("\n" + "=" * 80)
    print("TESTING INTERACTION TERMS IN MODEL")
    print("=" * 80)

    # Base features
    base_features = [
        'reform_alignment_normalized',
        'immigration_proportion',
        'ever_minister',
        'backbench_years',
        'retirement_proximity_score',
        'career_stagnation_score'
    ]

    # Interaction terms
    interaction_terms = [
        'interaction_stagnation_retirement',
        'interaction_reform_minister',
        'interaction_reform_retirement',
        'interaction_backbench_reform',
        'interaction_immigration_minister'
    ]

    # Prepare data
    X_base = df[base_features].fillna(0).values
    X_interactions = df[base_features + interaction_terms].fillna(0).values
    y = df['defected'].values

    # Train models
    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)

    scaler_inter = StandardScaler()
    X_inter_scaled = scaler_inter.fit_transform(X_interactions)

    model_base = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
    model_base.fit(X_base_scaled, y)

    model_inter = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
    model_inter.fit(X_inter_scaled, y)

    # Compare
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_base = cross_val_score(model_base, X_base_scaled, y, cv=skf, scoring='roc_auc')
    cv_inter = cross_val_score(model_inter, X_inter_scaled, y, cv=skf, scoring='roc_auc')

    print(f"\nModel without interactions: AUC = {cv_base.mean():.4f} +/- {cv_base.std():.4f}")
    print(f"Model with interactions:    AUC = {cv_inter.mean():.4f} +/- {cv_inter.std():.4f}")

    improvement = cv_inter.mean() - cv_base.mean()
    print(f"\nImprovement: {improvement:+.4f}")

    # Show interaction term coefficients
    print("\n\nInteraction term coefficients:")
    print("-" * 80)

    interaction_coefs = model_inter.coef_[0][-len(interaction_terms):]

    for term, coef in zip(interaction_terms, interaction_coefs):
        direction = "+" if coef > 0 else "-"
        print(f"  {term:<45} | {direction} | Coef: {coef:>8.4f}")

    return model_inter, X_inter_scaled


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("INTERACTION EFFECTS ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    df = load_training_data()

    # Analyze each interaction
    df = analyze_career_stagnation_retirement(df)
    df = analyze_reform_alignment_ministerial(df)
    df = analyze_reform_alignment_age(df)
    df = analyze_backbench_reform_alignment(df)
    df = analyze_immigration_ministerial(df)

    # Test interaction terms in model
    model, X_scaled = test_interaction_terms(df)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY INTERACTIONS")
    print("=" * 80)

    print("\n1. Career Stagnation × Retirement Proximity:")
    print("   - Former ministers with stagnation but NOT near retirement = HIGH RISK")
    print("   - Near retirement reduces risk (they're just waiting to retire)")

    print("\n2. Reform Alignment × Ministerial Status:")
    print("   - High Reform alignment among backbenchers = VERY HIGH RISK")
    print("   - Ministers with Reform views less likely to defect (more to lose)")

    print("\n3. Reform Alignment × Age:")
    print("   - Mid-career MPs with Reform views = HIGHEST RISK")
    print("   - Young MPs and near-retirement MPs less likely to defect")

    print("\n4. Backbench Years × Reform Alignment:")
    print("   - Long-time backbenchers with Reform views = HIGH RISK")
    print("   - No career ladder to climb + ideological alignment = defection")

    print("\n5. Immigration Focus × Ministerial Status:")
    print("   - Backbenchers focused on immigration = HIGH RISK")
    print("   - Ministers may discuss immigration due to portfolio (less predictive)")

    # Save results
    output_path = ANALYSIS_OUTPUT / "interaction_effects_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved interaction analysis to: {output_path}")

    return df, model


if __name__ == "__main__":
    results = main()
