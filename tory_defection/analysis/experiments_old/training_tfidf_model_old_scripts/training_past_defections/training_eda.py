"""
Training Data Exploratory Data Analysis
========================================

Analyzes the training dataset to compare defectors vs non-defectors
across speech features and demographic features.

Performs:
1. Descriptive statistics
2. Univariate analysis (t-tests, effect sizes)
3. Correlation analysis
4. Preliminary interaction effect identification
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
    print(f"  Loaded {len(df)} MPs")
    print(f"  Defectors: {df['defected'].sum()}")
    print(f"  Non-defectors: {(df['defected'] == 0).sum()}")

    return df


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_statistics(df):
    """Calculate descriptive statistics for defectors vs non-defectors."""

    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    defectors = df[df['defected'] == 1]
    non_defectors = df[df['defected'] == 0]

    # Speech features
    speech_features = [
        'reform_alignment_raw',
        'reform_alignment_normalized',
        'radicalization_slope',
        'extremism_percentile',
        'immigration_proportion',
        'total_speeches'
    ]

    # Demographic features (excluding age since it's unavailable)
    demographic_features = [
        'estimated_years_as_mp',
        'backbench_years',
        'ever_minister',
        'total_minister_years',
        'highest_ministerial_rank',
        'years_since_last_ministerial_role',
        'cabinet_level',
        'career_stagnation_score'
    ]

    results = []

    for feature in speech_features + demographic_features:
        if feature not in df.columns:
            continue

        defector_mean = defectors[feature].mean()
        defector_std = defectors[feature].std()
        non_defector_mean = non_defectors[feature].mean()
        non_defector_std = non_defectors[feature].std()
        difference = defector_mean - non_defector_mean

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(defectors) - 1) * defector_std**2 +
             (len(non_defectors) - 1) * non_defector_std**2) /
            (len(defectors) + len(non_defectors) - 2)
        )
        cohens_d = difference / pooled_std if pooled_std > 0 else 0

        # T-test
        if defectors[feature].std() > 0 and non_defectors[feature].std() > 0:
            t_stat, p_value = stats.ttest_ind(defectors[feature], non_defectors[feature])
        else:
            t_stat, p_value = 0, 1.0

        results.append({
            'feature': feature,
            'defector_mean': defector_mean,
            'defector_std': defector_std,
            'non_defector_mean': non_defector_mean,
            'non_defector_std': non_defector_std,
            'difference': difference,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value
        })

    results_df = pd.DataFrame(results)

    print("\nSpeech Features:")
    print("-" * 80)
    for _, row in results_df[results_df['feature'].isin(speech_features)].iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:<35}")
        print(f"  Defectors: {row['defector_mean']:.4f} +/- {row['defector_std']:.4f}")
        print(f"  Non-defectors: {row['non_defector_mean']:.4f} +/- {row['non_defector_std']:.4f}")
        print(f"  Difference: {row['difference']:+.4f} | Cohen's d: {row['cohens_d']:.3f} | p: {row['p_value']:.4f} {sig}")
        print()

    print("\nDemographic Features:")
    print("-" * 80)
    for _, row in results_df[results_df['feature'].isin(demographic_features)].iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:<35}")
        print(f"  Defectors: {row['defector_mean']:.4f} +/- {row['defector_std']:.4f}")
        print(f"  Non-defectors: {row['non_defector_mean']:.4f} +/- {row['non_defector_std']:.4f}")
        print(f"  Difference: {row['difference']:+.4f} | Cohen's d: {row['cohens_d']:.3f} | p: {row['p_value']:.4f} {sig}")
        print()

    # Save to CSV
    output_path = ANALYSIS_OUTPUT / "training_eda_univariate.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved univariate analysis to: {output_path}")

    return results_df


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(df):
    """Analyze correlations between features and defection."""

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    features = [
        'reform_alignment_raw',
        'reform_alignment_normalized',
        'radicalization_slope',
        'extremism_percentile',
        'immigration_proportion',
        'total_speeches',
        'estimated_years_as_mp',
        'backbench_years',
        'ever_minister',
        'total_minister_years',
        'highest_ministerial_rank',
        'years_since_last_ministerial_role',
        'cabinet_level',
        'career_stagnation_score',
        'defected'
    ]

    available_features = [f for f in features if f in df.columns]
    corr_df = df[available_features].corr()

    print("\nCorrelations with defection:")
    defection_corr = corr_df['defected'].drop('defected').sort_values(ascending=False)

    for feature, corr in defection_corr.items():
        print(f"  {feature:<40} {corr:+.4f}")

    # Save full correlation matrix
    output_path = ANALYSIS_OUTPUT / "training_eda_correlation.csv"
    corr_df.to_csv(output_path)
    print(f"\nSaved correlation matrix to: {output_path}")

    return corr_df


# =============================================================================
# INTERACTION EFFECTS (PRELIMINARY)
# =============================================================================

def preliminary_interaction_analysis(df):
    """Identify preliminary interaction effects."""

    print("\n" + "=" * 80)
    print("PRELIMINARY INTERACTION EFFECTS")
    print("=" * 80)

    # Interaction 1: Career stagnation × ministerial experience
    print("\n1. Career Stagnation × Ministerial Experience")
    print("-" * 80)

    # Create groups
    df['has_career_stagnation'] = df['career_stagnation_score'] > 0.2
    df['was_minister'] = df['ever_minister'] > 0

    groups = df.groupby(['was_minister', 'has_career_stagnation'])['defected'].agg(['mean', 'count'])
    print(groups)

    # Interaction 2: Reform alignment × ministerial experience
    print("\n2. Reform Alignment × Ministerial Experience")
    print("-" * 80)

    df['high_reform'] = df['reform_alignment_raw'] > df['reform_alignment_raw'].median()

    groups = df.groupby(['was_minister', 'high_reform'])['defected'].agg(['mean', 'count'])
    print(groups)

    # Interaction 3: Backbench years × ministerial experience
    print("\n3. Backbench Years × Ministerial Experience")
    print("-" * 80)

    df['long_backbench'] = df['backbench_years'] > df['backbench_years'].median()

    groups = df.groupby(['was_minister', 'long_backbench'])['defected'].agg(['mean', 'count'])
    print(groups)


# =============================================================================
# TOP RISK MPS
# =============================================================================

def analyze_top_risk_mps(df):
    """Analyze MPs with highest defection risk profile."""

    print("\n" + "=" * 80)
    print("TOP RISK MPs (Highest Reform Alignment)")
    print("=" * 80)

    # Top 20 by raw Reform alignment
    top_raw = df.nlargest(20, 'reform_alignment_raw')[
        ['speaker_name', 'reform_alignment_raw', 'reform_alignment_normalized',
         'ever_minister', 'career_stagnation_score', 'defected']
    ]

    print("\nTop 20 by Raw Reform Alignment:")
    for _, row in top_raw.iterrows():
        defected_label = "DEFECTED" if row['defected'] else ""
        minister_label = "Minister" if row['ever_minister'] else "Backbencher"
        print(f"  {row['speaker_name']:<30} | Raw: {row['reform_alignment_raw']:.3f} | "
              f"Norm: {row['reform_alignment_normalized']:.3f} | {minister_label:<12} | {defected_label}")

    # Top 20 by normalized Reform alignment
    top_norm = df.nlargest(20, 'reform_alignment_normalized')[
        ['speaker_name', 'reform_alignment_raw', 'reform_alignment_normalized',
         'ever_minister', 'career_stagnation_score', 'defected']
    ]

    print("\n" + "=" * 80)
    print("TOP RISK MPs (Highest Normalized Reform Alignment)")
    print("=" * 80)

    for _, row in top_norm.iterrows():
        defected_label = "DEFECTED" if row['defected'] else ""
        minister_label = "Minister" if row['ever_minister'] else "Backbencher"
        print(f"  {row['speaker_name']:<30} | Raw: {row['reform_alignment_raw']:.3f} | "
              f"Norm: {row['reform_alignment_normalized']:.3f} | {minister_label:<12} | {defected_label}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("TRAINING DATA EXPLORATORY ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    df = load_training_data()

    # Descriptive statistics
    univariate_results = descriptive_statistics(df)

    # Correlation analysis
    corr_matrix = correlation_analysis(df)

    # Interaction effects
    preliminary_interaction_analysis(df)

    # Top risk MPs
    analyze_top_risk_mps(df)

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey Findings:")

    # Most significant features
    significant_features = univariate_results[univariate_results['p_value'] < 0.05].sort_values('cohens_d', ascending=False)

    if len(significant_features) > 0:
        print("\nStatistically significant features (p < 0.05):")
        for _, row in significant_features.iterrows():
            print(f"  - {row['feature']}: Cohen's d = {row['cohens_d']:.3f}, p = {row['p_value']:.4f}")
    else:
        print("\nNo statistically significant features found (likely due to small sample size).")

    print("\nNote: With only 19 defectors, statistical power is limited.")
    print("Effect sizes (Cohen's d) are more informative than p-values.")
    print("  Small effect: |d| = 0.2")
    print("  Medium effect: |d| = 0.5")
    print("  Large effect: |d| = 0.8")

    return df, univariate_results, corr_matrix


if __name__ == "__main__":
    results = main()
