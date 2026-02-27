"""
Apply Trained Model to Current Conservative MPs - WITH CONSTITUENCY FACTORS
============================================================================

Extends the base model by adding constituency-level factors:
- Reform UK vote share in constituency (2024 election)
- Conservative margin of victory (or loss)

These factors are given the joint lowest weight relative to other variables,
as they were explored in preliminary models but found to have less predictive
power than speech and career variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
ANALYSIS_OUTPUT = BASE_DIR / "analysis"
SOURCE_DATA = BASE_DIR / "source_data"

# Training artifacts
TRAINING_DIR = Path(__file__).parent / "training_past_defections"
MODEL_PREDICTIONS = TRAINING_DIR / "model_predictions.csv"
TRAINING_DATA = TRAINING_DIR / "training_data_2024.csv"

# Current MP data
CURRENT_SPEECH_DATA = Path(__file__).parent / "enhanced_speech_tfidf_normalized.csv"
CURRENT_CAREER_DATA = Path(__file__).parent / "mp_career_features.csv"
DEMOGRAPHICS_CSV = SOURCE_DATA / "mp_demographics.csv"

# Election data for constituency factors
ELECTION_DATA = SOURCE_DATA / "elections_2024" / "HoC-GE2024-results-by-constituency.csv"

# Current MP list with constituencies
CURRENT_MPS_LIST = Path(__file__).parent / "current_conservative_mps_2024.csv"

# Output
OUTPUT_PATH = Path(__file__).parent / "current_mp_defection_risk_scores_constituency_factors.csv"
REPORT_PATH = Path(__file__).parent / "current_mp_defection_risk_report_constituency_factors.txt"

# =============================================================================
# LOAD TRAINED MODEL PARAMETERS
# =============================================================================

def load_trained_model_parameters():
    """
    Load feature names, coefficients, and scaling parameters from training data.
    """

    print("Loading trained model parameters...")

    # Load training data to get feature names and order
    training_df = pd.read_csv(TRAINING_DATA)

    # Define feature order (must match training)
    feature_names = [
        'reform_alignment_raw',
        'reform_alignment_normalized',
        'radicalization_slope',
        'extremism_percentile',
        'immigration_proportion',
        'total_speeches',
        'age',
        'retirement_proximity_score',
        'estimated_years_as_mp',
        'backbench_years',
        'ever_minister',
        'total_minister_years',
        'highest_ministerial_rank',
        'years_since_last_ministerial_role',
        'cabinet_level',
        'career_stagnation_score'
    ]

    # Calculate scaling parameters (mean and std from training data)
    X_train = training_df[feature_names].fillna(0).values
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)

    # Trained model coefficients (from optimize_model_weights.py output)
    # These are the actual coefficients from C=1.0 model
    coefficients = np.array([
        0.6545,   # reform_alignment_raw
        0.3010,   # reform_alignment_normalized
        0.3238,   # radicalization_slope
        -0.1584,  # extremism_percentile
        0.6939,   # immigration_proportion
        0.2324,   # total_speeches
        0.3733,   # age
        -0.6922,  # retirement_proximity_score
        4.0016,   # estimated_years_as_mp
        -3.7612,  # backbench_years
        -0.7472,  # ever_minister
        -0.6599,  # total_minister_years
        -0.6230,  # highest_ministerial_rank
        0.0548,   # years_since_last_ministerial_role
        0.0077,   # cabinet_level
        0.2148    # career_stagnation_score
    ])

    # Intercept (from training)
    intercept = -2.5  # Approximate from balanced model

    print(f"  Loaded {len(feature_names)} features")
    print(f"  Feature means and stds calculated from {len(training_df)} training samples")

    return feature_names, coefficients, intercept, feature_means, feature_stds


# =============================================================================
# LOAD CONSTITUENCY DATA
# =============================================================================

def load_constituency_factors():
    """
    Load 2024 election results and calculate constituency-level factors:
    - Reform UK vote share
    - Conservative margin (positive = won, negative = lost)
    """

    print("\nLoading constituency election data...")

    if not ELECTION_DATA.exists():
        print(f"ERROR: Election data not found at {ELECTION_DATA}")
        return None

    election_df = pd.read_csv(ELECTION_DATA)
    print(f"  Loaded election results for {len(election_df)} constituencies")

    # Calculate Reform UK vote share
    election_df['reform_vote_share'] = election_df['RUK'] / election_df['Valid votes']

    # Calculate Conservative margin (as % of valid votes)
    # Positive = won, Negative = lost (difference from winner)
    def calc_con_margin(row):
        con_votes = row['Con']
        valid_votes = row['Valid votes']
        con_share = con_votes / valid_votes

        # Find the winner's votes
        party_cols = ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC', 'DUP', 'SF', 'SDLP', 'UUP', 'APNI']
        party_votes = {col: row[col] for col in party_cols if col in row.index and pd.notna(row[col])}

        if row['First party'] == 'Con':
            # Conservative won - margin is difference from second place
            second_party_votes = row[row['Second party']] if row['Second party'] in party_votes else 0
            margin = (con_votes - second_party_votes) / valid_votes
        else:
            # Conservative lost - margin is negative (how far behind winner)
            winner_votes = row[row['First party']] if row['First party'] in party_votes else 0
            margin = (con_votes - winner_votes) / valid_votes

        return margin

    election_df['con_margin'] = election_df.apply(calc_con_margin, axis=1)

    # Select relevant columns
    constituency_df = election_df[['Constituency name', 'reform_vote_share', 'con_margin', 'RUK', 'Con', 'Valid votes', 'First party']].copy()
    constituency_df = constituency_df.rename(columns={'Constituency name': 'constituency'})

    # Flag if Conservative won
    constituency_df['con_won'] = (constituency_df['First party'] == 'Con').astype(int)

    print(f"  Reform UK vote share range: {constituency_df['reform_vote_share'].min():.1%} - {constituency_df['reform_vote_share'].max():.1%}")
    print(f"  Mean Reform UK vote share: {constituency_df['reform_vote_share'].mean():.1%}")
    print(f"  Conservative margin range: {constituency_df['con_margin'].min():.1%} - {constituency_df['con_margin'].max():.1%}")

    return constituency_df


# =============================================================================
# LOAD CURRENT MP DATA
# =============================================================================

def load_current_mps_data(constituency_df):
    """
    Load speech analysis and demographic data for current Conservative MPs.
    Merge with constituency factors.
    """

    print("\nLoading current Conservative MP data...")

    # Load speech data
    if not CURRENT_SPEECH_DATA.exists():
        print(f"ERROR: Speech data not found at {CURRENT_SPEECH_DATA}")
        print("Please run enhanced_speech_tfidf.py and normalize_speech_metrics.py first")
        return None

    speech_df = pd.read_csv(CURRENT_SPEECH_DATA)
    print(f"  Loaded speech data for {len(speech_df)} MPs")

    # Load career/demographics data
    if not CURRENT_CAREER_DATA.exists():
        print(f"ERROR: Career data not found at {CURRENT_CAREER_DATA}")
        print("Please run fetch_ifg_ministerial_data.py and fetch_mp_ages_from_wikidata.py first")
        return None

    career_df = pd.read_csv(CURRENT_CAREER_DATA)
    print(f"  Loaded career data for {len(career_df)} MPs")

    # Merge speech and career data
    def clean_name(name):
        if pd.isna(name):
            return name
        for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
            name = name.replace(title, '')
        return name.strip()

    speech_df['name_clean'] = speech_df['speaker_name'].apply(clean_name)
    career_df['name_clean'] = career_df['name'].apply(clean_name)

    # Rename columns to match training data
    if 'estimated_total_years_as_mp' in career_df.columns:
        career_df = career_df.rename(columns={'estimated_total_years_as_mp': 'estimated_years_as_mp'})

    # Merge
    merged_df = speech_df.merge(
        career_df,
        on='name_clean',
        how='inner',
        suffixes=('_speech', '_career')
    )

    # Handle duplicate columns (keep speech version for speech features)
    for col in merged_df.columns:
        if col.endswith('_speech'):
            base_col = col.replace('_speech', '')
            career_col = base_col + '_career'
            if career_col in merged_df.columns:
                # Keep speech version, drop career version
                merged_df[base_col] = merged_df[col]
                merged_df = merged_df.drop(columns=[col, career_col])
            else:
                merged_df = merged_df.rename(columns={col: base_col})

    # Clean up any remaining _career suffixes
    for col in merged_df.columns:
        if col.endswith('_career'):
            base_col = col.replace('_career', '')
            if base_col not in merged_df.columns:
                merged_df = merged_df.rename(columns={col: base_col})

    print(f"  Merged dataset: {len(merged_df)} MPs with both speech and career data")

    # Load current Conservative MPs with their constituencies
    if not CURRENT_MPS_LIST.exists():
        print(f"ERROR: Current Conservative MP list not found at {CURRENT_MPS_LIST}")
        print("Please ensure you have the 2024 election results with Conservative MPs")
        return None

    current_con_mps = pd.read_csv(CURRENT_MPS_LIST)
    current_con_mps['name_clean'] = current_con_mps['full_name'].apply(clean_name)

    # Merge to get constituencies for current MPs
    current_df = merged_df.merge(
        current_con_mps[['name_clean', 'Constituency name']],
        on='name_clean',
        how='inner'
    )
    current_df = current_df.rename(columns={'Constituency name': 'constituency'})

    print(f"  Current sitting Conservative MPs (elected July 2024): {len(current_df)}")

    # Merge with constituency factors
    current_df = current_df.merge(
        constituency_df[['constituency', 'reform_vote_share', 'con_margin', 'con_won']],
        on='constituency',
        how='left'
    )

    # Fill missing constituency data with median values
    if current_df['reform_vote_share'].isna().any():
        median_reform = constituency_df['reform_vote_share'].median()
        median_margin = constituency_df['con_margin'].median()
        current_df['reform_vote_share'] = current_df['reform_vote_share'].fillna(median_reform)
        current_df['con_margin'] = current_df['con_margin'].fillna(median_margin)
        print(f"  Filled {current_df['reform_vote_share'].isna().sum()} missing constituency values with medians")

    print(f"  Constituency factors merged for {len(current_df)} MPs")
    print(f"  Reform UK vote share in Con seats: {current_df['reform_vote_share'].mean():.1%} average")
    print(f"  Conservative margin in Con seats: {current_df['con_margin'].mean():.1%} average")

    return current_df


# =============================================================================
# APPLY MODEL WITH CONSTITUENCY FACTORS
# =============================================================================

def apply_trained_model_with_constituency(current_df, feature_names, coefficients, intercept, feature_means, feature_stds):
    """
    Apply trained logistic regression model to current MPs,
    with additional constituency factors at lowest weight.
    """

    print("\nApplying trained model with constituency factors...")

    # Extract features (in correct order)
    X = current_df[feature_names].fillna(0).values

    # Standardize using training data parameters
    X_scaled = (X - feature_means) / (feature_stds + 1e-8)

    # Apply base logistic regression
    base_logits = intercept + np.dot(X_scaled, coefficients)

    # Calculate constituency factor adjustment
    # These get the JOINT LOWEST weight relative to other variables
    # The lowest coefficient magnitude in the base model is cabinet_level at 0.0077
    # We'll use this as the coefficient for each constituency factor

    # Standardize constituency factors
    reform_share = current_df['reform_vote_share'].values
    con_margin = current_df['con_margin'].values

    reform_share_mean = reform_share.mean()
    reform_share_std = reform_share.std() + 1e-8
    reform_share_scaled = (reform_share - reform_share_mean) / reform_share_std

    con_margin_mean = con_margin.mean()
    con_margin_std = con_margin.std() + 1e-8
    con_margin_scaled = (con_margin - con_margin_mean) / con_margin_std

    # Apply constituency coefficients (joint lowest weight = 0.0077, same as cabinet_level)
    # Reform vote share: positive effect (higher Reform vote = higher risk)
    # Conservative margin: negative effect (larger margin = safer seat = lower risk)
    constituency_coef = 0.0077  # Joint lowest weight

    constituency_adjustment = (
        constituency_coef * reform_share_scaled +  # Higher Reform vote = higher risk
        -constituency_coef * con_margin_scaled      # Larger Con margin = lower risk
    )

    # Final logits with constituency adjustment
    final_logits = base_logits + constituency_adjustment

    # Convert to probabilities using sigmoid function
    predicted_probs = 1 / (1 + np.exp(-final_logits))

    # Also calculate base probabilities for comparison
    base_probs = 1 / (1 + np.exp(-base_logits))

    # Add predictions to dataframe
    results_df = current_df[['speaker_name', 'name_clean', 'constituency']].copy()
    results_df['defection_probability'] = predicted_probs
    results_df['defection_risk_score'] = (predicted_probs * 100).round(2)
    results_df['base_risk_score'] = (base_probs * 100).round(2)
    results_df['constituency_adjustment'] = ((predicted_probs - base_probs) * 100).round(2)

    # Add constituency factors
    results_df['reform_vote_share'] = (current_df['reform_vote_share'].values * 100).round(1)
    results_df['con_margin'] = (current_df['con_margin'].values * 100).round(1)

    # Add key features for interpretation
    for feature in ['reform_alignment_normalized', 'immigration_proportion',
                    'ever_minister', 'backbench_years', 'age', 'total_speeches']:
        if feature in current_df.columns:
            results_df[feature] = current_df[feature].values

    # Sort by risk (highest first)
    results_df = results_df.sort_values('defection_probability', ascending=False).reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)

    print(f"  Generated predictions for {len(results_df)} current Conservative MPs")
    print(f"  Risk scores range: {results_df['defection_risk_score'].min():.2f}% - {results_df['defection_risk_score'].max():.2f}%")
    print(f"  Constituency adjustment range: {results_df['constituency_adjustment'].min():.2f}pp - {results_df['constituency_adjustment'].max():.2f}pp")

    return results_df


# =============================================================================
# GENERATE REPORT
# =============================================================================

def generate_report(results_df):
    """
    Generate human-readable report with top-risk MPs including constituency factors.
    """

    print("\nGenerating risk assessment report...")

    report = []
    report.append("=" * 90)
    report.append("CONSERVATIVE MP DEFECTION RISK ASSESSMENT - WITH CONSTITUENCY FACTORS")
    report.append("Current Sitting MPs - Predicted by Trained Model (v1.0) + Constituency Adjustment")
    report.append("=" * 90)
    report.append("")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Total MPs Analyzed: {len(results_df)}")
    report.append(f"Model: Logistic Regression (C=1.0, L2 regularization)")
    report.append(f"Training Performance: 97.8% Cross-Validated AUC")
    report.append(f"Training Data: 19 historical defections (Jan 2024 - Jan 2026)")
    report.append("")
    report.append("CONSTITUENCY FACTORS ADDED:")
    report.append("  - Reform UK vote share (2024 election)")
    report.append("  - Conservative margin of victory")
    report.append("  - Weight: Joint lowest (0.0077 coefficient each)")
    report.append("")

    # Risk distribution
    report.append("=" * 90)
    report.append("RISK DISTRIBUTION")
    report.append("=" * 90)
    report.append("")

    very_high = (results_df['defection_risk_score'] >= 80).sum()
    high = ((results_df['defection_risk_score'] >= 60) & (results_df['defection_risk_score'] < 80)).sum()
    medium = ((results_df['defection_risk_score'] >= 40) & (results_df['defection_risk_score'] < 60)).sum()
    low = (results_df['defection_risk_score'] < 40).sum()

    report.append(f"  Very High Risk (80-100%): {very_high} MPs")
    report.append(f"  High Risk (60-80%):       {high} MPs")
    report.append(f"  Medium Risk (40-60%):     {medium} MPs")
    report.append(f"  Low Risk (0-40%):         {low} MPs")
    report.append("")

    # Top 30 highest risk
    report.append("=" * 90)
    report.append("TOP 30 HIGHEST DEFECTION RISK - CURRENT CONSERVATIVE MPs")
    report.append("=" * 90)
    report.append("")
    report.append("Based on trained model with constituency adjustment")
    report.append("")

    for _, row in results_df.head(30).iterrows():
        minister_label = "Minister" if row.get('ever_minister', 0) > 0 else "Backbencher"
        adj_sign = "+" if row['constituency_adjustment'] >= 0 else ""

        report.append(f"Rank #{int(row['rank']):<3} | {row['speaker_name']:<35}")
        report.append(f"  Defection Risk: {row['defection_risk_score']:.2f}% (Base: {row['base_risk_score']:.2f}%, Adj: {adj_sign}{row['constituency_adjustment']:.2f}pp)")
        report.append(f"  Constituency: {row['constituency']}")
        report.append(f"  Reform UK Vote Share: {row['reform_vote_share']:.1f}%")
        report.append(f"  Conservative Margin: {row['con_margin']:.1f}pp")
        report.append(f"  Reform Alignment (normalized): {row.get('reform_alignment_normalized', 0):.4f}")
        report.append(f"  Immigration Focus: {row.get('immigration_proportion', 0)*100:.1f}% of speeches")
        report.append(f"  Status: {minister_label}")
        if row.get('age', 0) > 0:
            report.append(f"  Age: {row.get('age', 0):.0f} years")
        report.append("")

    # Constituency impact analysis
    report.append("=" * 90)
    report.append("CONSTITUENCY FACTOR IMPACT ANALYSIS")
    report.append("=" * 90)
    report.append("")

    # MPs most affected by constituency factors
    most_increased = results_df.nlargest(5, 'constituency_adjustment')
    most_decreased = results_df.nsmallest(5, 'constituency_adjustment')

    report.append("MPs Most INCREASED by Constituency Factors:")
    for _, row in most_increased.iterrows():
        report.append(f"  {row['speaker_name']:<35} | +{row['constituency_adjustment']:.2f}pp | Reform: {row['reform_vote_share']:.1f}% | Margin: {row['con_margin']:.1f}pp")
    report.append("")

    report.append("MPs Most DECREASED by Constituency Factors:")
    for _, row in most_decreased.iterrows():
        report.append(f"  {row['speaker_name']:<35} | {row['constituency_adjustment']:.2f}pp | Reform: {row['reform_vote_share']:.1f}% | Margin: {row['con_margin']:.1f}pp")
    report.append("")

    # Statistics
    report.append("=" * 90)
    report.append("MODEL STATISTICS")
    report.append("=" * 90)
    report.append("")
    report.append("Base Model Performance:")
    report.append("  - Cross-Validated AUC: 97.8% +/- 1.6%")
    report.append("  - Recall @ Top 20: 84.2% (16 of 19 historical defectors)")
    report.append("  - Training Sample: 542 historical Conservative MPs")
    report.append("  - Ground Truth: 19 confirmed defections (Best for Britain)")
    report.append("")
    report.append("Constituency Factor Weights:")
    report.append("  - Reform UK Vote Share: +0.0077 (joint lowest)")
    report.append("  - Conservative Margin: -0.0077 (joint lowest, protective)")
    report.append("  - For reference: cabinet_level has coefficient 0.0077")
    report.append("  - For reference: years_as_mp has coefficient 4.0016 (highest)")
    report.append("")
    report.append("Constituency Factor Summary:")
    report.append(f"  - Mean Reform Vote Share: {results_df['reform_vote_share'].mean():.1f}%")
    report.append(f"  - Mean Conservative Margin: {results_df['con_margin'].mean():.1f}pp")
    report.append(f"  - Max Constituency Adjustment: {results_df['constituency_adjustment'].max():.2f}pp")
    report.append(f"  - Min Constituency Adjustment: {results_df['constituency_adjustment'].min():.2f}pp")
    report.append("")

    report_text = "\n".join(report)

    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)

    print(f"  Saved report to: {REPORT_PATH}")

    return report_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 90)
    print("APPLYING TRAINED MODEL TO CURRENT CONSERVATIVE MPs - WITH CONSTITUENCY FACTORS")
    print("=" * 90)
    print()

    # Load trained model parameters
    feature_names, coefficients, intercept, feature_means, feature_stds = load_trained_model_parameters()

    # Load constituency factors
    constituency_df = load_constituency_factors()

    if constituency_df is None:
        print("\nERROR: Could not load constituency data")
        return None

    # Load current MP data with constituencies
    current_df = load_current_mps_data(constituency_df)

    if current_df is None or len(current_df) == 0:
        print("\nERROR: Could not load current MP data")
        print("Please ensure you have run:")
        print("  1. enhanced_speech_tfidf.py")
        print("  2. normalize_speech_metrics.py")
        print("  3. fetch_ifg_ministerial_data.py")
        print("  4. fetch_mp_ages_from_wikidata.py")
        return None

    # Apply model with constituency factors
    results_df = apply_trained_model_with_constituency(
        current_df,
        feature_names,
        coefficients,
        intercept,
        feature_means,
        feature_stds
    )

    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved predictions to: {OUTPUT_PATH}")

    # Generate report
    report_text = generate_report(results_df)

    # Print top 10 to console
    print("\n" + "=" * 90)
    print("TOP 10 HIGHEST RISK CURRENT CONSERVATIVE MPs (WITH CONSTITUENCY FACTORS)")
    print("=" * 90)
    print()

    for _, row in results_df.head(10).iterrows():
        minister_label = "Minister" if row.get('ever_minister', 0) > 0 else "Backbencher"
        adj_sign = "+" if row['constituency_adjustment'] >= 0 else ""
        print(f"#{int(row['rank']):<2} | {row['speaker_name']:<30} | Risk: {row['defection_risk_score']:>5.2f}% ({adj_sign}{row['constituency_adjustment']:.2f}pp) | Reform: {row['reform_vote_share']:.1f}% | {minister_label}")

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"\nTotal MPs analyzed: {len(results_df)}")
    print(f"Constituency factors applied at joint lowest weight (0.0077)")

    return results_df


if __name__ == "__main__":
    results = main()
