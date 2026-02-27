"""
Prepare Training Dataset
=========================

Merges speech features, defection ground truth, and basic demographics
to create the training dataset for model optimization.

Note: Ministerial data unavailable from API, so we'll use the ministerial
rank data from the defection tracker and basic age/demographics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

# This script is in training_past_defections/, data is in parent and grandparent dirs
SCRIPT_DIR = Path(__file__).parent  # training_past_defections/
MODEL_DIR = SCRIPT_DIR.parent  # training_tfidf_model_final_spec/
BASE_DIR = MODEL_DIR.parent.parent  # tory_defection/
SOURCE_DATA = BASE_DIR / "source_data"

SPEECH_NORMALIZED = MODEL_DIR / "enhanced_speech_tfidf_normalized.csv"
DEFECTIONS = SOURCE_DATA / "defection_tracker" / "defections_2024.csv"
TRAINING_POPULATION = MODEL_DIR / "training_population_2019_2024.csv"
CAREER_FEATURES = MODEL_DIR / "mp_career_features.csv"
AGE_DATA = MODEL_DIR / "mp_ages_from_wikidata.csv"
OUTPUT_DIR = SCRIPT_DIR  # Save training data in training_past_defections/

# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_data():
    """Load speech features, defections, training population, career data, and ages."""

    print("Loading data sources...")

    # Speech features (normalized)
    speech_df = pd.read_csv(SPEECH_NORMALIZED)
    print(f"  Loaded {len(speech_df)} MPs with speech analysis")

    # Defections
    defections_df = pd.read_csv(DEFECTIONS)
    print(f"  Loaded {len(defections_df)} defections")

    # Training population (2019-2024 Conservative MPs)
    training_pop_df = pd.read_csv(TRAINING_POPULATION)
    print(f"  Loaded {len(training_pop_df)} MPs in training population (2019-2024)")
    print(f"    {training_pop_df['is_defector'].sum()} marked as defectors")

    # Career features (from IFG ministerial database)
    career_df = pd.read_csv(CAREER_FEATURES)
    print(f"  Loaded {len(career_df)} MPs with career features")
    print(f"    {career_df['ever_minister'].sum()} held ministerial roles")

    # Age data (from Wikidata)
    age_df = pd.read_csv(AGE_DATA)
    print(f"  Loaded {len(age_df)} MPs with age data")

    return speech_df, defections_df, training_pop_df, career_df, age_df


# =============================================================================
# MERGE DATASETS
# =============================================================================

def clean_name_for_matching(name):
    """Clean MP names for matching across datasets."""
    if pd.isna(name):
        return name

    # Remove titles
    for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
        name = name.replace(title, '')

    return name.strip()


def merge_training_data(speech_df, defections_df, training_pop_df, career_df, age_df):
    """Merge all data sources into training dataset."""

    print("\nMerging datasets...")

    # Clean names for matching
    speech_df['name_clean'] = speech_df['speaker_name'].apply(clean_name_for_matching)
    defections_df['name_clean'] = defections_df['name'].apply(clean_name_for_matching)
    training_pop_df['name_clean'] = training_pop_df['name'].apply(clean_name_for_matching)
    age_df['name_clean'] = age_df['name'].apply(clean_name_for_matching)

    # Start with training population (2019-2024 Conservative MPs)
    # This is the authoritative list of who to include
    training_df = training_pop_df.copy()
    print(f"  Starting with {len(training_df)} MPs from training population")

    # Merge with speech features
    training_df = training_df.merge(
        speech_df,
        on='name_clean',
        how='left'
    )

    # Count how many have speech data
    has_speech = training_df['reform_alignment_raw'].notna().sum()
    print(f"  {has_speech} MPs have speech analysis data")

    # Add defection labels from training population (is_defector column)
    training_df['defected'] = training_df['is_defector'].astype(int)

    # Merge with career features (ministerial data from IFG)
    training_df = training_df.merge(
        career_df[['name_clean', 'ever_minister', 'total_minister_years',
                   'highest_ministerial_rank', 'portfolio_count',
                   'years_since_last_ministerial_role', 'cabinet_level',
                   'backbench_years', 'career_stagnation_score']],
        on='name_clean',
        how='left',
        suffixes=('', '_career')
    )

    # Merge with age data
    training_df = training_df.merge(
        age_df[['name_clean', 'age', 'retirement_proximity_score']],
        on='name_clean',
        how='left',
        suffixes=('', '_age')
    )

    # Use age data if available
    if 'age_age' in training_df.columns:
        training_df['age'] = training_df['age_age'].fillna(training_df.get('age', np.nan))
        training_df = training_df.drop(columns=['age_age'], errors='ignore')
    if 'retirement_proximity_score_age' in training_df.columns:
        training_df['retirement_proximity_score'] = training_df['retirement_proximity_score_age'].fillna(
            training_df.get('retirement_proximity_score', 0)
        )
        training_df = training_df.drop(columns=['retirement_proximity_score_age'], errors='ignore')

    # Fill missing values for non-matched MPs
    for col in ['ever_minister', 'total_minister_years', 'highest_ministerial_rank',
                'portfolio_count', 'cabinet_level']:
        if col in training_df.columns:
            training_df[col] = training_df[col].fillna(0)

    for col in ['backbench_years', 'career_stagnation_score', 'retirement_proximity_score']:
        if col in training_df.columns:
            training_df[col] = training_df[col].fillna(0)

    training_df['years_since_last_ministerial_role'] = training_df['years_since_last_ministerial_role'].fillna(0)

    # CRITICAL FIX: Override ministerial data for defectors using defection tracker
    # The IFG data only covers current MPs, so many former MP defectors are missing
    # The defection tracker has authoritative ministerial data for defectors
    print("\n  Applying defector ministerial data corrections...")
    for _, defector in defections_df.iterrows():
        mask = training_df['name_clean'] == defector['name_clean']
        if mask.any() and 'ever_minister' in defector.index:
            old_val = training_df.loc[mask, 'ever_minister'].values[0]
            new_val = defector['ever_minister']
            if old_val != new_val:
                training_df.loc[mask, 'ever_minister'] = new_val
                if 'ministerial_rank' in defector.index:
                    training_df.loc[mask, 'highest_ministerial_rank'] = defector['ministerial_rank']
                print(f"    Corrected {defector['name']}: ever_minister {old_val} -> {new_val}")

    # Estimate years as MP for those without career data
    training_df['estimated_years_as_mp'] = np.maximum(
        training_df['age'].fillna(55) - 30,
        training_df['total_minister_years']
    )

    # Filter to only MPs with speech data for model training
    training_df = training_df[training_df['reform_alignment_raw'].notna()].copy()

    print(f"\nMerged dataset (MPs with speech data):")
    print(f"  Total MPs: {len(training_df)}")
    print(f"  Defectors: {training_df['defected'].sum()}")
    print(f"  Non-defectors: {(training_df['defected'] == 0).sum()}")
    print(f"  MPs with ministerial history: {training_df['ever_minister'].sum()}")

    return training_df


# =============================================================================
# SELECT FEATURES
# =============================================================================

def select_training_features(training_df):
    """Select and organize features for training."""

    print("\nSelecting training features...")

    # Speech features (Reform/Brexit alignment + radicalization trajectory)
    speech_features = [
        'reform_alignment_raw',
        'reform_alignment_normalized',
        'radicalization_slope',
        'extremism_percentile',
        'immigration_proportion',
        'total_speeches'
    ]

    # Career trajectory features (no age/gender - too much missing data)
    career_features = [
        'backbench_years',
        'ever_minister',
        'total_minister_years',
        'highest_ministerial_rank',
        'years_since_last_ministerial_role',
        'cabinet_level',
        'career_stagnation_score'
    ]

    # Label
    label = 'defected'

    # Identifiers
    identifiers = ['speaker_name', 'name_clean']

    # Select columns (speech + career, no age/gender)
    all_features = identifiers + speech_features + career_features + [label]

    training_data = training_df[all_features].copy()

    # Fill missing values
    for col in speech_features + career_features:
        if col in training_data.columns:
            training_data[col] = training_data[col].fillna(0)

    print(f"\nFeature summary:")
    print(f"  Speech features: {len(speech_features)}")
    print(f"  Career trajectory features: {len(career_features)}")
    print(f"  Total features: {len(speech_features) + len(career_features)}")

    return training_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("PREPARING TRAINING DATASET")
    print("=" * 80)
    print()

    # Load data
    speech_df, defections_df, training_pop_df, career_df, age_df = load_all_data()

    # Merge datasets
    training_df = merge_training_data(speech_df, defections_df, training_pop_df, career_df, age_df)

    # Select features
    training_data = select_training_features(training_df)

    # Save output
    output_path = OUTPUT_DIR / "training_data_2024.csv"
    training_data.to_csv(output_path, index=False)
    print(f"\nSaved training data to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("TRAINING DATA SUMMARY")
    print("=" * 80)

    print(f"\nDataset size:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Defectors (positive class): {training_data['defected'].sum()} ({training_data['defected'].mean()*100:.1f}%)")
    print(f"  Non-defectors (negative class): {(training_data['defected'] == 0).sum()} ({(1-training_data['defected'].mean())*100:.1f}%)")

    print(f"\nSpeech feature statistics:")
    print(f"  Mean raw Reform alignment: {training_data['reform_alignment_raw'].mean():.3f}")
    print(f"  Mean normalized Reform alignment: {training_data['reform_alignment_normalized'].mean():.3f}")
    print(f"  Mean radicalization slope: {training_data['radicalization_slope'].mean():.4f}")

    print(f"\nCareer feature statistics:")
    print(f"  MPs who held ministerial roles: {training_data['ever_minister'].sum()}")
    print(f"  Mean backbench years: {training_data['backbench_years'].mean():.1f}")
    print(f"  Mean career stagnation score: {training_data['career_stagnation_score'].mean():.3f}")

    # Compare defectors vs non-defectors
    print("\n" + "=" * 80)
    print("DEFECTORS VS NON-DEFECTORS")
    print("=" * 80)

    defectors = training_data[training_data['defected'] == 1]
    non_defectors = training_data[training_data['defected'] == 0]

    print(f"\nReform alignment (raw):")
    print(f"  Defectors: {defectors['reform_alignment_raw'].mean():.3f}")
    print(f"  Non-defectors: {non_defectors['reform_alignment_raw'].mean():.3f}")
    print(f"  Difference: {defectors['reform_alignment_raw'].mean() - non_defectors['reform_alignment_raw'].mean():+.3f}")

    print(f"\nReform alignment (normalized):")
    print(f"  Defectors: {defectors['reform_alignment_normalized'].mean():.3f}")
    print(f"  Non-defectors: {non_defectors['reform_alignment_normalized'].mean():.3f}")
    print(f"  Difference: {defectors['reform_alignment_normalized'].mean() - non_defectors['reform_alignment_normalized'].mean():+.3f}")

    print(f"\nBackbench years:")
    print(f"  Defectors: {defectors['backbench_years'].mean():.1f}")
    print(f"  Non-defectors: {non_defectors['backbench_years'].mean():.1f}")
    print(f"  Difference: {defectors['backbench_years'].mean() - non_defectors['backbench_years'].mean():+.1f}")

    print(f"\nMinisterial experience:")
    print(f"  Defectors who were ministers: {defectors['ever_minister'].sum()} ({defectors['ever_minister'].mean()*100:.1f}%)")
    print(f"  Non-defectors who were ministers: {non_defectors['ever_minister'].sum()} ({non_defectors['ever_minister'].mean()*100:.1f}%)")

    # List defectors
    print("\n" + "=" * 80)
    print("DEFECTORS IN TRAINING SET")
    print("=" * 80)

    defector_list = defectors[['speaker_name', 'reform_alignment_raw', 'reform_alignment_normalized', 'backbench_years', 'ever_minister']].sort_values('reform_alignment_raw', ascending=False)
    for _, row in defector_list.iterrows():
        minister_label = "Minister" if row['ever_minister'] else "Backbencher"
        print(f"  {row['speaker_name']:<30} | Raw: {row['reform_alignment_raw']:.3f} | Norm: {row['reform_alignment_normalized']:.3f} | Backbench: {row['backbench_years']:.0f}yrs | {minister_label}")

    return training_data


if __name__ == "__main__":
    results = main()
