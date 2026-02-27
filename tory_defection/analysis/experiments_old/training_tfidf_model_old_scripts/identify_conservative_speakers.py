"""
Identify Conservative Speakers for Model Training and Prediction
================================================================

Creates two verified lists of Conservative MP names:
1. TRAINING POPULATION: 2019-2024 Parliament Conservative MPs
   - Source: December 2019 Election Results (365 Conservative MPs)
   - All 23 defectors are from this parliament (per defections_2024.csv)

2. PREDICTION POPULATION: 2024-Present Parliament Conservative MPs
   - Source: July 2024 Election Results (121 Conservative MPs)
   - These are the MPs we want to predict defection risk for

This ensures clean data with NO contamination from other parties.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent  # tory_defection/
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = Path(__file__).parent  # training_tfidf_model_final_spec/

# Election data (updated path)
ELECTIONS_2019 = SOURCE_DATA / "elections" / "HoC-GE2019-results-by-candidate.xlsx"
ELECTIONS_2024 = SOURCE_DATA / "elections" / "MPs-elected.xlsx"

# Other data
DEFECTIONS_CSV = SOURCE_DATA / "defection_tracker" / "defections_2024.csv"
HANSARD_CSV = SOURCE_DATA / "hansard" / "all_speeches_extended.csv"  # Use extended dataset

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_name(name):
    """Clean names for matching."""
    if pd.isna(name):
        return name
    for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ', 'The ']:
        name = name.replace(title, '')
    return name.strip()


def get_2019_conservative_mps():
    """
    Extract all Conservative MPs elected in December 2019.

    Returns set of names in 'Firstname Surname' format.
    """
    print("Loading 2019 election results...")

    # Read with header on row 3 (0-indexed: row 2)
    df = pd.read_excel(ELECTIONS_2019, header=2)

    # Find winners - get candidate with most votes in each constituency
    winners = df.loc[df.groupby('Constituency name')['Votes'].idxmax()]

    # Filter to Conservative winners
    con_mps = winners[winners['Party abbreviation'] == 'Con'].copy()

    # Create full name
    con_mps['name'] = con_mps['Candidate first name'] + ' ' + con_mps['Candidate surname']

    print(f"  Found {len(con_mps)} Conservative MPs elected in December 2019")

    return set(con_mps['name'].values), con_mps


def get_2024_conservative_mps():
    """
    Extract all Conservative MPs elected in July 2024.

    Returns set of names in 'Firstname Surname' format.
    """
    print("Loading 2024 election results...")

    df = pd.read_excel(ELECTIONS_2024)

    # Filter to Conservative MPs
    con_mps = df[df['party_abbreviation'] == 'Con'].copy()

    # Create full name
    con_mps['name'] = con_mps['firstname'] + ' ' + con_mps['surname']

    print(f"  Found {len(con_mps)} Conservative MPs elected in July 2024")

    return set(con_mps['name'].values), con_mps


def get_defectors():
    """
    Load all verified defectors from defections_2024.csv.

    These were all 2019-2024 Parliament MPs (regardless of when they defected).
    """
    print("Loading defectors from Best for Britain tracker...")

    df = pd.read_csv(DEFECTIONS_CSV)

    print(f"  Found {len(df)} verified defectors")

    return set(df['name'].values), df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("IDENTIFYING CONSERVATIVE SPEAKERS FOR HANSARD ANALYSIS")
    print("Defining TRAINING (2019-2024) and PREDICTION (2024-present) populations")
    print("=" * 80)
    print()

    # =========================================================================
    # LOAD ALL DATA SOURCES
    # =========================================================================

    # 2019 election MPs (TRAINING population base)
    mps_2019, mps_2019_df = get_2019_conservative_mps()

    # 2024 election MPs (PREDICTION population)
    mps_2024, mps_2024_df = get_2024_conservative_mps()

    # Defectors (all from 2019-2024 Parliament)
    defectors, defectors_df = get_defectors()

    # Load Hansard data for speech verification
    print("\nLoading Hansard speech data...")
    hansard = pd.read_csv(HANSARD_CSV)
    hansard_speakers = set(hansard['speaker_name'].unique())
    print(f"  Found {len(hansard_speakers)} unique speakers in Hansard")

    # =========================================================================
    # DEFINE TRAINING POPULATION (2019-2024 Parliament)
    # =========================================================================

    print("\n" + "=" * 80)
    print("TRAINING POPULATION: 2019-2024 Parliament Conservative MPs")
    print("=" * 80)

    # Training population = 2019 election MPs + any defectors not captured
    training_population = mps_2019.copy()

    # Add any defectors not in 2019 election results (shouldn't be many)
    missing_defectors = defectors - mps_2019
    if missing_defectors:
        print(f"\n  Defectors not in 2019 election results (adding to training):")
        for name in sorted(missing_defectors):
            print(f"    - {name}")
        training_population.update(missing_defectors)

    print(f"\n  Total training population: {len(training_population)}")
    print(f"    - From 2019 election: {len(mps_2019)}")
    print(f"    - Additional defectors: {len(missing_defectors)}")

    # Check how many have Hansard speeches
    training_with_speeches = [n for n in training_population if n in hansard_speakers]
    print(f"    - With Hansard speeches: {len(training_with_speeches)}")

    # Verify defectors in training set
    defectors_in_training = [n for n in defectors if n in hansard_speakers]
    print(f"    - Defectors with speeches: {len(defectors_in_training)} of {len(defectors)}")

    # =========================================================================
    # DEFINE PREDICTION POPULATION (2024-Present Parliament)
    # =========================================================================

    print("\n" + "=" * 80)
    print("PREDICTION POPULATION: 2024-Present Parliament Conservative MPs")
    print("=" * 80)

    # Remove any known defectors from prediction population
    # (Danny Kruger and Robert Jenrick are in 2024 MPs but defected)
    prediction_population = mps_2024 - defectors

    print(f"\n  Total prediction population: {len(prediction_population)}")
    print(f"    - From 2024 election: {len(mps_2024)}")
    print(f"    - Minus defectors: {len(mps_2024 & defectors)}")

    # Check how many have Hansard speeches
    prediction_with_speeches = [n for n in prediction_population if n in hansard_speakers]
    print(f"    - With Hansard speeches: {len(prediction_with_speeches)}")

    # =========================================================================
    # DATA QUALITY CHECK
    # =========================================================================

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)

    known_non_conservatives = [
        'Angela Rayner', 'Rachel Reeves', 'Keir Starmer', 'Ed Miliband',
        'Yvette Cooper', 'Lisa Nandy', 'David Lammy', 'Wes Streeting',
        'Ed Davey', 'Tim Farron', 'Layla Moran', 'Richard Burgon'
    ]

    # Check training population
    training_contamination = [n for n in known_non_conservatives if n in training_population]
    if training_contamination:
        print(f"\n  WARNING: Training population has {len(training_contamination)} non-Conservatives:")
        for name in training_contamination:
            print(f"    - {name}")
    else:
        print("\n  [OK] Training population: No known non-Conservative MPs found")

    # Check prediction population
    prediction_contamination = [n for n in known_non_conservatives if n in prediction_population]
    if prediction_contamination:
        print(f"\n  WARNING: Prediction population has {len(prediction_contamination)} non-Conservatives:")
        for name in prediction_contamination:
            print(f"    - {name}")
    else:
        print("  [OK] Prediction population: No known non-Conservative MPs found")

    # =========================================================================
    # SAVE OUTPUT FILES
    # =========================================================================

    print("\n" + "=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)

    # 1. Training population (2019-2024 MPs)
    training_df = pd.DataFrame({
        'name': sorted(list(training_population)),
    })
    training_df['is_defector'] = training_df['name'].isin(defectors)
    training_df['has_hansard_speeches'] = training_df['name'].isin(hansard_speakers)
    training_df['parliament'] = '2019-2024'

    training_path = ANALYSIS_OUTPUT / "training_population_2019_2024.csv"
    training_df.to_csv(training_path, index=False)
    print(f"\n  Saved training population to: {training_path}")
    print(f"    - Total: {len(training_df)}")
    print(f"    - Defectors: {training_df['is_defector'].sum()}")
    print(f"    - With speeches: {training_df['has_hansard_speeches'].sum()}")

    # 2. Prediction population (2024-present MPs)
    prediction_df = pd.DataFrame({
        'name': sorted(list(prediction_population)),
    })
    prediction_df['is_defector'] = False  # We're predicting, so no defectors here
    prediction_df['has_hansard_speeches'] = prediction_df['name'].isin(hansard_speakers)
    prediction_df['parliament'] = '2024-present'

    prediction_path = ANALYSIS_OUTPUT / "prediction_population_2024_present.csv"
    prediction_df.to_csv(prediction_path, index=False)
    print(f"\n  Saved prediction population to: {prediction_path}")
    print(f"    - Total: {len(prediction_df)}")
    print(f"    - With speeches: {prediction_df['has_hansard_speeches'].sum()}")

    # 3. Combined list for backward compatibility
    all_conservatives = training_population | prediction_population
    combined_df = pd.DataFrame({
        'name': sorted(list(all_conservatives)),
    })
    combined_df['is_2019_mp'] = combined_df['name'].isin(training_population)
    combined_df['is_2024_mp'] = combined_df['name'].isin(prediction_population)
    combined_df['is_defector'] = combined_df['name'].isin(defectors)
    combined_df['has_hansard_speeches'] = combined_df['name'].isin(hansard_speakers)

    combined_path = ANALYSIS_OUTPUT / "verified_conservative_mps.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n  Saved combined list to: {combined_path}")
    print(f"    - Total unique MPs: {len(combined_df)}")

    # 4. Speakers with Hansard speeches (for speech analysis)
    speakers_df = combined_df[combined_df['has_hansard_speeches']][['name', 'has_hansard_speeches']].copy()
    speakers_path = ANALYSIS_OUTPUT / "conservative_speakers_in_hansard.csv"
    speakers_df.to_csv(speakers_path, index=False)
    print(f"\n  Saved speakers list to: {speakers_path}")
    print(f"    - Total with speeches: {len(speakers_df)}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
TRAINING DATA (2019-2024 Parliament):
  - Conservative MPs: {len(training_population)}
  - Defectors: {len(defectors)} (label=1)
  - Non-defectors: {len(training_population) - len(defectors)} (label=0)
  - Defection rate: {len(defectors)/len(training_population)*100:.1f}%

PREDICTION DATA (2024-Present Parliament):
  - Conservative MPs: {len(prediction_population)}
  - Already defected: 0 (excluded)
  - These are the MPs we predict defection risk for

KEY EXPECTED HIGH-RISK MPs (should appear in prediction population):
  - Esther McVey: {'Yes' if 'Esther McVey' in prediction_population else 'No'}
  - Suella Braverman: {'Yes' if 'Suella Braverman' in prediction_population else 'No'}
  - Mark Francois: {'Yes' if 'Mark Francois' in prediction_population else 'No'}
  - Andrew Rosindell: {'Yes' if 'Andrew Rosindell' in prediction_population else 'No'}
  - John Hayes: {'Yes' if 'John Hayes' in prediction_population else 'No'}
""")

    # =========================================================================
    # LIST DEFECTORS WITH HANSARD PRESENCE
    # =========================================================================

    print("\n" + "=" * 80)
    print("DEFECTORS IN TRAINING DATA")
    print("=" * 80)

    for _, defector in defectors_df.iterrows():
        name = defector['name']
        has_speeches = name in hansard_speakers
        minister = "Minister" if defector['ever_minister'] else "Backbencher"
        status = "[OK]" if has_speeches else "[NO SPEECHES]"
        print(f"  {status} {name:<25} | {minister:<12} | Defected: {defector['defection_date']}")

    return {
        'training_population': training_population,
        'prediction_population': prediction_population,
        'defectors': defectors,
        'training_df': training_df,
        'prediction_df': prediction_df,
    }


if __name__ == "__main__":
    results = main()
