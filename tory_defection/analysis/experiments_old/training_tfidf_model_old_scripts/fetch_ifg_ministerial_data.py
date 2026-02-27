"""
Fetch IFG Ministerial Data
===========================

Downloads and parses ministerial appointment data from the
Institute for Government (IfG) Ministers Database on GitHub.

Source: https://github.com/instituteforgov/ifg-ministers-database-public
License: CC-BY-4.0
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import numpy as np

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent  # tory_defection/
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = Path(__file__).parent  # training_tfidf_model_final_spec/

IFG_DATA_DIR = SOURCE_DATA / "ifg_ministers"
IFG_DATA_DIR.mkdir(exist_ok=True)

# Use the training population from 2019 election (Conservative MPs only)
TRAINING_POPULATION_CSV = ANALYSIS_OUTPUT / "training_population_2019_2024.csv"
PREDICTION_POPULATION_CSV = ANALYSIS_OUTPUT / "prediction_population_2024_present.csv"

# Defections for correcting ministerial data
DEFECTIONS_CSV = SOURCE_DATA / "defection_tracker" / "defections_2024.csv"

# IfG GitHub raw data URLs
IFG_BASE_URL = "https://raw.githubusercontent.com/instituteforgov/ifg-ministers-database-public/main/data"
IFG_FILES = {
    'appointments': f"{IFG_BASE_URL}/appointment.csv",
    'persons': f"{IFG_BASE_URL}/person.csv",
    'posts': f"{IFG_BASE_URL}/post.csv",
    'organisations': f"{IFG_BASE_URL}/organisation.csv",
}

# =============================================================================
# DOWNLOAD IFG DATA
# =============================================================================

def download_ifg_data():
    """Download CSV files from IfG GitHub repository."""

    print("Downloading IfG Ministers Database...")

    dataframes = {}

    for name, url in IFG_FILES.items():
        print(f"  Fetching {name}...")
        try:
            df = pd.read_csv(url)
            dataframes[name] = df
            print(f"    Loaded {len(df)} records")

            # Save locally
            local_path = IFG_DATA_DIR / f"{name}.csv"
            df.to_csv(local_path, index=False)

        except Exception as e:
            print(f"    ERROR: {e}")
            dataframes[name] = pd.DataFrame()

    return dataframes


# =============================================================================
# PARSE MINISTERIAL HISTORY
# =============================================================================

def clean_name_for_matching(name):
    """Clean names for matching."""
    if pd.isna(name):
        return name
    for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ', 'The ']:
        name = name.replace(title, '')
    return name.strip()


def get_ministerial_rank(post_name):
    """Determine ministerial rank from post name."""
    if not post_name or pd.isna(post_name):
        return 0

    post_upper = str(post_name).upper()

    # Shadow/opposition don't count
    if "SHADOW" in post_upper or "OPPOSITION" in post_upper:
        return 0

    # Rank hierarchy
    if "PRIME MINISTER" in post_upper:
        return 4
    if "SECRETARY OF STATE" in post_upper or "CHANCELLOR" in post_upper:
        return 4
    if "MINISTER OF STATE" in post_upper:
        return 3
    if "UNDER-SECRETARY" in post_upper or "UNDER SECRETARY" in post_upper:
        return 2
    if "PARLIAMENTARY SECRETARY" in post_upper:
        return 2
    if "PPS" in post_upper or "PARLIAMENTARY PRIVATE SECRETARY" in post_upper:
        return 1
    if "MINISTER" in post_upper:
        return 3  # Generic minister

    return 0


def parse_ministerial_careers(ifg_data, conservative_names_set):
    """Parse ministerial careers for Conservative MPs.

    Args:
        ifg_data: Dictionary of IFG dataframes
        conservative_names_set: Set of Conservative MP names to process
    """

    print("\nParsing ministerial careers...")

    appointments = ifg_data['appointments']
    persons = ifg_data['persons']
    posts = ifg_data['posts']
    orgs = ifg_data['organisations']

    # Use the provided Conservative names set
    conservative_names = conservative_names_set

    # Merge appointments with persons to get names
    appointments = appointments.merge(
        persons[['id', 'display_name']],
        left_on='person_id',
        right_on='id',
        how='left',
        suffixes=('', '_person')
    )

    # Merge with posts to get post titles
    appointments = appointments.merge(
        posts[['id', 'name', 'organisation_id']],
        left_on='post_id',
        right_on='id',
        how='left',
        suffixes=('', '_post')
    )

    # Clean person names
    appointments['name_clean'] = appointments['display_name'].apply(clean_name_for_matching)

    # Filter to Conservative MPs
    appointments = appointments[appointments['name_clean'].isin(conservative_names)]

    # Parse dates
    appointments['start_date'] = pd.to_datetime(appointments['start_date'], errors='coerce')
    appointments['end_date'] = pd.to_datetime(appointments['end_date'], errors='coerce')

    # Calculate duration
    appointments['end_date_calc'] = appointments['end_date'].fillna(pd.Timestamp.now())
    appointments['duration_years'] = (
        (appointments['end_date_calc'] - appointments['start_date']).dt.days / 365.25
    )

    # Get ministerial rank
    appointments['rank'] = appointments['name'].apply(get_ministerial_rank)

    # Filter to actual ministerial roles (rank > 0)
    ministerial_appointments = appointments[appointments['rank'] > 0].copy()

    print(f"  Found {len(ministerial_appointments)} ministerial appointments")
    print(f"  For {ministerial_appointments['name_clean'].nunique()} Conservative MPs")

    # Calculate career metrics per MP
    mp_metrics = []

    for mp_name_clean in conservative_names:
        mp_appointments = ministerial_appointments[ministerial_appointments['name_clean'] == mp_name_clean]

        if len(mp_appointments) == 0:
            # Never held ministerial role
            metrics = {
                'name_clean': mp_name_clean,
                'ever_minister': 0,
                'total_minister_years': 0.0,
                'highest_ministerial_rank': 0,
                'portfolio_count': 0,
                'years_since_last_ministerial_role': None,
                'current_minister': 0,
                'most_recent_position': None,
                'most_recent_start_date': None,
                'most_recent_end_date': None,
                'cabinet_level': 0
            }
        else:
            total_years = mp_appointments['duration_years'].sum()
            highest_rank = mp_appointments['rank'].max()
            portfolio_count = mp_appointments['organisation_id'].nunique()
            current_minister = int(mp_appointments['end_date'].isna().any())
            cabinet_level = int(highest_rank == 4)

            # Most recent appointment
            latest = mp_appointments.sort_values('start_date', ascending=False).iloc[0]

            if pd.notna(latest['end_date']):
                years_since = (datetime.now() - latest['end_date']).days / 365.25
            else:
                years_since = 0  # Still in role

            metrics = {
                'name_clean': mp_name_clean,
                'ever_minister': 1,
                'total_minister_years': round(total_years, 2),
                'highest_ministerial_rank': int(highest_rank),
                'portfolio_count': int(portfolio_count),
                'years_since_last_ministerial_role': round(years_since, 2) if pd.notna(latest['end_date']) else 0,
                'current_minister': current_minister,
                'most_recent_position': latest['name'],
                'most_recent_start_date': latest['start_date'].strftime('%Y-%m-%d') if pd.notna(latest['start_date']) else None,
                'most_recent_end_date': latest['end_date'].strftime('%Y-%m-%d') if pd.notna(latest['end_date']) else None,
                'cabinet_level': cabinet_level
            }

        mp_metrics.append(metrics)

    metrics_df = pd.DataFrame(mp_metrics)

    # Add name column for consistency
    metrics_df['name'] = metrics_df['name_clean']

    # Load actual tenure data from Parliament API (if available)
    tenure_data_path = ANALYSIS_OUTPUT / "mp_tenure_data.csv"
    if tenure_data_path.exists():
        tenure_df = pd.read_csv(tenure_data_path)
        tenure_df['name_clean'] = tenure_df['name'].apply(clean_name_for_matching)
        print(f"\n  Loaded tenure data for {len(tenure_df)} MPs from Parliament API")

        # Merge tenure data
        metrics_df = metrics_df.merge(
            tenure_df[['name_clean', 'years_as_mp']],
            on='name_clean',
            how='left'
        )

        # Use actual tenure data where available, otherwise default
        metrics_df['estimated_total_years_as_mp'] = metrics_df['years_as_mp'].fillna(10.0)  # Conservative default
        metrics_df = metrics_df.drop(columns=['years_as_mp'], errors='ignore')

        actual_tenure_count = metrics_df['estimated_total_years_as_mp'].notna().sum()
        print(f"    {actual_tenure_count} MPs have actual tenure data")
    else:
        print("\n  WARNING: No tenure data file found - using defaults")
        metrics_df['estimated_total_years_as_mp'] = 10.0

    # Backbench years = total years as MP minus ministerial years
    metrics_df['backbench_years'] = (
        metrics_df['estimated_total_years_as_mp'] -
        metrics_df['total_minister_years']
    ).clip(lower=0)

    # Career stagnation
    metrics_df['career_stagnation_score'] = 0.0

    stagnation_mask = (
        (metrics_df['ever_minister'] == 1) &
        (metrics_df['years_since_last_ministerial_role'] > 2)
    )

    metrics_df.loc[stagnation_mask, 'career_stagnation_score'] = (
        (metrics_df.loc[stagnation_mask, 'years_since_last_ministerial_role'] / 10).clip(upper=1)
    )

    print(f"\n  Processed {len(metrics_df)} Conservative MPs:")
    print(f"    {metrics_df['ever_minister'].sum()} held ministerial roles")
    print(f"    {metrics_df['cabinet_level'].sum()} served at Cabinet level")
    print(f"    {metrics_df['current_minister'].sum()} currently in ministerial roles")

    return metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("FETCHING IFG MINISTERIAL DATA FOR CONSERVATIVE MPs")
    print("=" * 80)
    print()

    # Download IFG data
    ifg_data = download_ifg_data()

    # Load Conservative MP populations
    print("\nLoading Conservative MP populations...")

    training_df = pd.read_csv(TRAINING_POPULATION_CSV)
    training_names = set(training_df['name'].values)
    print(f"  Training population (2019-2024): {len(training_names)}")

    prediction_df = pd.read_csv(PREDICTION_POPULATION_CSV)
    prediction_names = set(prediction_df['name'].values)
    print(f"  Prediction population (2024+): {len(prediction_names)}")

    # Combine all Conservative names
    all_conservative_names = training_names | prediction_names
    print(f"  Total unique Conservative MPs: {len(all_conservative_names)}")

    # Also clean names for IFG matching
    all_names_cleaned = set(clean_name_for_matching(n) for n in all_conservative_names)

    # Parse ministerial careers
    metrics_df = parse_ministerial_careers(ifg_data, all_names_cleaned)

    # Load defections for ministerial corrections
    print("\nApplying defector ministerial corrections...")
    defections_df = pd.read_csv(DEFECTIONS_CSV)

    for _, defector in defections_df.iterrows():
        defector_name = clean_name_for_matching(defector['name'])
        mask = metrics_df['name_clean'] == defector_name

        if mask.any():
            old_ever_minister = metrics_df.loc[mask, 'ever_minister'].values[0]
            new_ever_minister = defector['ever_minister']

            if old_ever_minister != new_ever_minister:
                metrics_df.loc[mask, 'ever_minister'] = new_ever_minister
                metrics_df.loc[mask, 'highest_ministerial_rank'] = defector['ministerial_rank']
                print(f"    Corrected {defector['name']}: ever_minister {old_ever_minister} -> {new_ever_minister}")

    # Recalculate backbench years after corrections
    metrics_df['backbench_years'] = (
        metrics_df['estimated_total_years_as_mp'] -
        metrics_df['total_minister_years']
    ).clip(lower=0)

    # For MPs with ever_minister=1 but total_minister_years=0 (from defector corrections),
    # set some reasonable defaults
    correction_mask = (metrics_df['ever_minister'] == 1) & (metrics_df['total_minister_years'] == 0)
    metrics_df.loc[correction_mask, 'total_minister_years'] = 2.0  # Assume 2 years average tenure
    metrics_df.loc[correction_mask, 'backbench_years'] = 23.0

    # Save output
    output_path = ANALYSIS_OUTPUT / "mp_career_features.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved career metrics to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nRank distribution:")
    print(f"  Rank 4 (Cabinet): {(metrics_df['highest_ministerial_rank'] == 4).sum()}")
    print(f"  Rank 3 (Minister of State): {(metrics_df['highest_ministerial_rank'] == 3).sum()}")
    print(f"  Rank 2 (Under-Secretary): {(metrics_df['highest_ministerial_rank'] == 2).sum()}")
    print(f"  Rank 1 (PPS): {(metrics_df['highest_ministerial_rank'] == 1).sum()}")
    print(f"  Rank 0 (Never minister): {(metrics_df['highest_ministerial_rank'] == 0).sum()}")

    ministers = metrics_df[metrics_df['ever_minister'] == 1]
    if len(ministers) > 0:
        print(f"\nMinisterial statistics:")
        print(f"  Mean minister years: {ministers['total_minister_years'].mean():.1f}")
        print(f"  Max minister years: {ministers['total_minister_years'].max():.1f}")
        print(f"  Mean portfolios: {ministers['portfolio_count'].mean():.1f}")
        print(f"  Mean years since last role: {ministers[ministers['years_since_last_ministerial_role'] > 0]['years_since_last_ministerial_role'].mean():.1f}")

    print(f"\nCareer stagnation:")
    print(f"  MPs with stagnation score > 0.3: {(metrics_df['career_stagnation_score'] > 0.3).sum()}")
    print(f"  Max stagnation score: {metrics_df['career_stagnation_score'].max():.2f}")

    # Spot checks
    print("\n" + "=" * 80)
    print("SPOT CHECKS (Known Ministers)")
    print("=" * 80)

    known_names = ['Rishi Sunak', 'Suella Braverman', 'Nadine Dorries', 'Robert Jenrick', 'Kemi Badenoch']
    for check_name in known_names:
        check_name_clean = clean_name_for_matching(check_name)
        matches = metrics_df[metrics_df['name_clean'] == check_name_clean]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f"\n{row['name']}:")
            print(f"  Ever minister: {row['ever_minister']}")
            print(f"  Total minister years: {row['total_minister_years']:.1f}")
            print(f"  Highest rank: {row['highest_ministerial_rank']}")
            print(f"  Most recent position: {row['most_recent_position']}")
            print(f"  Years since last role: {row['years_since_last_ministerial_role']}")
            print(f"  Career stagnation: {row['career_stagnation_score']:.2f}")
        else:
            print(f"\n{check_name}: NOT FOUND")

    return metrics_df


if __name__ == "__main__":
    results = main()
