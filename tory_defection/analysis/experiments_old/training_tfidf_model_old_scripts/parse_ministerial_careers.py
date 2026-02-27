"""
Parse Ministerial Career Data
==============================

Extracts ministerial history, backbench duration, and career metrics
for Conservative MPs from JSON files.

Output: analysis/mp_career_features.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"

MINISTERS_JSON = SOURCE_DATA / "mp_careers" / "ministers.json"
PEOPLE_JSON = SOURCE_DATA / "mp_careers" / "people.json"
DEMOGRAPHICS_CSV = SOURCE_DATA / "mp_demographics.csv"

# =============================================================================
# MINISTERIAL RANK HIERARCHY
# =============================================================================

RANK_MAP = {
    # Cabinet level (4)
    "Prime Minister": 4,
    "Secretary of State": 4,
    "Chancellor of the Exchequer": 4,
    "Foreign Secretary": 4,

    # Minister of State (3)
    "Minister of State": 3,
    "Minister for": 3,

    # Parliamentary Under-Secretary (2)
    "Parliamentary Under-Secretary": 2,
    "Parliamentary Secretary": 2,
    "Under-Secretary": 2,

    # PPS (1)
    "PPS": 1,
    "Parliamentary Private Secretary": 1,

    # Shadow roles (not counted as ministerial)
    "Shadow": 0,
    "Spokespersons": 0,
}

def get_ministerial_rank(role_name):
    """Determine ministerial rank from role name."""
    if not role_name or pd.isna(role_name):
        return 0

    role_upper = role_name.upper()

    # Check for shadow roles first (these don't count)
    if "SHADOW" in role_upper or "SPOKESPERSON" in role_upper:
        return 0

    # Check hierarchy
    if "PRIME MINISTER" in role_upper:
        return 4
    if "SECRETARY OF STATE" in role_upper or "CHANCELLOR" in role_upper:
        return 4
    if "MINISTER OF STATE" in role_upper:
        return 3
    if "UNDER-SECRETARY" in role_upper or "PARLIAMENTARY SECRETARY" in role_upper:
        return 2
    if "PPS" in role_upper or "PARLIAMENTARY PRIVATE SECRETARY" in role_upper:
        return 1

    return 0


# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_data():
    """Load ministers, people, and demographics data."""

    print("Loading JSON and CSV data...")

    # Ministers
    with open(MINISTERS_JSON, 'r', encoding='utf-8') as f:
        ministers_data = json.load(f)
    ministers_df = pd.DataFrame(ministers_data['memberships'])
    print(f"Loaded {len(ministers_df)} ministerial appointments")

    # People (for name mapping)
    with open(PEOPLE_JSON, 'r', encoding='utf-8') as f:
        people_data = json.load(f)
    persons = people_data['persons']

    # Create person_id → name mapping
    person_names = {}
    for person in persons:
        person_id = person['id']
        if 'other_names' in person and person['other_names']:
            names = person['other_names']
            for name_obj in names:
                if 'note' in name_obj and name_obj['note'] == 'Main':
                    given = name_obj.get('given_name', '')
                    family = name_obj.get('family_name', '')
                    full_name = f"{given} {family}".strip()
                    if full_name:
                        person_names[person_id] = full_name
                        break
            if person_id not in person_names and names:
                # Fallback to first name
                given = names[0].get('given_name', '')
                family = names[0].get('family_name', '')
                full_name = f"{given} {family}".strip()
                if full_name:
                    person_names[person_id] = full_name

    print(f"Mapped {len(person_names)} person IDs to names")

    # Demographics (for age data and Conservative filter)
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    print(f"Loaded {len(demographics)} MP demographics")

    # Clean names in demographics (remove titles)
    def clean_name(name):
        if pd.isna(name):
            return name
        # Remove common titles
        for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
            name = name.replace(title, '')
        return name.strip()

    demographics['name_clean'] = demographics['name'].apply(clean_name)

    return ministers_df, person_names, demographics


# =============================================================================
# PARSE MINISTERIAL CAREERS
# =============================================================================

def parse_ministerial_careers(ministers_df, person_names, demographics):
    """Extract ministerial career features for each MP."""

    print("\nParsing ministerial careers...")

    # Filter to Conservative MPs only (those in demographics CSV)
    conservative_names = set(demographics['name_clean'].values)

    # Convert person_id to name in ministers_df
    ministers_df['name'] = ministers_df['person_id'].map(person_names)
    ministers_df = ministers_df[ministers_df['name'].notna()]

    # Parse dates
    ministers_df['start_date'] = pd.to_datetime(ministers_df['start_date'], errors='coerce')
    ministers_df['end_date'] = pd.to_datetime(ministers_df['end_date'], errors='coerce')

    # Calculate duration in years
    ministers_df['duration_years'] = (
        (ministers_df['end_date'] - ministers_df['start_date']).dt.days / 365.25
    )

    # Get ministerial rank
    ministers_df['rank'] = ministers_df['role'].apply(get_ministerial_rank)

    # Filter to actual ministerial roles (rank > 0, not shadow)
    ministers_df = ministers_df[ministers_df['rank'] > 0]

    print(f"Filtered to {len(ministers_df)} actual ministerial roles (non-shadow)")

    # Aggregate by MP
    mp_features = []

    for name in conservative_names:
        mp_roles = ministers_df[ministers_df['name'] == name]

        if len(mp_roles) == 0:
            # Never held ministerial role
            features = {
                'name': name,
                'person_id': None,
                'ever_minister': 0,
                'total_minister_years': 0.0,
                'highest_ministerial_rank': 0,
                'portfolio_count': 0,
                'years_since_last_ministerial_role': None,
                'most_recent_role': None,
                'most_recent_department': None
            }
        else:
            # Extract features
            person_id = mp_roles['person_id'].iloc[0]
            total_years = mp_roles['duration_years'].sum()
            highest_rank = mp_roles['rank'].max()
            portfolio_count = mp_roles['organization_id'].nunique()

            # Most recent role
            latest_role = mp_roles.sort_values('end_date', ascending=False).iloc[0]
            most_recent_end = latest_role['end_date']

            if pd.notna(most_recent_end):
                years_since = (datetime.now() - most_recent_end).days / 365.25
            else:
                years_since = 0  # Still in role

            features = {
                'name': name,
                'person_id': person_id,
                'ever_minister': 1,
                'total_minister_years': round(total_years, 2),
                'highest_ministerial_rank': int(highest_rank),
                'portfolio_count': int(portfolio_count),
                'years_since_last_ministerial_role': round(years_since, 2),
                'most_recent_role': latest_role['role'],
                'most_recent_department': latest_role['organization_id']
            }

        mp_features.append(features)

    mp_careers = pd.DataFrame(mp_features)

    # Merge with demographics for age/DOB
    mp_careers = mp_careers.merge(
        demographics[['name', 'name_clean', 'date_of_birth', 'age']],
        left_on='name',
        right_on='name_clean',
        how='left'
    )

    # Keep original name with title
    mp_careers['name'] = mp_careers['name_x'].fillna(mp_careers['name_y'])
    mp_careers = mp_careers.drop(columns=['name_x', 'name_y', 'name_clean'])

    print(f"Processed {len(mp_careers)} Conservative MPs")
    print(f"  - {mp_careers['ever_minister'].sum()} held ministerial roles")
    print(f"  - {(mp_careers['ever_minister'] == 0).sum()} never held ministerial roles")

    return mp_careers


# =============================================================================
# CALCULATE DERIVED FEATURES
# =============================================================================

def calculate_derived_features(mp_careers):
    """Calculate backbench years, retirement proximity, career stagnation."""

    print("\nCalculating derived features...")

    # For now, we don't have exact MP start dates from all-members.json
    # Use a proxy: assume MPs have been in parliament for (age - 30) years max
    # This is approximate - ideally we'd fetch from Parliament API
    mp_careers['estimated_total_years_as_mp'] = np.maximum(
        mp_careers['age'].fillna(55) - 30,
        mp_careers['total_minister_years']
    )

    # Backbench years = total years - minister years
    mp_careers['backbench_years'] = (
        mp_careers['estimated_total_years_as_mp'] -
        mp_careers['total_minister_years']
    )
    mp_careers['backbench_years'] = mp_careers['backbench_years'].clip(lower=0)

    # Retirement proximity (0 at age 55, 1.0 at age 70)
    mp_careers['retirement_proximity_score'] = (
        (mp_careers['age'].fillna(55) - 55) / 15
    ).clip(lower=0, upper=1)

    # Career stagnation score
    # High if sidelined after ministerial role AND not near retirement
    mp_careers['career_stagnation_score'] = 0.0

    stagnation_mask = (
        (mp_careers['ever_minister'] == 1) &
        (mp_careers['years_since_last_ministerial_role'] > 2)
    )

    mp_careers.loc[stagnation_mask, 'career_stagnation_score'] = (
        (mp_careers.loc[stagnation_mask, 'years_since_last_ministerial_role'] / 10).clip(upper=1) *
        (1 - mp_careers.loc[stagnation_mask, 'retirement_proximity_score'])
    )

    print(f"Calculated features:")
    print(f"  - Mean backbench years: {mp_careers['backbench_years'].mean():.1f}")
    print(f"  - Mean retirement proximity: {mp_careers['retirement_proximity_score'].mean():.2f}")
    print(f"  - MPs with career stagnation > 0: {(mp_careers['career_stagnation_score'] > 0).sum()}")

    return mp_careers


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("PARSING MINISTERIAL CAREER DATA")
    print("=" * 80)
    print()

    # Load data
    ministers_df, person_names, demographics = load_all_data()

    # Parse ministerial careers
    mp_careers = parse_ministerial_careers(ministers_df, person_names, demographics)

    # Calculate derived features
    mp_careers = calculate_derived_features(mp_careers)

    # Save output
    output_path = ANALYSIS_OUTPUT / "mp_career_features.csv"
    mp_careers.to_csv(output_path, index=False)
    print(f"\nSaved career features to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal MPs: {len(mp_careers)}")
    print(f"MPs who held ministerial roles: {mp_careers['ever_minister'].sum()}")
    print(f"MPs who never held ministerial roles: {(mp_careers['ever_minister'] == 0).sum()}")
    print(f"\nMinisterial statistics:")
    print(f"  Mean minister years: {mp_careers[mp_careers['ever_minister'] == 1]['total_minister_years'].mean():.1f}")
    print(f"  Max minister years: {mp_careers['total_minister_years'].max():.1f}")
    print(f"  MPs with rank 4 (Cabinet): {(mp_careers['highest_ministerial_rank'] == 4).sum()}")
    print(f"  MPs with rank 3 (Minister of State): {(mp_careers['highest_ministerial_rank'] == 3).sum()}")
    print(f"  MPs with rank 2 (Under-Secretary): {(mp_careers['highest_ministerial_rank'] == 2).sum()}")
    print(f"  MPs with rank 1 (PPS): {(mp_careers['highest_ministerial_rank'] == 1).sum()}")

    print(f"\nCareer stagnation:")
    print(f"  MPs with stagnation score > 0.3: {(mp_careers['career_stagnation_score'] > 0.3).sum()}")
    print(f"  Max stagnation score: {mp_careers['career_stagnation_score'].max():.2f}")

    # Spot check known MPs
    print("\n" + "=" * 80)
    print("SPOT CHECKS (Known MPs)")
    print("=" * 80)

    known_mps = ['Rishi Sunak', 'Suella Braverman', 'Nadine Dorries', 'Robert Jenrick']
    for mp_name in known_mps:
        mp_data = mp_careers[mp_careers['name'] == mp_name]
        if len(mp_data) > 0:
            row = mp_data.iloc[0]
            print(f"\n{mp_name}:")
            print(f"  Ever minister: {row['ever_minister']}")
            print(f"  Total minister years: {row['total_minister_years']:.1f}")
            print(f"  Highest rank: {row['highest_ministerial_rank']}")
            print(f"  Most recent role: {row['most_recent_role']}")
            print(f"  Years since last role: {row['years_since_last_ministerial_role']:.1f}")
            print(f"  Career stagnation: {row['career_stagnation_score']:.2f}")
        else:
            print(f"\n{mp_name}: NOT FOUND in dataset")

    return mp_careers


if __name__ == "__main__":
    results = main()
