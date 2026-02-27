"""
Fetch Ministerial Data by Member ID
====================================

Fetches government positions for specific MPs from demographics CSV
using the Parliament Members API.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"

DEMOGRAPHICS_CSV = SOURCE_DATA / "mp_demographics.csv"

# Parliament API
MEMBERS_API = "https://members-api.parliament.uk/api"

# =============================================================================
# MINISTERIAL RANK HIERARCHY
# =============================================================================

def get_ministerial_rank(position_name):
    """Determine ministerial rank from position name."""
    if not position_name:
        return 0

    pos_upper = position_name.upper()

    # Shadow positions don't count
    if "SHADOW" in pos_upper or "OPPOSITION" in pos_upper:
        return 0

    # Rank hierarchy
    if "PRIME MINISTER" in pos_upper:
        return 4
    if "SECRETARY OF STATE" in pos_upper or "CHANCELLOR" in pos_upper:
        return 4
    if "MINISTER OF STATE" in pos_upper:
        return 3
    if "UNDER-SECRETARY" in pos_upper or "PARLIAMENTARY SECRETARY" in pos_upper:
        return 2
    if "PPS" in pos_upper or "PARLIAMENTARY PRIVATE SECRETARY" in pos_upper:
        return 1
    if "MINISTER" in pos_upper:
        return 3  # Generic minister

    return 0


# =============================================================================
# FETCH MINISTERIAL DATA
# =============================================================================

def fetch_government_positions(member_id):
    """Fetch government positions for a specific member."""
    url = f"{MEMBERS_API}/Members/{member_id}/GovernmentPosts"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('value', [])
    except Exception as e:
        return []


def fetch_ministerial_data_for_all_mps(demographics_df):
    """Fetch ministerial data for all MPs in demographics CSV."""

    print(f"Fetching ministerial data for {len(demographics_df)} MPs...")

    all_positions = []

    for idx, row in demographics_df.iterrows():
        member_id = row['member_id']
        name = row['name']

        if pd.isna(member_id):
            continue

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(demographics_df)} MPs...")

        # Fetch government positions
        positions = fetch_government_positions(member_id)

        for pos in positions:
            position_name = pos.get('name', '')
            rank = get_ministerial_rank(position_name)

            # Only count actual ministerial positions
            if rank > 0:
                all_positions.append({
                    'member_id': member_id,
                    'name': name,
                    'position': position_name,
                    'rank': rank,
                    'start_date': pos.get('startDate'),
                    'end_date': pos.get('endDate'),
                    'is_current': pos.get('endDate') is None or pos.get('endDate') == ''
                })

        time.sleep(0.1)  # Rate limiting

    positions_df = pd.DataFrame(all_positions)
    print(f"\nFound {len(positions_df)} ministerial appointments for {positions_df['name'].nunique() if len(positions_df) > 0 else 0} MPs")

    return positions_df


def calculate_career_metrics(positions_df, demographics_df):
    """Calculate ministerial career metrics."""

    print("\nCalculating career metrics...")

    # Clean names in demographics (remove titles)
    def clean_name(name):
        if pd.isna(name):
            return name
        for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
            name = name.replace(title, '')
        return name.strip()

    demographics_df['name_clean'] = demographics_df['name'].apply(clean_name)

    metrics_list = []

    for _, row in demographics_df.iterrows():
        name = row['name']
        name_clean = row['name_clean']

        # Find positions for this MP (match on either name or name_clean)
        mp_positions = positions_df[
            (positions_df['name'] == name) |
            (positions_df['name'].apply(clean_name) == name_clean)
        ].copy()

        if len(mp_positions) == 0:
            # Never held ministerial role
            metrics = {
                'name': name,
                'member_id': row['member_id'],
                'ever_minister': 0,
                'total_minister_years': 0.0,
                'highest_ministerial_rank': 0,
                'portfolio_count': 0,
                'years_since_last_ministerial_role': None,
                'current_minister': 0,
                'most_recent_position': None,
                'most_recent_start_date': None,
                'most_recent_end_date': None
            }
        else:
            # Calculate metrics
            mp_positions['start_date'] = pd.to_datetime(mp_positions['start_date'], errors='coerce')
            mp_positions['end_date'] = pd.to_datetime(mp_positions['end_date'], errors='coerce')

            # For positions still active, use current date as end date
            mp_positions['end_date_calc'] = mp_positions['end_date'].fillna(pd.Timestamp.now())

            # Calculate duration
            mp_positions['duration_years'] = (
                (mp_positions['end_date_calc'] - mp_positions['start_date']).dt.days / 365.25
            )

            total_years = mp_positions['duration_years'].sum()
            highest_rank = mp_positions['rank'].max()
            portfolio_count = mp_positions['position'].nunique()
            current_minister = int(mp_positions['is_current'].any())

            # Most recent position
            latest_pos = mp_positions.sort_values('start_date', ascending=False).iloc[0]

            if pd.notna(latest_pos['end_date']):
                years_since = (datetime.now() - latest_pos['end_date']).days / 365.25
            else:
                years_since = 0  # Still in role

            metrics = {
                'name': name,
                'member_id': row['member_id'],
                'ever_minister': 1,
                'total_minister_years': round(total_years, 2),
                'highest_ministerial_rank': int(highest_rank),
                'portfolio_count': int(portfolio_count),
                'years_since_last_ministerial_role': round(years_since, 2) if pd.notna(latest_pos['end_date']) else 0,
                'current_minister': current_minister,
                'most_recent_position': latest_pos['position'],
                'most_recent_start_date': latest_pos['start_date'].strftime('%Y-%m-%d') if pd.notna(latest_pos['start_date']) else None,
                'most_recent_end_date': latest_pos['end_date'].strftime('%Y-%m-%d') if pd.notna(latest_pos['end_date']) else None
            }

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # Merge with demographics for age
    metrics_df = metrics_df.merge(
        demographics_df[['name', 'date_of_birth', 'age']],
        on='name',
        how='left'
    )

    # Calculate derived features
    metrics_df['retirement_proximity_score'] = (
        (metrics_df['age'].fillna(55) - 55) / 15
    ).clip(lower=0, upper=1)

    # Career stagnation score
    metrics_df['career_stagnation_score'] = 0.0

    stagnation_mask = (
        (metrics_df['ever_minister'] == 1) &
        (metrics_df['years_since_last_ministerial_role'] > 2)
    )

    metrics_df.loc[stagnation_mask, 'career_stagnation_score'] = (
        (metrics_df.loc[stagnation_mask, 'years_since_last_ministerial_role'] / 10).clip(upper=1) *
        (1 - metrics_df.loc[stagnation_mask, 'retirement_proximity_score'])
    )

    # Estimate backbench years (approximate, as we don't have exact MP start date)
    metrics_df['estimated_total_years_as_mp'] = np.maximum(
        metrics_df['age'].fillna(55) - 30,
        metrics_df['total_minister_years']
    )

    metrics_df['backbench_years'] = (
        metrics_df['estimated_total_years_as_mp'] -
        metrics_df['total_minister_years']
    ).clip(lower=0)

    print(f"\nProcessed {len(metrics_df)} MPs:")
    print(f"  - {metrics_df['ever_minister'].sum()} held ministerial roles")
    print(f"  - {metrics_df['current_minister'].sum()} currently in ministerial roles")
    print(f"  - {(metrics_df['ever_minister'] == 0).sum()} never held ministerial roles")

    return metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FETCHING MINISTERIAL DATA BY MEMBER ID")
    print("=" * 80)
    print()

    # Load demographics
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    print(f"Loaded {len(demographics)} MPs from demographics CSV")

    # Fetch ministerial positions
    positions_df = fetch_ministerial_data_for_all_mps(demographics)

    # Calculate metrics
    metrics_df = calculate_career_metrics(positions_df, demographics)

    # Save outputs
    positions_output = ANALYSIS_OUTPUT / "ministerial_positions_detailed.csv"
    positions_df.to_csv(positions_output, index=False)
    print(f"\nSaved detailed positions to: {positions_output}")

    metrics_output = ANALYSIS_OUTPUT / "mp_career_features.csv"
    metrics_df.to_csv(metrics_output, index=False)
    print(f"Saved career metrics to: {metrics_output}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nRank distribution:")
    print(f"  Rank 4 (Cabinet): {(metrics_df['highest_ministerial_rank'] == 4).sum()}")
    print(f"  Rank 3 (Minister of State): {(metrics_df['highest_ministerial_rank'] == 3).sum()}")
    print(f"  Rank 2 (Under-Secretary): {(metrics_df['highest_ministerial_rank'] == 2).sum()}")
    print(f"  Rank 1 (PPS): {(metrics_df['highest_ministerial_rank'] == 1).sum()}")

    ministers = metrics_df[metrics_df['ever_minister'] == 1]
    if len(ministers) > 0:
        print(f"\nMinisterial statistics:")
        print(f"  Mean minister years: {ministers['total_minister_years'].mean():.1f}")
        print(f"  Max minister years: {ministers['total_minister_years'].max():.1f}")
        print(f"  Mean portfolios: {ministers['portfolio_count'].mean():.1f}")

    print(f"\nCareer stagnation:")
    print(f"  MPs with stagnation score > 0.3: {(metrics_df['career_stagnation_score'] > 0.3).sum()}")

    # Spot checks
    print("\n" + "=" * 80)
    print("SPOT CHECKS (Known Ministers)")
    print("=" * 80)

    known_names = ['Rishi Sunak', 'Suella Braverman', 'Kemi Badenoch', 'Jeremy Hunt', 'Robert Jenrick']
    for check_name in known_names:
        matches = metrics_df[metrics_df['name'].str.contains(check_name, case=False, na=False)]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f"\n{row['name']}:")
            print(f"  Ever minister: {row['ever_minister']}")
            print(f"  Total minister years: {row['total_minister_years']:.1f}")
            print(f"  Highest rank: {row['highest_ministerial_rank']}")
            print(f"  Most recent position: {row['most_recent_position']}")
            print(f"  Years since last role: {row['years_since_last_ministerial_role']}")
        else:
            print(f"\n{check_name}: NOT FOUND")

    return metrics_df


if __name__ == "__main__":
    import numpy as np
    results = main()
