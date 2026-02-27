"""
Fetch Current Ministerial Data from Parliament API
===================================================

Fetches government positions and ministerial roles for current/recent MPs
from the official UK Parliament Members API.
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"

# Parliament API endpoints
MEMBERS_API = "https://members-api.parliament.uk/api"
MEMBERS_SEARCH = f"{MEMBERS_API}/Members/Search"

# =============================================================================
# FETCH GOVERNMENT POSITIONS
# =============================================================================

def fetch_all_conservative_mps():
    """Fetch all Conservative MPs (current and recent)."""

    print("Fetching Conservative MPs from Parliament API...")

    # Get all MPs, then filter to Conservative
    params = {
        'House': 'Commons',
        'skip': 0,
        'take': 20  # API limit per request
    }

    all_mps = []

    while True:
        try:
            response = requests.get(MEMBERS_SEARCH, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get('items', [])
            if not items:
                break

            # Filter to Conservative party
            conservative_items = [
                mp for mp in items
                if mp.get('latestParty', {}).get('name') == 'Conservative'
            ]

            all_mps.extend(conservative_items)

            print(f"Fetched {params['skip'] + len(items)} MPs so far... ({len(all_mps)} Conservative)")

            # Check if we've reached the end
            if len(items) < params['take']:
                break

            params['skip'] += params['take']
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"Error fetching MPs: {e}")
            break

    print(f"\nTotal Conservative MPs found: {len(all_mps)}")
    return all_mps


def fetch_government_positions(member_id):
    """Fetch government positions for a specific member."""

    url = f"{MEMBERS_API}/Members/{member_id}/GovernmentPosts"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('value', [])
    except Exception as e:
        # Not all MPs have government positions
        return []


def fetch_opposition_positions(member_id):
    """Fetch opposition positions for a specific member."""

    url = f"{MEMBERS_API}/Members/{member_id}/OppositionPosts"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('value', [])
    except Exception as e:
        return []


# =============================================================================
# PARSE MINISTERIAL POSITIONS
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


def parse_ministerial_history(mps_data):
    """Parse ministerial history for all MPs."""

    print("\nFetching government positions for each MP...")

    ministerial_records = []

    for idx, mp in enumerate(mps_data, 1):
        member_id = mp['value']['id']
        name = mp['value']['nameDisplayAs']

        if idx % 10 == 0:
            print(f"Processing {idx}/{len(mps_data)}: {name}")

        # Fetch government positions
        gov_positions = fetch_government_positions(member_id)

        for pos in gov_positions:
            position_name = pos.get('name', '')
            rank = get_ministerial_rank(position_name)

            # Only count actual ministerial positions (not shadow)
            if rank > 0:
                start_date = pos.get('startDate')
                end_date = pos.get('endDate')

                ministerial_records.append({
                    'member_id': member_id,
                    'name': name,
                    'position': position_name,
                    'rank': rank,
                    'start_date': start_date,
                    'end_date': end_date,
                    'is_current': end_date is None or end_date == ''
                })

        time.sleep(0.2)  # Rate limiting

    df = pd.DataFrame(ministerial_records)
    print(f"\nFound {len(df)} ministerial appointments for {df['name'].nunique()} MPs")

    return df


# =============================================================================
# CALCULATE CAREER METRICS
# =============================================================================

def calculate_ministerial_metrics(ministerial_df, all_mps):
    """Calculate ministerial career metrics for each MP."""

    print("\nCalculating ministerial career metrics...")

    # Get all unique Conservative MPs
    all_mp_names = [mp['value']['nameDisplayAs'] for mp in all_mps]

    mp_metrics = []

    for mp_name in all_mp_names:
        mp_positions = ministerial_df[ministerial_df['name'] == mp_name]

        if len(mp_positions) == 0:
            # Never held ministerial role
            metrics = {
                'name': mp_name,
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
                'name': mp_name,
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

        mp_metrics.append(metrics)

    metrics_df = pd.DataFrame(mp_metrics)

    print(f"\nProcessed {len(metrics_df)} Conservative MPs:")
    print(f"  - {metrics_df['ever_minister'].sum()} held ministerial roles")
    print(f"  - {metrics_df['current_minister'].sum()} currently in ministerial roles")
    print(f"  - {(metrics_df['ever_minister'] == 0).sum()} never held ministerial roles")

    return metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("FETCHING CURRENT MINISTERIAL DATA FROM PARLIAMENT API")
    print("=" * 80)
    print()

    # Fetch all Conservative MPs
    mps_data = fetch_all_conservative_mps()

    if not mps_data:
        print("ERROR: No MPs fetched")
        return None

    # Parse ministerial history
    ministerial_df = parse_ministerial_history(mps_data)

    # Calculate metrics
    metrics_df = calculate_ministerial_metrics(ministerial_df, mps_data)

    # Save outputs
    ministerial_output = ANALYSIS_OUTPUT / "ministerial_positions_current.csv"
    ministerial_df.to_csv(ministerial_output, index=False)
    print(f"\nSaved detailed ministerial positions to: {ministerial_output}")

    metrics_output = ANALYSIS_OUTPUT / "mp_ministerial_metrics_current.csv"
    metrics_df.to_csv(metrics_output, index=False)
    print(f"Saved ministerial metrics to: {metrics_output}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nRank distribution:")
    print(f"  Rank 4 (Cabinet): {(metrics_df['highest_ministerial_rank'] == 4).sum()}")
    print(f"  Rank 3 (Minister of State): {(metrics_df['highest_ministerial_rank'] == 3).sum()}")
    print(f"  Rank 2 (Under-Secretary): {(metrics_df['highest_ministerial_rank'] == 2).sum()}")
    print(f"  Rank 1 (PPS): {(metrics_df['highest_ministerial_rank'] == 1).sum()}")

    print(f"\nMean ministerial years: {metrics_df[metrics_df['ever_minister'] == 1]['total_minister_years'].mean():.1f}")
    print(f"Max ministerial years: {metrics_df['total_minister_years'].max():.1f}")

    # Spot checks
    print("\n" + "=" * 80)
    print("SPOT CHECKS (Known Ministers)")
    print("=" * 80)

    known_ministers = ['Rishi Sunak', 'Suella Braverman', 'Kemi Badenoch', 'Jeremy Hunt']
    for minister_name in known_ministers:
        mp_data = metrics_df[metrics_df['name'].str.contains(minister_name, case=False, na=False)]
        if len(mp_data) > 0:
            row = mp_data.iloc[0]
            print(f"\n{row['name']}:")
            print(f"  Ever minister: {row['ever_minister']}")
            print(f"  Total minister years: {row['total_minister_years']:.1f}")
            print(f"  Highest rank: {row['highest_ministerial_rank']}")
            print(f"  Most recent position: {row['most_recent_position']}")
            print(f"  Currently minister: {row['current_minister']}")
        else:
            print(f"\n{minister_name}: NOT FOUND")

    return metrics_df


if __name__ == "__main__":
    results = main()
