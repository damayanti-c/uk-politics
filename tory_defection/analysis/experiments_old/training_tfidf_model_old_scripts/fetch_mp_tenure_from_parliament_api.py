"""
Fetch MP Tenure Data from UK Parliament API
============================================

Uses the UK Parliament Members API to get accurate membership start dates
for calculating actual years as MP and backbench years.

API Documentation: https://members-api.parliament.uk/index.html
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent  # tory_defection/
ANALYSIS_OUTPUT = Path(__file__).parent  # training_tfidf_model_final_spec/

TRAINING_POPULATION_CSV = ANALYSIS_OUTPUT / "training_population_2019_2024.csv"
PREDICTION_POPULATION_CSV = ANALYSIS_OUTPUT / "prediction_population_2024_present.csv"
OUTPUT_CSV = ANALYSIS_OUTPUT / "mp_tenure_data.csv"

# =============================================================================
# UK PARLIAMENT API
# =============================================================================

API_BASE = "https://members-api.parliament.uk/api/Members"

def search_member_by_name(name):
    """Search for a member by name and return their details."""

    url = f"{API_BASE}/Search"
    params = {
        'Name': name,
        'House': 'Commons',
        'take': 5
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('items'):
            return data['items']
        return []
    except Exception as e:
        print(f"  Error searching for {name}: {e}")
        return []


def get_member_details(member_id):
    """Get detailed member information including membership history."""

    url = f"{API_BASE}/{member_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error getting member {member_id}: {e}")
        return None


def parse_membership_dates(member_data):
    """Extract membership start date and calculate years as MP."""

    if not member_data or 'value' not in member_data:
        return None, None

    value = member_data['value']

    # Get membership start date
    latest_membership = value.get('latestHouseMembership', {})
    start_date_str = latest_membership.get('membershipStartDate')

    if not start_date_str:
        return None, None

    try:
        # Parse date (format: 2019-12-12T00:00:00)
        start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        years_as_mp = (datetime.now() - start_date.replace(tzinfo=None)).days / 365.25
        return start_date, round(years_as_mp, 2)
    except Exception as e:
        print(f"  Error parsing date {start_date_str}: {e}")
        return None, None


def fetch_tenure_for_mps(mp_names):
    """Fetch tenure data for a list of MP names."""

    results = []
    total = len(mp_names)

    for i, name in enumerate(mp_names):
        if i % 20 == 0:
            print(f"  Processing {i}/{total}...")

        # Search for the member
        search_results = search_member_by_name(name)

        if not search_results:
            results.append({
                'name': name,
                'member_id': None,
                'membership_start_date': None,
                'years_as_mp': None,
                'found': False
            })
            continue

        # Use first result (most relevant match)
        member_info = search_results[0]
        member_id = member_info['value']['id']

        # Get full details
        member_data = get_member_details(member_id)

        if member_data:
            start_date, years_as_mp = parse_membership_dates(member_data)

            results.append({
                'name': name,
                'member_id': member_id,
                'api_name': member_info['value'].get('nameDisplayAs', name),
                'membership_start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
                'years_as_mp': years_as_mp,
                'found': True
            })
        else:
            results.append({
                'name': name,
                'member_id': member_id,
                'membership_start_date': None,
                'years_as_mp': None,
                'found': False
            })

        # Rate limiting
        time.sleep(0.1)

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("FETCHING MP TENURE DATA FROM UK PARLIAMENT API")
    print("=" * 80)
    print()

    # Load MP populations
    print("Loading Conservative MP populations...")

    training_df = pd.read_csv(TRAINING_POPULATION_CSV)
    training_names = set(training_df['name'].values)
    print(f"  Training population: {len(training_names)}")

    prediction_df = pd.read_csv(PREDICTION_POPULATION_CSV)
    prediction_names = set(prediction_df['name'].values)
    print(f"  Prediction population: {len(prediction_names)}")

    all_names = list(training_names | prediction_names)
    print(f"  Total unique MPs: {len(all_names)}")

    # Fetch tenure data
    print("\nFetching tenure data from Parliament API...")
    tenure_df = fetch_tenure_for_mps(all_names)

    # Summary
    found_count = tenure_df['found'].sum()
    print(f"\n  Found data for {found_count}/{len(all_names)} MPs")

    # Save output
    tenure_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved tenure data to: {OUTPUT_CSV}")

    # Statistics
    valid_years = tenure_df[tenure_df['years_as_mp'].notna()]['years_as_mp']
    if len(valid_years) > 0:
        print("\n" + "=" * 80)
        print("TENURE STATISTICS")
        print("=" * 80)
        print(f"\n  Mean years as MP: {valid_years.mean():.1f}")
        print(f"  Median years as MP: {valid_years.median():.1f}")
        print(f"  Min: {valid_years.min():.1f}")
        print(f"  Max: {valid_years.max():.1f}")

    return tenure_df


if __name__ == "__main__":
    results = main()
