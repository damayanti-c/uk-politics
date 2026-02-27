"""
Fetch MP demographic data (age, date of birth) from Parliament API.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

def calculate_age(date_of_birth: str) -> int:
    """Calculate age from date of birth string."""
    try:
        dob = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except:
        return None

def fetch_mp_details(mp_id: int) -> dict:
    """Fetch detailed information for a single MP."""
    url = f"https://members-api.parliament.uk/api/Members/{mp_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        value = data.get('value', {})

        return {
            'member_id': mp_id,
            'name': value.get('nameDisplayAs'),
            'date_of_birth': value.get('dateOfBirth'),
            'gender': value.get('gender')
        }
    except Exception as e:
        print(f"Error fetching MP {mp_id}: {e}")
        return None

def fetch_all_current_mps() -> pd.DataFrame:
    """Fetch all current MPs with demographic details."""
    # First get list of current MPs
    url = "https://members-api.parliament.uk/api/Members/Search"
    params = {
        'house': 1,  # Commons
        'isCurrentMember': True,
        'skip': 0,
        'take': 20
    }

    all_mps = []

    print("Fetching current MPs list...")

    while True:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        items = data.get('items', [])
        if not items:
            break

        all_mps.extend(items)

        total_results = data.get('totalResults', 0)
        print(f"Fetched {len(all_mps)} of {total_results} MPs")

        if len(all_mps) >= total_results:
            break

        params['skip'] += params['take']
        time.sleep(0.5)  # Be polite to the API

    print(f"\nFetching detailed demographics for {len(all_mps)} MPs...")

    # Now fetch details for each MP
    demographics = []
    for i, mp in enumerate(all_mps, 1):
        mp_id = mp['value']['id']
        details = fetch_mp_details(mp_id)

        if details:
            # Add age calculation
            if details['date_of_birth']:
                details['age'] = calculate_age(details['date_of_birth'])
            else:
                details['age'] = None

            demographics.append(details)

        if i % 50 == 0:
            print(f"Processed {i}/{len(all_mps)} MPs")
            time.sleep(1)  # Longer pause every 50 requests
        else:
            time.sleep(0.2)  # Short pause between requests

    df = pd.DataFrame(demographics)
    print(f"\nSuccessfully retrieved demographics for {len(df)} MPs")

    return df

def save_mp_demographics():
    """Fetch and save MP demographics to CSV."""
    df = fetch_all_current_mps()

    if not df.empty:
        output_path = Path(__file__).parent / 'source_data' / 'mp_demographics.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nSaved demographics to: {output_path}")

        # Print summary statistics
        print(f"\nAge statistics:")
        print(f"Average age: {df['age'].mean():.1f}")
        print(f"Median age: {df['age'].median():.1f}")
        print(f"Youngest: {df['age'].min()} years")
        print(f"Oldest: {df['age'].max()} years")

        print(f"\nGender breakdown:")
        print(df['gender'].value_counts())

        return df
    else:
        print("No demographics retrieved")
        return None

if __name__ == '__main__':
    save_mp_demographics()
