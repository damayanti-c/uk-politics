import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time
import os

# pull list of 2019-2024 Conservative MPs
base_dir = Path(__file__).parent.parent
elections_2024 = base_dir / "elections" / "MPs-elected.xlsx"
output_path = Path(__file__).parent / "sitting_mp_tenure.csv"

df = pd.read_excel(elections_2024)
con_mps = df[df['party_abbreviation'] == 'Con']

# Load existing results if file exists
if output_path.exists():
    existing_df = pd.read_csv(output_path)
    already_done = set(existing_df['name'].tolist())
    print(f"Already have {len(already_done)} MPs")
else:
    already_done = set()

results = []
missing = []
batch_count = 0

# loop through MPs and fetch tenure data

for _, row in con_mps.iterrows():
    name = f"{row['firstname']} {row['surname']}"

    if name in already_done:
        continue

    for attempt in range(3):
        try:
            search_url = f'https://members-api.parliament.uk/api/Members/Search?Name={name}&take=5'
            resp = requests.get(search_url, timeout=10)
            data = resp.json()
            items = data.get('items', [])
            break
        except:
            time.sleep(2)
    else:
        print(f"Failed after retries: {name}")
        missing.append(name)
        continue

    if not items:
        print(f"No results for {name}")
        missing.append(name)
        continue

    member_id = items[0]['value']['id']

    for attempt in range(3):
        try:
            bio_url = f'https://members-api.parliament.uk/api/Members/{member_id}/Biography'
            bio_resp = requests.get(bio_url, timeout=10)
            bio_data = bio_resp.json()
            break
        except:
            time.sleep(2)
    else:
        print(f"Failed to get bio: {name}")
        missing.append(name)
        continue

    house_memberships = bio_data.get('value', {}).get('houseMemberships', [])

    total_days = 0
    first_start = None
    last_end = None

    for membership in house_memberships:
        if membership.get('name') == 'Commons':
            start = membership.get('startDate')
            end = membership.get('endDate')
            if start:
                start_date = datetime.fromisoformat(start)
                end_date = datetime.fromisoformat(end) if end else datetime.now()
                total_days += (end_date - start_date).days

                if first_start is None or start_date < first_start:
                    first_start = start_date
                if last_end is None or end_date > last_end:
                    last_end = end_date

    total_years = total_days / 365.25

    results.append({
        'name': name,
        'member_id': member_id,
        'membership_start': first_start.strftime('%Y-%m-%d') if first_start else None,
        'membership_end': last_end.strftime('%Y-%m-%d') if last_end else None,
        'total_years_as_mp': round(total_years, 2)
    })

    print(f"{name}: {total_years:.2f} years")
    batch_count += 1

    # Save every 10 MPs
    if batch_count >= 10:
        new_df = pd.DataFrame(results)
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(output_path, index=False)
        print(f"Saved batch of {len(results)} MPs (total: {len(combined)})")
        results = []
        batch_count = 0
        time.sleep(2)

    time.sleep(0.5)

# Save any remaining
if results:
    new_df = pd.DataFrame(results)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)
    print(f"Saved final batch of {len(results)} MPs (total: {len(combined)})")

final_df = pd.read_csv(output_path)
print(f"\nTotal saved: {len(final_df)} MPs")
print(f"Missing {len(missing)} MPs: {missing}")


# Try alternative names for missing MPs
alt_names = {
    'David T. C. Davies': 'David Davies',
    'Stephen Barclay': 'Steve Barclay',
    'Thérèse Coffey': 'Therese Coffey',
    'Jacqueline Doyle-Price': 'Jackie Doyle-Price',
    'Thomas Tugendhat': 'Tom Tugendhat',
    'Gregory Clark': 'Greg Clark',
    'Matthew Hancock': 'Matt Hancock',
    'Stephen Brine': 'Steve Brine',
}

for original, alt in alt_names.items():
    if original not in missing:
        continue

    try:
        search_url = f'https://members-api.parliament.uk/api/Members/Search?Name={alt}&take=5'
        resp = requests.get(search_url, timeout=10)
        data = resp.json()
        items = data.get('items', [])
    except:
        print(f"Failed: {original}")
        continue

    if not items:
        print(f"No results for {original} (tried {alt})")
        continue

    member_id = items[0]['value']['id']

    try:
        bio_url = f'https://members-api.parliament.uk/api/Members/{member_id}/Biography'
        bio_resp = requests.get(bio_url, timeout=10)
        bio_data = bio_resp.json()
    except:
        print(f"Failed to get bio: {original}")
        continue

    house_memberships = bio_data.get('value', {}).get('houseMemberships', [])

    total_days = 0
    first_start = None
    last_end = None

    for membership in house_memberships:
        if membership.get('name') == 'Commons':
            start = membership.get('startDate')
            end = membership.get('endDate')
            if start:
                start_date = datetime.fromisoformat(start)
                end_date = datetime.fromisoformat(end) if end else datetime.now()
                total_days += (end_date - start_date).days

                if first_start is None or start_date < first_start:
                    first_start = start_date
                if last_end is None or end_date > last_end:
                    last_end = end_date

    total_years = total_days / 365.25

    new_row = pd.DataFrame([{
        'name': original,
        'member_id': member_id,
        'membership_start': first_start.strftime('%Y-%m-%d') if first_start else None,
        'membership_end': last_end.strftime('%Y-%m-%d') if last_end else None,
        'total_years_as_mp': round(total_years, 2)
    }])

    existing_df = pd.read_csv(output_path)
    combined = pd.concat([existing_df, new_row], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Added {original} (as {alt}): {total_years:.2f} years")
    missing.remove(original)
    time.sleep(0.5)

final_df = pd.read_csv(output_path)

# Impute mean tenure for remaining missing MPs
all_mp_names = set(con_mps['firstname'] + ' ' + con_mps['surname'])
found_names = set(final_df['name'].tolist())
still_missing = all_mp_names - found_names

if still_missing:
    mean_tenure = final_df['total_years_as_mp'].mean()
    print(f"\nImputing mean tenure ({mean_tenure:.2f} years) for {len(still_missing)} missing MPs:")

    for name in still_missing:
        new_row = pd.DataFrame([{
            'name': name,
            'member_id': None,
            'membership_start': None,
            'membership_end': None,
            'total_years_as_mp': round(mean_tenure, 2)
        }])
        final_df = pd.concat([final_df, new_row], ignore_index=True)
        print(f"  Imputed: {name}")

    final_df.to_csv(output_path, index=False)

print(f"MPs with imputed tenure: {len(still_missing) if still_missing else 0}")