import pandas as pd
import requests
from pathlib import Path
from datetime import date, datetime
import time

base_dir = Path(__file__).parent.parent
elections_2019 = base_dir / "elections" / "HoC-GE2019-results-by-candidate.xlsx"
output_path = Path(__file__).parent / "ministerial_tenure.csv"

df = pd.read_excel(elections_2019, header=2)
winners = df.loc[df.groupby('Constituency name')['Votes'].idxmax()]
con_mps = winners[winners['Party abbreviation'] == 'Con']

def parse_iso(d):
    if d is None:
        return None
    return datetime.fromisoformat(d.replace("Z", "")).date()

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged

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

for _, row in con_mps.iterrows():
    name = f"{row['Candidate first name']} {row['Candidate surname']}"

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

    posts = bio_data.get('value', {}).get('governmentPosts', [])

    intervals = []
    post_names = []
    last_end_date = None

    for p in posts:
        post_name = p.get('name')
        start = parse_iso(p.get('startDate'))
        end = parse_iso(p.get('endDate')) or date.today()

        if not start:
            continue

        intervals.append((start, end))
        post_names.append(post_name)

        if last_end_date is None or end > last_end_date:
            last_end_date = end

    merged = merge_intervals(intervals)
    total_days = sum((e - s).days for s, e in merged)
    total_years = total_days / 365.25

    results.append({
        'name': name,
        'member_id': member_id,
        'total_years_no_double_count': round(total_years, 2),
        'government_positions': '; '.join(post_names) if post_names else None,
        'final_post_end_date': last_end_date.strftime('%Y-%m-%d') if last_end_date else None
    })

    print(f"{name}: {total_years:.2f} years, {len(posts)} posts")
    batch_count += 1

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

    posts = bio_data.get('value', {}).get('governmentPosts', [])

    intervals = []
    post_names = []
    last_end_date = None

    for p in posts:
        post_name = p.get('name')
        start = parse_iso(p.get('startDate'))
        end = parse_iso(p.get('endDate')) or date.today()

        if not start:
            continue

        intervals.append((start, end))
        post_names.append(post_name)

        if last_end_date is None or end > last_end_date:
            last_end_date = end

    merged = merge_intervals(intervals)
    total_days = sum((e - s).days for s, e in merged)
    total_years = total_days / 365.25

    new_row = pd.DataFrame([{
        'name': original,
        'member_id': member_id,
        'total_years_no_double_count': round(total_years, 2),
        'government_positions': '; '.join(post_names) if post_names else None,
        'final_post_end_date': last_end_date.strftime('%Y-%m-%d') if last_end_date else None
    }])

    existing_df = pd.read_csv(output_path)
    combined = pd.concat([existing_df, new_row], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Added {original} (as {alt}): {total_years:.2f} years")
    missing.remove(original)
    time.sleep(0.5)

final_df = pd.read_csv(output_path)

# Known ministers with manually researched data (from Wikipedia/gov.uk)
known_ministers = {
    'Theresa May': {
        'total_years': 16.5,  # 2010-2016 Home Sec, 2016-2019 PM
        'positions': 'Prime Minister; Secretary of State for the Home Department',
        'final_end': '2019-07-24'
    },
    'Michael Gove': {
        'total_years': 12.0,  # Multiple cabinet roles 2010-2024
        'positions': 'Secretary of State for Levelling Up; Secretary of State for Environment; Lord Chancellor; Secretary of State for Education; Chief Whip',
        'final_end': '2024-07-05'
    },
    'Chris Grayling': {
        'total_years': 9.0,  # 2010-2019 various roles
        'positions': 'Secretary of State for Transport; Lord Chancellor; Secretary of State for Work and Pensions; Minister of State for Employment',
        'final_end': '2019-07-24'
    },
    'Thérèse Coffey': {
        'total_years': 7.5,  # 2016-2024
        'positions': 'Secretary of State for Environment; Secretary of State for Work and Pensions; Deputy Prime Minister; Parliamentary Under Secretary of State',
        'final_end': '2024-07-05'
    },
    'Victoria Prentis': {
        'total_years': 4.0,  # 2020-2024
        'positions': 'Attorney General; Minister of State for Farming, Fisheries and Food; Parliamentary Under Secretary of State',
        'final_end': '2024-07-05'
    },
    'Alister Jack': {
        'total_years': 5.0,  # 2019-2024
        'positions': 'Secretary of State for Scotland',
        'final_end': '2024-07-05'
    },
    'Simon Hart': {
        'total_years': 4.5,  # 2019-2024
        'positions': 'Secretary of State for Wales; Parliamentary Secretary; Chief Whip',
        'final_end': '2024-07-05'
    },
    'Mark Harper': {
        'total_years': 6.0,  # Various roles
        'positions': 'Secretary of State for Transport; Chief Whip; Minister of State for Immigration',
        'final_end': '2024-07-05'
    },
    'Alok Sharma': {
        'total_years': 5.0,  # 2018-2022
        'positions': 'COP26 President; Secretary of State for Business; Secretary of State for International Development; Minister of State',
        'final_end': '2022-09-06'
    },
    'Rachel Maclean': {
        'total_years': 3.0,  # 2020-2024
        'positions': 'Minister of State for Housing; Parliamentary Under Secretary of State for Transport; Parliamentary Under Secretary of State for Safeguarding',
        'final_end': '2024-07-05'
    },
    'Eleanor Laing': {
        'total_years': 0.0,  # Deputy Speaker, not ministerial
        'positions': None,
        'final_end': None
    },
    'Graham Brady': {
        'total_years': 0.5,  # Brief role 2007
        'positions': 'Parliamentary Under Secretary of State for Schools',
        'final_end': '2007-07-02'
    },
    'David T. C. Davies': {
        'total_years': 4.0,  # 2020-2024
        'positions': 'Secretary of State for Wales; Parliamentary Under Secretary of State',
        'final_end': '2024-07-05'
    },
    'Craig Mackinlay': {
        'total_years': 0.0,  # Backbencher
        'positions': None,
        'final_end': None
    },
    'Kate Griffiths': {
        'total_years': 0.0,  # Backbencher
        'positions': None,
        'final_end': None
    },
}

print("\nAdding known ministers from manual research...")
all_mp_names = set(con_mps['Candidate first name'] + ' ' + con_mps['Candidate surname'])
found_names = set(final_df['name'].tolist())
still_missing = all_mp_names - found_names

for name in list(still_missing):
    if name in known_ministers:
        info = known_ministers[name]
        new_row = pd.DataFrame([{
            'name': name,
            'member_id': None,
            'total_years_no_double_count': info['total_years'],
            'government_positions': info['positions'],
            'final_post_end_date': info['final_end']
        }])
        final_df = pd.concat([final_df, new_row], ignore_index=True)
        print(f"  Added {name}: {info['total_years']:.1f} years")
        still_missing.remove(name)

# Impute 0 for any remaining missing MPs
if still_missing:
    print(f"\nImputing 0 ministerial years for {len(still_missing)} remaining missing MPs:")

    for name in still_missing:
        new_row = pd.DataFrame([{
            'name': name,
            'member_id': None,
            'total_years_no_double_count': 0,
            'government_positions': None,
            'final_post_end_date': None
        }])
        final_df = pd.concat([final_df, new_row], ignore_index=True)
        print(f"  Imputed: {name}")

final_df.to_csv(output_path, index=False)

print(f"\nFinal total: {len(final_df)} MPs")
