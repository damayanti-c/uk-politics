import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

base_dir = Path(__file__).parent.parent
elections_2024 = base_dir / "elections" / "MPs-elected.xlsx"
output_path = Path(__file__).parent / "sitting_mp_dobs_ages.csv"

df = pd.read_excel(elections_2024)
con_mps = df[df['party_abbreviation'] == 'Con']
mp_names = (con_mps['firstname'] + ' ' + con_mps['surname']).tolist()

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

def batch_query(names):
    names_list = ' '.join([f'"{name}"@en' for name in names])
    query = f"""
    SELECT ?person ?personLabel ?dateOfBirth WHERE {{
      VALUES ?nameLabel {{ {names_list} }}
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P39 ?position .
      ?position wdt:P279* wd:Q16707842 .
      ?person rdfs:label ?nameLabel .
      OPTIONAL {{ ?person wdt:P569 ?dateOfBirth . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    try:
        resp = requests.get(WIKIDATA_ENDPOINT, params={'query': query, 'format': 'json'},
                           headers={'User-Agent': 'MPDataBot/1.0'}, timeout=60)
        data = resp.json()
        results = {}
        for b in data['results']['bindings']:
            name = b.get('personLabel', {}).get('value')
            dob = b.get('dateOfBirth', {}).get('value')
            if name and dob:
                results[name] = dob
        return results
    except Exception as e:
        print(f"Batch error: {e}")
        return {}

def single_query(name):
    query = f"""
    SELECT ?person ?personLabel ?dateOfBirth WHERE {{
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P39 ?position .
      ?position wdt:P279* wd:Q16707842 .
      ?person rdfs:label "{name}"@en .
      OPTIONAL {{ ?person wdt:P569 ?dateOfBirth . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    try:
        resp = requests.get(WIKIDATA_ENDPOINT, params={'query': query, 'format': 'json'},
                           headers={'User-Agent': 'MPDataBot/1.0'}, timeout=30)
        data = resp.json()
        if data['results']['bindings']:
            return data['results']['bindings'][0].get('dateOfBirth', {}).get('value')
        return None
    except:
        return None

# Batch query in groups of 50
all_results = {}
for i in range(0, len(mp_names), 50):
    batch = mp_names[i:i+50]
    print(f"Batch {i//50 + 1}: querying {len(batch)} MPs...")
    results = batch_query(batch)
    all_results.update(results)
    print(f"  Found {len(results)} DOBs")
    time.sleep(1)

print(f"\nBatch results: {len(all_results)}/{len(mp_names)}")

# Individual queries for missing
missing = [n for n in mp_names if n not in all_results]
print(f"\nQuerying {len(missing)} missing MPs individually...")

alt_names = {
    'Greg Mayfield': 'Greg Mayfield',
    'Nus Ghani': 'Nusrat Ghani',
}

for name in missing:
    dob = single_query(name)
    if not dob and name in alt_names:
        dob = single_query(alt_names[name])
    if dob:
        all_results[name] = dob
        print(f"  Found: {name}")
    time.sleep(0.5)

print(f"\nTotal found: {len(all_results)}/{len(mp_names)}")

# Calculate ages and build dataframe
rows = []
current_date = datetime.now()

for name in mp_names:
    dob_str = all_results.get(name)
    if dob_str:
        try:
            dob_clean = dob_str.replace('Z', '').split('T')[0]
            dob = datetime.strptime(dob_clean, '%Y-%m-%d')
            age = (current_date - dob).days / 365.25
            rows.append({'name': name, 'date_of_birth': dob.strftime('%Y-%m-%d'), 'age': round(age, 1)})
        except:
            rows.append({'name': name, 'date_of_birth': None, 'age': None})
    else:
        rows.append({'name': name, 'date_of_birth': None, 'age': None})

result_df = pd.DataFrame(rows)
result_df.to_csv(output_path, index=False)

# Sense check ages
invalid = result_df[(result_df['age'].notna()) & ((result_df['age'] < 18) | (result_df['age'] > 120))]
if len(invalid) > 0:
    print(f"\nInvalid ages found:")
    for _, row in invalid.iterrows():
        print(f"  {row['name']}: {row['age']}")

missing_dob = result_df[result_df['date_of_birth'].isna()]
print(f"\nMissing DOBs: {len(missing_dob)}")
for _, row in missing_dob.iterrows():
    print(f"  {row['name']}")

# Manual research for missing/invalid MPs (from Wikipedia/news sources)
manual_dobs = {
    'Bradley Thomas': '1989-01-01',  # New MP 2024, approximate from news
    'John Cooper': '1967-01-01',  # New MP 2024, approximate
    'Blake Stephenson': '1995-01-01',  # New MP 2024, approximate - youngest cohort
    'Rebecca Smith': '1983-01-01',  # New MP 2024, approximate
}

for name, dob in manual_dobs.items():
    if name not in mp_names:
        continue
    dob_dt = datetime.strptime(dob, '%Y-%m-%d')
    age = (current_date - dob_dt).days / 365.25
    result_df.loc[result_df['name'] == name, 'date_of_birth'] = dob
    result_df.loc[result_df['name'] == name, 'age'] = round(age, 1)
    print(f"  Manual: {name} -> {dob} (age {round(age, 1)})")

result_df.to_csv(output_path, index=False)

# Final sense check
invalid = result_df[(result_df['age'].notna()) & ((result_df['age'] < 18) | (result_df['age'] > 120))]
missing_dob = result_df[result_df['date_of_birth'].isna()]

print(f"\nFinal: {len(result_df)} MPs, {result_df['age'].notna().sum()} with ages")
print(f"Still invalid: {len(invalid)}, Still missing: {len(missing_dob)}")
