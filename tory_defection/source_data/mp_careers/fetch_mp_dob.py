import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

base_dir = Path(__file__).parent.parent
elections_2019 = base_dir / "elections" / "HoC-GE2019-results-by-candidate.xlsx"
output_path = Path(__file__).parent / "mp_dobs_ages.csv"

df = pd.read_excel(elections_2019, header=2)
winners = df.loc[df.groupby('Constituency name')['Votes'].idxmax()]
con_mps = winners[winners['Party abbreviation'] == 'Con']
mp_names = (con_mps['Candidate first name'] + ' ' + con_mps['Candidate surname']).tolist()

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
    'David T. C. Davies': 'David Davies',
    'Stephen Barclay': 'Steve Barclay',
    'Thérèse Coffey': 'Therese Coffey',
    'Jacqueline Doyle-Price': 'Jackie Doyle-Price',
    'Thomas Tugendhat': 'Tom Tugendhat',
    'Gregory Clark': 'Greg Clark',
    'Matthew Hancock': 'Matt Hancock',
    'Stephen Brine': 'Steve Brine',
    'Robert Halfon': 'Rob Halfon',
    'Geoffrey Cox': 'Geoffrey Cox QC',
    'Robert Buckland': 'Robert Buckland QC',
    'Christopher Pincher': 'Chris Pincher',
    'Nicholas Gibb': 'Nick Gibb',
    'Edward Argar': 'Edward Argar',
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

# Manual research for missing/invalid MPs (from Wikipedia)
manual_dobs = {
    # Missing from Wikidata
    'Jamie Wallis': '1988-05-07',
    'Robert Neill': '1952-06-24',
    'Kate Griffiths': '1972-01-01',  # Year only available
    'Elizabeth Truss': '1975-07-26',
    'William Cash': '1940-05-10',
    'Nusrat Ghani': '1972-09-01',
    # Invalid ages (wrong person matched)
    'David Jones': '1952-03-22',  # MP for Clwyd West
    'Henry Smith': '1969-05-14',  # MP for Crawley
    'Philip Dunne': '1958-08-14',  # MP for Ludlow
    'David Morris': '1966-01-03',  # MP for Morecambe and Lunesdale
    'Anthony Browne': '1967-06-11',  # MP for South Cambridgeshire
}

for name, dob in manual_dobs.items():
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
