"""
Fetch MP Ages from Wikidata
============================

Queries Wikidata for date of birth information for all MPs in the training dataset.
Calculates age and retirement proximity scores.

Uses SPARQL queries to the Wikidata Query Service.
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_OUTPUT = BASE_DIR / "analysis"
SOURCE_DATA = BASE_DIR / "source_data"

CONSERVATIVE_SPEAKERS = ANALYSIS_OUTPUT / "conservative_speakers_in_hansard.csv"
DEMOGRAPHICS_CSV = SOURCE_DATA / "mp_demographics.csv"

# Wikidata Query Service endpoint
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# =============================================================================
# WIKIDATA QUERIES
# =============================================================================

def query_wikidata_for_mp(mp_name):
    """Query Wikidata for MP's date of birth."""

    # SPARQL query to find MP by name and get date of birth
    query = f"""
    SELECT ?person ?personLabel ?dateOfBirth WHERE {{
      ?person wdt:P31 wd:Q5 .                    # instance of human
      ?person wdt:P39 ?position .                # position held
      ?position wdt:P279* wd:Q16707842 .         # member of UK Parliament (or subclass)
      ?person rdfs:label "{mp_name}"@en .        # name matches
      OPTIONAL {{ ?person wdt:P569 ?dateOfBirth . }}  # date of birth
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={'query': query, 'format': 'json'},
            headers={'User-Agent': 'DefectionAnalysisBot/1.0'},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data['results']['bindings']:
            result = data['results']['bindings'][0]
            dob = result.get('dateOfBirth', {}).get('value')
            return dob

        return None

    except Exception as e:
        print(f"  Error querying {mp_name}: {e}")
        return None


def batch_query_wikidata(mp_names):
    """Batch query Wikidata for multiple MPs."""

    # Build SPARQL query for multiple names
    # Use VALUES clause to query multiple MPs at once
    names_list = ' '.join([f'"{name}"@en' for name in mp_names])

    query = f"""
    SELECT ?person ?personLabel ?dateOfBirth WHERE {{
      VALUES ?nameLabel {{ {names_list} }}
      ?person wdt:P31 wd:Q5 .                    # instance of human
      ?person wdt:P39 ?position .                # position held
      ?position wdt:P279* wd:Q16707842 .         # member of UK Parliament
      ?person rdfs:label ?nameLabel .
      OPTIONAL {{ ?person wdt:P569 ?dateOfBirth . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={'query': query, 'format': 'json'},
            headers={'User-Agent': 'DefectionAnalysisBot/1.0'},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        results = {}
        for binding in data['results']['bindings']:
            name = binding.get('personLabel', {}).get('value')
            dob = binding.get('dateOfBirth', {}).get('value')
            if name and dob:
                results[name] = dob

        return results

    except Exception as e:
        print(f"  Batch query error: {e}")
        return {}


# =============================================================================
# FETCH AGES FOR ALL MPS
# =============================================================================

def fetch_all_mp_ages():
    """Fetch date of birth for all Conservative MPs."""

    print("Loading Conservative MP list...")
    conservative_mps = pd.read_csv(CONSERVATIVE_SPEAKERS)
    mp_names = conservative_mps['name'].tolist()
    print(f"  Found {len(mp_names)} MPs to query")

    # Try batch querying in chunks of 50 to avoid timeout
    print("\nQuerying Wikidata in batches...")

    all_results = {}
    batch_size = 50

    for i in range(0, len(mp_names), batch_size):
        batch = mp_names[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(mp_names)-1)//batch_size + 1}: Querying {len(batch)} MPs...")

        batch_results = batch_query_wikidata(batch)
        all_results.update(batch_results)

        print(f"    Found {len(batch_results)} dates of birth in this batch")

        # Rate limiting - be nice to Wikidata
        time.sleep(1)

    print(f"\nTotal dates of birth found: {len(all_results)}/{len(mp_names)} ({len(all_results)/len(mp_names)*100:.1f}%)")

    # For MPs not found in batch query, try individual queries
    missing_mps = [name for name in mp_names if name not in all_results]

    if missing_mps and len(missing_mps) <= 100:
        print(f"\nQuerying {len(missing_mps)} missing MPs individually...")

        for i, mp_name in enumerate(missing_mps):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(missing_mps)}")

            dob = query_wikidata_for_mp(mp_name)
            if dob:
                all_results[mp_name] = dob

            time.sleep(0.5)  # Rate limiting

    print(f"\nFinal results: {len(all_results)}/{len(mp_names)} MPs with dates of birth ({len(all_results)/len(mp_names)*100:.1f}%)")

    return all_results


# =============================================================================
# CALCULATE AGE METRICS
# =============================================================================

def calculate_age_metrics(dob_dict):
    """Calculate age and retirement proximity from dates of birth."""

    print("\nCalculating age metrics...")

    results = []
    current_date = datetime.now()

    for name, dob_str in dob_dict.items():
        try:
            # Parse date of birth (format: 1975-01-15T00:00:00Z)
            # Remove timezone info to make it naive
            dob_str_clean = dob_str.replace('Z', '').split('T')[0]
            dob = datetime.strptime(dob_str_clean, '%Y-%m-%d')

            # Calculate age
            age = (current_date - dob).days / 365.25

            # Calculate retirement proximity (0 at age 55, 1.0 at age 70)
            retirement_proximity = max(0, (age - 55) / 15)
            retirement_proximity = min(1.0, retirement_proximity)

            results.append({
                'name': name,
                'date_of_birth': dob.strftime('%Y-%m-%d'),
                'age': round(age, 1),
                'retirement_proximity_score': round(retirement_proximity, 3)
            })

        except Exception as e:
            print(f"  Error parsing date for {name}: {e}")

    df = pd.DataFrame(results)
    print(f"  Calculated ages for {len(df)} MPs")
    print(f"  Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    print(f"  Mean age: {df['age'].mean():.1f}")

    return df


# =============================================================================
# UPDATE DEMOGRAPHICS CSV
# =============================================================================

def update_demographics_with_ages(age_df):
    """Update demographics CSV with age data from Wikidata."""

    print("\nUpdating demographics CSV...")

    # Load existing demographics
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    print(f"  Loaded {len(demographics)} MPs from demographics CSV")

    # Merge with age data
    demographics = demographics.drop(columns=['date_of_birth', 'age'], errors='ignore')
    demographics = demographics.merge(
        age_df[['name', 'date_of_birth', 'age']],
        on='name',
        how='left'
    )

    # Save updated demographics
    demographics.to_csv(DEMOGRAPHICS_CSV, index=False)
    print(f"  Updated demographics CSV with {age_df['age'].notna().sum()} ages")

    # Save age data separately
    age_output = ANALYSIS_OUTPUT / "mp_ages_from_wikidata.csv"
    age_df.to_csv(age_output, index=False)
    print(f"  Saved age data to: {age_output}")

    return demographics


# =============================================================================
# UPDATE CAREER FEATURES
# =============================================================================

def update_career_features_with_ages(age_df):
    """Update career features CSV with age data."""

    print("\nUpdating career features CSV...")

    career_features_path = ANALYSIS_OUTPUT / "mp_career_features.csv"

    if not career_features_path.exists():
        print("  Career features CSV not found, skipping")
        return None

    career = pd.read_csv(career_features_path)
    print(f"  Loaded {len(career)} MPs from career features CSV")

    # Create clean names for matching
    def clean_name(name):
        if pd.isna(name):
            return name
        for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
            name = name.replace(title, '')
        return name.strip()

    career['name_clean'] = career['name'].apply(clean_name)
    age_df['name_clean'] = age_df['name'].apply(clean_name)

    # Drop old age columns
    career = career.drop(columns=['date_of_birth', 'age', 'retirement_proximity_score'], errors='ignore')

    # Merge with new age data
    career = career.merge(
        age_df[['name_clean', 'date_of_birth', 'age', 'retirement_proximity_score']],
        on='name_clean',
        how='left'
    )

    # Recalculate career stagnation with real ages
    career['career_stagnation_score'] = 0.0

    stagnation_mask = (
        (career['ever_minister'] == 1) &
        (career['years_since_last_ministerial_role'] > 2) &
        (career['retirement_proximity_score'].notna())
    )

    career.loc[stagnation_mask, 'career_stagnation_score'] = (
        (career.loc[stagnation_mask, 'years_since_last_ministerial_role'] / 10).clip(upper=1) *
        (1 - career.loc[stagnation_mask, 'retirement_proximity_score'])
    )

    # Recalculate backbench years with real ages
    career['estimated_total_years_as_mp'] = career['age'].fillna(55) - 30
    career['backbench_years'] = (
        career['estimated_total_years_as_mp'] - career['total_minister_years']
    ).clip(lower=0)

    # Save updated career features
    career.to_csv(career_features_path, index=False)
    print(f"  Updated career features with {age_df['age'].notna().sum()} ages")
    print(f"  Recalculated retirement proximity and career stagnation scores")

    return career


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("FETCHING MP AGES FROM WIKIDATA")
    print("=" * 80)
    print()

    # Fetch dates of birth from Wikidata
    dob_dict = fetch_all_mp_ages()

    if not dob_dict:
        print("\nNo dates of birth found. Exiting.")
        return None

    # Calculate age metrics
    age_df = calculate_age_metrics(dob_dict)

    # Update demographics CSV
    demographics = update_demographics_with_ages(age_df)

    # Update career features CSV
    career = update_career_features_with_ages(age_df)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nSuccessfully fetched {len(age_df)} dates of birth from Wikidata")
    print(f"Age statistics:")
    print(f"  Mean: {age_df['age'].mean():.1f} years")
    print(f"  Median: {age_df['age'].median():.1f} years")
    print(f"  Range: {age_df['age'].min():.1f} - {age_df['age'].max():.1f} years")

    print(f"\nRetirement proximity:")
    near_retirement = (age_df['retirement_proximity_score'] > 0.5).sum()
    print(f"  MPs near retirement (score > 0.5): {near_retirement} ({near_retirement/len(age_df)*100:.1f}%)")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE MPs WITH AGES")
    print("=" * 80)

    sample = age_df.sample(min(20, len(age_df))).sort_values('age', ascending=False)
    for _, row in sample.iterrows():
        print(f"  {row['name']:<30} | Age: {row['age']:.0f} | DOB: {row['date_of_birth']} | Retirement: {row['retirement_proximity_score']:.2f}")

    return age_df


if __name__ == "__main__":
    results = main()
