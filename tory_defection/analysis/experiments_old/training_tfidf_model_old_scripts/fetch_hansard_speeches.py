"""
Fetch and Parse Hansard Speeches from ParlParse
================================================

Downloads debate XML files from TheyWorkForYou's public archive
and extracts speeches for Conservative MPs.

No API key required - uses free public data.
"""

import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from collections import defaultdict
import pandas as pd

BASE_DIR = Path(__file__).parent
HANSARD_DIR = BASE_DIR / "source_data" / "hansard"
DEBATES_URL = "http://www.theyworkforyou.com/pwdata/scrapedxml/debates/"


def get_available_debate_files(year: int = 2025) -> list:
    """Scrape the debates directory to find available XML files for a year."""
    print(f"Fetching list of debate files for {year}...")

    try:
        response = requests.get(DEBATES_URL, timeout=30)
        response.raise_for_status()

        # Find all XML files for the specified year
        pattern = rf'debates{year}-\d{{2}}-\d{{2}}[a-z]\.xml'
        files = re.findall(pattern, response.text)

        # Deduplicate and sort
        files = sorted(set(files))
        print(f"Found {len(files)} debate files for {year}")
        return files

    except requests.RequestException as e:
        print(f"Error fetching file list: {e}")
        return []


def download_debate_file(filename: str) -> Path:
    """Download a single debate XML file."""
    local_path = HANSARD_DIR / filename

    if local_path.exists():
        return local_path

    url = DEBATES_URL + filename
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        return local_path

    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None


def parse_debate_file(filepath: Path) -> list:
    """
    Parse a debate XML file and extract speeches.

    Returns list of dicts with:
    - person_id: PublicWhip person identifier
    - speaker_name: Name as shown
    - text: Speech content
    - date: Date of debate
    - speech_id: Unique speech identifier
    """
    speeches = []

    try:
        # Parse XML - handle entities by replacing them
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract date from filename (e.g., debates2025-12-17b.xml -> 2025-12-17)
        match = re.search(r'debates(\d{4}-\d{2}-\d{2})', filepath.name)
        debate_date = match.group(1) if match else "unknown"

        # Parse XML - use a more lenient approach
        # Remove DOCTYPE which causes issues
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
        content = re.sub(r'<!ENTITY[^>]*>', '', content)

        root = ET.fromstring(content)

        for speech in root.findall('.//speech'):
            person_id = speech.get('person_id', '')
            speaker_name = speech.get('speakername', '')
            speech_id = speech.get('id', '')

            # Skip if no speaker identified
            if not person_id or not speaker_name:
                continue

            # Extract all paragraph text
            paragraphs = []
            for p in speech.findall('.//p'):
                if p.text:
                    paragraphs.append(p.text.strip())
                # Also get tail text after child elements
                for child in p:
                    if child.tail:
                        paragraphs.append(child.tail.strip())

            text = ' '.join(paragraphs)

            if text:
                speeches.append({
                    'person_id': person_id,
                    'speaker_name': speaker_name,
                    'text': text,
                    'date': debate_date,
                    'speech_id': speech_id,
                })

    except Exception as e:
        print(f"Error parsing {filepath.name}: {e}")

    return speeches


def load_mp_person_ids() -> dict:
    """
    Load ParlParse people.json to map person_ids to MP names.
    Returns dict of {person_id: {name, party, constituency, ...}}
    """
    people_path = BASE_DIR / "source_data" / "mp_careers" / "people.json"

    if not people_path.exists():
        print("Warning: people.json not found")
        return {}

    with open(people_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build person lookup
    persons = {}
    for person in data.get('persons', []):
        pid = person.get('id', '')
        names = person.get('other_names', [])
        if names:
            # Get most recent name
            name = names[-1].get('name', '') if names else ''
            persons[pid] = {
                'name': name,
                'person_data': person
            }

    return persons


def fetch_recent_debates(months: int = 6, years: int = None) -> pd.DataFrame:
    """
    Fetch and parse debates from the last N months or N years.
    Returns DataFrame with all speeches.

    Args:
        months: Number of months to fetch (default 6)
        years: If provided, overrides months and fetches N years of data
    """
    HANSARD_DIR.mkdir(parents=True, exist_ok=True)

    all_speeches = []

    current_year = datetime.now().year

    if years:
        # Fetch full years of data
        year_range = range(current_year - years + 1, current_year + 1)
        cutoff_date = None
        print(f"Fetching {years} years of debates ({min(year_range)}-{max(year_range)})...")
    else:
        # Fetch recent months
        year_range = [current_year - 1, current_year]
        cutoff_date = datetime.now() - timedelta(days=months * 30)

    for year in year_range:
        files = get_available_debate_files(year)

        for filename in files:
            # Filter by cutoff date if set
            if cutoff_date:
                match = re.search(r'debates(\d{4})-(\d{2})-(\d{2})', filename)
                if match:
                    file_date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    if file_date < cutoff_date:
                        continue

            # Skip 'a' versions which are usually redirects
            if filename.endswith('a.xml'):
                continue

            print(f"Processing {filename}...")
            local_path = download_debate_file(filename)

            if local_path and local_path.exists():
                speeches = parse_debate_file(local_path)
                all_speeches.extend(speeches)
                print(f"  Extracted {len(speeches)} speeches")

    df = pd.DataFrame(all_speeches)
    print(f"\nTotal speeches extracted: {len(df)}")

    return df


def filter_conservative_speeches(speeches_df: pd.DataFrame, tory_mps_df: pd.DataFrame) -> pd.DataFrame:
    """Filter speeches to only Conservative MPs."""

    # Load people.json to get person_id mappings
    persons = load_mp_person_ids()

    # Try to match by name
    tory_names = set(tory_mps_df['name'].str.lower().str.strip())

    def is_tory_speaker(row):
        speaker = row.get('speaker_name', '').lower().strip()
        # Remove titles
        speaker = re.sub(r'^(sir|dame|mr|mrs|ms|dr)\s+', '', speaker)
        return speaker in tory_names or any(speaker in name for name in tory_names)

    # Filter speeches
    tory_speeches = speeches_df[speeches_df.apply(is_tory_speaker, axis=1)]

    print(f"Filtered to {len(tory_speeches)} Conservative MP speeches")
    return tory_speeches


def analyze_speech_content(speeches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze speech content for Reform-aligned keywords.
    """
    REFORM_KEYWORDS = [
        'immigration', 'migrant', 'migrants', 'border', 'borders',
        'small boat', 'small boats', 'channel crossing',
        'asylum', 'asylum seeker', 'refugee',
        'rwanda', 'deportation', 'deport',
        'illegal migration', 'illegal immigrant',
        'brexit', 'sovereignty', 'european union',
        'net zero', 'climate target',
        'woke', 'free speech', 'cancel culture',
    ]

    results = []

    for speaker, group in speeches_df.groupby('speaker_name'):
        all_text = ' '.join(group['text'].fillna('')).lower()
        total_speeches = len(group)

        # Count keyword mentions
        keyword_counts = {}
        total_keywords = 0
        for keyword in REFORM_KEYWORDS:
            count = all_text.count(keyword)
            if count > 0:
                keyword_counts[keyword] = count
                total_keywords += count

        # Calculate metrics
        immigration_mentions = sum(
            all_text.count(kw) for kw in
            ['immigration', 'migrant', 'border', 'asylum', 'boat', 'rwanda', 'deportation']
        )

        results.append({
            'speaker_name': speaker,
            'total_speeches': total_speeches,
            'total_reform_keywords': total_keywords,
            'keywords_per_speech': total_keywords / max(total_speeches, 1),
            'immigration_mentions': immigration_mentions,
            'immigration_per_speech': immigration_mentions / max(total_speeches, 1),
            'top_keywords': dict(sorted(keyword_counts.items(), key=lambda x: -x[1])[:5]),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("HANSARD SPEECH ANALYSIS FOR TORY DEFECTION MODEL")
    print("(Using free ParlParse data - no API required)")
    print("=" * 70)
    print()

    # 1. Fetch 5 years of debates
    print("STEP 1: Fetching 5 years of debate files (2021-2026)...")
    print("-" * 40)
    speeches_df = fetch_recent_debates(years=5)

    if speeches_df.empty:
        print("No speeches found")
        return

    # Save raw speeches
    speeches_path = HANSARD_DIR / "all_speeches.csv"
    speeches_df.to_csv(speeches_path, index=False)
    print(f"Saved {len(speeches_df)} speeches to {speeches_path}")

    # 2. Load Conservative MPs
    print("\n" + "=" * 70)
    print("STEP 2: Loading Conservative MPs...")
    print("-" * 40)

    mp_scores_path = BASE_DIR / "mp_defection_risk_scores.csv"
    if mp_scores_path.exists():
        tory_mps = pd.read_csv(mp_scores_path)
        print(f"Loaded {len(tory_mps)} Conservative MPs")
    else:
        print("Warning: mp_defection_risk_scores.csv not found")
        print("Run run_analysis.py first to generate MP list")
        return

    # 3. Filter to Conservative speeches
    print("\n" + "=" * 70)
    print("STEP 3: Filtering to Conservative MP speeches...")
    print("-" * 40)
    tory_speeches = filter_conservative_speeches(speeches_df, tory_mps)

    # 4. Analyze speech content
    print("\n" + "=" * 70)
    print("STEP 4: Analyzing speech content for Reform keywords...")
    print("-" * 40)
    speech_analysis = analyze_speech_content(tory_speeches)

    # Save analysis
    analysis_path = BASE_DIR / "speech_analysis.csv"
    speech_analysis.to_csv(analysis_path, index=False)
    print(f"Saved speech analysis to {analysis_path}")

    # 5. Show top MPs by Reform keyword usage
    print("\n" + "=" * 70)
    print("TOP 10 TORY MPs BY REFORM-ALIGNED KEYWORD USAGE")
    print("-" * 40)

    top_reform = speech_analysis.nlargest(10, 'keywords_per_speech')
    for i, (_, row) in enumerate(top_reform.iterrows(), 1):
        print(f"{i:2}. {row['speaker_name']}")
        print(f"    Speeches: {row['total_speeches']}, Keywords/speech: {row['keywords_per_speech']:.2f}")
        print(f"    Immigration mentions: {row['immigration_mentions']}")
        if row['top_keywords']:
            top_kw = ', '.join(f"{k}({v})" for k, v in list(row['top_keywords'].items())[:3])
            print(f"    Top keywords: {top_kw}")
        print()

    # 6. Merge with existing risk scores
    print("=" * 70)
    print("STEP 5: Merging with defection risk model...")
    print("-" * 40)

    # Merge speech features with existing scores
    merged = tory_mps.merge(
        speech_analysis[['speaker_name', 'total_speeches', 'keywords_per_speech', 'immigration_per_speech']],
        left_on='name',
        right_on='speaker_name',
        how='left'
    )

    # Update risk score to include speech data
    merged['speech_risk'] = merged['keywords_per_speech'].fillna(0) / 10  # Normalize
    merged['updated_risk_score'] = (
        merged['risk_score'] * 0.7 +  # Original constituency-based score
        merged['speech_risk'] * 0.3    # Add speech-based score
    )

    # Save updated scores
    updated_path = BASE_DIR / "mp_defection_risk_scores_with_speech.csv"
    merged.to_csv(updated_path, index=False)
    print(f"Saved updated risk scores to {updated_path}")

    # Show updated top 10
    print("\n" + "=" * 70)
    print("UPDATED TOP 10 MPs AT HIGHEST DEFECTION RISK (with speech data)")
    print("-" * 40)

    top_10 = merged.nlargest(10, 'updated_risk_score')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2}. {row['name']}")
        print(f"    Constituency: {row['constituency']}")
        print(f"    Updated Risk: {row['updated_risk_score']:.3f} (was {row['risk_score']:.3f})")
        if pd.notna(row.get('keywords_per_speech')):
            print(f"    Reform keywords/speech: {row['keywords_per_speech']:.2f}")
        print()

    print("=" * 70)
    print("DONE!")


if __name__ == "__main__":
    main()
