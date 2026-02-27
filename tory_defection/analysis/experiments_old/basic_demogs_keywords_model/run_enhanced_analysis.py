"""
Enhanced Tory MP Defection Risk Analysis
=========================================

Updated model with weighted features:
- Speech Analysis (35%): Reform keywords, immigration mentions
- Voting Rebellion (30%): Overall rebellion rate, Rwanda Bill votes
- Constituency Risk (25%): Reform vote share, majority size
- Demographics (10%): Gender, tenure, age

Data sources:
- House of Commons Library 2024 election results
- ParlParse Hansard speeches (5 years)
- Public Whip rebellion rates
- Rwanda Bill division votes
- Parliament Members API
"""

import json
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Optional imports
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent  # Go up one level from analysis/ folder
SOURCE_DATA = BASE_DIR / "source_data"
HANSARD_DATA = SOURCE_DATA / "hansard"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"  # Output folder for results


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_election_results() -> pd.DataFrame:
    """Load 2024 election results from Commons Library CSV."""
    csv_path = SOURCE_DATA / "elections_2024" / "HoC-GE2024-results-by-constituency.csv"

    if not csv_path.exists():
        print(f"ERROR: Election results not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Calculate vote shares
    df['total_votes'] = df['Valid votes']
    df['con_pct'] = (df['Con'] / df['total_votes'] * 100).round(2)
    df['reform_pct'] = (df['RUK'] / df['total_votes'] * 100).round(2)
    df['majority_pct'] = (df['Majority'] / df['total_votes'] * 100).round(2)

    print(f"Loaded {len(df)} constituencies from 2024 election results")
    return df


def load_speech_analysis() -> pd.DataFrame:
    """Load Hansard speech analysis with Reform keyword counts."""
    csv_path = ANALYSIS_OUTPUT / "speech_analysis.csv"

    if not csv_path.exists():
        print(f"Warning: Speech analysis not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded speech analysis for {len(df)} MPs")
    return df


def load_rebellion_data() -> pd.DataFrame:
    """Load MP rebellion rates from Public Whip."""
    csv_path = SOURCE_DATA / "voting_records" / "rebellion_rates.csv"

    if not csv_path.exists():
        print(f"Warning: Rebellion data not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded rebellion data for {len(df)} MPs")
    return df


def load_rwanda_votes() -> pd.DataFrame:
    """Load Rwanda Bill voting records."""
    csv_path = SOURCE_DATA / "voting_records" / "rwanda_bill_votes.csv"

    if not csv_path.exists():
        print(f"Warning: Rwanda Bill votes not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded Rwanda Bill votes for {len(df)} MPs")
    return df


def load_mp_demographics() -> pd.DataFrame:
    """Load MP demographics (age, date of birth)."""
    csv_path = SOURCE_DATA / "mp_demographics.csv"

    if not csv_path.exists():
        print(f"Warning: Demographics not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded demographics for {len(df)} MPs")
    return df


def fetch_current_mps_from_api() -> pd.DataFrame:
    """Fetch current MPs from Parliament API."""
    print("Fetching current MPs from Parliament API...")

    url = "https://members-api.parliament.uk/api/Members/Search?House=Commons&IsCurrentMember=true"

    all_members = []
    skip = 0
    take = 20

    try:
        while True:
            response = requests.get(f"{url}&skip={skip}&take={take}", timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                member = item.get("value", {})
                all_members.append({
                    "parliament_id": member.get("id"),
                    "name": member.get("nameDisplayAs"),
                    "party": member.get("latestParty", {}).get("name"),
                    "constituency": member.get("latestHouseMembership", {}).get("membershipFrom"),
                    "start_date": member.get("latestHouseMembership", {}).get("membershipStartDate"),
                    "gender": member.get("gender"),
                })

            skip += take
            if skip >= data.get("totalResults", 0):
                break

        df = pd.DataFrame(all_members)
        print(f"Fetched {len(df)} current MPs from Parliament API")
        return df

    except requests.RequestException as e:
        print(f"Error fetching from Parliament API: {e}")
        return pd.DataFrame()


def get_known_defectors() -> pd.DataFrame:
    """Manual list of known Tory-to-Reform defectors."""
    defectors = [
        {"name": "Lee Anderson", "defection_date": "2024-03-11"},
        {"name": "Danny Kruger", "defection_date": "2025-09-01"},
    ]

    df = pd.DataFrame(defectors)
    print(f"Using manual list of {len(df)} known defectors")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_enhanced_risk_features(mps_df: pd.DataFrame,
                                   election_df: pd.DataFrame,
                                   speech_df: pd.DataFrame,
                                   rebellion_df: pd.DataFrame,
                                   rwanda_df: pd.DataFrame,
                                   demographics_df: pd.DataFrame,
                                   defectors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive risk features with new weighting:
    - Speech Analysis: 35%
    - Voting Rebellion: 30%
    - Constituency Risk: 25%
    - Demographics: 10%
    """

    df = mps_df.copy()
    current_year = datetime.now().year

    # Calculate years as MP
    df['years_as_mp'] = df['start_date'].apply(lambda x:
        current_year - int(x[:4]) if pd.notna(x) and len(str(x)) >= 4 else None
    )

    # Create defector flag
    defector_names = set(defectors_df['name'].str.lower().str.strip())
    df['is_defector'] = df['name'].str.lower().str.strip().isin(defector_names).astype(int)

    # ==========================================================================
    # CATEGORY 1: CONSTITUENCY RISK FEATURES (25% total weight)
    # ==========================================================================

    # Merge election data
    con_seats = election_df[election_df['First party'] == 'Con'].copy()
    df = df.merge(
        con_seats[['Constituency name', 'Majority', 'majority_pct', 'con_pct', 'reform_pct']],
        left_on='constituency',
        right_on='Constituency name',
        how='left'
    )

    # Feature: Reform UK presence (0-1 scale)
    df['reform_presence_risk'] = df['reform_pct'].apply(
        lambda x: 0 if pd.isna(x) else min(1, x / 30)
    )

    # Feature: Majority vulnerability (0-1 scale)
    df['majority_risk'] = df['majority_pct'].apply(
        lambda x: 1 if pd.isna(x) else max(0, (20 - x) / 20)
    )

    # Feature: Conservative weakness (0-1 scale)
    df['con_weakness_risk'] = df['con_pct'].apply(
        lambda x: 0 if pd.isna(x) else max(0, (40 - x) / 40)
    )

    # ==========================================================================
    # CATEGORY 2: SPEECH ANALYSIS FEATURES (35% total weight)
    # ==========================================================================

    if not speech_df.empty:
        # Merge speech analysis
        df = df.merge(
            speech_df[['speaker_name', 'total_speeches', 'keywords_per_speech',
                       'immigration_mentions', 'total_reform_keywords']],
            left_on='name',
            right_on='speaker_name',
            how='left'
        )

        # Feature: Reform keywords intensity (normalized 0-1)
        max_keywords = speech_df['keywords_per_speech'].max()
        df['speech_reform_intensity'] = df['keywords_per_speech'].apply(
            lambda x: 0 if pd.isna(x) or max_keywords == 0 else min(1, x / max_keywords)
        )

        # Feature: Immigration focus (normalized 0-1)
        max_immigration = speech_df['immigration_mentions'].max()
        df['speech_immigration_focus'] = df['immigration_mentions'].apply(
            lambda x: 0 if pd.isna(x) or max_immigration == 0 else min(1, x / max_immigration)
        )

        # Feature: Speech volume (normalized 0-1)
        max_speeches = speech_df['total_speeches'].max()
        df['speech_volume'] = df['total_speeches'].apply(
            lambda x: 0 if pd.isna(x) or max_speeches == 0 else min(1, x / max_speeches)
        )
    else:
        df['speech_reform_intensity'] = 0
        df['speech_immigration_focus'] = 0
        df['speech_volume'] = 0

    # ==========================================================================
    # CATEGORY 3: VOTING REBELLION FEATURES (30% total weight)
    # ==========================================================================

    if not rebellion_df.empty:
        # Merge rebellion data
        df = df.merge(
            rebellion_df[['mp_name', 'rebellion_rate']],
            left_on='name',
            right_on='mp_name',
            how='left'
        )

        # Feature: Overall rebellion rate (normalized 0-1)
        df['rebellion_risk'] = df['rebellion_rate'].apply(
            lambda x: 0 if pd.isna(x) else min(1, x / 10)  # Cap at 10% rebellion
        )
    else:
        df['rebellion_risk'] = 0

    if not rwanda_df.empty:
        # Merge Rwanda Bill votes
        df = df.merge(
            rwanda_df[['mp_name', 'vote_type']],
            left_on='name',
            right_on='mp_name',
            how='left',
            suffixes=('', '_rwanda')
        )

        # Feature: Rwanda Bill rebellion (binary)
        df['rwanda_rebellion'] = df['vote_type'].apply(
            lambda x: 1 if x == 'rebellion' else 0
        )
    else:
        df['rwanda_rebellion'] = 0

    # ==========================================================================
    # CATEGORY 4: DEMOGRAPHIC FEATURES (10% total weight)
    # ==========================================================================

    # Feature: Gender risk
    df['gender_risk'] = df['gender'].apply(lambda x: 0.7 if x == 'Male' else 0.3)

    # Feature: Tenure risk
    df['tenure_risk'] = df['years_as_mp'].apply(
        lambda x: 0 if pd.isna(x) else min(1, x / 25)
    )

    # Merge demographics for age
    if not demographics_df.empty:
        df = df.merge(
            demographics_df[['name', 'age']],
            left_on='name',
            right_on='name',
            how='left',
            suffixes=('', '_demo')
        )

        # Feature: Age risk (younger MPs may be more willing to switch)
        df['age_risk'] = df['age'].apply(
            lambda x: 0.5 if pd.isna(x) else max(0, (50 - x) / 50) if x < 50 else 0.3
        )
    else:
        df['age_risk'] = 0.5  # Default neutral

    return df


def calculate_enhanced_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall risk score with new weights:
    - Speech Analysis (35%): Reform keywords, immigration focus, volume
    - Voting Rebellion (30%): Overall rebellion + Rwanda Bill
    - Constituency Risk (25%): Reform presence, majority, Con weakness
    - Demographics (10%): Gender, tenure, age
    """

    df = df.copy()

    # SPEECH ANALYSIS (35% total)
    df['speech_score'] = (
        df['speech_reform_intensity'].fillna(0) * 0.50 +  # 17.5% of total
        df['speech_immigration_focus'].fillna(0) * 0.35 + # 12.25% of total
        df['speech_volume'].fillna(0) * 0.15              # 5.25% of total
    ) * 0.35

    # VOTING REBELLION (30% total)
    df['voting_score'] = (
        df['rebellion_risk'].fillna(0) * 0.70 +           # 21% of total
        df['rwanda_rebellion'].fillna(0) * 0.30           # 9% of total
    ) * 0.30

    # CONSTITUENCY RISK (25% total)
    df['constituency_score'] = (
        df['reform_presence_risk'].fillna(0) * 0.50 +     # 12.5% of total
        df['majority_risk'].fillna(0) * 0.30 +            # 7.5% of total
        df['con_weakness_risk'].fillna(0) * 0.20          # 5% of total
    ) * 0.25

    # DEMOGRAPHICS (10% total)
    df['demographics_score'] = (
        df['gender_risk'].fillna(0.5) * 0.50 +            # 5% of total
        df['tenure_risk'].fillna(0) * 0.30 +              # 3% of total
        df['age_risk'].fillna(0.5) * 0.20                 # 2% of total
    ) * 0.10

    # OVERALL RISK SCORE
    df['risk_score'] = (
        df['speech_score'] +
        df['voting_score'] +
        df['constituency_score'] +
        df['demographics_score']
    )

    # Categorize risk
    df['risk_category'] = pd.cut(
        df['risk_score'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    return df


# =============================================================================
# REPORTING
# =============================================================================

def generate_enhanced_report(df: pd.DataFrame, output_path: Path = None) -> str:
    """Generate comprehensive risk report with new features."""

    tory_mps = df[df['party'] == 'Conservative'].copy()

    lines = [
        "=" * 80,
        "ENHANCED TORY MP DEFECTION RISK REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 80,
        "",
        "METHODOLOGY",
        "-" * 80,
        "Risk Score Composition (updated model):",
        "",
        "  1. SPEECH ANALYSIS (35% weight)",
        "     - Reform keyword intensity per speech (50% of category = 17.5% total)",
        "     - Immigration focus mentions (35% of category = 12.25% total)",
        "     - Speech volume/activity (15% of category = 5.25% total)",
        "     Data: 5 years of Hansard speeches (199,701 speeches analyzed)",
        "",
        "  2. VOTING REBELLION (30% weight)",
        "     - Overall rebellion rate vs party (70% of category = 21% total)",
        "     - Rwanda Bill rebellion/abstention (30% of category = 9% total)",
        "     Data: Public Whip rebellion rates, Rwanda Bill divisions",
        "",
        "  3. CONSTITUENCY RISK (25% weight)",
        "     - Reform UK vote share 2024 (50% of category = 12.5% total)",
        "     - Majority size vulnerability (30% of category = 7.5% total)",
        "     - Conservative vote weakness (20% of category = 5% total)",
        "     Data: House of Commons Library 2024 election results",
        "",
        "  4. DEMOGRAPHICS (10% weight)",
        "     - Gender (male MPs historically more likely) (50% of category = 5% total)",
        "     - Parliamentary tenure (30% of category = 3% total)",
        "     - Age (younger MPs may be more willing) (20% of category = 2% total)",
        "     Data: Parliament Members API",
        "",
        "=" * 80,
        "",
        "SUMMARY STATISTICS",
        "-" * 80,
        f"Total Conservative MPs analyzed: {len(tory_mps)}",
    ]

    if 'risk_category' in tory_mps.columns:
        risk_counts = tory_mps['risk_category'].value_counts()
        lines.extend(["", "Risk distribution:"])
        for cat in ['Very High', 'High', 'Medium', 'Low']:
            count = risk_counts.get(cat, 0)
            pct = count / len(tory_mps) * 100 if len(tory_mps) > 0 else 0
            lines.append(f"  {cat}: {count} MPs ({pct:.1f}%)")

    # Top 25 at risk
    lines.extend([
        "",
        "=" * 80,
        "",
        "TOP 25 MPS AT HIGHEST DEFECTION RISK",
        "-" * 80,
    ])

    top_25 = tory_mps.nlargest(25, 'risk_score')

    for i, (_, row) in enumerate(top_25.iterrows(), 1):
        name = row.get('name', 'Unknown')
        constituency = row.get('constituency', 'Unknown')
        score = row['risk_score']

        lines.append(f"\n{i:2}. {name} ({constituency})")
        lines.append(f"    Overall Risk Score: {score:.3f} ({row.get('risk_category', 'N/A')})")
        lines.append(f"    Breakdown:")
        lines.append(f"      - Speech Analysis: {row.get('speech_score', 0):.3f} (35% weight)")
        lines.append(f"      - Voting Rebellion: {row.get('voting_score', 0):.3f} (30% weight)")
        lines.append(f"      - Constituency Risk: {row.get('constituency_score', 0):.3f} (25% weight)")
        lines.append(f"      - Demographics: {row.get('demographics_score', 0):.3f} (10% weight)")

        # Additional context
        reform_pct = row.get('reform_pct', 0)
        rebellion_rate = row.get('rebellion_rate', 0)
        keywords_per_speech = row.get('keywords_per_speech', 0)

        lines.append(f"    Key Metrics:")
        if pd.notna(keywords_per_speech):
            lines.append(f"      - Reform keywords per speech: {keywords_per_speech:.2f}")
        if pd.notna(rebellion_rate):
            lines.append(f"      - Rebellion rate: {rebellion_rate:.1f}%")
        if pd.notna(reform_pct):
            lines.append(f"      - Constituency Reform vote: {reform_pct:.1f}%")

    lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("ENHANCED TORY MP DEFECTION RISK ANALYSIS")
    print("=" * 80)
    print()

    # Load all data sources
    print("STEP 1: Loading data sources")
    print("-" * 80)

    election_df = load_election_results()
    speech_df = load_speech_analysis()
    rebellion_df = load_rebellion_data()
    rwanda_df = load_rwanda_votes()
    demographics_df = load_mp_demographics()
    current_mps = fetch_current_mps_from_api()
    defectors_df = get_known_defectors()

    if current_mps.empty:
        print("ERROR: Could not fetch current MPs")
        return

    # Filter to Conservative MPs
    print("\n" + "=" * 80)
    print("STEP 2: Filtering to Conservative MPs")
    print("-" * 80)

    tory_mps = current_mps[current_mps['party'] == 'Conservative'].copy()
    print(f"Found {len(tory_mps)} Conservative MPs")

    # Create enhanced features
    print("\n" + "=" * 80)
    print("STEP 3: Creating enhanced risk features")
    print("-" * 80)

    mps_with_features = create_enhanced_risk_features(
        tory_mps, election_df, speech_df, rebellion_df,
        rwanda_df, demographics_df, defectors_df
    )

    print(f"Created {len(mps_with_features)} MP profiles with enhanced features")

    # Calculate risk scores
    print("\n" + "=" * 80)
    print("STEP 4: Calculating enhanced risk scores")
    print("-" * 80)

    mps_scored = calculate_enhanced_risk_scores(mps_with_features)

    # Generate report
    print("\n" + "=" * 80)
    print("STEP 5: Generating enhanced report")
    print("-" * 80)

    report_path = ANALYSIS_OUTPUT / "enhanced_defection_risk_report.txt"
    report = generate_enhanced_report(mps_scored, report_path)

    # Save data
    csv_path = ANALYSIS_OUTPUT / "mp_enhanced_risk_scores.csv"
    tory_scored = mps_scored[mps_scored['party'] == 'Conservative'].copy()
    tory_scored = tory_scored.sort_values('risk_score', ascending=False)

    # Select key columns for CSV
    output_cols = [
        'name', 'constituency', 'party', 'gender', 'years_as_mp',
        'risk_score', 'risk_category',
        'speech_score', 'voting_score', 'constituency_score', 'demographics_score',
        'keywords_per_speech', 'immigration_mentions', 'total_speeches',
        'rebellion_rate', 'rwanda_rebellion',
        'reform_pct', 'majority_pct', 'con_pct'
    ]

    # Only include columns that exist
    output_cols = [col for col in output_cols if col in tory_scored.columns]

    tory_scored[output_cols].to_csv(csv_path, index=False)
    print(f"Risk scores saved to: {csv_path}")

    # Print summary
    print("\n" + report)

    return mps_scored


if __name__ == "__main__":
    results = main()
