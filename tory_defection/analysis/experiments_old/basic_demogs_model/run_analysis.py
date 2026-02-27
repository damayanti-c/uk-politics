"""
Tory MP Defection Risk Analysis
===============================

This script runs the full analysis using free data sources only:
- House of Commons Library 2024 election results
- ParlParse people.json (MP career data)
- Parliament Members API (current MPs)
- Best for Britain defection tracker (scraped)

No TheyWorkForYou API required.
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
    print("Note: beautifulsoup4 not installed - defection tracker scraping disabled")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: scikit-learn not installed - model training disabled")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
SOURCE_DATA = BASE_DIR / "source_data"
ELECTIONS_DIR = SOURCE_DATA / "elections_2024"
MP_CAREERS_DIR = SOURCE_DATA / "mp_careers"
DEFECTION_DIR = SOURCE_DATA / "defection_tracker"


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_election_results() -> pd.DataFrame:
    """Load 2024 election results from Commons Library CSV."""
    csv_path = ELECTIONS_DIR / "HoC-GE2024-results-by-constituency.csv"

    if not csv_path.exists():
        print(f"ERROR: Election results not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Calculate vote shares
    df['total_votes'] = df['Valid votes']
    df['con_pct'] = (df['Con'] / df['total_votes'] * 100).round(2)
    df['lab_pct'] = (df['Lab'] / df['total_votes'] * 100).round(2)
    df['reform_pct'] = (df['RUK'] / df['total_votes'] * 100).round(2)
    df['majority_pct'] = (df['Majority'] / df['total_votes'] * 100).round(2)

    print(f"Loaded {len(df)} constituencies from 2024 election results")
    return df


def load_parlparse_people() -> dict:
    """Load ParlParse people.json with full MP career data."""
    json_path = MP_CAREERS_DIR / "people.json"

    if not json_path.exists():
        print(f"ERROR: people.json not found at {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded ParlParse data with {len(data.get('memberships', []))} membership records")
    return data


def load_ministers() -> dict:
    """Load ministerial appointments from ParlParse."""
    json_path = MP_CAREERS_DIR / "ministers.json"

    if not json_path.exists():
        print(f"Warning: ministers.json not found")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded ministerial data")
    return data


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
                    "thumbnail": member.get("thumbnailUrl"),
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


def scrape_defection_tracker() -> pd.DataFrame:
    """Scrape Best for Britain defection tracker."""
    if not HAS_BS4:
        print("BeautifulSoup not available - using manual defector list")
        return get_known_defectors()

    url = "https://www.bestforbritain.org/tory_reformuk_defection_tracker"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find tables with defection data
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            if len(rows) > 1:
                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                data = []
                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if cells:
                        data.append(cells)

                if data and len(data) > 5:  # Likely the right table
                    df = pd.DataFrame(data)
                    if len(headers) == len(df.columns):
                        df.columns = headers
                    print(f"Scraped {len(df)} defection records from Best for Britain")
                    return df

        print("Could not parse defection tracker - using manual list")
        return get_known_defectors()

    except Exception as e:
        print(f"Error scraping defection tracker: {e}")
        return get_known_defectors()


def get_known_defectors() -> pd.DataFrame:
    """Manual list of known Tory-to-Reform defectors (MPs and former MPs)."""
    # Based on public reporting as of January 2026
    defectors = [
        # Sitting/Former MPs who defected
        {"name": "Lee Anderson", "type": "MP", "defection_date": "2024-03-11", "constituency": "Ashfield"},
        {"name": "Danny Kruger", "type": "MP", "defection_date": "2025-09-01", "constituency": "East Wiltshire"},
        {"name": "Marco Longhi", "type": "Former MP", "defection_date": "2025", "constituency": "Dudley North"},
        {"name": "Andrea Jenkyns", "type": "Former MP", "defection_date": "2024", "constituency": "Morley and Outwood"},
        {"name": "Lucy Allan", "type": "Former MP", "defection_date": "2024", "constituency": "Telford"},
        {"name": "Mark Reckless", "type": "Former MP", "defection_date": "2024", "constituency": "Rochester and Strood"},
        {"name": "David Jones", "type": "Former MP", "defection_date": "2024", "constituency": "Clwyd West"},
        {"name": "Nadine Dorries", "type": "Former MP", "defection_date": "2025", "constituency": "Mid Bedfordshire"},
        {"name": "Jonathan Gullis", "type": "Former MP", "defection_date": "2025", "constituency": "Stoke-on-Trent North"},
        {"name": "Nadhim Zahawi", "type": "Former MP", "defection_date": "2026-01", "constituency": "Stratford-on-Avon"},
        {"name": "Jake Berry", "type": "Former MP", "defection_date": "2025", "constituency": "Rossendale and Darwen"},
        {"name": "Adam Holloway", "type": "Former MP", "defection_date": "2025", "constituency": "Gravesham"},
        {"name": "Anne Marie Morris", "type": "Former MP", "defection_date": "2025", "constituency": "Newton Abbot"},
        {"name": "Maria Caulfield", "type": "Former MP", "defection_date": "2025", "constituency": "Lewes"},
        {"name": "Sarah Atherton", "type": "Former MP", "defection_date": "2025", "constituency": "Wrexham"},
        {"name": "Lia Nici", "type": "Former MP", "defection_date": "2025", "constituency": "Great Grimsby"},
        {"name": "Chris Green", "type": "Former MP", "defection_date": "2025", "constituency": "Bolton West"},
        {"name": "Ross Thomson", "type": "Former MP", "defection_date": "2025", "constituency": "Aberdeen South"},
        {"name": "Aidan Burley", "type": "Former MP", "defection_date": "2024", "constituency": "Cannock Chase"},
    ]

    df = pd.DataFrame(defectors)
    print(f"Using manual list of {len(df)} known defectors")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def extract_career_features(mps_df: pd.DataFrame, parlparse_data: dict) -> pd.DataFrame:
    """Extract career features from Parliament API and ParlParse data."""
    current_year = datetime.now().year
    features = []

    # Build lookup from ParlParse memberships
    memberships = parlparse_data.get('memberships', [])
    persons = parlparse_data.get('persons', [])

    # Create person lookup
    person_lookup = {}
    for person in persons:
        pid = person.get('id', '')
        person_lookup[pid] = person

    # Create membership history by person
    membership_by_person = {}
    for m in memberships:
        pid = m.get('person_id', '')
        if pid not in membership_by_person:
            membership_by_person[pid] = []
        membership_by_person[pid].append(m)

    for _, row in mps_df.iterrows():
        name = row.get('name', '')

        # Calculate years as MP from Parliament API
        start_date = row.get('start_date')
        if start_date:
            try:
                if isinstance(start_date, str):
                    start_year = int(start_date[:4])
                else:
                    start_year = start_date.year
                years_as_mp = current_year - start_year
            except (ValueError, TypeError):
                years_as_mp = None
        else:
            years_as_mp = None

        # Try to find additional info from ParlParse
        # (Would need name matching - simplified for now)

        features.append({
            'name': name,
            'party': row.get('party'),
            'constituency': row.get('constituency'),
            'parliament_id': row.get('parliament_id'),
            'gender': row.get('gender'),
            'years_as_mp': years_as_mp,
            'first_elected_year': start_year if start_date else None,
        })

    return pd.DataFrame(features)


def extract_constituency_features(mps_df: pd.DataFrame, election_df: pd.DataFrame) -> pd.DataFrame:
    """Extract constituency-level features from election results."""

    # Filter to Conservative-held seats
    con_seats = election_df[election_df['First party'] == 'Con'].copy()

    # Merge with MP data
    merged = mps_df.merge(
        con_seats[['Constituency name', 'Majority', 'majority_pct', 'con_pct',
                   'reform_pct', 'lab_pct', 'total_votes', 'Electorate']],
        left_on='constituency',
        right_on='Constituency name',
        how='left'
    )

    # Also get Reform performance in non-Tory held seats (for former MPs)
    all_constituencies = election_df[['Constituency name', 'Majority', 'majority_pct',
                                       'con_pct', 'reform_pct', 'lab_pct', 'First party']].copy()

    return merged


def create_defection_risk_features(mps_with_features: pd.DataFrame, defectors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically designed to predict defection risk.

    Key risk factors based on research:
    1. Small majority (vulnerable seat)
    2. High Reform vote share in constituency
    3. Longer tenure (potentially frustrated with career progression)
    4. Male MPs (83% of defectors are male per YouGov)
    5. Previously held ministerial roles but no longer (passed over)
    """

    df = mps_with_features.copy()

    # Create defector flag for training
    defector_names = set(defectors_df['name'].str.lower().str.strip())
    df['is_defector'] = df['name'].str.lower().str.strip().isin(defector_names).astype(int)

    # Feature: Seat vulnerability (smaller majority = higher risk)
    df['majority_risk'] = df['majority_pct'].apply(
        lambda x: 1 if pd.isna(x) else max(0, (20 - x) / 20)  # Risk increases as majority decreases
    )

    # Feature: Reform UK presence (higher Reform vote = higher risk of MP seeing opportunity)
    df['reform_presence_risk'] = df['reform_pct'].apply(
        lambda x: 0 if pd.isna(x) else min(1, x / 30)  # Normalize to 0-1, cap at 30%
    )

    # Feature: Gender risk (male MPs more likely to defect)
    df['gender_risk'] = df['gender'].apply(lambda x: 0.7 if x == 'Male' else 0.3)

    # Feature: Tenure risk (longer tenure could mean frustration)
    df['tenure_risk'] = df['years_as_mp'].apply(
        lambda x: 0 if pd.isna(x) else min(1, x / 25)  # Risk increases with tenure, caps at 25 years
    )

    # Feature: Conservative vote collapse (low Con % despite holding seat)
    df['con_weakness_risk'] = df['con_pct'].apply(
        lambda x: 0 if pd.isna(x) else max(0, (40 - x) / 40)  # Risk if Con vote under 40%
    )

    return df


# =============================================================================
# MODEL TRAINING AND PREDICTION
# =============================================================================

def train_defection_model(df: pd.DataFrame) -> tuple:
    """Train a model to predict defection risk."""

    if not HAS_SKLEARN:
        print("scikit-learn not available - using rule-based scoring")
        return None, None

    # Feature columns
    feature_cols = ['majority_risk', 'reform_presence_risk', 'gender_risk',
                    'tenure_risk', 'con_weakness_risk']

    # Only use rows with complete data
    df_model = df.dropna(subset=feature_cols + ['is_defector'])

    if len(df_model) < 10:
        print(f"Insufficient data for model training ({len(df_model)} rows)")
        return None, None

    X = df_model[feature_cols]
    y = df_model['is_defector']

    print(f"\nTraining data: {len(X)} MPs, {y.sum()} defectors")

    if y.sum() < 2:
        print("Not enough defectors in current MP data for proper model training")
        print("Using rule-based scoring instead")
        return None, feature_cols

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=min(5, int(y.sum())), scoring='roc_auc')
    print(f"Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Fit final model
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return model, feature_cols


def calculate_risk_scores(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    """Calculate defection risk scores for all MPs."""

    df = df.copy()

    if model is not None and HAS_SKLEARN:
        # Use model predictions
        X = df[feature_cols].fillna(0)
        df['risk_score'] = model.predict_proba(X)[:, 1]
    else:
        # Rule-based scoring (weighted average of risk factors)
        weights = {
            'reform_presence_risk': 0.30,  # Reform doing well in constituency
            'majority_risk': 0.25,         # Small majority
            'con_weakness_risk': 0.20,     # Low Conservative vote share
            'gender_risk': 0.15,           # Male MPs more likely
            'tenure_risk': 0.10,           # Longer tenure
        }

        df['risk_score'] = sum(
            df[col].fillna(0) * weight
            for col, weight in weights.items()
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

def generate_report(df: pd.DataFrame, output_path: Path = None) -> str:
    """Generate a comprehensive defection risk report."""

    # Filter to Conservative MPs only
    tory_mps = df[df['party'] == 'Conservative'].copy()

    lines = [
        "=" * 70,
        "TORY MP DEFECTION RISK REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "METHODOLOGY",
        "-" * 40,
        "Risk factors considered:",
        "  - Reform UK vote share in constituency (30% weight)",
        "  - Majority size vulnerability (25% weight)",
        "  - Conservative vote weakness (20% weight)",
        "  - Gender (male MPs historically more likely) (15% weight)",
        "  - Parliamentary tenure (10% weight)",
        "",
        "Data sources:",
        "  - House of Commons Library 2024 election results",
        "  - Parliament Members API (current MPs)",
        "  - ParlParse (career history)",
        "  - Best for Britain defection tracker (ground truth)",
        "",
        "=" * 70,
        "",
        "SUMMARY STATISTICS",
        "-" * 40,
        f"Total Conservative MPs analyzed: {len(tory_mps)}",
    ]

    if 'risk_category' in tory_mps.columns:
        risk_counts = tory_mps['risk_category'].value_counts()
        lines.extend([
            "",
            "Risk distribution:",
        ])
        for cat in ['Very High', 'High', 'Medium', 'Low']:
            count = risk_counts.get(cat, 0)
            pct = count / len(tory_mps) * 100
            lines.append(f"  {cat}: {count} MPs ({pct:.1f}%)")

    # Top 20 at risk
    lines.extend([
        "",
        "=" * 70,
        "",
        "TOP 20 MPs AT HIGHEST DEFECTION RISK",
        "-" * 40,
    ])

    top_20 = tory_mps.nlargest(20, 'risk_score')

    for i, (_, row) in enumerate(top_20.iterrows(), 1):
        name = row.get('name', 'Unknown')
        constituency = row.get('constituency', 'Unknown')
        score = row['risk_score']
        reform_pct = row.get('reform_pct', 0)
        majority_pct = row.get('majority_pct', 0)

        lines.append(f"\n{i:2}. {name}")
        lines.append(f"    Constituency: {constituency}")
        lines.append(f"    Risk Score: {score:.3f} ({row.get('risk_category', 'N/A')})")
        lines.append(f"    Reform UK vote: {reform_pct:.1f}% | Majority: {majority_pct:.1f}%")

    # Key risk factors for top MPs
    lines.extend([
        "",
        "=" * 70,
        "",
        "KEY INSIGHTS",
        "-" * 40,
    ])

    # Constituencies where Reform did best
    if 'reform_pct' in tory_mps.columns:
        high_reform = tory_mps.nlargest(5, 'reform_pct')
        lines.append("\nConstituencies with highest Reform UK vote (held by Tories):")
        for _, row in high_reform.iterrows():
            lines.append(f"  - {row['constituency']}: {row['reform_pct']:.1f}% Reform ({row['name']})")

    # Most vulnerable majorities
    if 'majority_pct' in tory_mps.columns:
        vulnerable = tory_mps.nsmallest(5, 'majority_pct')
        lines.append("\nMost vulnerable Tory majorities:")
        for _, row in vulnerable.iterrows():
            lines.append(f"  - {row['constituency']}: {row['majority_pct']:.1f}% majority ({row['name']})")

    lines.extend([
        "",
        "=" * 70,
        "",
        "NOTES AND LIMITATIONS",
        "-" * 40,
        "1. This analysis uses constituency-level data as proxy for MP views",
        "2. Speech/voting data not included (would require TheyWorkForYou API)",
        "3. Private negotiations between MPs and Reform UK not observable",
        "4. Model trained on small sample of actual defectors",
        "5. Political events can rapidly change defection calculus",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
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
    print("=" * 70)
    print("TORY MP DEFECTION RISK ANALYSIS")
    print("(Using free data sources only - no TheyWorkForYou API)")
    print("=" * 70)
    print()

    # 1. Load all data sources
    print("STEP 1: Loading data sources")
    print("-" * 40)

    election_df = load_election_results()
    parlparse_data = load_parlparse_people()
    ministers_data = load_ministers()
    current_mps = fetch_current_mps_from_api()
    defectors_df = scrape_defection_tracker()

    if current_mps.empty:
        print("ERROR: Could not fetch current MPs")
        return

    # 2. Filter to Conservative MPs
    print("\n" + "=" * 70)
    print("STEP 2: Filtering to Conservative MPs")
    print("-" * 40)

    tory_mps = current_mps[current_mps['party'] == 'Conservative'].copy()
    print(f"Found {len(tory_mps)} Conservative MPs")

    # 3. Extract features
    print("\n" + "=" * 70)
    print("STEP 3: Extracting features")
    print("-" * 40)

    career_features = extract_career_features(tory_mps, parlparse_data)
    mps_with_constituency = extract_constituency_features(career_features, election_df)
    mps_with_risk = create_defection_risk_features(mps_with_constituency, defectors_df)

    print(f"Created {len(mps_with_risk)} MP profiles with features")

    # 4. Train model / calculate scores
    print("\n" + "=" * 70)
    print("STEP 4: Calculating defection risk scores")
    print("-" * 40)

    model, feature_cols = train_defection_model(mps_with_risk)

    if feature_cols:
        mps_scored = calculate_risk_scores(mps_with_risk, model, feature_cols)
    else:
        mps_scored = mps_with_risk
        mps_scored['risk_score'] = 0.5  # Default

    # 5. Generate report
    print("\n" + "=" * 70)
    print("STEP 5: Generating report")
    print("-" * 40)

    report_path = BASE_DIR / "defection_risk_report.txt"
    report = generate_report(mps_scored, report_path)

    # Also save data
    csv_path = BASE_DIR / "mp_defection_risk_scores.csv"
    tory_scored = mps_scored[mps_scored['party'] == 'Conservative'].copy()
    tory_scored = tory_scored.sort_values('risk_score', ascending=False)
    tory_scored.to_csv(csv_path, index=False)
    print(f"Risk scores saved to: {csv_path}")

    # Print summary
    print("\n" + report)

    return mps_scored


if __name__ == "__main__":
    results = main()
