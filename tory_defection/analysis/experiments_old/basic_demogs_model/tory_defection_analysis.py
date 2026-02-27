"""
Tory MP Defection Risk Prediction Model
========================================

This script builds a prediction model to assess which Conservative MPs are at risk
of defecting to Reform UK, combining speech/voting data with parliamentary career
characteristics.

Author: SHGH Economics Team
Date: January 2026
"""

import os
import json
import requests
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Optional imports - install as needed
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# MODULE 1: DATA SOURCES CONFIGURATION
# =============================================================================

@dataclass
class DataSource:
    """Configuration for a single data source."""
    name: str
    url: str
    description: str
    format: str  # 'api', 'csv', 'xml', 'mysql', 'scrape'
    coverage: str
    access_type: str  # 'free', 'api_key', 'subscription'
    api_key_env: Optional[str] = None
    rate_limit: Optional[str] = None


# All available data sources for this analysis
DATA_SOURCES: Dict[str, DataSource] = {
    # --- Speech & Statement Data ---
    "parlparse_hansard": DataSource(
        name="MySociety ParlParse Hansard",
        url="https://data.mysociety.org/datasets/uk-hansard/",
        description="XML files of debates in Commons/Lords from 1918. Speeches labelled with unique IDs.",
        format="xml",
        coverage="1918-present",
        access_type="free",
    ),
    "theyworkforyou_api": DataSource(
        name="TheyWorkForYou API",
        url="https://www.theyworkforyou.com/api/",
        description="Hansard data, MP info, and voting records via JSON API.",
        format="api",
        coverage="1997-present",
        access_type="api_key",
        api_key_env="THEYWORKFORYOU_API_KEY",
        rate_limit="1000 calls/month (free tier)",
    ),
    "historic_hansard": DataSource(
        name="Historic Hansard API",
        url="https://api.parliament.uk/historic-hansard/api",
        description="REST API for historical Hansard 1803-2005.",
        format="api",
        coverage="1803-2005",
        access_type="free",
    ),
    "official_hansard": DataSource(
        name="Official Hansard Website",
        url="https://hansard.parliament.uk/",
        description="Current debates and contributions. Requires web scraping.",
        format="scrape",
        coverage="Current parliament",
        access_type="free",
    ),

    # --- Voting Records ---
    "public_whip": DataSource(
        name="Public Whip",
        url="https://www.publicwhip.org.uk/project/data.php",
        description="MySQL dump of all MP votes since 1997.",
        format="mysql",
        coverage="1997-present",
        access_type="free",
    ),
    "votes_parliament": DataSource(
        name="Votes in Parliament",
        url="https://votes.parliament.uk/",
        description="Official vote results from Commons and Lords.",
        format="csv",
        coverage="2010-present",
        access_type="free",
    ),

    # --- MP Career & Biographical Data ---
    "parliament_members_api": DataSource(
        name="Parliament Members Data Platform",
        url="https://explore.data.parliament.uk/",
        description="Current MPs with full career history, roles, and contact info.",
        format="api",
        coverage="Current",
        access_type="free",
    ),
    "wikidata_politicians": DataSource(
        name="Wikidata British Politicians",
        url="https://www.wikidata.org/wiki/Wikidata:WikiProject_British_Politicians",
        description="Biographical data via SPARQL queries.",
        format="api",
        coverage="Historical",
        access_type="free",
    ),
    "commons_library_mps": DataSource(
        name="House of Commons Library - MP Data",
        url="https://commonslibrary.parliament.uk/data-tools-and-resources/parliament-elections-data/",
        description="PM lists, ministerial office holders, parliamentary activities.",
        format="csv",
        coverage="1918-present",
        access_type="free",
    ),

    # --- Constituency & Election Data ---
    "election_results_2024": DataSource(
        name="2024 General Election Results",
        url="https://commonslibrary.parliament.uk/research-briefings/cbp-10009/",
        description="Detailed results by constituency and candidate (CSV/Excel).",
        format="csv",
        coverage="2024",
        access_type="free",
    ),
    "electoral_calculus": DataSource(
        name="Electoral Calculus",
        url="https://www.electoralcalculus.co.uk/flatfile.html",
        description="Historical election results flat files.",
        format="csv",
        coverage="Historical",
        access_type="free",
    ),
    "uk_parliament_results": DataSource(
        name="UK Parliament Election Results",
        url="https://electionresults.parliament.uk/",
        description="Official results database with CSV downloads.",
        format="csv",
        coverage="2010-present",
        access_type="free",
    ),

    # --- Defection Tracking (Ground Truth) ---
    "best_for_britain_tracker": DataSource(
        name="Best for Britain Defection Tracker",
        url="https://www.bestforbritain.org/tory_reformuk_defection_tracker",
        description="Database tracking Tory-to-Reform defections. 110+ tracked as of Jan 2026.",
        format="scrape",
        coverage="2024-present",
        access_type="free",
    ),
}


def list_data_sources() -> pd.DataFrame:
    """Return a DataFrame summarizing all available data sources."""
    records = []
    for key, source in DATA_SOURCES.items():
        records.append({
            "key": key,
            "name": source.name,
            "format": source.format,
            "coverage": source.coverage,
            "access": source.access_type,
            "url": source.url,
        })
    return pd.DataFrame(records)


# =============================================================================
# MODULE 2: DATA INGESTION FUNCTIONS
# =============================================================================

class DataIngestion:
    """Functions to fetch data from various sources."""

    def __init__(self, data_dir: str = "source_data"):
        self.data_dir = data_dir
        self.theyworkforyou_api_key = os.getenv("THEYWORKFORYOU_API_KEY")

    # --- Hansard Speeches ---

    def fetch_parlparse_hansard(self, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Download ParlParse Hansard XML files.

        The full dataset is available at:
        https://github.com/mysociety/parlparse/tree/master/scrapedxml

        Returns dict with download URLs for different sections.
        """
        urls = {
            "debates": "https://github.com/mysociety/parlparse/tree/master/scrapedxml/debates",
            "wrans": "https://github.com/mysociety/parlparse/tree/master/scrapedxml/wrans",
            "lords": "https://github.com/mysociety/parlparse/tree/master/scrapedxml/lordspages",
            "westminster_hall": "https://github.com/mysociety/parlparse/tree/master/scrapedxml/westminhall",
            "members": "https://github.com/mysociety/parlparse/tree/master/members",
        }

        print("ParlParse Hansard data available at:")
        for section, url in urls.items():
            print(f"  {section}: {url}")

        return urls

    def fetch_theyworkforyou_speeches(
        self,
        person_id: Optional[int] = None,
        search: Optional[str] = None,
        num: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch speeches from TheyWorkForYou API.

        Args:
            person_id: TheyWorkForYou person ID for specific MP
            search: Search term to filter speeches
            num: Number of results to return

        Returns:
            DataFrame with speech records
        """
        if not self.theyworkforyou_api_key:
            print("WARNING: THEYWORKFORYOU_API_KEY not set. Get one at:")
            print("  https://www.theyworkforyou.com/api/key")
            return None

        base_url = "https://www.theyworkforyou.com/api/getHansard"
        params = {
            "key": self.theyworkforyou_api_key,
            "output": "json",
            "num": num,
        }

        if person_id:
            params["person"] = person_id
        if search:
            params["search"] = search

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "rows" in data:
                return pd.DataFrame(data["rows"])
            return pd.DataFrame(data)
        except requests.RequestException as e:
            print(f"Error fetching TheyWorkForYou data: {e}")
            return None

    def fetch_theyworkforyou_mps(self, party: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch list of MPs from TheyWorkForYou API.

        Args:
            party: Filter by party name (e.g., 'Conservative')

        Returns:
            DataFrame with MP records
        """
        if not self.theyworkforyou_api_key:
            print("WARNING: THEYWORKFORYOU_API_KEY not set.")
            return None

        base_url = "https://www.theyworkforyou.com/api/getMPs"
        params = {
            "key": self.theyworkforyou_api_key,
            "output": "json",
        }

        if party:
            params["party"] = party

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except requests.RequestException as e:
            print(f"Error fetching MPs: {e}")
            return None

    # --- Voting Records ---

    def fetch_public_whip_data(self) -> Dict[str, str]:
        """
        Get instructions for downloading Public Whip MySQL dump.

        The data includes:
        - pw_mp: MP biographical info
        - pw_vote: Individual votes
        - pw_division: Division (vote) metadata
        """
        info = {
            "download_url": "https://www.publicwhip.org.uk/project/data.php",
            "format": "MySQL dump (gzipped)",
            "tables": [
                "pw_mp - MP info (name, constituency, party, dates)",
                "pw_vote - Individual vote records",
                "pw_division - Division metadata (date, topic, result)",
                "pw_moffice - Ministerial offices held",
            ],
            "instructions": """
            1. Download the MySQL dump from the URL
            2. Import into MySQL/MariaDB or convert to SQLite
            3. Alternatively, use pandas read_sql or convert to CSV
            """,
        }

        print("Public Whip data download:")
        print(f"  URL: {info['download_url']}")
        print(f"  Format: {info['format']}")
        print("  Tables available:")
        for table in info['tables']:
            print(f"    - {table}")

        return info

    def fetch_votes_parliament(self, house: str = "commons") -> Dict[str, str]:
        """
        Get download URLs for official Votes in Parliament data.

        Args:
            house: 'commons' or 'lords'
        """
        urls = {
            "main_site": "https://votes.parliament.uk/",
            "commons_api": "https://commonsvotes-api.parliament.uk/swagger/ui/index",
            "lords_api": "https://lordsvotes-api.parliament.uk/swagger/ui/index",
        }

        print(f"Votes in Parliament ({house}):")
        print(f"  Main site: {urls['main_site']}")
        print(f"  Commons API: {urls['commons_api']}")
        print(f"  Lords API: {urls['lords_api']}")

        return urls

    # --- MP Career Data ---

    def fetch_parliament_members(self, house: str = "Commons") -> Optional[pd.DataFrame]:
        """
        Fetch current members from Parliament API.

        Args:
            house: 'Commons' or 'Lords'
        """
        url = f"https://members-api.parliament.uk/api/Members/Search?House={house}&IsCurrentMember=true"

        try:
            all_members = []
            skip = 0
            take = 20

            while True:
                response = requests.get(f"{url}&skip={skip}&take={take}")
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    member = item.get("value", {})
                    all_members.append({
                        "id": member.get("id"),
                        "name": member.get("nameDisplayAs"),
                        "party": member.get("latestParty", {}).get("name"),
                        "constituency": member.get("latestHouseMembership", {}).get("membershipFrom"),
                        "start_date": member.get("latestHouseMembership", {}).get("membershipStartDate"),
                        "gender": member.get("gender"),
                    })

                skip += take
                if skip >= data.get("totalResults", 0):
                    break

            return pd.DataFrame(all_members)
        except requests.RequestException as e:
            print(f"Error fetching Parliament members: {e}")
            return None

    def fetch_wikidata_mps(self) -> str:
        """
        Return SPARQL query for fetching British MPs from Wikidata.

        Execute at: https://query.wikidata.org/
        """
        query = """
        SELECT ?mp ?mpLabel ?partyLabel ?constituencyLabel ?startDate ?endDate ?birthDate
        WHERE {
          ?mp wdt:P39 wd:Q16707842.  # position held: Member of Parliament (UK)
          OPTIONAL { ?mp wdt:P102 ?party. }
          OPTIONAL { ?mp wdt:P768 ?constituency. }
          OPTIONAL { ?mp p:P39 ?statement.
                     ?statement pq:P580 ?startDate.
                     ?statement pq:P582 ?endDate. }
          OPTIONAL { ?mp wdt:P569 ?birthDate. }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        LIMIT 10000
        """

        print("Wikidata SPARQL query for UK MPs:")
        print("Execute at: https://query.wikidata.org/")
        print("-" * 50)
        print(query)

        return query

    # --- Election Results ---

    def fetch_election_results_2024(self) -> Dict[str, str]:
        """
        Get download URLs for 2024 General Election results.

        The House of Commons Library provides comprehensive CSV/Excel files.
        """
        urls = {
            "commons_library": "https://commonslibrary.parliament.uk/research-briefings/cbp-10009/",
            "files": {
                "by_constituency_csv": "Detailed results by constituency (113 KB CSV)",
                "by_constituency_xlsx": "Detailed results by constituency (152 KB Excel)",
                "by_candidate_csv": "Detailed results by candidate (660 KB CSV)",
                "by_candidate_xlsx": "Detailed results by candidate (564 KB Excel)",
            },
            "parliament_results": "https://electionresults.parliament.uk/",
            "electoral_calculus": "https://www.electoralcalculus.co.uk/flatfile.html",
        }

        print("2024 Election Results downloads:")
        print(f"  Commons Library: {urls['commons_library']}")
        print("  Available files:")
        for name, desc in urls['files'].items():
            print(f"    - {desc}")
        print(f"  Parliament Results: {urls['parliament_results']}")
        print(f"  Electoral Calculus: {urls['electoral_calculus']}")

        return urls

    # --- Defection Ground Truth ---

    def fetch_defection_tracker(self) -> Optional[pd.DataFrame]:
        """
        Scrape Best for Britain defection tracker.

        Note: Requires BeautifulSoup. Install with: pip install beautifulsoup4
        """
        if not HAS_BS4:
            print("WARNING: beautifulsoup4 not installed. Install with:")
            print("  pip install beautifulsoup4")
            return None

        url = "https://www.bestforbritain.org/tory_reformuk_defection_tracker"

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find table(s) with defection data
            tables = soup.find_all('table')

            if tables:
                # Try to parse the first table with data
                for table in tables:
                    rows = table.find_all('tr')
                    if len(rows) > 1:
                        headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                        data = []
                        for row in rows[1:]:
                            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                            if cells:
                                data.append(cells)

                        if data:
                            df = pd.DataFrame(data, columns=headers if len(headers) == len(data[0]) else None)
                            return df

            print("Could not parse defection tracker table. Manual download may be required.")
            print(f"Visit: {url}")
            return None

        except requests.RequestException as e:
            print(f"Error fetching defection tracker: {e}")
            return None


# =============================================================================
# MODULE 3: FEATURE ENGINEERING
# =============================================================================

class FeatureEngineering:
    """Extract features for defection prediction model."""

    # Keywords associated with Reform UK talking points
    REFORM_KEYWORDS = [
        "immigration", "borders", "small boats", "illegal migration",
        "rwanda", "deportation", "asylum",
        "brexit", "sovereignty", "eu", "european union",
        "tax cuts", "lower taxes", "nhs waiting lists",
        "net zero", "climate targets", "energy bills",
        "woke", "free speech", "cancel culture",
    ]

    # Keywords for rebellion detection
    REBELLION_TOPICS = [
        "rwanda bill", "illegal migration bill",
        "net zero", "climate",
        "tax", "spending",
    ]

    def extract_speech_features(self, speeches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract NLP features from MP speeches.

        Features:
        - immigration_mentions: Count of immigration-related terms
        - reform_alignment: Score of alignment with Reform UK talking points
        - leadership_criticism: Mentions of party leadership criticism
        - sentiment_score: Overall sentiment polarity

        Args:
            speeches_df: DataFrame with columns ['mp_id', 'text', 'date']

        Returns:
            DataFrame with features per MP
        """
        if speeches_df is None or speeches_df.empty:
            print("No speeches data provided")
            return pd.DataFrame()

        features = []

        # Group by MP
        for mp_id, group in speeches_df.groupby('mp_id'):
            all_text = ' '.join(group['text'].fillna('').astype(str)).lower()

            # Count Reform-aligned keyword mentions
            immigration_count = sum(
                all_text.count(kw) for kw in
                ['immigration', 'migrant', 'border', 'asylum', 'boat']
            )

            reform_keyword_count = sum(
                all_text.count(kw) for kw in self.REFORM_KEYWORDS
            )

            # Normalize by speech count
            speech_count = len(group)

            features.append({
                'mp_id': mp_id,
                'total_speeches': speech_count,
                'immigration_mentions': immigration_count,
                'immigration_per_speech': immigration_count / max(speech_count, 1),
                'reform_keyword_count': reform_keyword_count,
                'reform_alignment_score': reform_keyword_count / max(speech_count, 1),
            })

        return pd.DataFrame(features)

    def extract_voting_features(self, votes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract voting behavior features.

        Features:
        - rebellion_rate: Proportion of votes against party whip
        - key_vote_alignment: Votes on Rwanda, Brexit, immigration bills
        - absence_rate: Proportion of missed votes

        Args:
            votes_df: DataFrame with columns ['mp_id', 'vote', 'party_position', 'division_topic']

        Returns:
            DataFrame with voting features per MP
        """
        if votes_df is None or votes_df.empty:
            print("No voting data provided")
            return pd.DataFrame()

        features = []

        for mp_id, group in votes_df.groupby('mp_id'):
            total_votes = len(group)

            # Calculate rebellion rate (votes against party position)
            if 'party_position' in group.columns and 'vote' in group.columns:
                rebellions = (group['vote'] != group['party_position']).sum()
                rebellion_rate = rebellions / max(total_votes, 1)
            else:
                rebellion_rate = None

            # Calculate absence rate
            if 'vote' in group.columns:
                absences = group['vote'].isna().sum()
                absence_rate = absences / max(total_votes, 1)
            else:
                absence_rate = None

            features.append({
                'mp_id': mp_id,
                'total_votes': total_votes,
                'rebellion_rate': rebellion_rate,
                'absence_rate': absence_rate,
            })

        return pd.DataFrame(features)

    def extract_career_features(self, mps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract career and biographical features.

        Features:
        - years_as_mp: Total years in Parliament
        - ministerial_experience: Boolean/count of ministerial roles
        - age: Current age
        - years_to_retirement: Estimated years (based on avg retirement age ~70)

        Args:
            mps_df: DataFrame with MP career info

        Returns:
            DataFrame with career features
        """
        if mps_df is None or mps_df.empty:
            print("No MP career data provided")
            return pd.DataFrame()

        features = []
        current_year = datetime.now().year

        for _, row in mps_df.iterrows():
            mp_id = row.get('id') or row.get('mp_id')

            # Calculate years as MP
            start_date = row.get('start_date') or row.get('first_elected')
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

            # Calculate age and years to retirement
            birth_date = row.get('birth_date') or row.get('dob')
            if birth_date:
                try:
                    if isinstance(birth_date, str):
                        birth_year = int(birth_date[:4])
                    else:
                        birth_year = birth_date.year
                    age = current_year - birth_year
                    years_to_retirement = max(0, 70 - age)  # Assume 70 as retirement age
                except (ValueError, TypeError):
                    age = None
                    years_to_retirement = None
            else:
                age = None
                years_to_retirement = None

            features.append({
                'mp_id': mp_id,
                'name': row.get('name') or row.get('nameDisplayAs'),
                'party': row.get('party'),
                'constituency': row.get('constituency'),
                'years_as_mp': years_as_mp,
                'age': age,
                'years_to_retirement': years_to_retirement,
                'gender': row.get('gender'),
            })

        return pd.DataFrame(features)

    def extract_constituency_features(self, election_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract constituency-level features.

        Features:
        - majority_2024: Vote majority in 2024
        - majority_pct: Majority as % of votes
        - reform_vote_share: Reform UK vote share in constituency
        - swing_2019_2024: Change in vote share from 2019
        - leave_vote_2016: EU referendum Leave vote %

        Args:
            election_df: DataFrame with 2024 election results by constituency

        Returns:
            DataFrame with constituency features
        """
        if election_df is None or election_df.empty:
            print("No election data provided")
            return pd.DataFrame()

        features = []

        for _, row in election_df.iterrows():
            constituency = row.get('constituency') or row.get('Constituency')

            features.append({
                'constituency': constituency,
                'winner': row.get('winner') or row.get('First party'),
                'majority': row.get('majority') or row.get('Majority'),
                'majority_pct': row.get('majority_pct') or row.get('Majority %'),
                'reform_vote_share': row.get('reform_pct') or row.get('Reform UK %'),
                'conservative_vote_share': row.get('con_pct') or row.get('Conservative %'),
                'turnout': row.get('turnout') or row.get('Turnout'),
            })

        return pd.DataFrame(features)


# =============================================================================
# MODULE 4: MODEL TRAINING
# =============================================================================

class DefectionModel:
    """Train and evaluate defection prediction model."""

    def __init__(self):
        if not HAS_SKLEARN:
            print("WARNING: scikit-learn not installed. Install with:")
            print("  pip install scikit-learn")

        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_columns = []

    def prepare_training_data(
        self,
        speech_features: pd.DataFrame,
        voting_features: pd.DataFrame,
        career_features: pd.DataFrame,
        constituency_features: pd.DataFrame,
        defection_labels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Combine all features and labels into training dataset.

        Args:
            speech_features: NLP features from speeches
            voting_features: Voting behavior features
            career_features: Career and biographical features
            constituency_features: Constituency-level features
            defection_labels: DataFrame with ['mp_id', 'defected'] columns

        Returns:
            Tuple of (feature DataFrame, label Series)
        """
        # Start with career features as base
        df = career_features.copy()

        # Merge speech features
        if not speech_features.empty:
            df = df.merge(speech_features, on='mp_id', how='left')

        # Merge voting features
        if not voting_features.empty:
            df = df.merge(voting_features, on='mp_id', how='left')

        # Merge constituency features
        if not constituency_features.empty and 'constituency' in df.columns:
            df = df.merge(constituency_features, on='constituency', how='left')

        # Merge defection labels
        df = df.merge(defection_labels[['mp_id', 'defected']], on='mp_id', how='left')

        # Fill missing defection labels with 0 (not defected)
        df['defected'] = df['defected'].fillna(0).astype(int)

        # Select numeric features for model
        self.feature_columns = [
            col for col in df.columns
            if col not in ['mp_id', 'name', 'party', 'constituency', 'winner', 'defected', 'gender']
            and df[col].dtype in ['int64', 'float64']
        ]

        X = df[self.feature_columns].fillna(0)
        y = df['defected']

        return X, y

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
    ) -> Dict[str, Any]:
        """
        Train defection prediction model.

        Args:
            X: Feature DataFrame
            y: Label Series (0 = stayed, 1 = defected)
            model_type: 'logistic', 'random_forest', or 'gradient_boosting'

        Returns:
            Dict with model metrics
        """
        if not HAS_SKLEARN:
            print("scikit-learn required for model training")
            return {}

        # Select model
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:  # default: random_forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='roc_auc')

        metrics = {
            'model_type': model_type,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            print("Model not trained yet")
            return pd.DataFrame()

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance,
        }).sort_values('importance', ascending=False)


# =============================================================================
# MODULE 5: PREDICTION & OUTPUT
# =============================================================================

class DefectionPredictor:
    """Generate predictions and risk reports."""

    def __init__(self, model: DefectionModel):
        self.model = model

    def predict_defection_risk(self, X: pd.DataFrame, mp_info: pd.DataFrame) -> pd.DataFrame:
        """
        Predict defection risk for all MPs.

        Args:
            X: Feature DataFrame (same format as training)
            mp_info: DataFrame with MP identifiers (mp_id, name, constituency)

        Returns:
            DataFrame with risk scores
        """
        if self.model.model is None:
            print("Model not trained")
            return pd.DataFrame()

        # Scale features
        X_scaled = self.model.scaler.transform(X.fillna(0))

        # Get probabilities
        if hasattr(self.model.model, 'predict_proba'):
            risk_scores = self.model.model.predict_proba(X_scaled)[:, 1]
        else:
            risk_scores = self.model.model.predict(X_scaled)

        # Combine with MP info
        results = mp_info.copy()
        results['defection_risk_score'] = risk_scores
        results['risk_category'] = pd.cut(
            risk_scores,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        return results.sort_values('defection_risk_score', ascending=False)

    def generate_risk_report(
        self,
        predictions: pd.DataFrame,
        feature_importance: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate comprehensive risk report.

        Args:
            predictions: DataFrame from predict_defection_risk()
            feature_importance: DataFrame from get_feature_importance()
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 70,
            "TORY MP DEFECTION RISK REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
        ]

        # Summary statistics
        if not predictions.empty:
            high_risk = predictions[predictions['risk_category'].isin(['High', 'Very High'])]
            report_lines.extend([
                f"Total MPs analyzed: {len(predictions)}",
                f"High/Very High risk: {len(high_risk)} ({100*len(high_risk)/len(predictions):.1f}%)",
                f"Average risk score: {predictions['defection_risk_score'].mean():.3f}",
                "",
            ])

            # Top 10 at risk
            report_lines.extend([
                "TOP 10 MPs AT RISK OF DEFECTION",
                "-" * 40,
            ])

            top_10 = predictions.head(10)
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                name = row.get('name', 'Unknown')
                constituency = row.get('constituency', 'Unknown')
                score = row['defection_risk_score']
                report_lines.append(f"{i:2}. {name} ({constituency}): {score:.3f}")

        # Feature importance
        if not feature_importance.empty:
            report_lines.extend([
                "",
                "KEY RISK FACTORS (by importance)",
                "-" * 40,
            ])

            for _, row in feature_importance.head(10).iterrows():
                report_lines.append(f"  - {row['feature']}: {row['importance']:.4f}")

        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function demonstrating the full pipeline."""

    print("=" * 70)
    print("TORY MP DEFECTION RISK PREDICTION MODEL")
    print("=" * 70)
    print()

    # 1. List available data sources
    print("STEP 1: Available Data Sources")
    print("-" * 40)
    sources_df = list_data_sources()
    print(sources_df.to_string(index=False))
    print()

    # 2. Initialize data ingestion
    print("STEP 2: Data Ingestion")
    print("-" * 40)
    ingestion = DataIngestion()

    # Show data source instructions
    print("\n[Hansard/Speeches]")
    ingestion.fetch_parlparse_hansard()

    print("\n[Voting Records]")
    ingestion.fetch_public_whip_data()

    print("\n[Election Results]")
    ingestion.fetch_election_results_2024()

    # Try to fetch current MPs
    print("\n[Fetching Current MPs from Parliament API...]")
    mps_df = ingestion.fetch_parliament_members(house="Commons")
    if mps_df is not None:
        tory_mps = mps_df[mps_df['party'] == 'Conservative']
        print(f"  Found {len(tory_mps)} Conservative MPs")

    # 3. Feature engineering setup
    print("\n" + "=" * 70)
    print("STEP 3: Feature Engineering Framework")
    print("-" * 40)
    fe = FeatureEngineering()
    print("Feature categories:")
    print("  - Speech features (NLP): immigration mentions, Reform alignment")
    print("  - Voting features: rebellion rate, key vote positions")
    print("  - Career features: tenure, age, ministerial experience")
    print("  - Constituency features: majority, Reform vote share")
    print()
    print("Reform UK keyword list:")
    for kw in fe.REFORM_KEYWORDS[:10]:
        print(f"  - {kw}")
    print("  ... and more")

    # 4. Model training setup
    print("\n" + "=" * 70)
    print("STEP 4: Model Training Framework")
    print("-" * 40)
    if HAS_SKLEARN:
        print("scikit-learn available. Models supported:")
        print("  - Logistic Regression")
        print("  - Random Forest Classifier")
        print("  - Gradient Boosting Classifier")
    else:
        print("WARNING: Install scikit-learn for model training")
        print("  pip install scikit-learn")

    # 5. Summary
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("-" * 40)
    print("""
To run the full pipeline:

1. Set up API keys:
   export THEYWORKFORYOU_API_KEY=your_key_here

2. Download data files:
   - Commons Library 2024 results CSV
   - Public Whip MySQL dump
   - ParlParse Hansard XML files

3. Place data in source_data/ subdirectories

4. Run feature extraction and model training:

   # Example code:
   ingestion = DataIngestion()
   fe = FeatureEngineering()
   model = DefectionModel()

   # Load data
   speeches_df = pd.read_csv('source_data/hansard/speeches.csv')
   votes_df = pd.read_csv('source_data/voting/votes.csv')

   # Extract features
   speech_features = fe.extract_speech_features(speeches_df)
   voting_features = fe.extract_voting_features(votes_df)

   # Train model
   X, y = model.prepare_training_data(...)
   metrics = model.train_model(X, y)

   # Generate predictions
   predictor = DefectionPredictor(model)
   predictions = predictor.predict_defection_risk(X, mp_info)
   report = predictor.generate_risk_report(predictions, ...)
""")

    print("=" * 70)
    print("Pipeline setup complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
