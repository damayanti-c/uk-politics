"""
Training Speech Analysis Using TF-IDF
======================================

Analyzes 10 years of Hansard speeches for 2019-2024 Conservative MPs.
Measures Reform UK rhetoric alignment, extremism, and radicalization trajectories.

Outputs metrics for use in defection prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from hansard_data_source import load_hansard_dataframe

# =============================================================================
# PATHS
# =============================================================================

base_dir = Path(__file__).parent.parent.parent
source_data = base_dir / "source_data"
hansard_csv = source_data / "hansard" / "all_speeches_extended.csv"
training_data_csv = Path(__file__).parent / "training_data.csv"
output_csv = Path(__file__).parent / "training_speech_features.csv"

# =============================================================================
# REFORM UK RHETORIC TEMPLATES
# =============================================================================

REFORM_IMMIGRATION_RHETORIC = [
    "We must stop the boats and take back control of our borders",
    "Mass immigration is putting unsustainable pressure on public services and housing",
    "The Rwanda scheme is necessary to deter illegal channel crossings",
    "We need to leave the European Court of Human Rights to control immigration properly",
    "Small boat crossings are an invasion that threatens our national sovereignty",
    "The asylum system is broken and being abused by economic migrants",
    "Net zero immigration should be our target to protect British jobs and wages",
    "We must prioritize British workers over unlimited immigration from Europe",
    "Illegal immigrants should be detained and deported immediately",
    "The government has lost control of our borders completely",
    "We are being flooded with illegal migrants who bypass safe countries",
    "The migrant crisis is costing taxpayers billions in hotel accommodation",
    "Failed asylum seekers must be removed from the country swiftly",
    "We need an Australian-style points system with strict caps on numbers",
    "The small boats trade is organized criminal activity that must be stopped",
    "Channel crossings are illegal immigration that undermines our sovereignty",
    "Mass migration threatens our national identity and cultural cohesion",
    "We cannot sustain unlimited numbers of people crossing the channel",
    "The immigration system is completely out of control and broken",
    "British people should come first in housing jobs and public services"
]

# Hardline anti-immigration keywords
HARDLINE_KEYWORDS = [
    'illegal', 'crisis', 'flood', 'flooding', 'invasion', 'invading', 'control our borders',
    'small boats', 'small boat', 'channel crossing', 'channel crossings',
    'failed', 'broken system', 'stop the boats', 'take back control',
    'sovereignty', 'mass immigration', 'mass migration', 'deport', 'deportation',
    'detention', 'detain', 'remove', 'removal', 'echr', 'rwanda scheme',
    'economic migrants', 'economic migrant', 'abuse', 'abusing', 'exploiting',
    'criminal gangs', 'criminal gang', 'out of control', 'lost control',
    'unsustainable', 'overwhelm', 'overwhelming', 'threat', 'threatens',
    'prioritize british', 'british first', 'british people first',
    # Less hardline but still immigration-skeptic parliamentary language
    'reduce immigration', 'reduce numbers', 'cut immigration', 'lower immigration',
    'cap on immigration', 'immigration cap', 'limit immigration', 'immigration limit',
    'sustainable levels', 'sustainable immigration', 'controlled immigration',
    'firm but fair', 'managed migration', 'points-based', 'points based',
    'tens of thousands', 'too many', 'too high', 'excessive immigration',
    'burden on', 'strain on', 'pressure on services', 'pressure on housing',
    'queue jump', 'queue-jump', 'unfair on', 'british workers',
    'wage compression', 'downward pressure on wages', 'undercut'
]

# Pro-immigration / compassionate keywords (to avoid catching these MPs)
PRO_IMMIGRATION_KEYWORDS = [
    'humanitarian', 'refuge', 'refugee', 'compassion', 'compassionate',
    'safe routes', 'safe route', 'international obligations', 'persecution',
    'protection', 'protect', 'family reunion', 'sanctuary', 'welcome',
    'diversity', 'multicultural', 'contribution', 'fleeing', 'vulnerable',
    'human rights', 'asylum seekers right', 'legitimate',
    'war-torn', 'displaced', 'shelter'
]

# Keywords to identify immigration-related speeches
IMMIGRATION_FILTER_KEYWORDS = ['immigra', 'asylum', 'border', 'migrant', 'rwanda', 'boat', 'channel', 'refugee']

# Name mapping: election data name -> Hansard name
NAME_TO_HANSARD = {
    'Nick Fletcher': 'Nicholas Fletcher',
    'Gregory Clark': 'Greg Clark',
    'Robert Neill': 'Bob Neill',
    'Dan Poulter': 'Daniel Poulter',
    'Jacqueline Doyle-Price': 'Jackie Doyle-Price',
    'Steve Baker': 'Steven Baker',
    'William Cash': 'Bill Cash',
    'Caroline Johnson': 'Dr Caroline Johnson',
    'David T. C. Davies': 'David Davies',
    'Thérèse Coffey': 'Therese Coffey',
    'Stephen Brine': 'Steve Brine',
}


# =============================================================================
# TF-IDF FUNCTIONS
# =============================================================================

def calculate_reform_alignment(speech_text, vectorizer, reform_vectors):
    """Calculate similarity between speech and Reform UK rhetoric."""
    if not speech_text or len(speech_text.strip()) < 50:
        return 0.0

    sentences = [s.strip() for s in speech_text.split('.') if len(s.strip()) > 20]
    immigration_sentences = [
        s for s in sentences
        if any(kw in s.lower() for kw in IMMIGRATION_FILTER_KEYWORDS)
    ]

    if not immigration_sentences:
        return 0.0

    immigration_sentences = immigration_sentences[:50]

    try:
        mp_vectors = vectorizer.transform(immigration_sentences)
        similarities = cosine_similarity(mp_vectors, reform_vectors)
        max_similarities = np.max(similarities, axis=1)
        return float(np.mean(max_similarities))
    except:
        return 0.0


def calculate_hardline_ratio(speech_text):
    """Calculate ratio of hardline vs compassionate language (-1 to 1)."""
    text_lower = speech_text.lower()

    hardline_count = sum(1 for kw in HARDLINE_KEYWORDS if kw in text_lower)
    compassion_count = sum(1 for kw in PRO_IMMIGRATION_KEYWORDS if kw in text_lower)

    total = hardline_count + compassion_count
    if total == 0:
        return 0.0

    return (hardline_count - compassion_count) / total


def count_immigration_speeches(mp_speeches):
    """Count speeches that mention immigration topics."""
    immigration_count = 0
    for text in mp_speeches['text'].values:
        if any(kw in text.lower() for kw in IMMIGRATION_FILTER_KEYWORDS):
            immigration_count += 1
    return immigration_count


def calculate_radicalization_trajectory(mp_speeches, vectorizer, reform_vectors):
    """Analyze if MP is becoming more anti-immigration over time."""
    if len(mp_speeches) < 10:
        return {
            'radicalization_slope': 0.0,
            'trend_strength': 0.0,
            'current_alignment': 0.0,
            'starting_alignment': 0.0,
            'total_change': 0.0,
            'quarters_tracked': 0
        }

    mp_speeches = mp_speeches.copy()
    mp_speeches['date'] = pd.to_datetime(mp_speeches['date'], errors='coerce')
    mp_speeches = mp_speeches.dropna(subset=['date'])
    mp_speeches['quarter'] = mp_speeches['date'].dt.to_period('Q')

    quarterly_scores = []
    for quarter, group in mp_speeches.groupby('quarter'):
        combined_text = ' '.join(group['text'].values)
        reform_score = calculate_reform_alignment(combined_text, vectorizer, reform_vectors)
        quarterly_scores.append({
            'quarter': str(quarter),
            'reform_alignment': reform_score,
            'speech_count': len(group)
        })

    df_timeline = pd.DataFrame(quarterly_scores).sort_values('quarter')

    if len(df_timeline) < 4:
        return {
            'radicalization_slope': 0.0,
            'trend_strength': 0.0,
            'current_alignment': df_timeline.iloc[-1]['reform_alignment'] if len(df_timeline) > 0 else 0.0,
            'starting_alignment': df_timeline.iloc[0]['reform_alignment'] if len(df_timeline) > 0 else 0.0,
            'total_change': 0.0,
            'quarters_tracked': len(df_timeline)
        }

    x = np.arange(len(df_timeline))
    y = df_timeline['reform_alignment'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'radicalization_slope': float(slope),
        'trend_strength': float(r_value ** 2),
        'current_alignment': float(df_timeline.iloc[-1]['reform_alignment']),
        'starting_alignment': float(df_timeline.iloc[0]['reform_alignment']),
        'total_change': float(df_timeline.iloc[-1]['reform_alignment'] - df_timeline.iloc[0]['reform_alignment']),
        'quarters_tracked': len(df_timeline)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TRAINING SPEECH ANALYSIS - TF-IDF")
    print("=" * 80)
    print()

    # Load training data to get MP list
    print("Loading training data...")
    training_data = pd.read_csv(training_data_csv)
    mp_names = set(training_data['name'].values)
    print(f"  {len(mp_names)} MPs in training data")

    # Load Hansard speeches (local CSV or Snowflake, based on env)
    print("\nLoading Hansard speeches...")
    all_speeches_df = load_hansard_dataframe(default_local_csv_path=hansard_csv)
    print(f"  {len(all_speeches_df)} total speeches in Hansard")

    # Initialize TF-IDF
    print("\nInitializing TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95
    )

    # Fit on ALL speeches + Reform rhetoric (broader parliamentary baseline)
    print("Fitting TF-IDF model on ALL speeches (broader baseline)...")
    # Sample from all speeches to get broader parliamentary context
    sample_size = min(50000, len(all_speeches_df))
    sample_texts = list(all_speeches_df['text'].sample(n=sample_size, random_state=42).values)
    sample_texts.extend(REFORM_IMMIGRATION_RHETORIC)
    vectorizer.fit(sample_texts)
    print(f"  Fitted on {len(sample_texts)} documents (all-party parliamentary baseline)")

    # Now filter to training MPs for analysis
    print("\nFiltering to training MPs...")
    hansard_names = set(mp_names) | set(NAME_TO_HANSARD.values())
    speeches_df = all_speeches_df[all_speeches_df['speaker_name'].isin(hansard_names)]
    print(f"  {len(speeches_df)} speeches from training MPs")

    # Check date range
    speeches_df['date'] = pd.to_datetime(speeches_df['date'], errors='coerce')
    print(f"  Date range: {speeches_df['date'].min()} to {speeches_df['date'].max()}")

    # MPs found in Hansard
    mps_found = set(speeches_df['speaker_name'].unique())
    print(f"  {len(mps_found)}/{len(mp_names)} MPs found in Hansard")

    # Transform Reform rhetoric
    reform_vectors = vectorizer.transform(REFORM_IMMIGRATION_RHETORIC)
    print(f"  Encoded {len(REFORM_IMMIGRATION_RHETORIC)} Reform rhetoric templates")

    # Analyze each MP
    print("\nAnalyzing MPs...")
    results = []
    all_alignments = {}

    for idx, mp_name in enumerate(mp_names, 1):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(mp_names)}: {mp_name}")

        # Use name mapping if available
        hansard_name = NAME_TO_HANSARD.get(mp_name, mp_name)
        mp_speeches = speeches_df[speeches_df['speaker_name'] == hansard_name]

        if len(mp_speeches) == 0:
            results.append({
                'name': mp_name,
                'total_speeches': 0,
                'immigration_speeches': 0,
                'immigration_speech_proportion': None,
                'reform_alignment': 0.0,
                'hardline_ratio': 0.0,
                'radicalization_slope': 0.0,
                'trend_strength': 0.0,
                'current_alignment': 0.0,
                'starting_alignment': 0.0,
                'alignment_change': 0.0,
                'extremism_percentile': None
            })
            continue

        # Combine all speeches
        all_text = ' '.join(mp_speeches['text'].values)

        # Reform alignment
        reform_alignment = calculate_reform_alignment(all_text, vectorizer, reform_vectors)
        all_alignments[mp_name] = reform_alignment

        # Hardline ratio
        hardline_ratio = calculate_hardline_ratio(all_text)

        # Immigration speech proportion
        total_speeches = len(mp_speeches)
        immigration_speeches = count_immigration_speeches(mp_speeches)
        immigration_proportion = immigration_speeches / total_speeches if total_speeches > 0 else 0

        # Radicalization trajectory
        trajectory = calculate_radicalization_trajectory(mp_speeches, vectorizer, reform_vectors)

        results.append({
            'name': mp_name,
            'total_speeches': total_speeches,
            'immigration_speeches': immigration_speeches,
            'immigration_speech_proportion': round(immigration_proportion, 4),
            'reform_alignment': round(reform_alignment, 4),
            'hardline_ratio': round(hardline_ratio, 4),
            'radicalization_slope': round(trajectory['radicalization_slope'], 4),
            'trend_strength': round(trajectory['trend_strength'], 4),
            'current_alignment': round(trajectory['current_alignment'], 4),
            'starting_alignment': round(trajectory['starting_alignment'], 4),
            'alignment_change': round(trajectory['total_change'], 4),
            'extremism_percentile': None
        })

    # Calculate extremism percentiles
    print("\nCalculating extremism percentiles...")
    scores = sorted(all_alignments.values())
    for r in results:
        if r['name'] in all_alignments:
            score = all_alignments[r['name']]
            rank_position = sum(1 for s in scores if s < score)
            r['extremism_percentile'] = round((rank_position / len(scores)) * 100, 1) if len(scores) > 0 else None

    # Create output DataFrame
    results_df = pd.DataFrame(results)

    # Save
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results_df)} MPs to {output_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid = results_df[results_df['total_speeches'] > 0]
    print(f"\nMPs with speeches: {len(valid)}/{len(results_df)}")
    print(f"Mean Reform alignment: {valid['reform_alignment'].mean():.4f}")
    print(f"Max Reform alignment: {valid['reform_alignment'].max():.4f}")

    print(f"\nMean immigration speech proportion: {valid['immigration_speech_proportion'].mean()*100:.2f}%")

    radicalizing = valid[valid['radicalization_slope'] > 0.01]
    print(f"MPs becoming more extreme: {len(radicalizing)}")

    print("\n" + "=" * 80)
    print("TOP 20 BY REFORM ALIGNMENT")
    print("=" * 80)
    top20 = results_df.nlargest(20, 'reform_alignment')
    for idx, row in enumerate(top20.iterrows(), 1):
        _, r = row
        print(f"{idx:2}. {r['name']:<30} | Alignment: {r['reform_alignment']:.4f} | Hardline: {r['hardline_ratio']:+.2f} | Imm%: {(r['immigration_speech_proportion'] or 0)*100:.1f}%")

    return results_df


if __name__ == "__main__":
    results = main()
