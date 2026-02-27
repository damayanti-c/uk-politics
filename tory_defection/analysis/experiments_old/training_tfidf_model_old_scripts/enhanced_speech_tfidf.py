"""
Enhanced Speech Analysis Using TF-IDF Vectorization
===================================================

Alternative to transformer models - uses TF-IDF with cosine similarity.
No PyTorch required, works without admin privileges.

Still provides:
1. Reform UK rhetoric alignment (semantic similarity via TF-IDF)
2. Radicalization trajectories over time
3. Hardline vs compassionate language detection
4. Extremism percentile ranking

Dependencies (no admin needed):
pip install scikit-learn scipy numpy pandas --user
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

# =============================================================================
# PATHS
# =============================================================================

TORY_DEFECTION_DIR = Path(__file__).parent.parent.parent
SOURCE_DATA = TORY_DEFECTION_DIR / "source_data"
# Use extended dataset (2015-2026) for fuller coverage
HANSARD_CSV = SOURCE_DATA / "hansard" / "all_speeches_extended.csv"
ANALYSIS_OUTPUT = Path(__file__).parent


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

HARDLINE_KEYWORDS = [
    'illegal', 'crisis', 'flood', 'flooding', 'invasion', 'invading', 'control our borders',
    'small boats', 'small boat', 'channel crossing', 'channel crossings',
    'failed', 'broken system', 'stop the boats', 'take back control',
    'sovereignty', 'mass immigration', 'mass migration', 'deport', 'deportation',
    'detention', 'detain', 'remove', 'removal', 'echr', 'rwanda scheme',
    'economic migrants', 'economic migrant', 'abuse', 'abusing', 'exploiting',
    'criminal gangs', 'criminal gang', 'out of control', 'lost control',
    'unsustainable', 'overwhelm', 'overwhelming', 'threat', 'threatens',
    'prioritize british', 'british first', 'british people first'
]

PRO_IMMIGRATION_KEYWORDS = [
    'humanitarian', 'refuge', 'refugee', 'compassion', 'compassionate',
    'safe routes', 'safe route', 'international obligations', 'persecution',
    'protection', 'protect', 'family reunion', 'sanctuary', 'welcome',
    'diversity', 'multicultural', 'contribution', 'fleeing', 'vulnerable',
    'human rights', 'asylum seekers right', 'legitimate', 'persecution',
    'war-torn', 'displaced', 'shelter'
]

# Filter keywords for topic detection
IMMIGRATION_FILTER_KEYWORDS = ['immigra', 'asylum', 'border', 'migrant', 'rwanda', 'boat', 'channel', 'refugee']


# =============================================================================
# TF-IDF VECTORIZATION FUNCTIONS
# =============================================================================

def initialize_tfidf_model():
    """Create TF-IDF vectorizer."""
    print("Initializing TF-IDF vectorizer...")

    # Use 1-3 word phrases, remove common English stopwords
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Capture phrases up to 3 words
        max_features=5000,    # Limit vocabulary size
        stop_words='english',
        lowercase=True,
        min_df=2,            # Ignore very rare terms
        max_df=0.95          # Ignore very common terms
    )

    print("OK TF-IDF vectorizer ready")
    return vectorizer


def calculate_reform_alignment_tfidf(speech_text, vectorizer, reform_vectors):
    """
    Calculate semantic similarity between speech and Reform UK rhetoric using TF-IDF.

    Returns:
        float: 0-1 score (0=completely different, 1=very similar)
    """
    if not speech_text or len(speech_text.strip()) < 50:
        return 0.0

    # Split into sentences
    sentences = [s.strip() for s in speech_text.split('.') if len(s.strip()) > 20]

    # Filter to immigration-related sentences
    immigration_sentences = [
        s for s in sentences
        if any(kw in s.lower() for kw in
               ['immigra', 'asylum', 'border', 'migrant', 'rwanda', 'boat', 'channel'])
    ]

    if not immigration_sentences:
        return 0.0

    # Limit to 50 sentences for performance
    immigration_sentences = immigration_sentences[:50]

    try:
        # Transform MP's sentences
        mp_vectors = vectorizer.transform(immigration_sentences)

        # Calculate cosine similarity to Reform statements
        similarities = cosine_similarity(mp_vectors, reform_vectors)

        # Return average of max similarities
        max_similarities = np.max(similarities, axis=1)
        return float(np.mean(max_similarities))

    except Exception as e:
        print(f"Error calculating alignment: {e}")
        return 0.0


def calculate_hardline_vs_compassionate_ratio(speech_text):
    """
    Calculate ratio of hardline vs compassionate immigration language.

    Returns:
        float: -1 to 1 (-1=very compassionate, 0=neutral, 1=very hardline)
    """
    text_lower = speech_text.lower()

    hardline_count = sum(1 for keyword in HARDLINE_KEYWORDS if keyword in text_lower)
    compassion_count = sum(1 for keyword in PRO_IMMIGRATION_KEYWORDS if keyword in text_lower)

    total = hardline_count + compassion_count
    if total == 0:
        return 0.0

    return (hardline_count - compassion_count) / total


# =============================================================================
# TEMPORAL RADICALIZATION ANALYSIS
# =============================================================================

def analyze_radicalization_trajectory(speeches_df, mp_name, vectorizer, reform_vectors):
    """
    Detect if MP is becoming more anti-immigration over time.

    Returns:
        dict: Radicalization metrics
    """
    mp_speeches = speeches_df[speeches_df['speaker_name'] == mp_name].copy()

    if len(mp_speeches) < 10:
        return {
            'radicalization_slope': 0.0,
            'trend_strength': 0.0,
            'current_alignment': 0.0,
            'starting_alignment': 0.0,
            'total_change': 0.0,
            'recent_acceleration': 0.0,
            'quarters_tracked': 0,
            'insufficient_data': True
        }

    # Parse dates and create quarters
    mp_speeches['date'] = pd.to_datetime(mp_speeches['date'], errors='coerce')
    mp_speeches = mp_speeches.dropna(subset=['date'])
    mp_speeches['quarter'] = mp_speeches['date'].dt.to_period('Q')

    # Calculate metrics per quarter
    quarterly_scores = []

    for quarter, group in mp_speeches.groupby('quarter'):
        combined_text = ' '.join(group['text'].values)

        # Reform alignment
        reform_score = calculate_reform_alignment_tfidf(combined_text, vectorizer, reform_vectors)

        # Hardline ratio
        hardline_ratio = calculate_hardline_vs_compassionate_ratio(combined_text)

        # Immigration intensity
        total_words = len(combined_text.split())
        immigration_words = sum(1 for word in combined_text.lower().split()
                               if any(kw in word for kw in ['immigra', 'asylum', 'migrant', 'border', 'rwanda']))
        immigration_intensity = (immigration_words / total_words * 1000) if total_words > 0 else 0

        quarterly_scores.append({
            'quarter': str(quarter),
            'reform_alignment': reform_score,
            'hardline_ratio': hardline_ratio,
            'immigration_intensity': immigration_intensity,
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
            'recent_acceleration': 0.0,
            'quarters_tracked': len(df_timeline),
            'insufficient_data': True
        }

    # Linear regression on Reform alignment over time
    x = np.arange(len(df_timeline))
    y = df_timeline['reform_alignment'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate recent acceleration
    recent_avg = df_timeline.iloc[-4:]['reform_alignment'].mean()
    early_avg = df_timeline.iloc[:4]['reform_alignment'].mean()

    return {
        'radicalization_slope': float(slope),
        'trend_strength': float(r_value ** 2),
        'trend_significance': float(p_value),
        'current_alignment': float(df_timeline.iloc[-1]['reform_alignment']),
        'starting_alignment': float(df_timeline.iloc[0]['reform_alignment']),
        'total_change': float(df_timeline.iloc[-1]['reform_alignment'] -
                             df_timeline.iloc[0]['reform_alignment']),
        'recent_acceleration': float(recent_avg - early_avg),
        'quarters_tracked': len(df_timeline),
        'timeline': df_timeline.to_dict('records'),
        'insufficient_data': False
    }


# =============================================================================
# EXTREMISM RANKING
# =============================================================================

def calculate_extremism_percentile(mp_name, all_mp_alignments):
    """Calculate where MP falls on immigration hardline spectrum."""
    if mp_name not in all_mp_alignments:
        return {
            'extremism_percentile': 50.0,
            'reform_alignment_score': 0.0,
            'rank': 0,
            'total_mps': 0
        }

    target_score = all_mp_alignments[mp_name]
    scores = sorted(all_mp_alignments.values())

    rank_position = sum(1 for score in scores if score < target_score)
    percentile = (rank_position / len(scores)) * 100 if len(scores) > 0 else 50.0
    rank = len(scores) - rank_position

    return {
        'extremism_percentile': float(percentile),
        'reform_alignment_score': float(target_score),
        'rank': int(rank),
        'total_mps': len(scores)
    }


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_tfidf_analysis():
    """Main pipeline for TF-IDF-based speech analysis - Immigration focus."""
    print("=" * 80)
    print("ENHANCED SPEECH ANALYSIS - TF-IDF METHOD")
    print("Immigration Rhetoric Analysis")
    print("=" * 80)
    print()

    # Initialize TF-IDF vectorizer
    vectorizer = initialize_tfidf_model()

    # Load Hansard speeches
    print("\nLoading Hansard speeches...")
    speeches_path = HANSARD_CSV

    if not speeches_path.exists():
        print(f"ERROR: Speeches file not found at {speeches_path}")
        return None

    speeches_df = pd.read_csv(speeches_path)
    print(f"Loaded {len(speeches_df)} speeches")

    # Filter to Conservative MPs (using comprehensive list including historical MPs and defectors)
    print("\nFiltering to Conservative MPs...")
    conservative_speakers_path = ANALYSIS_OUTPUT / "conservative_speakers_in_hansard.csv"
    if conservative_speakers_path.exists():
        conservative_speakers_df = pd.read_csv(conservative_speakers_path)
        conservative_speakers = set(conservative_speakers_df['name'].values)
        speeches_df = speeches_df[speeches_df['speaker_name'].isin(conservative_speakers)]
        print(f"Filtered to {len(speeches_df)} Conservative speeches from {len(conservative_speakers)} MPs")
        print(f"  (Includes 2019-2024 MPs, 2024+ MPs, and defectors)")
    else:
        print(f"WARNING: {conservative_speakers_path} not found")
        print("  Run identify_conservative_speakers.py first to generate comprehensive MP list")
        print("  Falling back to all speakers in Hansard data...")
        conservative_speakers = set(speeches_df['speaker_name'].unique())

    # Fit TF-IDF on all speeches + Reform immigration rhetoric
    print("\nFitting TF-IDF model on corpus...")
    all_texts = list(speeches_df['text'].values[:15000])  # Larger sample for extended data
    all_texts.extend(REFORM_IMMIGRATION_RHETORIC)
    vectorizer.fit(all_texts)
    print(f"OK TF-IDF model fitted on {len(all_texts)} documents")

    # Transform rhetoric templates
    print("\nTransforming rhetoric templates...")
    immigration_vectors = vectorizer.transform(REFORM_IMMIGRATION_RHETORIC)
    print(f"OK Encoded {len(REFORM_IMMIGRATION_RHETORIC)} immigration statements")

    # Calculate Reform alignment for all MPs
    print("\nCalculating Reform UK immigration rhetoric alignment for all MPs...")
    print("(Immigration analysis - may take 5-10 minutes)")

    all_mp_alignments = {}
    mp_detailed_metrics = []

    unique_mps = speeches_df['speaker_name'].unique()
    total_mps = len(unique_mps)

    for idx, mp_name in enumerate(unique_mps, 1):
        if idx % 20 == 0:
            print(f"Processing MP {idx}/{total_mps}: {mp_name}")

        mp_speeches = speeches_df[speeches_df['speaker_name'] == mp_name]

        # Combine speeches for overall alignment (more speeches for extended data)
        all_text = ' '.join(mp_speeches['text'].values[:150])

        # Immigration alignment (Reform UK focus)
        immigration_alignment = calculate_reform_alignment_tfidf(all_text, vectorizer, immigration_vectors)

        # Hardline ratio
        immigration_hardline_ratio = calculate_hardline_vs_compassionate_ratio(all_text)

        # Calculate speech proportions (topic focus)
        total_words = len(all_text.split())

        immigration_words = sum(1 for word in all_text.lower().split()
                               if any(kw in word for kw in IMMIGRATION_FILTER_KEYWORDS))
        immigration_proportion = (immigration_words / total_words) if total_words > 0 else 0

        # Radicalization trajectory
        trajectory = analyze_radicalization_trajectory(speeches_df, mp_name, vectorizer, immigration_vectors)

        # Store alignment for percentile calculation
        all_mp_alignments[mp_name] = immigration_alignment

        mp_detailed_metrics.append({
            'speaker_name': mp_name,
            # Immigration metrics (primary Reform UK focus)
            'reform_alignment_raw': immigration_alignment,
            'immigration_hardline_ratio': immigration_hardline_ratio,
            'immigration_proportion': immigration_proportion,
            # Trajectory
            'radicalization_slope': trajectory['radicalization_slope'],
            'radicalization_strength': trajectory['trend_strength'],
            'current_alignment': trajectory['current_alignment'],
            'starting_alignment': trajectory['starting_alignment'],
            'alignment_change': trajectory['total_change'],
            'recent_acceleration': trajectory['recent_acceleration'],
            'quarters_tracked': trajectory['quarters_tracked'],
            'total_speeches': len(mp_speeches),
            'insufficient_data': trajectory['insufficient_data']
        })

    # Calculate extremism percentiles
    print("\nCalculating extremism percentiles...")
    for metric in mp_detailed_metrics:
        percentile_data = calculate_extremism_percentile(
            metric['speaker_name'],
            all_mp_alignments
        )
        metric['extremism_percentile'] = percentile_data['extremism_percentile']
        metric['rank'] = percentile_data['rank']
        metric['total_mps'] = percentile_data['total_mps']

    # Create DataFrame
    results_df = pd.DataFrame(mp_detailed_metrics)
    results_df = results_df.sort_values('reform_alignment_raw', ascending=False)

    # Calculate enhanced speech score (immigration-focused formula)
    print("\nCalculating enhanced speech scores...")
    results_df['enhanced_speech_score'] = (
        results_df['reform_alignment_raw'] * 0.40 +
        results_df['radicalization_slope'].clip(0, 1) * 10 * 0.25 +
        ((results_df['immigration_hardline_ratio'] + 1) / 2) * 0.20 +
        (results_df['extremism_percentile'] / 100) * 0.15
    )

    # Save results
    output_path = ANALYSIS_OUTPUT / "enhanced_speech_tfidf.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nOK Results saved to: {output_path}")

    # Generate summary report
    generate_summary_report(results_df)

    return results_df


def generate_summary_report(results_df):
    """Generate human-readable summary report."""

    report_path = ANALYSIS_OUTPUT / "enhanced_speech_tfidf_report.txt"

    lines = [
        "=" * 80,
        "ENHANCED SPEECH ANALYSIS REPORT - TF-IDF METHOD",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 80,
        "",
        "METHODOLOGY",
        "-" * 80,
        "Uses TF-IDF vectorization with cosine similarity to measure:",
        "1. Immigration rhetoric alignment (similarity to Reform UK immigration talking points)",
        "2. Radicalization trajectory (becoming more extreme over time)",
        "3. Hardline vs compassionate language ratios",
        "4. Extremism percentile ranking among Conservative MPs",
        "",
        "Method: TF-IDF (1-3 word phrases) + Cosine Similarity",
        "",
        "=" * 80,
        "",
        "TOP 25 MPS BY REFORM UK IMMIGRATION ALIGNMENT",
        "-" * 80,
        ""
    ]

    top_25 = results_df.head(25)

    for idx, (_, row) in enumerate(top_25.iterrows(), 1):
        lines.append(f"\n{idx:2}. {row['speaker_name']}")
        lines.append(f"    Reform Immigration Alignment: {row['reform_alignment_raw']:.4f}")
        lines.append(f"    Enhanced Speech Score: {row['enhanced_speech_score']:.3f}")
        lines.append(f"    Extremism Percentile: {row['extremism_percentile']:.1f}% (rank #{row['rank']}/{row['total_mps']})")
        lines.append(f"    Immigration Focus: {row['immigration_proportion']*100:.1f}%")

        if not row['insufficient_data']:
            lines.append(f"    Radicalization Slope: {row['radicalization_slope']:+.4f} per quarter")
        else:
            lines.append(f"    Radicalization: Insufficient data ({row['total_speeches']} speeches)")

    lines.extend([
        "",
        "=" * 80,
        "",
        "KEY FINDINGS",
        "-" * 80,
        ""
    ])

    lines.append(f"Total MPs analyzed: {len(results_df)}")
    lines.append("")
    lines.append("Reform Immigration Alignment:")
    lines.append(f"  Mean: {results_df['reform_alignment_raw'].mean():.4f}")
    lines.append(f"  Median: {results_df['reform_alignment_raw'].median():.4f}")
    lines.append(f"  Max: {results_df['reform_alignment_raw'].max():.4f}")
    lines.append("")

    # Radicalizing MPs
    radicalizing = results_df[
        (~results_df['insufficient_data']) &
        (results_df['radicalization_slope'] > 0.01)
    ]
    lines.append(f"MPs becoming MORE extreme over time: {len(radicalizing)}")
    if len(radicalizing) > 0:
        lines.append("  Top 5 radicalizing:")
        for _, row in radicalizing.nlargest(5, 'radicalization_slope').iterrows():
            lines.append(f"    - {row['speaker_name']}: +{row['radicalization_slope']:.4f} per quarter")

    lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])

    report = "\n".join(lines)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"OK Summary report saved to: {report_path}")

    # Print top 15 to console
    print("\n" + "=" * 80)
    print("TOP 15 MPS BY REFORM UK IMMIGRATION ALIGNMENT")
    print("=" * 80)
    for idx, (_, row) in enumerate(results_df.head(15).iterrows(), 1):
        print(f"\n{idx:2}. {row['speaker_name']}")
        print(f"    Reform Alignment: {row['reform_alignment_raw']:.4f} | Immigration Focus: {row['immigration_proportion']*100:.1f}%")


if __name__ == "__main__":
    results = run_tfidf_analysis()
