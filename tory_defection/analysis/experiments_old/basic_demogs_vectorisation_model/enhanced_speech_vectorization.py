"""
Enhanced Speech Vectorization Analysis
======================================

Uses semantic similarity and temporal analysis to detect:
1. Reform UK rhetoric alignment
2. Radicalization trajectories over time
3. Immigration hardline positioning
4. Extremism percentile ranking

Dependencies:
pip install sentence-transformers torch scikit-learn scipy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: sentence-transformers not installed")
    print("Run: pip install sentence-transformers torch")

from scipy import stats

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
HANSARD_DATA = SOURCE_DATA / "hansard"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"


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
    "The small boats trade is organized criminal activity that must be stopped"
]

HARDLINE_KEYWORDS = [
    'illegal', 'crisis', 'flood', 'invasion', 'control our borders',
    'small boats', 'channel crossings', 'failed', 'broken system',
    'stop the boats', 'take back control', 'sovereignty', 'mass immigration',
    'deport', 'detention', 'remove', 'echr', 'rwanda scheme',
    'economic migrants', 'abuse', 'exploiting', 'criminal gangs'
]

PRO_IMMIGRATION_KEYWORDS = [
    'humanitarian', 'refuge', 'compassion', 'safe routes',
    'international obligations', 'persecution', 'protection',
    'family reunion', 'sanctuary', 'welcome', 'diversity'
]


# =============================================================================
# SEMANTIC SIMILARITY FUNCTIONS
# =============================================================================

def initialize_model():
    """Load sentence transformer model."""
    if not HAS_TRANSFORMERS:
        return None

    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
    print("Model loaded successfully")
    return model


def calculate_reform_alignment_score(speech_text, model, reform_embeddings):
    """
    Calculate semantic similarity between speech and Reform UK rhetoric.

    Returns:
        float: 0-1 score (0=completely different, 1=identical rhetoric)
    """
    if not model or not speech_text or len(speech_text.strip()) < 50:
        return 0.0

    # Split into sentences
    sentences = [s.strip() for s in speech_text.split('.') if len(s.strip()) > 20]

    # Filter to immigration-related sentences only
    immigration_sentences = [
        s for s in sentences
        if any(kw in s.lower() for kw in
               ['immigra', 'asylum', 'border', 'migrant', 'rwanda', 'boat', 'channel'])
    ]

    if not immigration_sentences:
        return 0.0

    # Limit to 50 sentences max (for performance)
    immigration_sentences = immigration_sentences[:50]

    try:
        # Encode MP's immigration sentences
        mp_embeddings = model.encode(immigration_sentences, convert_to_tensor=True)

        # Calculate cosine similarity to each Reform statement
        similarities = util.cos_sim(mp_embeddings, reform_embeddings)

        # Return average of max similarities for each sentence
        max_similarities = torch.max(similarities, dim=1).values
        return float(torch.mean(max_similarities))
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

def analyze_radicalization_trajectory(speeches_df, mp_name, model, reform_embeddings):
    """
    Detect if MP is becoming more anti-immigration over time.

    Returns:
        dict: Radicalization metrics including slope, trend strength, acceleration
    """
    mp_speeches = speeches_df[speeches_df['speaker_name'] == mp_name].copy()

    if len(mp_speeches) < 10:  # Need minimum speeches for trend analysis
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

    # Calculate Reform alignment per quarter
    quarterly_scores = []

    for quarter, group in mp_speeches.groupby('quarter'):
        # Combine speeches for this quarter
        combined_text = ' '.join(group['speech'].values)

        # Calculate Reform alignment
        reform_score = calculate_reform_alignment_score(combined_text, model, reform_embeddings)

        # Calculate hardline ratio
        hardline_ratio = calculate_hardline_vs_compassionate_ratio(combined_text)

        # Immigration mention intensity
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

    if len(df_timeline) < 4:  # Need at least 4 quarters for trend
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

    # Calculate recent acceleration (last 4 quarters vs first 4 quarters)
    recent_avg = df_timeline.iloc[-4:]['reform_alignment'].mean()
    early_avg = df_timeline.iloc[:4]['reform_alignment'].mean()

    return {
        'radicalization_slope': float(slope),  # Change per quarter
        'trend_strength': float(r_value ** 2),  # R-squared
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
    """
    Calculate where MP falls on the immigration hardline spectrum.

    Returns:
        dict: Percentile, rank, and score
    """
    if mp_name not in all_mp_alignments:
        return {
            'extremism_percentile': 50.0,
            'reform_alignment_score': 0.0,
            'rank': 0,
            'total_mps': 0
        }

    target_score = all_mp_alignments[mp_name]
    scores = sorted(all_mp_alignments.values())

    # Find percentile (0=most pro-immigration, 100=most anti-immigration)
    rank_position = sum(1 for score in scores if score < target_score)
    percentile = (rank_position / len(scores)) * 100 if len(scores) > 0 else 50.0

    # Rank (1 = most extreme)
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

def run_enhanced_speech_analysis():
    """
    Main pipeline for enhanced speech vectorization analysis.
    """
    print("=" * 80)
    print("ENHANCED SPEECH VECTORIZATION ANALYSIS")
    print("=" * 80)
    print()

    # Check dependencies
    if not HAS_TRANSFORMERS:
        print("ERROR: sentence-transformers not installed")
        print("Run: pip install sentence-transformers torch")
        return None

    # Load model
    model = initialize_model()
    if not model:
        print("ERROR: Could not load sentence transformer model")
        return None

    # Encode Reform rhetoric once (reuse for all MPs)
    print("\nEncoding Reform UK rhetoric templates...")
    reform_embeddings = model.encode(REFORM_IMMIGRATION_RHETORIC, convert_to_tensor=True)
    print(f"Encoded {len(REFORM_IMMIGRATION_RHETORIC)} Reform UK rhetoric examples")

    # Load Hansard speeches
    print("\nLoading Hansard speeches...")
    speeches_path = HANSARD_DATA / "all_speeches.csv"

    if not speeches_path.exists():
        print(f"ERROR: Speeches file not found at {speeches_path}")
        return None

    speeches_df = pd.read_csv(speeches_path)
    print(f"Loaded {len(speeches_df)} speeches")

    # Filter to Conservative MPs only
    print("\nFiltering to Conservative MPs...")
    # Load existing speech analysis to get Conservative MP list
    speech_analysis_path = ANALYSIS_OUTPUT / "speech_analysis.csv"
    if speech_analysis_path.exists():
        existing_analysis = pd.read_csv(speech_analysis_path)
        conservative_speakers = set(existing_analysis['speaker_name'].values)
        speeches_df = speeches_df[speeches_df['speaker_name'].isin(conservative_speakers)]
        print(f"Filtered to {len(speeches_df)} Conservative speeches from {len(conservative_speakers)} MPs")

    # Calculate Reform alignment for all MPs
    print("\nCalculating Reform UK rhetoric alignment for all MPs...")
    print("(This may take 5-10 minutes for 200k speeches)")

    all_mp_alignments = {}
    mp_detailed_metrics = []

    unique_mps = speeches_df['speaker_name'].unique()
    total_mps = len(unique_mps)

    for idx, mp_name in enumerate(unique_mps, 1):
        if idx % 10 == 0:
            print(f"Processing MP {idx}/{total_mps}: {mp_name}")

        mp_speeches = speeches_df[speeches_df['speaker_name'] == mp_name]

        # Combine all speeches for overall alignment
        all_text = ' '.join(mp_speeches['speech'].values[:100])  # Sample max 100 speeches
        overall_alignment = calculate_reform_alignment_score(all_text, model, reform_embeddings)

        # Calculate hardline ratio
        hardline_ratio = calculate_hardline_vs_compassionate_ratio(all_text)

        # Calculate radicalization trajectory
        trajectory = analyze_radicalization_trajectory(speeches_df, mp_name, model, reform_embeddings)

        # Store overall alignment for percentile calculation
        all_mp_alignments[mp_name] = overall_alignment

        # Store detailed metrics
        mp_detailed_metrics.append({
            'speaker_name': mp_name,
            'reform_alignment_score': overall_alignment,
            'hardline_compassion_ratio': hardline_ratio,
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
        metric.update(percentile_data)

    # Create DataFrame
    results_df = pd.DataFrame(mp_detailed_metrics)

    # Sort by Reform alignment (most extreme first)
    results_df = results_df.sort_values('reform_alignment_score', ascending=False)

    # Calculate enhanced speech score (combining all metrics)
    print("\nCalculating enhanced speech scores...")
    results_df['enhanced_speech_score'] = (
        results_df['reform_alignment_score'] * 0.35 +  # Current alignment
        results_df['radicalization_slope'].clip(0, 1) * 10 * 0.25 +  # Trajectory (scaled)
        ((results_df['hardline_compassion_ratio'] + 1) / 2) * 0.20 +  # Hardline ratio normalized
        (results_df['extremism_percentile'] / 100) * 0.15 +  # Relative position
        (results_df['total_speeches'].clip(0, 100) / 100) * 0.05  # Volume
    )

    # Save results
    output_path = ANALYSIS_OUTPUT / "enhanced_speech_vectorization.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Generate summary report
    generate_summary_report(results_df)

    return results_df


def generate_summary_report(results_df):
    """Generate human-readable summary report."""

    report_path = ANALYSIS_OUTPUT / "enhanced_speech_report.txt"

    lines = [
        "=" * 80,
        "ENHANCED SPEECH VECTORIZATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 80,
        "",
        "METHODOLOGY",
        "-" * 80,
        "Uses semantic similarity (sentence transformers) to measure:",
        "1. Reform UK rhetoric alignment (cosine similarity to Reform talking points)",
        "2. Radicalization trajectory (becoming more extreme over time)",
        "3. Hardline vs compassionate language ratio",
        "4. Extremism percentile ranking among Conservative MPs",
        "",
        "Enhanced Speech Score = (Alignment×0.35) + (Radicalization×0.25) + ",
        "                        (Hardline×0.20) + (Percentile×0.15) + (Volume×0.05)",
        "",
        "=" * 80,
        "",
        "TOP 25 MPs BY REFORM UK RHETORIC ALIGNMENT",
        "-" * 80,
        ""
    ]

    top_25 = results_df.head(25)

    for idx, (_, row) in enumerate(top_25.iterrows(), 1):
        lines.append(f"\n{idx:2}. {row['speaker_name']}")
        lines.append(f"    Reform Alignment Score: {row['reform_alignment_score']:.3f}")
        lines.append(f"    Enhanced Speech Score: {row['enhanced_speech_score']:.3f}")
        lines.append(f"    Extremism Percentile: {row['extremism_percentile']:.1f}% (rank #{row['rank']}/{row['total_mps']})")
        lines.append(f"    Hardline/Compassion: {row['hardline_compassion_ratio']:+.2f}")

        if not row['insufficient_data']:
            lines.append(f"    Radicalization Slope: {row['radicalization_slope']:+.4f} per quarter")
            lines.append(f"    Alignment Change: {row['alignment_change']:+.3f} over {row['quarters_tracked']} quarters")
            lines.append(f"    Recent Acceleration: {row['recent_acceleration']:+.3f}")
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

    # Summary statistics
    lines.append(f"Total MPs analyzed: {len(results_df)}")
    lines.append(f"Average Reform alignment: {results_df['reform_alignment_score'].mean():.3f}")
    lines.append(f"Median Reform alignment: {results_df['reform_alignment_score'].median():.3f}")
    lines.append(f"Highest alignment: {results_df['reform_alignment_score'].max():.3f}")
    lines.append("")

    # MPs with positive radicalization
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

    print(f"Summary report saved to: {report_path}")

    # Print top 10 to console
    print("\n" + "=" * 80)
    print("TOP 10 MPs BY REFORM UK RHETORIC ALIGNMENT")
    print("=" * 80)
    for idx, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n{idx:2}. {row['speaker_name']}")
        print(f"    Reform Alignment: {row['reform_alignment_score']:.3f} | Enhanced Score: {row['enhanced_speech_score']:.3f}")
        print(f"    Rank: #{row['rank']}/{row['total_mps']} | Percentile: {row['extremism_percentile']:.1f}%")


if __name__ == "__main__":
    results = run_enhanced_speech_analysis()
