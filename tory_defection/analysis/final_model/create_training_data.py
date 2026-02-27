"""
Create training data
=============================================

This script defines the training data, i.e. the features of defectors and non-defectors 
from the Conservative Party to Reform UK,all of whom were MPs in the 2019-2024 Parliament,
which forms the basis of our dataset. To this we merge data on MP careers, ministerial tenure,
ages, speech analysis features, and Conservative Party faction memberships based on secondary research.

We then use these base variables to create a number of additional scoreing variables in training.
This is to assist the supervised learning model in identifying complex patterns associated with defection 
risk - e.g., previous ministers may be less likely to defect overall, but disgruntled middle-aged ex-ministers
looking at a stalling career in the Conservative Party are probably more likely to defect.

"""

import pandas as pd
from pathlib import Path

# Import desk research - faction membership data
from faction_membership_data import get_all_faction_data

# set up training dataset with 2019-2024 Conservative MPs
base_dir = Path(__file__).parent.parent.parent
mp_careers = base_dir / "source_data" / "mp_careers"
elections_2019 = base_dir / "source_data" / "elections" / "HoC-GE2019-results-by-candidate.xlsx"

df = pd.read_excel(elections_2019, header=2)
winners = df.loc[df.groupby('Constituency name')['Votes'].idxmax()]
con_mps = winners[winners['Party abbreviation'] == 'Con']

training_data = pd.DataFrame({
    'name': con_mps['Candidate first name'] + ' ' + con_mps['Candidate surname'],
    'constituency': con_mps['Constituency name']
})

# merge MP tenure
mp_tenure = pd.read_csv(mp_careers / "mp_tenure.csv")
training_data = training_data.merge(
    mp_tenure[['name', 'membership_end', 'total_years_as_mp']],
    on='name', how='left'
)

# merge ministerial tenure
ministerial = pd.read_csv(mp_careers / "ministerial_tenure.csv")
training_data = training_data.merge(
    ministerial[['name', 'total_years_no_double_count', 'government_positions', 'final_post_end_date']],
    on='name', how='left'
)
training_data.rename(columns={'total_years_no_double_count': 'ministerial_years'}, inplace=True)

# parse highest ministerial rank from government_positions
# Rank 5: Prime Minister
# Rank 4: Cabinet (Secretary of State, Chancellor)
# Rank 3: Minister of State
# Rank 2: Parliamentary Under-Secretary
# Rank 1: PPS/Whip/Assistant Whip
# Rank 0: No ministerial role
def parse_highest_ministerial_rank(positions):
    if pd.isna(positions) or positions == '':
        return 0
    # split by semicolon and check each role
    roles = [r.strip().upper() for r in str(positions).split(';')]
    max_rank = 0
    for role in roles:
        # check for actual Prime Minister (starts with "Prime Minister")
        if role.startswith('PRIME MINISTER'):
            return 5
        if 'SECRETARY OF STATE' in role or 'CHANCELLOR OF THE EXCHEQUER' in role or role.startswith('FIRST SECRETARY'):
            max_rank = max(max_rank, 4)
        # Deputy PM is cabinet level
        elif 'DEPUTY PRIME MINISTER' in role:
            max_rank = max(max_rank, 4)
        elif 'MINISTER OF STATE' in role or role.startswith('MINISTER FOR'):
            max_rank = max(max_rank, 3)
        elif 'UNDER-SECRETARY' in role or 'PARLIAMENTARY SECRETARY' in role:
            max_rank = max(max_rank, 2)
        elif 'WHIP' in role or 'PPS' in role or 'COMMISSIONER' in role:
            max_rank = max(max_rank, 1)
    return max_rank

training_data['highest_ministerial_rank'] = training_data['government_positions'].apply(parse_highest_ministerial_rank)
training_data.drop(columns=['government_positions'], inplace=True)

# merge ages
ages = pd.read_csv(mp_careers / "mp_dobs_ages.csv")
training_data = training_data.merge(ages[['name', 'date_of_birth', 'age']], on='name', how='left')

# calculate years on backbench since last ministerial appointment
training_data['membership_end'] = pd.to_datetime(training_data['membership_end'])
training_data['final_post_end_date'] = pd.to_datetime(training_data['final_post_end_date'])

training_data['years_on_backbench_since_last_ministerial'] = None
for idx, row in training_data.iterrows():
    if pd.isna(row['final_post_end_date']) or row['ministerial_years'] == 0:
        # never had ministerial appointment
        training_data.at[idx, 'years_on_backbench_since_last_ministerial'] = None
    elif pd.notna(row['membership_end']) and pd.notna(row['final_post_end_date']):
        days = (row['membership_end'] - row['final_post_end_date']).days
        training_data.at[idx, 'years_on_backbench_since_last_ministerial'] = round(max(0, days) / 365.25, 2)

# calculate total backbench years (total MP years minus ministerial years)
training_data['total_backbench_years'] = (
    training_data['total_years_as_mp'] - training_data['ministerial_years'].fillna(0)
).round(2)

# create binary flag for ever having held ministerial role
training_data['ever_minister'] = (training_data['ministerial_years'].fillna(0) > 0).astype(int)

# clean up date columns for output
training_data['membership_end'] = training_data['membership_end'].dt.strftime('%Y-%m-%d')
training_data['final_post_end_date'] = training_data['final_post_end_date'].dt.strftime('%Y-%m-%d')

# merge speech analysis features
speech_feature_cols = [
    'name', 'total_speeches', 'immigration_speeches', 'immigration_speech_proportion',
    'reform_alignment', 'hardline_ratio', 'radicalization_slope', 'trend_strength',
    'current_alignment', 'starting_alignment', 'alignment_change', 'extremism_percentile'
]
speech_features_path = Path(__file__).parent / "training_speech_features.csv"
if speech_features_path.exists():
    speech_features = pd.read_csv(speech_features_path)
    training_data = training_data.merge(
        speech_features[speech_feature_cols],
        on='name', how='left'
    )
else:
    print("training_speech_features.csv not found; bootstrapping with zeroed speech features.")
    for col in speech_feature_cols:
        if col != "name":
            training_data[col] = 0.0

# merge defection data (binary target variable)
defections = pd.read_csv(base_dir / "source_data" / "defection_tracker" / "training_defections.csv")
defector_names = set(defections['name'].values)
training_data['defected'] = training_data['name'].apply(lambda x: 1 if x in defector_names else 0)

# =============================================================================
# INTERACTION FEATURES
# =============================================================================
# We create several interaction features to help capture more complex patterns associated with defection risk.

# Interaction: reform alignment * years on backbench since leaving ministerial role
# Captures: ideological rebels who've been pushed out of government
training_data['reform_x_backbench_since_ministerial'] = (
    training_data['reform_alignment'] *
    training_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# Interaction: reform alignment * total backbench years (for those never in government)
# Captures: long-serving backbenchers with reform sympathies
training_data['reform_x_total_backbench'] = (
    training_data['reform_alignment'] * training_data['total_backbench_years']
)

# Interaction: extremism percentile * years since ministerial role
# Captures: extreme rhetoric from sidelined ex-ministers
training_data['extremism_x_backbench_since_ministerial'] = (
    training_data['extremism_percentile'] *
    training_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# Party loyalty indicator: currently/recently in senior leadership reduces defection risk
# Rank 5 = PM, Rank 4 = Cabinet - these are party loyalists unlikely to defect
# Scale: higher = more loyal to party establishment
training_data['senior_party_stake'] = (
    (training_data['highest_ministerial_rank'] >= 4).astype(int) *
    (training_data['years_on_backbench_since_last_ministerial'].fillna(99) < 2).astype(int)
)

# Interaction: ever_minister * years on backbench since ministerial role
# Captures: sidelined ex-ministers (longer time = more disaffected)
# Only applies to those who were ever ministers
training_data['sidelined_minister_years'] = (
    training_data['ever_minister'] *
    training_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# =============================================================================
# FACTION MEMBERSHIP FEATURES
# =============================================================================
# Research-backed faction membership data from faction_membership_data.py
# Official and unofficial membership of factions allows an additional dimension by which
# to assess ideological alignment, especially for MPs who speak less in the Commons.
# Sources: Wikipedia, The Spectator, Conservative Home, etc.

faction_df = get_all_faction_data(training_data['name'])
training_data = training_data.merge(faction_df, on='name', how='left')

# Fill any missing faction values (MPs not in any faction)
faction_cols = ['is_erg', 'is_csg', 'is_new_conservative', 'is_rwanda_rebel',
                'is_one_nation', 'is_trg', 'is_party_leader', 'faction_composite_score']
for col in faction_cols:
    if col in training_data.columns:
        training_data[col] = training_data[col].fillna(0)

print(f"  Faction membership in training data:")
print(f"    ERG members: {training_data['is_erg'].sum():.0f}")
print(f"    Common Sense Group: {training_data['is_csg'].sum():.0f}")
print(f"    New Conservatives: {training_data['is_new_conservative'].sum():.0f}")
print(f"    Rwanda rebels: {training_data['is_rwanda_rebel'].sum():.0f}")
print(f"    One Nation: {training_data['is_one_nation'].sum():.0f}")
print(f"    TRG: {training_data['is_trg'].sum():.0f}")
print(f"    Party leaders: {training_data['is_party_leader'].sum():.0f}")

# =============================================================================
# COMPOSITE SCORE FEATURE SET (aligned with modeling scripts)
# =============================================================================
# These engineered features match the composite score inputs used in training/testing.

from faction_membership_data import FACTION_WEIGHTS

training_data['career_stagnation'] = (
    (training_data['ever_minister'] == 0) |
    (training_data['sidelined_minister_years'] >= 2)
).astype(float)

training_data['rightwing_intensity'] = (
    training_data['extremism_percentile'] / 100 +
    training_data['hardline_ratio'].clip(0, 1) +
    training_data['reform_alignment'] * 2
) / 3

training_data['backbench_frustration'] = (
    training_data['total_backbench_years'] / 20 *
    training_data['rightwing_intensity']
)

training_data['sidelined_rebel'] = (
    training_data['ever_minister'] *
    training_data['sidelined_minister_years'] / 10 *
    training_data['rightwing_intensity']
)

training_data['immigration_focus'] = training_data['immigration_speech_proportion'] * 10
training_data['radicalizing'] = training_data['radicalization_slope'].clip(0, None) * 100
training_data.drop(columns=['radicalization_slope'], inplace=True)
training_data['never_minister_rebel'] = (
    (training_data['ever_minister'] == 0) * training_data['rightwing_intensity'] * 2
)

training_data['establishment_loyalty'] = (
    (training_data['highest_ministerial_rank'] >= 4) &
    (training_data['sidelined_minister_years'] < 2)
).astype(float) * -1

training_data['rightwing_faction_score'] = (
    training_data['is_erg'] * FACTION_WEIGHTS['ergs'] +
    training_data['is_csg'] * FACTION_WEIGHTS['csg'] +
    training_data['is_new_conservative'] * FACTION_WEIGHTS['new_cons'] +
    training_data['is_rwanda_rebel'] * FACTION_WEIGHTS['rwanda_rebel']
)

training_data['moderate_faction_score'] = (
    training_data['is_one_nation'] * abs(FACTION_WEIGHTS['one_nation']) +
    training_data['is_trg'] * abs(FACTION_WEIGHTS['trg'])
)

training_data['net_faction_score'] = (
    training_data['rightwing_faction_score'] - training_data['moderate_faction_score']
)

training_data['party_leader_penalty'] = training_data['is_party_leader'] * FACTION_WEIGHTS['party_leader']

training_data.to_csv(Path(__file__).parent / "training_data.csv", index=False)
print(f"Saved {len(training_data)} MPs to training_data.csv")
print(f"  Defectors: {training_data['defected'].sum()}")
