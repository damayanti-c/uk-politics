"""
Create test data
=============================================

This script defines the test data, i.e. sitting Conservative MPs in the current (2024) Parliament,
which is the population we want to score for defection risk. To this we merge data on MP careers,
ministerial tenure, ages, speech analysis features, and Conservative Party faction memberships
based on secondary research.

We then use these base variables to create a number of additional scoring variables in testing.
This mirrors the feature engineering in create_training_data.py so the trained model can be applied
consistently to the current sitting MPs.

"""

import pandas as pd
from pathlib import Path

# Import desk research - faction membership data
from faction_membership_data import get_all_faction_data

# =============================================================================
# ADD BASE FEATURES
# =============================================================================

# set up test dataset with 2024+ sitting Conservative MPs
base_dir = Path(__file__).parent.parent.parent
mp_careers = base_dir / "source_data" / "mp_careers"
elections_2024 = base_dir / "source_data" / "elections" / "HoC-GE2024-results-by-constituency.csv"

df = pd.read_csv(elections_2024)
con_mps = df[df['First party'] == 'Con']

test_data = pd.DataFrame({
    'name': con_mps['Member first name'] + ' ' + con_mps['Member surname'],
    'constituency': con_mps['Constituency name']
})

# merge MP tenure
mp_tenure = pd.read_csv(mp_careers / "sitting_mp_tenure.csv")
test_data = test_data.merge(
    mp_tenure[['name', 'membership_end', 'total_years_as_mp']],
    on='name', how='left'
)

# merge ministerial tenure
ministerial = pd.read_csv(mp_careers / "sitting_ministerial_tenure.csv")
test_data = test_data.merge(
    ministerial[['name', 'total_years_no_double_count', 'government_positions', 'final_post_end_date']],
    on='name', how='left'
)
test_data.rename(columns={'total_years_no_double_count': 'ministerial_years'}, inplace=True)

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

test_data['highest_ministerial_rank'] = test_data['government_positions'].apply(parse_highest_ministerial_rank)
test_data.drop(columns=['government_positions'], inplace=True)

# merge ages
ages = pd.read_csv(mp_careers / "sitting_mp_dobs_ages.csv")
test_data = test_data.merge(ages[['name', 'date_of_birth', 'age']], on='name', how='left')

# calculate years on backbench since last ministerial appointment
test_data['membership_end'] = pd.to_datetime(test_data['membership_end'])
test_data['final_post_end_date'] = pd.to_datetime(test_data['final_post_end_date'])

test_data['years_on_backbench_since_last_ministerial'] = None
for idx, row in test_data.iterrows():
    if pd.isna(row['final_post_end_date']) or row['ministerial_years'] == 0:
        # never had ministerial appointment
        test_data.at[idx, 'years_on_backbench_since_last_ministerial'] = None
    elif pd.notna(row['membership_end']) and pd.notna(row['final_post_end_date']):
        days = (row['membership_end'] - row['final_post_end_date']).days
        test_data.at[idx, 'years_on_backbench_since_last_ministerial'] = round(max(0, days) / 365.25, 2)

# calculate total backbench years (total MP years minus ministerial years)
test_data['total_backbench_years'] = (
    test_data['total_years_as_mp'] - test_data['ministerial_years'].fillna(0)
).round(2)

# create binary flag for ever having held ministerial role
test_data['ever_minister'] = (test_data['ministerial_years'].fillna(0) > 0).astype(int)

# clean up date columns for output
test_data['membership_end'] = test_data['membership_end'].dt.strftime('%Y-%m-%d')
test_data['final_post_end_date'] = test_data['final_post_end_date'].dt.strftime('%Y-%m-%d')

# merge speech analysis features
speech_features = pd.read_csv(Path(__file__).parent / "test_speech_features.csv")
test_data = test_data.merge(
    speech_features[[
        'name', 'total_speeches', 'immigration_speeches', 'immigration_speech_proportion',
        'reform_alignment', 'hardline_ratio', 'radicalization_slope', 'trend_strength',
        'current_alignment', 'starting_alignment', 'alignment_change', 'extremism_percentile'
    ]],
    on='name', how='left'
)

# =============================================================================
# INTERACTION FEATURES
# =============================================================================
# We create several interaction features to help capture more complex patterns associated with defection risk.

# Interaction: reform alignment * years on backbench since leaving ministerial role
# Theory: captures ideological rebels who've been pushed out of government
test_data['reform_x_backbench_since_ministerial'] = (
    test_data['reform_alignment'] *
    test_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# Interaction: reform alignment * total backbench years (for those never in government)
# Theory: captures long-serving backbenchers with reform sympathies
test_data['reform_x_total_backbench'] = (
    test_data['reform_alignment'] * test_data['total_backbench_years']
)

# Interaction: extremism percentile * years since ministerial role
# Theory: captures extreme rhetoric from sidelined ex-ministers
test_data['extremism_x_backbench_since_ministerial'] = (
    test_data['extremism_percentile'] *
    test_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# Party loyalty indicator: currently/recently in senior leadership reduces defection risk
# Rank 5 = PM, Rank 4 = Cabinet - these are party loyalists unlikely to defect
# Theory: captures higher = more loyal to party establishment
test_data['senior_party_stake'] = (
    (test_data['highest_ministerial_rank'] >= 4).astype(int) *
    (test_data['years_on_backbench_since_last_ministerial'].fillna(99) < 2).astype(int)
)

# Interaction: ever_minister * years on backbench since ministerial role
# Theory: captures sidelined ex-ministers (longer time = more disaffected)
# Only applies to those who were ever ministers
test_data['sidelined_minister_years'] = (
    test_data['ever_minister'] *
    test_data['years_on_backbench_since_last_ministerial'].fillna(0)
)

# =============================================================================
# FACTION MEMBERSHIP FEATURES
# =============================================================================
# Research-backed faction membership data from faction_membership_data.py
# Official and unofficial membership of factions allows an additional dimension by which
# to assess ideological alignment, especially for MPs who speak less in the Commons.
# Sources: Wikipedia, The Spectator, Conservative Home, etc.

from faction_membership_data import FACTION_WEIGHTS

faction_df = get_all_faction_data(test_data['name'])
test_data = test_data.merge(faction_df, on='name', how='left')

# Fill any missing faction values (MPs not in any faction)
faction_cols = ['is_erg', 'is_csg', 'is_new_conservative', 'is_rwanda_rebel',
                'is_one_nation', 'is_trg', 'is_party_leader', 'faction_composite_score']
for col in faction_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna(0)

# Create faction score features

test_data['rightwing_faction_score'] = (
    test_data['is_erg'] * FACTION_WEIGHTS['ergs'] +
    test_data['is_csg'] * FACTION_WEIGHTS['csg'] +
    test_data['is_new_conservative'] * FACTION_WEIGHTS['new_cons'] +
    test_data['is_rwanda_rebel'] * FACTION_WEIGHTS['rwanda_rebel']
)

test_data['moderate_faction_score'] = (
    test_data['is_one_nation'] * abs(FACTION_WEIGHTS['one_nation']) +
    test_data['is_trg'] * abs(FACTION_WEIGHTS['trg'])
)

test_data['net_faction_score'] = (
    test_data['rightwing_faction_score'] - test_data['moderate_faction_score']
)

test_data['party_leader_penalty'] = test_data['is_party_leader'] * FACTION_WEIGHTS['party_leader']

print(f"  Faction membership in test data:")
print(f"    ERG members: {test_data['is_erg'].sum():.0f}")
print(f"    Common Sense Group: {test_data['is_csg'].sum():.0f}")
print(f"    New Conservatives: {test_data['is_new_conservative'].sum():.0f}")
print(f"    Rwanda rebels: {test_data['is_rwanda_rebel'].sum():.0f}")
print(f"    One Nation: {test_data['is_one_nation'].sum():.0f}")
print(f"    TRG: {test_data['is_trg'].sum():.0f}")
print(f"    Party leaders: {test_data['is_party_leader'].sum():.0f}")

# =============================================================================
# ENGINEERED FEATURES
# =============================================================================
# We create these additional engineered features from our input data to reflect variables 
# that are not directly observable but are important for modelling defection risk
# according to our theory of change

# Career stagnation: never made minister or career has slowed down since getting a ministerial role
test_data['career_stagnation'] = (
    (test_data['ever_minister'] == 0) |
    (test_data['sidelined_minister_years'] >= 2)
).astype(float)

# Right-wing intensity score: combines Reform alignment, hardline attitudes to immigration, 
# and how extreme they are on immgiration relative to other MPs
test_data['rightwing_intensity'] = (
    test_data['extremism_percentile'] / 100 +
    test_data['hardline_ratio'].clip(0, 1) +
    test_data['reform_alignment'] * 2
) / 3

# Right-wing MP feeling stuck on bachbenches
test_data['backbench_frustration'] = (
    test_data['total_backbench_years'] / 20 *
    test_data['rightwing_intensity']
)

# Ex-ministers with high right-wing intensity who have been out of government for a while
test_data['sidelined_rebel'] = (
    test_data['ever_minister'] *
    test_data['sidelined_minister_years'] / 10 *
    test_data['rightwing_intensity']
)

# Immigration is key issue for MP
test_data['immigration_focus'] = test_data['immigration_speech_proportion'] * 10

# MP had become more right wing over time
test_data['radicalizing'] = test_data['radicalization_slope'].clip(0, None) * 100
test_data.drop(columns=['radicalization_slope'], inplace=True)

# MP is a right-wing backbencher who has never held ministerial office
test_data['never_minister_rebel'] = (
    (test_data['ever_minister'] == 0) * test_data['rightwing_intensity'] * 2
)
# MP is a loyalist establishment figure close to establishment who has had a good career
test_data['establishment_loyalty'] = (
    (test_data['highest_ministerial_rank'] >= 4) &
    (test_data['sidelined_minister_years'] < 2)
).astype(float) * -1


# =============================================================================
test_data.to_csv(Path(__file__).parent / "test_data.csv", index=False)
print(f"Saved {len(test_data)} MPs to test_data.csv")
