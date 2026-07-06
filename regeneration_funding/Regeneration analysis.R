library(tidyverse)
library(dplyr)
library(ggplot2)
library(readr)
library(sf)
library(readxl)
library(purrr)
library(stringr)
library(MatchIt)
library(tidyr)
library(grf)
library(Matching)
library(rgenoud)
library(rlemon)
library(optmatch)

# load data ####

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


funding <- read.csv("./Data for political and demographic analysis/Regeneration funding initatives - Funding per LAD per capita.csv")

funding$funding <- parse_number(funding$Per.capita.funding..exc..outliers...islands.and.the.Highlands.)
funding$lab_funding <- parse_number(funding$Per.capita.funding.from.Labour.governments.per.year.in.power)
funding$con_funding <- parse_number(funding$Per.capita.funding.from.Conservative.governments.per.year.in.power)

lad_shp <- read_sf("./Data for political and demographic analysis/LAD_MAY_2024_UK_BGC.shp")

deprivation_ew <- read.csv("./Data for political and demographic analysis/custom-filtered-2025-11-13T17_20_37Z.csv")
deprivation_ni <- read.csv("./Data for political and demographic analysis/ni-census21-household-lgd14+hh_deprivation-a858ca61.csv")
deprivation_s <- read.csv("./Data for political and demographic analysis/table_2025-12-09_12-24-59.csv")

mrp_q4_2024 <- read_excel("./Data for political and demographic analysis/Political_MRP_by_quarter.xlsx", sheet = "Q4 2024")
mrp_q1_2025 <- read_excel("./Data for political and demographic analysis/Political_MRP_by_quarter.xlsx", sheet = "Q1 2025")
mrp_q2_2025 <- read_excel("./Data for political and demographic analysis/Political_MRP_by_quarter.xlsx", sheet = "Q2 2025")
mrp_q3_2025 <- read_excel("./Data for political and demographic analysis/Political_MRP_by_quarter.xlsx", sheet = "Q3 2025")
mrp_q4_2025 <- read_excel("./Data for political and demographic analysis/Political_MRP_by_quarter.xlsx", sheet = "Q4 2025")

lad2_to_lad23_24 <- read.csv("./Data for political and demographic analysis/Regeneration funding initatives - LAD22 to LAD23_24.csv")
scot_DZ_to_lad <- read.csv("./Data for political and demographic analysis/DataZone2011lookup_2024-12-16.csv")
scot_ward_lad_lookup <- scot_DZ_to_lad %>% 
  dplyr::distinct(MMWard_Code, .keep_all = TRUE) %>% 
  dplyr::select(MMWard_Code, LA_Code, LA_Name)
ward_lad24_lookup <- read.csv("./Data for political and demographic analysis/Ward_to_Local_Authority_District_(May_2024)_Lookup_in_the_UK (1).csv")
ward_lad22_lookup <- read.csv("./Data for political and demographic analysis/ward to LAD.csv")


cob_ew <- read.csv("./Data for political and demographic analysis/custom-filtered-2025-12-10T16_45_43Z.csv")
cob_ni <- read.csv("./Data for political and demographic analysis/ni-census21-people-lgd14+cob_agg3-3d9d3527.csv")
cob_s <- read.csv("./Data for political and demographic analysis/table_2025-12-10_16-44-04.csv")

# maps ####

lad_map <- lad_shp %>%
  left_join(funding, by = "LAD24CD")

ggplot(lad_map) +
  geom_sf(aes(fill = funding), colour = NA) +
  scale_fill_gradient(low = "white", high = "orange", na.value = "grey90") +
  labs(
    title = "Per-capita funding",
    fill  = "£ per year"
  ) +
  coord_sf(expand = FALSE) +
  theme_void() 

ggplot(lad_map) +
  geom_sf(aes(fill = con_funding), colour = NA) +
  scale_fill_gradient(low = "white", high = "orange", na.value = "grey90") +
  labs(
    title = "Per-capita funding from Conservative governments",
    fill  = "£ per year"
  ) +
  coord_sf(expand = FALSE) +
  theme_void() 

ggplot(lad_map) +
  geom_sf(aes(fill = lab_funding), colour = NA) +
  scale_fill_gradient(low = "white", high = "orange", na.value = "grey90") +
  labs(
    title = "Per-capita funding from Labour governments",
    fill  = "£ per year"
  ) +
  coord_sf(expand = FALSE) +
  theme_void() 

# deprivation ####

# England and Wales

deprivation_ew <- deprivation_ew %>% rename(lad = Lower.tier.local.authorities.Code, lad_nm = Lower.tier.local.authorities)

all <- deprivation_ew %>% group_by(lad) %>% summarise(total = sum(Observation))

deprivation_ew <- deprivation_ew %>%
  group_by(lad, lad_nm, Household.deprivation..6.categories.) %>%
  summarise(Observation = sum(Observation), .groups = 'drop')

deprivation_ew <- deprivation_ew %>%
  pivot_wider(
    names_from = Household.deprivation..6.categories., 
    values_from = Observation)

deprivation_ew <- left_join(deprivation_ew, all, by = "lad")

deprivation_ew <- deprivation_ew %>%
  mutate(
    deprivation_mean = (.[[8]] * 0 + 
                          .[[5]] * 1 + 
                          .[[7]] * 2 + 
                          .[[6]] * 3 + 
                          .[[4]] * 4) / total)

deprivation_ew <- deprivation_ew %>% dplyr::select(lad, lad_nm, deprivation_mean)

# NI deprivation

deprivation_ni <- deprivation_ni %>% rename(lad = Local.Government.District.2014.Code, lad_nm = Local.Government.District.2014.Label)

all <- deprivation_ni %>% group_by(lad) %>% summarise(total = sum(Count))

deprivation_ni <- deprivation_ni %>%
  group_by(lad, lad_nm, Household.Deprivation.Label) %>%
  summarise(Count = sum(Count), .groups = 'drop')

deprivation_ni <- deprivation_ni %>%
  pivot_wider(
    names_from = Household.Deprivation.Label, 
    values_from = Count)

deprivation_ni <- left_join(deprivation_ni, all, by = "lad")

deprivation_ni <- deprivation_ni %>%
  mutate(
    deprivation_mean = (.[[7]] * 0 + 
                          .[[3]] * 1 + 
                          .[[4]] * 2 + 
                          .[[5]] * 3 + 
                          .[[6]] * 4) / total
  )

deprivation_ni <- deprivation_ni %>% dplyr::select(lad, lad_nm, deprivation_mean)

# S deprivation

deprivation_s <- deprivation_s %>% rename(lad = council_area_2019)

deprivation_s <- deprivation_s %>%
  mutate(
    deprivation_mean = (.[[2]] * 0 + 
                          .[[3]] * 1 + 
                          .[[4]] * 2 + 
                          .[[5]] * 3 + 
                          .[[6]] * 4) / Total
  )

deprivation_s <- deprivation_s %>% 
  dplyr::mutate(lad_nm = NA) %>% 
  dplyr::select(lad, lad_nm, deprivation_mean)

# stack

deprivation <- rbind(deprivation_ew, deprivation_ni, deprivation_s)

# compare with funding

merged_deprivation <- deprivation %>% 
  left_join(lad2_to_lad23_24, by = c("lad" = "LAD22CD")) %>% 
  left_join(funding, by = c("LAD23CD" = "LAD24CD"))

merged_deprivation <- merged_deprivation %>%
  mutate(funding = funding %>%
           gsub("£", "", .) %>%
           gsub(",", "", .) %>%
           as.numeric()
  )

ggplot(merged_deprivation, aes(x = deprivation_mean, y = funding)) +
  geom_point(alpha = 0.6) +
  labs(
    x = "Household deprivation",
    y = "Funding",
    title = "Relationship between deprivation and funding"
  ) +
  theme_minimal()


# vote choice ####

# add vote share columns to MRP data

vote_cols <- grep("Projected w/ TacticalProjected .* Vote Count$", names(mrp_q1_2025), value = TRUE)
vote_cols <- setdiff(vote_cols, "Projected w/ TacticalProjected Total Votes")

for (col in vote_cols) {
  
  # Clean the party name:
  # "Projected w/ TacticalProjected Conservative Vote Count" 
  #   -> "conservative"
  party_name <- col
  party_name <- sub("^Projected w/ TacticalProjected ", "", party_name)
  party_name <- sub(" Vote Count$", "", party_name)
  
  # Standardise to lower snake_case
  party_name <- tolower(gsub(" ", "_", party_name))
  
  # Build the output share column name
  share_col <- paste0(party_name, "_share")
  
  # Compute the share
  mrp_q4_2024[[share_col]] <- mrp_q4_2024[[col]] / mrp_q4_2024[["Projected w/ TacticalProjected Total Votes"]]
  mrp_q1_2025[[share_col]] <- mrp_q1_2025[[col]] / mrp_q1_2025[["Projected w/ TacticalProjected Total Votes"]]
  mrp_q2_2025[[share_col]] <- mrp_q2_2025[[col]] / mrp_q2_2025[["Projected w/ TacticalProjected Total Votes"]]
  mrp_q3_2025[[share_col]] <- mrp_q3_2025[[col]] / mrp_q3_2025[["Projected w/ TacticalProjected Total Votes"]]
  mrp_q4_2025[[share_col]] <- mrp_q4_2025[[col]] / mrp_q4_2025[["Projected w/ TacticalProjected Total Votes"]]
}

# add measure of seat safeness

mrp_q4_2024$seat_safeness <- mrp_q4_2024$`Projected w/ TacticalWin Margin` / mrp_q4_2024$`Projected w/ TacticalProjected Total Votes`
mrp_q1_2025$seat_safeness <- mrp_q1_2025$`Projected w/ TacticalWin Margin` / mrp_q1_2025$`Projected w/ TacticalProjected Total Votes`
mrp_q2_2025$seat_safeness <- mrp_q2_2025$`Projected w/ TacticalWin Margin` / mrp_q2_2025$`Projected w/ TacticalProjected Total Votes`
mrp_q3_2025$seat_safeness <- mrp_q3_2025$`Projected w/ TacticalWin Margin` / mrp_q3_2025$`Projected w/ TacticalProjected Total Votes`
mrp_q4_2025$seat_safeness <- mrp_q4_2025$`Projected w/ TacticalWin Margin` / mrp_q4_2025$`Projected w/ TacticalProjected Total Votes`

# aggregate to LAD

dataset_list <- c("mrp_q4_2024", "mrp_q1_2025", "mrp_q2_2025", "mrp_q3_2025", "mrp_q4_2025")

for (ds_name in dataset_list) {
  
  current_df <- get(ds_name)
  
  processed_df <- current_df %>%
    left_join(scot_ward_lad_lookup, by = c("Ward_code" = "MMWard_Code")) %>% # Scottish lookup to 2024 LADs
    left_join(ward_lad24_lookup, by = c("Ward_code" = "WD24CD")) %>% # E&W lookup from 2024 wards to 2024 LADs
    left_join(ward_lad22_lookup, by = c("Ward_code" = "WD22CD")) %>% # if no matching ward code, E&W lookup from 2022 wards to 2022 LADs 
    left_join(lad2_to_lad23_24, by = "LAD22CD") %>% 
    
    mutate(LAD = coalesce(LAD24CD, LAD23CD, LA_Code)) %>% 
    
    group_by(LAD) %>% 
    summarise(
      seat_safeness = weighted.mean(seat_safeness, w = `Projected w/ TacticalProjected Total Votes`, na.rm = TRUE),
      across(
        ends_with("_share"),
        ~ weighted.mean(.x, w = `Projected w/ TacticalProjected Total Votes`, na.rm = TRUE),
        .names = "{.col}"
      )
    )
  
  new_name <- paste0(ds_name, "_lad")
  assign(new_name, processed_df)
}

# compare with funding

mrp = mrp_q4_2024_lad # switch - select MRP here

merged_vote <- mrp %>% 
  left_join(funding, by = c("LAD" = "LAD24CD"))

merged_vote <- merged_vote %>%
  mutate(across(
    ends_with("share"),
    ~ as.numeric(.)
  ))

# test against vote choice

ggplot(merged_vote, aes(x = reform_uk_share, y = funding)) +
  geom_point(alpha = 0.6) +
  labs(
    x = "Reform UK vote share",
    y = "Funding",
    title = "Relationship between Reform UK vote share and funding"
  ) +
  theme_minimal()


ggplot(merged_vote, aes(x = conservative_share, y = con_funding)) +
  geom_point(alpha = 0.6) +
  labs(
    x = "Conservative vote share",
    y = "Funding from Cons",
    title = "Relationship between Conservative vote share and funding"
  ) +
  theme_minimal()

ggplot(merged_vote, aes(x = labour_share, y = lab_funding)) +
  geom_point(alpha = 0.6) +
  labs(
    x = "Labour vote share",
    y = "Funding from Lab",
    title = "Relationship between Labour vote share and funding"
  ) +
  theme_minimal()


# test against marginality 

ggplot(merged_vote, aes(x = seat_safeness, y = funding)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
  labs(
    x = "Safeness of seat",
    y = "Funding",
    title = "Relationship between marginality and funding"
  ) +
  theme_minimal()


# test against partisan marginality

share_cols <- grep("_share$", names(merged_vote), value = TRUE)

merged_vote <- merged_vote %>%
  rowwise() %>%
  mutate(
    shares_vec = list(c_across(all_of(share_cols))),
    order_idx  = list(order(-shares_vec)),
    biggest_share_col        = share_cols[order_idx[1]],
    second_biggest_share_col = share_cols[order_idx[2]],
    labour_seat = biggest_share_col == "labour_share", 
    conservative_seat = biggest_share_col == "conservative_share", 
    labour_top2 = biggest_share_col == "labour_share" |
      second_biggest_share_col == "labour_share",
    conservative_top2 = biggest_share_col == "conservative_share" |
      second_biggest_share_col == "conservative_share"
  ) %>%
  ungroup() %>%
  dplyr::select(-shares_vec, -order_idx)  # drop helper cols


ggplot(merged_vote, aes(x = seat_safeness, y = lab_funding, colour = labour_top2)) +
  geom_point(alpha = 0.6) +
  scale_colour_manual(
    values = c(`TRUE` = "red", `FALSE` = "grey70"),
    name   = "Labour in top 2\nvote shares"
  ) +
  labs(
    x = "Safeness of seat",
    y = "Labour funding",
    title = "Marginality vs Labour funding,\nhighlighting seats where Labour is 1st or 2nd"
  ) +
  theme_minimal()

ggplot(merged_vote, aes(x = seat_safeness, y = lab_funding, colour = labour_seat)) +
  geom_point(alpha = 0.6) +
  scale_colour_manual(
    values = c(`TRUE` = "red", `FALSE` = "grey70"),
    name   = "Labour LAD"
  ) +
  labs(
    x = "Safeness of seat",
    y = "Labour funding",
    title = "Marginality vs Labour funding,\nhighlighting seats where Labour is 1st"
  ) +
  theme_minimal()


ggplot(merged_vote, aes(x = seat_safeness, y = con_funding, colour = conservative_top2)) +
  geom_point(alpha = 0.6) +
  scale_colour_manual(
    values = c(`TRUE` = "blue", `FALSE` = "grey70"),
    name   = "Cons in top 2\nvote shares"
  ) +
  labs(
    x = "Safeness of seat",
    y = "Conservative funding",
    title = "Marginality vs Conservative funding,\nhighlighting seats where Conservative is 1st or 2nd"
  ) +
  theme_minimal()


ggplot(merged_vote, aes(x = seat_safeness, y = con_funding, colour = conservative_seat)) +
  geom_point(alpha = 0.6) +
  scale_colour_manual(
    values = c(`TRUE` = "blue", `FALSE` = "grey70"),
    name   = "Cons LAD"
  ) +
  labs(
    x = "Safeness of seat",
    y = "Conservative funding",
    title = "Marginality vs Conservative funding,\nhighlighting seats where Conservatives are 1st"
  ) +
  theme_minimal()


# regression with deprivation and vote choice ####

merged <- deprivation %>% 
  left_join(lad2_to_lad23_24, by = c("lad" = "LAD22CD")) %>% 
  left_join(funding, by = c("LAD23CD" = "LAD24CD")) %>% 
  left_join(mrp, by = c("LAD23CD" = "LAD"))


summary(lm(funding ~ deprivation_mean + 
             # labour_share +
             reform_uk_share + # +
             seat_safeness,
           data = merged))


# check relationship between deprivation and marginality itself

ggplot(merged, aes(x = seat_safeness, y = deprivation_mean)) +
  geom_point(alpha = 0.6) +
  labs(
    x = "Safeness of seat",
    y = "Household deprivaion",
    title = "Relationship between marginality and deprivation"
  ) +
  theme_minimal()


# check impact over time on vote choice - matching ####

# create dataset of Reform vote over time

dataset_names <- c(
  "mrp_q4_2024_lad", 
  "mrp_q1_2025_lad", 
  "mrp_q2_2025_lad", 
  "mrp_q3_2025_lad", 
  "mrp_q4_2025_lad")

combined_reform_share <- dataset_names %>%
  map(function(ds_name) {
    
    time_suffix <- str_extract(ds_name, "q[1-4]_[0-9]{4}") # use regex to extract the quarter and year
    new_col_name <- paste0("reform_uk_share_", time_suffix)
    
    get(ds_name) %>%
      dplyr::select(LAD, reform_uk_share) %>% #  get the data frame and select the relevant columns
      rename(!!new_col_name := reform_uk_share) # rename dynamically
  }) %>%
  
  reduce(full_join, by = "LAD")  # reduce list into df by joining them all

vote_impact <- deprivation %>%   # add together with deprivation indicator and regen funding levels
  left_join(lad2_to_lad23_24, by = c("lad" = "LAD22CD")) %>% 
  left_join(combined_reform_share, by = c("LAD23CD" = "LAD")) %>% 
  left_join(funding, by = c("LAD23CD" = "LAD24CD"))


# define outcome variables - key indicators are change in reform vote share, and absolute levels of support for Reform each quarter

vote_impact$reform_change = vote_impact$reform_uk_share_q4_2025 - vote_impact$reform_uk_share_q4_2024

# define treatment indicators

vote_impact <- vote_impact %>% # check vulnerability to mean v median
  mutate(
    treat_mean   = if_else(funding > mean(funding, na.rm = TRUE), 1, 0),
    treat_median = if_else(funding > median(funding, na.rm = TRUE), 1, 0))

# drop NAs from key variables 

vote_impact <- vote_impact %>% 
  drop_na(deprivation_mean, treat_mean, treat_median, reform_uk_share_q4_2024,
          reform_uk_share_q1_2025, reform_uk_share_q2_2025,
          reform_uk_share_q3_2025, reform_uk_share_q4_2025) 

# create models for different matching techniques - a robust result should be significant across all / most

mean.nn <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "nearest")  # mean funding, NN matching
median.nn <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "nearest")  # median funding, NN matching

mean.opt <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "optimal")  # mean funding, optimal matching
median.opt <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "optimal")  # median funding, optimal matching

mean.full <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "full")  # mean funding, full matching
median.full <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "full")  # median funding, full matching

mean.sub <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "subclass")  # mean funding, subclass matching
median.sub <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "subclass")  # median funding, subclass matching

mean.nn_strict <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "nearest", distance = "logit", caliper = 0.2, ratio = 2)  # mean funding, strict NN matching
median.nn_strict <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "nearest", distance = "logit", caliper = 0.2, ratio = 2)  # median funding, strict NN matching

mean.gen <- matchit(treat_mean ~ deprivation_mean, data = vote_impact, method = "genetic")  # mean funding, genetic matching
median.gen <- matchit(treat_median ~ deprivation_mean, data = vote_impact, method = "genetic")  # median funding, genetic matching

matched <- match.data(mean.nn_strict) # manual sanity check for matches

# make results table

outcomes <- c(
  "reform_change", # test if change in reform vote is significant
  "reform_uk_share_q4_2025", # test if overall level of reform support is significant
  "reform_uk_share_q3_2025",
  "reform_uk_share_q2_2025",
  "reform_uk_share_q1_2025",
  "reform_uk_share_q4_2024"
)

# list all specs
match_list <- list(
  mean_nearest        = list(obj = mean.nn,         treat = "treat_mean"),
  median_nearest      = list(obj = median.nn,       treat = "treat_median"),
  mean_optimal        = list(obj = mean.opt,        treat = "treat_mean"),
  median_optimal      = list(obj = median.opt,      treat = "treat_median"),
  mean_full           = list(obj = mean.full,       treat = "treat_mean"),
  median_full         = list(obj = median.full,     treat = "treat_median"),
  mean_subclass       = list(obj = mean.sub,        treat = "treat_mean"),
  median_subclass     = list(obj = median.sub,      treat = "treat_median"),
  mean_nearest_strict = list(obj = mean.nn_strict,  treat = "treat_mean"),
  median_nearest_strict = list(obj = median.nn_strict, treat = "treat_median"),
  mean_genetic        = list(obj = mean.gen,        treat = "treat_mean"),
  median_genetic      = list(obj = median.gen,      treat = "treat_median")
)

# function to run t-tests for a given model
run_tests_for_matching <- function(match_obj, matching_name, treat_var, outcomes) {
  df_matched <- match.data(match_obj)
  
  map_dfr(outcomes, function(outcome_var) {
    
    f <- as.formula(paste(outcome_var, "~", treat_var))
    
    tt <- t.test(f, data = df_matched)
    
    tibble(
      outcome       = outcome_var, # name of measure
      matching_spec = matching_name, # matching type
      treat_var     = treat_var, # name of treatment indicator
      mean_0        = unname(tt$estimate[1]), # mean reform support in control group (not much regen funding)
      mean_1        = unname(tt$estimate[2]), # mean reform support in treated group (lots of regen funding)
      p_value       = tt$p.value # significance of result
    )
  })
}

# loop over matching specs and collate into table
results_table <- imap_dfr(
  match_list,
  ~ run_tests_for_matching(match_obj = .x$obj,
                           matching_name = .y,
                           treat_var = .x$treat,
                           outcomes = outcomes))


avg_reform_suport <- results_table %>%
  filter(outcome != "reform_change") %>%
  group_by(outcome) %>%
  summarise(avg_mean_group0 = mean(mean_0, na.rm = TRUE),
            avg_mean_group1 = mean(mean_1, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(poll = str_replace(outcome, "reform_uk_share_", ""), # cleaner name for timings
         .before = outcome) %>% 
  dplyr::select(poll, avg_mean_group0, avg_mean_group1) %>% # pivot longer for easy plotting
  pivot_longer(cols = starts_with("avg_mean_group"),
               names_to = "group",
               values_to = "avg_mean") %>%
  mutate(poll = factor(poll, levels = c("q4_2024", "q1_2025", "q2_2025", "q3_2025", "q4_2025"))) # set chronology


ggplot(avg_reform_suport, aes(x = poll, y = avg_mean, color = group, group = group)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    x = "Date of MRP",
    y = "Average Reform vote share",
    colour = "Group") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")


# check impact over time on vote choice - matching with consideration of immigration ####

# assemble UK-wide data on country of birth 

# England and Wales 

cob_ew <- cob_ew %>% rename(lad = Lower.tier.local.authorities.Code, lad_nm = Lower.tier.local.authorities)

all <- cob_ew %>% group_by(lad) %>% summarise(total = sum(Observation))

cob_ew <- cob_ew %>%
  group_by(lad, lad_nm, Country.of.birth..UK...3.categories.) %>%
  summarise(Observation = sum(Observation), .groups = 'drop')

cob_ew <- cob_ew %>%
  pivot_wider(
    names_from = Country.of.birth..UK...3.categories., 
    values_from = Observation)

cob_ew <- left_join(cob_ew, all, by = "lad")

cob_ew <- cob_ew %>%
  mutate(foreign_born = `Born outside the UK`/ total)

cob_ew <- cob_ew %>% dplyr::select(lad, lad_nm, foreign_born)

# NI cob

cob_ni <- cob_ni %>% rename(lad = Local.Government.District.2014.Code, lad_nm = Local.Government.District.2014.Label)

all <- cob_ni %>% group_by(lad) %>% summarise(total = sum(Count))

cob_ni <- cob_ni %>%
  group_by(lad, lad_nm, Country.of.Birth...3.Categories.Label) %>%
  summarise(Count = sum(Count), .groups = 'drop')

cob_ni <- cob_ni %>%
  pivot_wider(
    names_from = Country.of.Birth...3.Categories.Label, 
    values_from = Count)

cob_ni <- left_join(cob_ni, all, by = "lad")

cob_ni <- cob_ni %>%
  mutate(foreign_born = (Other)/ total)

cob_ni <- cob_ni %>% dplyr::select(lad, lad_nm, foreign_born)

# S cob

cob_s <- cob_s %>% rename(lad = council_area_2019)

cob_s <- cob_s %>%
  mutate(foreign_born = Other / Total)

cob_s <- cob_s %>% 
  dplyr::mutate(lad_nm = NA) %>% 
  dplyr::select(lad, lad_nm, foreign_born)

# stack

cob <- rbind(cob_ew, cob_ni, cob_s)

# re run matching excluding all LADs with over 13% foreign born (UK mean 13%, median 9%) 

mean(cob$foreign_born, na.rm = TRUE)
median(cob$foreign_born, na.rm = TRUE)

vote_impact_immig <- vote_impact %>% 
  left_join(cob, by = "lad") %>%
  filter(foreign_born <= 0.13)

# define outcome variables - key indicators are change in reform vote share, and absolute levels of support for Reform each quarter

vote_impact_immig$reform_change = vote_impact_immig$reform_uk_share_q4_2025 - vote_impact_immig$reform_uk_share_q4_2024

# define treatment indicators

vote_impact_immig <- vote_impact_immig %>% # check vulnerability to mean v median
  mutate(
    treat_mean   = if_else(funding > mean(funding, na.rm = TRUE), 1, 0),
    treat_median = if_else(funding > median(funding, na.rm = TRUE), 1, 0))

# drop NAs from key variables 

vote_impact_immig <- vote_impact_immig %>% 
  drop_na(deprivation_mean, treat_mean, treat_median, reform_uk_share_q4_2024,
          reform_uk_share_q1_2025, reform_uk_share_q2_2025,
          reform_uk_share_q3_2025, reform_uk_share_q4_2025) 

# create datasets for different matching techniques - a robust result should be significant across all / most

mean.nn <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "nearest")  # mean funding, NN matching
median.nn <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "nearest")  # median funding, NN matching

mean.opt <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "optimal")  # mean funding, optimal matching
median.opt <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "optimal")  # median funding, optimal matching

mean.full <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "full")  # mean funding, full matching
median.full <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "full")  # median funding, full matching

mean.sub <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "subclass")  # mean funding, subclass matching
median.sub <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "subclass")  # median funding, subclass matching

mean.nn_strict <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "nearest", distance = "logit", caliper = 0.2, ratio = 2)  # mean funding, strict NN matching
median.nn_strict <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "nearest", distance = "logit", caliper = 0.2, ratio = 2)  # median funding, strict NN matching

mean.gen <- matchit(treat_mean ~ deprivation_mean, data = vote_impact_immig, method = "genetic")  # mean funding, genetic matching
median.gen <- matchit(treat_median ~ deprivation_mean, data = vote_impact_immig, method = "genetic")  # median funding, genetic matching

matched <- match.data(mean.nn_strict) # manual test to sense check the underlying matches

# make results table

outcomes <- c(
  "reform_change", # test if change in reform vote is significant
  "reform_uk_share_q4_2025", # test if overall level of reform support is significant
  "reform_uk_share_q3_2025",
  "reform_uk_share_q2_2025",
  "reform_uk_share_q1_2025",
  "reform_uk_share_q4_2024"
)

# list all specs
match_list <- list(
  mean_nearest        = list(obj = mean.nn,         treat = "treat_mean"),
  median_nearest      = list(obj = median.nn,       treat = "treat_median"),
  mean_optimal        = list(obj = mean.opt,        treat = "treat_mean"),
  median_optimal      = list(obj = median.opt,      treat = "treat_median"),
  mean_full           = list(obj = mean.full,       treat = "treat_mean"),
  median_full         = list(obj = median.full,     treat = "treat_median"),
  mean_subclass       = list(obj = mean.sub,        treat = "treat_mean"),
  median_subclass     = list(obj = median.sub,      treat = "treat_median"),
  mean_nearest_strict = list(obj = mean.nn_strict,  treat = "treat_mean"),
  median_nearest_strict = list(obj = median.nn_strict, treat = "treat_median"),
  mean_genetic        = list(obj = mean.gen,        treat = "treat_mean"),
  median_genetic      = list(obj = median.gen,      treat = "treat_median")
)

# function to run t-tests for a given model
run_tests_for_matching <- function(match_obj, matching_name, treat_var, outcomes) {
  df_matched <- match.data(match_obj)
  
  map_dfr(outcomes, function(outcome_var) {
    
    f <- as.formula(paste(outcome_var, "~", treat_var))
    
    tt <- t.test(f, data = df_matched)
    
    tibble(
      outcome       = outcome_var, # name of measure
      matching_spec = matching_name, # matching type
      treat_var     = treat_var, # name of treatment indicator
      mean_0        = unname(tt$estimate[1]), # mean reform support in control group (not much regen funding)
      mean_1        = unname(tt$estimate[2]), # mean reform support in treated group (lots of regen funding)
      p_value       = tt$p.value # significance of result
    )
  })
}

# loop over matching specs  and collate into table
results_table_immig <- imap_dfr(
  match_list,
  ~ run_tests_for_matching(match_obj = .x$obj,
                           matching_name = .y,
                           treat_var = .x$treat,
                           outcomes = outcomes))

# plot

avg_reform_suport <- results_table_immig %>%
  filter(outcome != "reform_change") %>%
  group_by(outcome) %>%
  summarise(avg_mean_group0 = mean(mean_0, na.rm = TRUE),
            avg_mean_group1 = mean(mean_1, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(poll = str_replace(outcome, "reform_uk_share_", ""), # cleaner name for timings
         .before = outcome) %>% 
  dplyr::select(poll, avg_mean_group0, avg_mean_group1) %>% # pivot longer for easy plotting
  pivot_longer(cols = starts_with("avg_mean_group"),
               names_to = "group",
               values_to = "avg_mean") %>%
  mutate(poll = factor(poll, levels = c("q4_2024", "q1_2025", "q2_2025", "q3_2025", "q4_2025"))) # set chronology


ggplot(avg_reform_suport, aes(x = poll, y = avg_mean, color = group, group = group)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    x = "Date of MRP",
    y = "Average Reform vote share",
    colour = "Group") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")
