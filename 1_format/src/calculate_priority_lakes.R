#' @param temp_dat all daily temperature observations from pipeline #1
#' @param n_min minimum number of total observations a lake must have to be considered a priority lake. This
#' filter is applied before n_years, years_with_7months, years_with_10days, and n_days.
#' @param n_years number of years of observation which qualifies a lake for priority status
#' @param years_with_7months number of years for which a lake has observations in 7 or more months, which qualifies a lake for priority status
#' @param years_with_10days number of years for which a lake has 10 or more days of observations, which qualifies a lake for priority status
#' @param n_days number of observation days a lake has over the entire period or record which qualifies a lake for priority status
calc_priority_lakes <- function(temp_dat_ind, n_min, n_years, years_with_7months, years_with_10days, n_days) {
  # This function calculates priority lakes based on data availability
  all_dat <- feather::read_feather(sc_retrieve(temp_dat_ind))

  stats <- all_dat %>%
    mutate(year = lubridate::year(date),
           month = lubridate::month(date)) %>%
    group_by(nhd_id, year, month, date) %>% # first group by date to compress depths
    summarize(n_depths = n()) %>% # keep track of n depths to get total obs
    group_by(nhd_id, year) %>% # now flatten to yearly stats
    summarize(mean_ndepths = mean(n_depths), # avg n depths measured per lake-date
              ntotal = sum(n_depths), # total obs per lake-year
              ndays_peryear = n(), # days of observation per lake-year
              nmonths_peryear = length(unique(month))) %>% # n months of obs per lake-year
    group_by(nhd_id) %>% # now flatten to lake stats
    summarize(mean_ndepths = mean(mean_ndepths),
              ntotal = sum(ntotal), # total obs per lake
              mean_ndays_peryear = mean(ndays_peryear), # avg number of obs days per lake-year
              ndays_total = sum(ndays_peryear), # total days of monitoring per lake
              nyears = n(), # total years of monitoring per lake
              nyears_10days = length(which(ndays_peryear>=10)), # n years with >= 10 days of monitoring per lake
              mean_nmonths_peryear = mean(nmonths_peryear), # avg number of monitoring months across all years per lake
              nyears_7months = length(which(nmonths_peryear>=7))) # n years with >= 7 months of monitoring per year

  first_pass <- filter(stats, ntotal > n_min) # first, lakes must meet n_min criteria

  lakes_years <- filter(first_pass, nyears >= n_years)
  lakes_months <- filter(first_pass, mean_nmonths_peryear >= nyears_7months)
  lakes_years_10days <- filter(first_pass, nyears_10days >= years_with_10days)
  lakes_days <- filter(first_pass, ndays_total >= n_days)

  # combine and find unique lakes that meet each criteria
  priority_lakes <- unique(c(lakes_years$nhd_id, lakes_months$nhd_id, lakes_years_10days$nhd_id, lakes_days$nhd_id))

  return(priority_lakes)

}

combine_priorities <- function(priority_lakes_by_choice, priority_lakes_by_data, truncate_lakes_for_dev = FALSE) {

  # combine the two lake lists (by union)
  all_lakes <- union(priority_lakes_by_choice, priority_lakes_by_data)

  # but during pipeline development, just use two lakes
  if(truncate_lakes_for_dev) {
    all_lakes <- c("nhd_1099476", "nhd_1099526")
  }

  # give warning if the selected lakes don't meet the priority_lakes_by_data criteria
  choice_lakes_dont_quality <- priority_lakes_by_choice[!priority_lakes_by_choice %in% priority_lakes_by_data]
  if(length(choice_lakes_dont_quality) > 0) {
    warning(
      length(choice_lakes_dont_quality),
      " chosen lakes don't meet data criteria: ",
      paste(choice_lakes_dont_quality, collapse=', '))
  }
  # attach data criteria info to selected IDs
  lake_selection <- tibble(
    site_id = all_lakes,
    meets_data_criteria = !(site_id %in% choice_lakes_dont_quality))

  return(lake_selection)
}

#' @param nldas_crosswalk_ind indicator file of the lake crosswalk, which has
#'   details on the cell x and cell y for each lake
add_lake_metadata <- function(lake_selection, nldas_crosswalk_ind) {
  # merge in data from the crosswalk
  crosswalk <- readRDS(sc_retrieve(nldas_crosswalk_ind)) %>%
    mutate(site_id = as.character(site_id))

  all_lakes_names <-  lake_selection %>%
    left_join(crosswalk, by='site_id') %>%
    mutate(lake_name = trimws(gsub('\\d+$', '', GNIS_Nm))) %>%
    #other columns from NLDAS file or master lake list could be saved here if useful
    select(site_id, lake_name, meets_data_criteria, nldas_coord_x, nldas_coord_y) %>%
    distinct()

  # use a google sheet to fill in any lake names that are missing from the crosswalk
  lakename_repair_sheet <- '1_format/in/missing_names_crosswalk'
  missing_names <- scipiper:::gd_locate_file(lakename_repair_sheet) %>%
    filter(name == basename(lakename_repair_sheet)) %>%
    pull(id) %>%
    googlesheets::gs_key() %>%
    googlesheets::gs_read(col_types = cols('c','c'))
  all_lakes_names_fixed <- left_join(all_lakes_names, missing_names, by = 'site_id') %>%
    mutate(lake_name = ifelse(is.na(lake_name.x), lake_name.y, lake_name.x)) %>%
    select(-lake_name.x, -lake_name.y)

  # make sure we filled in all the missing lake names (or give warning)
  if (any(is.na(all_lakes_names_fixed$lake_name))) {
    warning(paste0(
      'Some NHD ids are still missing lake names (site_id = ',
      paste(all_lakes_names_fixed$site_id[is.na(all_lakes_names_fixed$lake_name)], sep = ', '),
      sprintf('). Update the google sheet lake-temperature-neural-networks/%s.', lakename_repair_sheet)))
  }

  return(all_lakes_names_fixed)
}

#' Assign file paths to the three input files that we'll need per lake (before
#' munging to prepare them for input to the PGDL models)
#'
#' @param lakes_df a data.frame with one row per priority lake
#' @param obs_pattern an sprintf string with exactly one `%s`` wildcard for the
#'   lake id
#' @param glm_preds_pattern an sprintf string with exactly one `%s`` wildcard
#'   for the lake id
#' @param drivers_pattern an sprintf string with exactly three `%s`` wildcard
#'   for the time range, cell x, and cell y
#' @param drivers_time a string to be used as the time range in the
#'   drivers_pattern
assign_lake_files <- function(lakes_df, obs_pattern, geometry_pattern, glm_preds_pattern, drivers_pattern, drivers_time) {
  lakes_df %>%
    mutate(
      obs_file = sprintf(obs_pattern, site_id),
      geometry_file = sprintf(geometry_pattern, site_id),
      glm_preds_file = sprintf(glm_preds_pattern, site_id),
      drivers_file = sprintf(drivers_pattern, drivers_time, nldas_coord_x, nldas_coord_y)) %>%
    select(-nldas_coord_x, -nldas_coord_y)
}
