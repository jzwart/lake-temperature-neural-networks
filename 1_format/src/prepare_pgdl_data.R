# Goal: go from these data files from other pipelines:

# obs_file        "1_format/tmp/obs/nhd_1099476_obs.fst"
# geometry_file   "1_format/tmp/geometry/nhd_1099476_geometry.csv"
# glm_preds_file  "1_format/tmp/glm_preds/nhd_1099476_temperatures.feather"
# drivers_file    "1_format/tmp/drivers/NLDAS_time[0.350500]_x[254]_y[160].csv"

# to these PGDL-read input files:

# x_train_1: Input data for supervised learning
# y_train_1: Observations for supervised learning
# m_train_1: Observation mask for supervised learning
# p_f: Unnormalized input data for unsupervised learning (p = physics) = concat(p_train_1, p_test)
# x_test: Inputs data for testing
# y_test: Observations for testing
# m_test: Observation mask for testing
# depth_areas: cross-sectional area of each depth

prep_pgdl_data <- function(
  lake_id = 'nhd_1099476',
  drivers_file = '1_format/tmp/drivers/NLDAS_time[0.350500]_x[254]_y[160].csv',
  geometry_file = '1_format/tmp/geometry/nhd_1099476_geometry.csv',
  glm_preds_file = '1_format/tmp/glm_preds/nhd_1099476_temperatures.feather',
  obs_file = '1_format/tmp/obs/nhd_1099476_obs.fst') {

  # Read in data as tibbles
  geometry <-  readr::read_csv(geometry_file, col_types='dd')
  drivers <- readr::read_csv(drivers_file, col_types = 'Dddddddd')
  glm_preds <-  feather::read_feather(glm_preds_file)
  obs <- fst::read_fst(obs_file) %>% as_tibble()

  #### Harmonize the data files ####

  # Harmonize date formats
  drivers <- drivers %>%
    rename(date = time)
  glm_preds <- glm_preds %>%
    rename(date = DateTime) %>%
    mutate(date = as.Date(date))

  # Subset the driver and GLM data to extend no farther than the observations
  # TODO I think we don't really want this subsetting for lake-temps project
  start_date <- max(
    min(drivers$date),
    min(obs$date),
    min(glm_preds$date))
  end_date <- min(
    max(drivers$date),
    max(obs$date),
    max(glm_preds$date))
  # give messages about which dates are getting dropped
  report_drops <- function(extras_df, data_type) {
    other_data_types <- setdiff(c('observation', 'driver', 'GLM pred'), data_type)
    if(nrow(extras_df) > 0) {
      message(sprintf(
        '%s: dropping %d %s dates outside the date ranges of the %ss and/or %ss',
        lake_id, length(unique(extras_df$date)), data_type, other_data_types[1], other_data_types[2]))
    }
  }
  report_drops(obs %>% filter(date < start_date | date > end_date), 'observation')
  report_drops(drivers %>% filter(date < start_date | date > end_date), 'driver')
  report_drops(glm_preds %>% filter(date < start_date | date > end_date), 'GLM pred')
  # filter the datasets
  drivers <- dplyr::filter(drivers, date >= start_date, date <= end_date)
  glm_preds <- dplyr::filter(glm_preds, date >= start_date, date <= end_date)
  obs <- dplyr::filter(obs, date >= start_date, date <= end_date)

  # Require that the dates match exactly for the drivers and glm_preds
  stopifnot(all.equal(drivers$date, glm_preds$date))

  # Move glm_preds$ice into drivers
  drivers <- drivers %>%
    dplyr::left_join(select(glm_preds, date, ice), by='date') %>%
    dplyr::rename(Ice = ice)
  glm_preds <- glm_preds %>%
    dplyr::select(-ice)

  #### Munge GLM predictions ####

  # Reshape glm_preds into long form
  glm_preds <- glm_preds %>%
    tidyr::gather('temp_depth', 'TempC', starts_with('temp')) %>%
    dplyr::mutate(Depth = as.numeric(gsub('temp_', '', temp_depth))) %>%
    dplyr::select(date, Depth, TempC) %>%
    dplyr::arrange(date, Depth)

  # Define the PGDL depth dimension names based on glm_preds
  depths <- sort(unique(glm_preds$Depth))

  # Fill in any missing values in the very last depth of the GLM predictions;
  # throw error if missing values occur elsewhere
  missing_preds <- glm_preds %>% dplyr::filter(is.na(TempC))# %>% group_by(Depth) %>% tally()
  if(nrow(missing_preds) > 0) {
    # Throw an error if anything but the last depth is missing
    missing_at_disallowed_depth <- filter(missing_preds, !(Depth %in% max(depths)))
    if(nrow(missing_at_disallowed_depth) > 0) {
      stop(sprintf(
        '%s: %d missing GLM predictions above the deepest depth (%0.1fm) at %s m',
        lake_id, nrow(missing_at_disallowed_depth), max(depths),
        paste0(sprintf('%0.1f', sort(unique(missing_at_disallowed_depth$Depth))), collapse=', ')))
    }
    # Fill in any of those missing final depths by copying from the next-deepest layer
    glm_preds <- glm_preds %>%
      dplyr::arrange(date, Depth) %>% # we already did this above, but it's super important for tidyr::fill
      tidyr::fill(TempC, .direction='down')
  }

  #### Munge observations ####

  # Notice and report on any observations >= 0.5 m below the lowest GLM depth
  deep_obs <- obs %>% filter(depth >= max(depths) + 0.5)
  if(nrow(deep_obs) > 0) {
    message(sprintf(
      '%s: dropping %d observations deeper than max GLM depth (%0.1f) at %0.1f - %0.1f m',
      lake_id, nrow(deep_obs), max(depths), min(deep_obs$depth), max(deep_obs$depth)))
  }

  # Match/interpolate observations to nearest GLM/PGDL prediction depths
  obs <- obs %>%
    group_by(date) %>%
    do({
      date_df <- .
      date_df %>%
        mutate(
          # find the closest PGDL depth to the observation depth
          new_depth = purrr::map_dbl(depth, function(obsdep) depths[which.min(abs(obsdep - depths))]),
          # estimate temperature at the new depth using interpolation, or if
          # that's not possible, set to the nearest observed temperature
          new_temp = if(nrow(date_df) >= 2) {
            # if we have enough values on this date, interpolate
            approx(x=depth, y=temp, xout=new_depth, rule=2)$y
          } else {
            # if we can't interpolate, just use the nearest value
            temp
          },
          depth_diff = abs(depth - new_depth)) %>%
        # after approx(), trash any values at new_depth >= 0.5 m from the nearest observation
        filter(depth_diff < 0.5) %>%
        # only keep one estimate for each new_depth
        group_by(new_depth) %>%
        filter(depth_diff == min(depth_diff)) %>%
        ungroup()
    }) %>%
    ungroup() %>%
    # to see my work as columns, print out the result up to this point, e.g. by uncommenting
    # tail(20) %>% print(n=20)
    # now we clean up the columns
    select(date, depth=new_depth, temp=new_temp)

  #### Munge the lake geometry ####

  # interpolate the original geometry info to match the PGDL depths
  geometry <- tibble(
    Depth = depths,
    Area = approx(x=geometry$depths, y=geometry$areas, xout=Depth)$y)

  #### Reshape and format data into ~tidy format ####

  # Extract the dimension names
  dates <- drivers$date
  driver_names_norm <- sprintf('%sNorm', names(select(drivers, -date, -Ice))) # normalized drivers
  driver_names_phys <- names(select(drivers, -date)) # drivers in physical units
  # depths <- sort(unique(glm_preds$Depth)) # this was done above to use in munging the glm_preds, observations, and geometry

  # Define the dimension sizes
  n_dates <- length(dates)
  n_drivers_phys <- length(driver_names_norm)
  n_drivers_norm <- length(driver_names_phys)
  n_depths <- length(depths)

  # Expand drivers from one per date to n_depths per date; add Depth column
  drivers_long <- drivers %>%
    dplyr::slice(rep(1:n(), each = n_depths)) %>%
    tibble::add_column(Depth = rep(depths, times = n_dates), .before=2)

  # glm_preds is already the right (long) shape for this step
  glm_preds_long <- glm_preds

  # Expand observations to same shape and colnames as drivers (and glm_preds)
  obs_long <- obs %>%
    rename(Depth = depth, TempC = temp) %>%
    right_join(select(drivers_long, date, Depth), by = c('date', 'Depth'))

  # The first two columns should now be identical across the three datasets
  stopifnot(nrow(drivers_long) == n_dates * n_depths)
  stopifnot(nrow(glm_preds_long) == n_dates * n_depths)
  stopifnot(nrow(obs_long) == n_dates * n_depths)

  #### Split into training, tuning, and testing sets (before normalizing) ####

  # Choose the test/tune/train dates
  obs_dates <- sort(unique(obs$date))
  # test: include 50 observation dates from the shoulders
  test_dates <- dates[dates <= obs_dates[25] | dates >= obs_dates[length(obs_dates) - 24]]
  # tune: 50 obs dates from the shoulders after excluding test dates
  non_test_dates <- dates[!(dates %in% test_dates)]
  tune_dates <- non_test_dates[non_test_dates <= obs_dates[50] | non_test_dates >= obs_dates[length(obs_dates) - 49]]
  # train: the remainder of the dates
  train_dates <- dates[!(dates %in% c(tune_dates, test_dates))]
  date_split <- list(
    test = test_dates,
    tune = tune_dates,
    train = train_dates
  )

  # Split each of the long datasets
  split_fun <- function(date_subset, dat_long) {
    filter(dat_long, date %in% date_subset)
  }
  drivers_split <- lapply(date_split, split_fun, drivers_long)
  glm_preds_split <- lapply(date_split, split_fun, glm_preds_long)
  obs_split <- lapply(date_split, split_fun, obs_long)

  # Count the number of non-NA values and dates represented in these splits
  split_counts <- bind_rows(
    bind_cols(
      tibble(count_of='driver values'),
      as_tibble(lapply(drivers_split, nrow))),
    bind_cols(
      tibble(count_of='driver dates'),
      as_tibble(lapply(drivers_split, function(dat) { dat %>% pull(date) %>% unique() %>% length() }))),
    bind_cols(
      tibble(count_of='obs values'),
      as_tibble(lapply(obs_split, function(dat) { dat %>% filter(!is.na(TempC)) %>% nrow }))),
    bind_cols(
      tibble(count_of='obs dates'),
      as_tibble(lapply(obs_split, function(dat) { dat %>% filter(!is.na(TempC)) %>% pull(date) %>% unique() %>% length() })))
  )

  # Give warnings if there aren't very many observation values or observation dates
  obs_values_train <- filter(split_counts, count_of=='obs values') %>% pull(train)
  if(obs_values_train < 1000) {
    warning(sprintf(
      '%s: fewer than 1000 observation values in training set (%d)',
      lake_id, obs_values_train))
  }
  obs_dates_train <- filter(split_counts, count_of=='obs dates') %>% pull(train)
  if(obs_dates_train < 100) {
    warning(sprintf(
      '%s: fewer than 100 observation dates in training set (%d)',
      lake_id, obs_dates_train))
  }

  #### Normalize the feature data ####
  drivers_norm_split <- lapply(drivers_split, function(dsplit) {
    dsplit %>%
      select(-Ice) %>% # permanently remove the Ice column from the normalized features
      select(-date) %>% # temporarily remove the date column for scaling
      scale() %>%
      as_tibble() %>%
      tibble::add_column(date = dsplit$date, .before=1) # restore the date column
  })

  #### TODO Reshape tibbles to PGDL-ready format ####

  # n_batch = number of training batches (but we actually use them all at once?)
  # n_sec = number of windows per batch
  # dim(drivers) = c(n_depths*n_sec*n_batch, n_dates, n_drivers)
  # dim(drivers_norm = c(n_depths*n_sec*n_batch, n_dates, n_drivers_norm)
  # dim(glm_preds) = c(n_depths*n_sec*n_batch, n_dates)
  # dim(obs) = dim(glm_preds)
  # dim(mask) = dim(obs)

}
