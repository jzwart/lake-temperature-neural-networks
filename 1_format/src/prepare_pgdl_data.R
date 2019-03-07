# Goal: go from these data files from other pipelines:

# obs_file        "1_format/tmp/obs/nhd_1099476_obs.fst"
# geometry_file   "1_format/tmp/geometry/nhd_1099476_geometry.csv"
# glm_preds_file  "1_format/tmp/glm_preds/nhd_1099476_temperatures.feather"
# drivers_file    "1_format/tmp/drivers/NLDAS_time[0.350500]_x[254]_y[160].csv"

# to these PGDL-read input files:

# x_train_1: Input data for supervised learning
# y_train_1: Observations for supervised learning
# m_train_1: Observation mask for supervised learning
# x_f: Normalized input data for unsupervised learning
# p_f: Unnormalized input data for unsupervised learning (p = physics) = concat(p_train_1, p_test)
# x_test: Inputs data for testing
# y_test: Observations for testing
# m_test: Observation mask for testing
# depth_areas: cross-sectional area of each depth

tidy_pgdl_data <- function(
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

  # Move glm_preds$ice into drivers
  drivers <- drivers %>%
    dplyr::left_join(select(glm_preds, date, ice), by='date') %>%
    dplyr::rename(Ice = ice)
  glm_preds <- glm_preds %>%
    dplyr::select(-ice)

  # Subset the observations and GLM data to extend no farther than the drivers;
  # give messages about which dates are getting dropped
  report_drops <- function(extras_df, data_type) {
    if(nrow(extras_df) > 0) {
      message(sprintf(
        '%s: dropping %d %s dates outside the date range of the drivers',
        lake_id, length(unique(extras_df$date)), data_type))
    }
  }
  report_drops(obs %>% filter(date < min(drivers$date) | date > max(drivers$date)), 'observation')
  report_drops(glm_preds %>% filter(date < min(drivers$date) | date > max(drivers$date)), 'GLM pred')
  # filter the datasets
  obs <- obs %>% filter(date >= min(drivers$date), date <= max(drivers$date))
  glm_preds <- glm_preds %>% filter(date >= min(drivers$date), date <= max(drivers$date))

  #### Munge GLM predictions ####

  # Reshape glm_preds into long form
  glm_preds <- glm_preds %>%
    tidyr::gather('temp_depth', 'TempC', starts_with('temp')) %>%
    dplyr::mutate(Depth = as.numeric(gsub('temp_', '', temp_depth))) %>%
    dplyr::select(date, Depth, TempC) %>%
    dplyr::arrange(date, Depth)

  # If there's are missing GLM predictions near the lake bottom, truncate the
  # bottom layers of the dataset until no more than 1 layer is ever missing
  missing_preds <- glm_preds %>% dplyr::filter(is.na(TempC))
  if(nrow(missing_preds) > 0) {
    # Report and truncate if anything but the last depth is missing
    missing_at_disallowed_depth <- filter(missing_preds, !(Depth %in% max(glm_preds$Depth)))
    if(nrow(missing_at_disallowed_depth) > 0) {
      new_deepest <- min(missing_at_disallowed_depth$Depth)
      message(sprintf(
        '%s: %d missing GLM predictions above the deepest depth (%0.1fm) at %s m;\ntruncating to end at %0.1f',
        lake_id, nrow(missing_at_disallowed_depth), max(depths),
        paste0(sprintf('%0.1f', sort(unique(missing_at_disallowed_depth$Depth))), collapse=', '),
        new_deepest))
      glm_preds <- glm_preds %>%
        filter(Depth <= new_deepest)
    }
    # Fill in any of those missing final depths by copying from the next-deepest layer
    glm_preds <- glm_preds %>%
      dplyr::arrange(date, Depth) %>% # we already did this above, but it's super important for tidyr::fill
      tidyr::fill(TempC, .direction='down')
  }

  # Define the PGDL-prediction depths based on glm_preds
  depths <- sort(unique(glm_preds$Depth))

  #### Munge observations ####

  # Notice and report on any observations >= 0.5 m below the lowest GLM depth.
  # We will drop them after interpolation
  deep_obs <- obs %>% filter(depth >= max(depths) + 0.5)
  if(nrow(deep_obs) > 0) {
    message(sprintf(
      '%s: dropping %d observations deeper than max GLM depth (%0.1f) at %0.1f - %0.1f m',
      lake_id, nrow(deep_obs), max(depths), min(deep_obs$depth), max(deep_obs$depth)))
  }

  # Match/interpolate observations to nearest GLM/PGDL prediction depths. This
  # is not strictly necessary for the LSTM structure we're using, but it does
  # make the data prep a heck of a lot simpler and reduces the time required for
  # prediction.
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

  # Extract the date-time dimension names
  dates <- drivers$date
  # depths <- sort(unique(glm_preds$Depth)) # this was done above to use in munging the glm_preds, observations, and geometry

  # Define the dimension sizes
  n_dates <- length(dates)
  n_depths <- length(depths)

  # Expand drivers from one per date to n_depths per date; add Depth column
  drivers_long <- drivers %>%
    dplyr::slice(rep(1:n(), each = n_depths)) %>%
    tibble::add_column(Depth = rep(depths, times = length(unique(drivers$date))), .before=2)

  # glm_preds is already the right (long) shape for this step, but right-join
  # with the drivers date-times to fill in any mismatches with NAs, which we'll
  # handle later with a mask
  glm_preds_long <- glm_preds %>%
    right_join(select(drivers_long, date, Depth), by = c('date', 'Depth'))

  # Expand observations to same shape and colnames as drivers (and glm_preds).
  # This creates lots of NAs, which we'll handle later with a mask
  obs_long <- obs %>%
    rename(Depth = depth, TempC = temp) %>%
    right_join(select(drivers_long, date, Depth), by = c('date', 'Depth'))

  # The first two columns should now be identical across the three datasets
  stopifnot(nrow(drivers_long) == n_dates * n_depths)
  stopifnot(all.equal(select(glm_preds_long, date, Depth), select(drivers_long, date, Depth)))
  stopifnot(all.equal(select(obs_long, date, Depth), select(drivers_long, date, Depth)))

  return(list(
    drivers = drivers_long,
    glm_preds = glm_preds_long,
    obs = obs_long,
    geometry = geometry
  ))
}
tidied <- tidy_pgdl_data()

#' Split into training, tuning, and testing sets, then normalize
#'
#' @param tidied list containing long-form drivers, glm_preds, obs, and geometry
#' @param n_lead_dates integer number of dates before the first non-NA label for
#'   which to include driver data (if available). This should be approximately
#'   half or more of a window width, so that the LSTM predictions that we are
#'   comparing to observations (or GLM pretraining 'observations') are those in
#'   the ~2nd half of a prediction sequence
split_pgdl_data <- function(tidied, n_lead_dates=100) {

  # unpack the R list
  drivers <- tidied$drivers
  glm_preds <- tidied$glm_preds
  obs <- tidied$obs
  geometry <- tidied$geometry

  # function to normalize the driver data into features. We will use this
  # function right after creating the relevant splits. This takes the driver
  # data, removes the Ice column, and normalizes the remaining columns (subtract
  # mean and divide by standard devation of each column)
  normalize <- function(driver_split) {
    driver_split %>%
      select(-Ice) %>% # permanently remove the Ice column from the normalized features
      select(-date) %>% # temporarily remove the date column for scaling
      scale() %>%
      as_tibble() %>%
      tibble::add_column(date = driver_split$date, .before=1) # restore the date column
  }

  # function to restrict the first dataset (could be drivers, glm_preds, or obs)
  # to those date and Depth rows also represented in the second dataset
  # (glm_preds or obs), or dates in the n_lead_dates preceding the first
  # observation in the second dataset. I think (but am not completely sure) that
  # it's useful to go back in driver data as far as the window width before the
  # first non-NA obs or GLM prediction (if there are drivers before then),
  # because this means the first LSTM prediction of those early obs/GLM dates
  # will be toward the end of a sequence rather than the beginning.
  restrict_to <- function(dat_full, obs_split, n_lead_dates) {
    obs_range <- range(obs_split$date)
    dat_range <- c(obs_range[1] - as.difftime(n_lead_dates, units='days'), obs_range[2])
    dat_restricted <- dat_full %>%
      filter(date >= dat_range[1], date <= dat_range[2])
  }

  # pad dat_restricted to the left with NAs, using dat_full as a template for
  # the shape (dates, Depths, and column names), and padding with up to
  # n_lead_dates (but fewer if there aren't that many dates in dat_full)
  buffer <- function(dat_restricted, dat_full, n_lead_dates) {
    obs_range <- range(dat_restricted$date)
    # create a df of same shape as dat_full but with all NAs in columns other
    # than date and Depth
    na_mask <- dat_full
    # teh restricted and buffered data starts with a buffer of NAs and ends with the data as restricted by
    dat_restricted <- bind_rows(
      dat_full %>% # select just the buffer region
        filter(date >= obs_range[1] - as.difftime(n_lead_dates, units='days'), date < obs_range[1]) %>%
        group_by(date, Depth) %>% mutate_all(funs(.*NA)) %>% ungroup(), # convert values to NAs
      dat_restricted) # add in the restricted data
  }

  # function to generate an observation mask with 1s where an observation is
  # available and 0s otherwise
  mask <- function(obs_split) {
    obs_split %>%
      transmute(
        date = date,
        Depth = Depth,
        Multiplier = ifelse(is.na(TempC), 0, 1))
  }

  # function to double check that we have three identically dimensioned
  # data.frames
  require_equal_dims <- function(features, labels, mask) {
    stopifnot(all.equal(
      select(features, date),
      select(labels, date) # don't require equal Depth because features$Depth is normalized
    ))
    stopifnot(all.equal(
      select(labels, date, Depth),
      select(mask, date, Depth)
    ))
  }

  # Initialize an R list where we'll keep all the dataset splits
  datasets <- list()

  # For unsupervised learning, always use the full driver dataset (normalized
  # and unnormalized), but exclude days without Ice information because these
  # can't be tested for energy conservation. Do this both in pretraining and in
  # fine tuning (to prevent unrealistic states or budget closure during both
  # phases)
  datasets$unsup$physics <- restrict_to(drivers, filter(drivers, !is.na(Ice)), 0)
  datasets$unsup$features <- normalize(datasets$unsup$physics)

  # Pretrain on the full GLM predictions dataset (even though there could be
  # situations where the prediction in a specific year gets worse because we're
  # training on 1979 and climate has changed)
  glm_non_na <- filter(glm_preds, !is.na(TempC))
  # labels: extend up to n_lead_dates before the first non-NA GLM prediction
  datasets$pretrain$labels <- restrict_to(glm_preds, glm_non_na, n_lead_dates)
  datasets$pretrain$mask <- mask(datasets$pretrain$labels)
  # features: make these match the shape of the pretrain$labels exactly, then normalize
  datasets$pretrain$features <- restrict_to(drivers, datasets$pretrain$labels, 0) %>%
    normalize()
  # double check the dimensions
  require_equal_dims(datasets$pretrain$features, datasets$pretrain$labels, datasets$pretrain$mask)

  # For testing, hyperparameter tuning, and training = fine tuning, split things differently depending on how many observation
  # dates are available
  obs_non_na <- filter(obs, !is.na(TempC))
  obs_dates <- sort(unique(obs_non_na$date))
  obs_dates_rev <- rev(obs_dates)
  n_obs_dates <- length(obs_dates)

  #' Slice the obs and drivers into labels, mask, and features for one
  #' application (probably test, tune, or train)
  #'
  #' assumes availability of obs, drivers, obs_dates, obs_dates_rev, and
  #' n_lead_dates
  #'
  #' @param low_obs_num the index of the first observation to include (counting
  #'   first from the left and then from the right)
  #' @param high_obs_num NA if one data chunk is desired. if two chunks are
  #'   desired, this is the index of the final observation to include in each
  #'   shoulder (counting first from the left for shoulder 1 and then from the
  #'   right for shoulder 2)
  slice_labels_features_mask <- function(low_obs_num, high_obs_num) {

    one_slice <- list(labels=NA, mask=NA, features=NA)

    if(!is.na(high_obs_num)) {
      # define the 4 date cutoffs for the 2 shoulders
      date_cutoffs <- c(
        obs_dates[low_obs_num],
        obs_dates[high_obs_num],
        obs_dates_rev[high_obs_num],
        obs_dates_rev[low_obs_num]
      )
      # check the date cutoffs to make sure they're sensible
      if(any(as.numeric(diff(date_cutoffs), units='days') <= 0)) {
        stop(paste0('obs_nums imply non-increasing series of dates: ', paste0(date_cutoffs, collapse=', ')))
      }
      # pick out the two shoulders worth of labels
      one_slice$labels <- list(
        left = obs_non_na %>% filter(date >= date_cutoffs[1] & date <= date_cutoffs[2]) %>%
          restrict_to(obs, ., 0) %>%
          buffer(obs, n_lead_dates),
        right = obs_non_na %>% filter(date >= date_cutoffs[3] & date <= date_cutoffs[4]) %>%
          restrict_to(obs, ., 0) %>%
          buffer(obs, n_lead_dates))
      # pick out the two shoulders worth of features (based on the labels
      # shoulders) and bind them together
      one_slice$features <- bind_rows(
        left = restrict_to(drivers, one_slice$labels$left, 0) %>% normalize(),
        right = restrict_to(drivers, one_slice$labels$right, 0) %>% normalize())
      # bind together the two shoulders worth of labels
      one_slice$labels <- bind_rows(one_slice$labels)

    } else { # is.na(high_obs_num)
      # define the 2 date cutoffs for the 1 data chunk
      date_cutoffs <- c(
        obs_dates[low_obs_num],
        obs_dates_rev[low_obs_num]
      )
      # check the date cutoffs to make sure they're sensible
      if(any(as.numeric(diff(date_cutoffs), units='days') <= 0)) {
        stop(paste0('obs_nums imply non-increasing series of dates: ', paste0(date_cutoffs, collapse=', ')))
      }
      # pick out the labels
      one_slice$labels <- obs_non_na %>% filter(date >= date_cutoffs[1] & date <= date_cutoffs[2]) %>%
        restrict_to(obs, ., 0) %>%
        buffer(obs, n_lead_dates)
      # pick out the features based on the labels
      one_slice$features <- restrict_to(drivers, one_slice$labels, 0) %>% normalize()
    }

    # compute the mask based on the labels
    one_slice$mask <- mask(one_slice$labels)

    # check the dimensions to make sure we've ended up with three equally shaped data.frames
    require_equal_dims(one_slice$features, one_slice$labels, one_slice$mask)

    # return the labels, features, and mask for this one data slice
    return(one_slice)
  }

  # TODO we could use the pretrain data for datasets$tune (hyperparameter
  # tuning), in which case we'd want to not allocate any real obs to tuning
  # below. But for now we're allocating some obs to hyperparameter tuning.

  if(n_obs_dates <= 60) {
    # If there are fewer than 60 obs dates, use no test or tuning obs (or maybe
    # these will already have been dropped from the priority lakes list). As
    # with the GLM predictions, extend the drivers and observations up to
    # n_lead_dates before the first non-NA observation to allow the PGDL
    # sequence to warm up
    datasets$test <- NULL
    datasets$tune <- NULL
    datasets$train <- slice_labels_features_mask(1, NA)

  } else if(n_obs_dates <= 100) {
    # If there are 61 to 100 obs dates, only use 30 test obs, 30 tune obs that
    # overlap with training dataset, and remainder for training (between 30 and
    # 70)
    datasets$test <- slice_labels_features_mask(1, 15)
    datasets$tune <- slice_labels_features_mask(16, 30)
    datasets$train <- slice_labels_features_mask(16, NA)

  } else if(n_obs_dates <= 200) {
    # If there are 101-200 obs dates, use 30 test obs, 20 tune obs + 10 obs
    # overlapping with training, remainder for training (51+)
    datasets$test <- slice_labels_features_mask(1, 15)
    datasets$tune <- slice_labels_features_mask(16, 30)
    datasets$train <- slice_labels_features_mask(26, NA)

  } else {
    # If there are 201+ obs dates, use 50 test, 50 tune, remainder for training (101+)
    datasets$test <- slice_labels_features_mask(1, 25)
    datasets$tune <- slice_labels_features_mask(26, 50)
    datasets$train <- slice_labels_features_mask(51, NA)
  }

  # For prediction, use the full driver dataset (no need to restrict to just
  # those days with Ice estimates)
  datasets$predict$features <- drivers

  return(datasets)
}
splits <- split_pgdl_data(tidied, n_lead_dates=100)

summarize_pgdl_split <- function(splits, tidied) {
  split_obs_dates <-
    bind_rows(mutate(tidied$obs, Type='Obs'),
              mutate(splits$pretrain$labels, Type='Pretrain'),
              mutate(splits$train$labels, Type='Train'),
              mutate(splits$tune$labels, Type='Tune'),
              mutate(splits$test$labels, Type='Test')) %>%
    group_by(Type, date) %>%
    summarize(
      obs_date = any(!is.na(TempC)),
      n_obs = length(which(!is.na(TempC)))) %>%
    mutate(
      shoulder = {
        shoulder_end <- date[which(as.numeric(diff(date), units='days') > 1)]
        if(length(shoulder_end) > 1) stop('found more than 1 shoulder break, not ready for that')
        if(length(shoulder_end) == 1) {
          ifelse(date <= shoulder_end, 'left', 'right')
        } else 'all'
      }) %>%
    group_by(Type, shoulder) %>%
    mutate(buffer = cumsum(obs_date) == 0) %>%
    group_by(Type, shoulder, buffer) %>%
    summarize(
      start=min(date), end=max(date),
      n_obs=sum(n_obs),
      n_days=n(), n_obs_days=length(which(obs_date))) %>%
    arrange(Type, shoulder, desc(buffer))

  split_obs_dates
}
summarize_pgdl_split(splits, tidied)

plot_pgdl_split_days <- function(splits, tidied) {
  split_obs_dates <-
    bind_rows(mutate(tidied$obs, Type='Obs'),
              mutate(splits$pretrain$labels, Type='Pretrain'),
              mutate(splits$train$labels, Type='Train'),
              mutate(splits$tune$labels, Type='Tune'),
              mutate(splits$test$labels, Type='Test')) %>%
    group_by(date, Type) %>%
    summarize(obs_date = any(!is.na(TempC))) %>%
    ungroup() %>%
    mutate(Type = ordered(Type, levels=c('Pretrain','Obs','Train','Tune','Test')))
  ggplot(split_obs_dates, aes(x=date, y=1)) +
    geom_point(shape=NA) +
    geom_vline(data=filter(split_obs_dates, !obs_date), aes(xintercept=date, color=obs_date)) +
    geom_vline(data=filter(split_obs_dates, obs_date), aes(xintercept=date, color=obs_date)) +
    ylab('') + theme_classic() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
    facet_grid(Type ~ ., scales='free_y')
}
plot_pgdl_split_days(splits, tidied)

#' Reshape tibbles to PGDL-ready format
#'
#' n_batch = number of training batches (but we actually use them all at once?)
#'
#' n_sec = number of sequences per batch
#'
#' dim(drivers) = c(n_depths*n_sec*n_batch, n_dates, n_drivers)
#'
#' dim(drivers_norm) = c(n_depths*n_sec*n_batch, n_dates, n_drivers_norm)
#'
#' dim(glm_preds) = c(n_depths*n_sec*n_batch, n_dates)
#'
#' dim(obs) = dim(glm_preds) dim(mask) = dim(obs)
#'
#' x_train_1: Input data for supervised learning
#' y_train_1: Observations for supervised learning
#' m_train_1: Observation mask for supervised learning
#' x_f: Normalized input data for unsupervised learning
#' p_f: Unnormalized input data for unsupervised learning (p = physics) = concat(p_train_1, p_test)
#' x_test: Inputs data for testing
#' y_test: Observations for testing
#' m_test: Observation mask for testing
#' depth_areas: cross-sectional area of each depth
#'
#' @param inputs an R list of data
#' @param batch_size the maximum size per batch (will be adjusted downward to
#'   produce even-sized batches given actual data availaiblity)
#' @param sequence_length the number of dates per timeseries sequence passed in
#'   as a training example
#' @param sequence_offset the number of dates by which each subsequent sequence
#'   lags behind the previous sequence (per date)
reshape_data_for_pgdl <- function(inputs, batch_size, sequence_length) {

  # Extract the date-time dimension names
  dates <- drivers$date
  driver_names_norm <- sprintf('%sNorm', names(select(drivers, -date, -Ice))) # normalized drivers
  driver_names_phys <- names(select(drivers, -date)) # drivers in physical units
  # depths <- sort(unique(glm_preds$Depth)) # this was done above to use in munging the glm_preds, observations, and geometry

  # Define the dimension sizes
  n_dates <- length(dates)
  n_depths <- length(depths)

  n_drivers_phys <- length(driver_names_norm)
  n_drivers_norm <- length(driver_names_phys)


}
