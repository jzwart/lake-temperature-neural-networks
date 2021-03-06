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

#' Munge, organize, and interpolate the input data from other pipelines into a
#' list of four long-format tibbles
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
    dplyr::mutate(depth = as.numeric(gsub('temp_', '', temp_depth))) %>%
    dplyr::select(date, depth, TempC) %>%
    dplyr::arrange(date, depth)

  # If there's are missing GLM predictions near the lake bottom, truncate the
  # bottom layers of the dataset until no more than 1 layer is ever missing
  missing_preds <- glm_preds %>% dplyr::filter(is.na(TempC))
  if(nrow(missing_preds) > 0) {
    # Report and truncate if anything but the last depth is missing
    missing_at_disallowed_depth <- filter(missing_preds, !(depth %in% max(glm_preds$depth)))
    if(nrow(missing_at_disallowed_depth) > 0) {
      new_deepest <- min(missing_at_disallowed_depth$depth)
      message(sprintf(
        '%s: %d missing GLM predictions above the deepest depth (%0.1fm) at %s m;\ntruncating to end at %0.1f',
        lake_id, nrow(missing_at_disallowed_depth), max(glm_preds$depth),
        paste0(sprintf('%0.1f', sort(unique(missing_at_disallowed_depth$depth))), collapse=', '),
        new_deepest))
      glm_preds <- glm_preds %>%
        filter(depth <= new_deepest)
    }
    # Fill in any of those missing final depths by copying from the next-deepest layer
    glm_preds <- glm_preds %>%
      dplyr::arrange(date, depth) %>% # we already did this above, but it's super important for tidyr::fill
      tidyr::fill(TempC, .direction='down')
  }

  # Define the PGDL-prediction depths based on glm_preds
  depths <- sort(unique(glm_preds$depth))

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
    # take care of some duplicated depth-date combos
    group_by(date, depth) %>%
    summarize(temp = mean(temp)) %>%
    ungroup() %>%
    # interpolate
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
    select(date, depth=new_depth, temp=new_temp) %>%
    # take care of some duplicated depth-date combos
    group_by(date, depth) %>%
    summarize(temp = mean(temp)) %>%
    ungroup()

  #### Munge the lake geometry ####

  # interpolate the original geometry info to match the PGDL depths
  geometry <- tibble(
    Depth = depths,
    Area = approx(x=geometry$depths, y=geometry$areas, xout=Depth)$y)

  #### Reshape and format data into ~tidy format ####

  # Extract the date-time dimension names
  dates <- drivers$date
  # depths <- sort(unique(glm_preds$depth)) # this was done above to use in munging the glm_preds, observations, and geometry

  # Define the dimension sizes
  n_dates <- length(dates)
  n_depths <- length(depths)

  # Expand drivers from one per date to n_depths per date; add Depth column
  drivers_long <- drivers %>%
    dplyr::slice(rep(1:n(), each = n_depths)) %>%
    tibble::add_column(depth = rep(depths, times = length(unique(drivers$date))), .before=2)

  # glm_preds is already the right (long) shape for this step
  glm_preds_long <- glm_preds

  # Expand observations to same shape and colnames as drivers within the date
  # range covered by obs. This creates lots of NAs. We'll handle many NAs later
  # with a mask.
  drivers_dims_obs_range <- drivers_long %>%
    filter(date >= min(obs$date), date <= max((obs$date))) %>%
    select(date, depth)
  obs_long <- obs %>%
    rename(TempC = temp) %>%
    right_join(drivers_dims_obs_range, by = c('date', 'depth'))

  return(list(
    drivers = drivers_long,
    glm_preds = glm_preds_long,
    obs = obs_long,
    geometry = geometry
  ))
}

#' Split into training, tuning, and testing sets, then normalize
#'
#' Each split should contain an integer number of sequences
#'
#' @param tidied list containing long-form drivers, glm_preds, obs, and geometry
#' @param sequence_length integer number of dates within each sequence to be fed
#'   to the NN. In this function, sequence_length is only used to produce splits
#'   that are an even number of sequences long.
#' @param sequence_offset integer number of dates by which each sequence should
#'   lag behind its predecessor
split_pgdl_data <- function(tidied, sequence_length, sequence_offset) {

  # unpack the R list
  drivers <- tidied$drivers %>%
    tibble::add_column(Depth = tidied$drivers$depth, .after='depth') # copy into column to keep Depth as a predictor, not just a dimension
  glm_preds <- tidied$glm_preds
  obs <- tidied$obs
  geometry <- tidied$geometry

  #### Truncate drivers ####

  # truncate the drivers to a window that contains an integer number of
  # sequences, prefering to cut driver dates that have no corresponding GLM
  # predictions or observations, and prefering to cut observations from the left
  # (earlier dates) rather than the right (because the obs quality may have been
  # worse in the early monitoring years)
  truncate_drivers <- function(drivers, glm_preds, obs) {
    glm_date_range <- glm_preds %>%
      filter(!is.na(TempC)) %>%
      pull(date) %>%
      range()
    obs_date_range <- obs %>%
      filter(!is.na(TempC)) %>%
      pull(date) %>%
      range()
    driver_dates <- drivers %>%
      group_by(date) %>%
      summarize(driver_date = TRUE) %>%
      mutate(glm_shoulder = ifelse(date < glm_date_range[1], 'left', ifelse(date > glm_date_range[2], 'right', 'none'))) %>%
      mutate(obs_shoulder = ifelse(date < obs_date_range[1], 'left', ifelse(date > obs_date_range[2], 'right', 'none')))
    target_num_driver_dates <- sequence_length + sequence_offset *
      floor( # number of whole sequences possible after the first one
        (nrow(driver_dates) - sequence_length) / # the first sequence is always sequence_length
          sequence_offset) # subsequent sequences each require seq_off more obs
    stopifnot(target_num_driver_dates <= nrow(driver_dates)) # make sure we're not requiring more dates than we have
    # first drop as many as we can from the righthand side where there are no observations
    target_drop_driver_dates <- nrow(driver_dates) - target_num_driver_dates
    if(target_drop_driver_dates > 0) {
      easy_drops_right <- filter(driver_dates, glm_shoulder == 'right' & obs_shoulder == 'right')
      drops <- tail(easy_drops_right, min(nrow(easy_drops_right), target_drop_driver_dates)) %>% pull(date)
      driver_dates <- driver_dates %>% filter(!date %in% drops)
    }
    # then drop some from the lefthand side where there are no observations
    target_drop_driver_dates <- nrow(driver_dates) - target_num_driver_dates
    if(target_drop_driver_dates > 0) {
      easy_drops_left <- filter(driver_dates, glm_shoulder == 'left' & obs_shoulder == 'left')
      drops <- head(easy_drops_left, min(nrow(easy_drops_left), target_drop_driver_dates)) %>% pull(date)
      driver_dates <- driver_dates %>% filter(!date %in% drops)
    }
    # then drop some from the lefthand side for the remaining dates
    target_drop_driver_dates <- nrow(driver_dates) - target_num_driver_dates
    if(target_drop_driver_dates > 0) {
      drops <- head(driver_dates, min(nrow(driver_dates), target_drop_driver_dates)) %>% pull(date)
      driver_dates <- driver_dates %>% filter(!date %in% drops)
    }
    # actually truncate the data
    return(filter(drivers, date %in% driver_dates$date))
  }

  #### Helper functions for splitting ####

  # function to normalize the driver data into features. We will use this
  # function right after creating the relevant splits. This takes the driver
  # data, removes the Ice column, and normalizes the remaining columns (subtract
  # mean and divide by standard devation of each column)
  normalize <- function(driver_split) {
    driver_split %>%
      select(-Ice) %>% # permanently remove the Ice column from the normalized features
      select(-date, -depth) %>% # temporarily remove the date column for scaling
      scale() %>%
      as_tibble() %>%
      tibble::add_column(depth = driver_split$depth, .before=1) %>% # restore the depth column
      tibble::add_column(date = driver_split$date, .before=1) # restore the date column
  }

  # function to truncate a dataset to exactly the date range spanned by the
  # given dates. unlike truncate_drivers, this function does not also truncate
  # to an integer number of sequences; that's the job of truncate_drivers() or
  # calc_padding() + buffer()
  restrict <- function(dat, to) {
    date_range <- range(to$date)
    dat %>% filter(date >= date_range[1], date <= date_range[2])
  }

  # pad dat_restricted with NAs so that the output is the length of an integer
  # number of sequences. Pad to the left as much as possible but no farther than
  # the available drivers, then pad to the right
  buffer <- function(dat_restricted) {

    ## Compute Padding ##

    # count dates
    n_core_dates <- length(unique(dat_restricted$date))

    # count driver dates to the left and right available for buffering
    n_driver_dates <- length(unique(drivers$date))
    n_driver_dates_left <- length(unique(filter(drivers, date < min(dat_restricted$date))$date))
    n_driver_dates_right <- length(unique(filter(drivers, date > max(dat_restricted$date))$date))

    # be realistic about the minimum buffer to the left; can't have more buffer
    # than we have drivers to the left
    try_buffer <- sequence_length/2 # seems like half a sequence length would be pretty good
    min_buffer <- min(try_buffer, n_driver_dates_left)

    # optimistically round up to the nearest number of dates that will fill an
    # integer number of sequences
    target_n_dates <- max(n_core_dates + min_buffer, sequence_length) # should be at least sequence_length long
    target_n_dates <- sequence_length + # there's always one first sequence (see previous line)
      ceiling((target_n_dates - sequence_length) / sequence_offset) * sequence_offset # add sequences to cover the core dates + min_buffer

    # pessimistically round down to the max number of driver dates that would fill an integer number of sequences
    round_n_driver_dates <- sequence_length + # there's always one first sequence (see previous line)
      floor((n_driver_dates - sequence_length) / sequence_offset) * sequence_offset # add sequences to cover the core dates + min_buffer

    # be realistic about the final buffer; can't have more total length than we
    # have drivers that divide evenly into sequences
    target_n_dates <- min(target_n_dates, round_n_driver_dates)

    # compute padding to the left and right, prioritizing padding left first
    target_n_pad_dates <- target_n_dates - n_core_dates
    pad_left <- min(n_driver_dates_left, target_n_pad_dates)
    pad_right <- min(n_driver_dates_right, target_n_pad_dates - pad_left)

    ## Do Padding ##

    dat_range <- range(dat_restricted$date)
    as_days <- function(n) { as.difftime(n, units='days') }
    no_date <- Sys.Date()[c()]
    pad_left_dates <- if(pad_left > 0) seq(dat_range[1] - as_days(pad_left), dat_range[1] - as_days(1), by=as_days(1)) else no_date
    pad_right_dates <- if(pad_right > 0) seq(dat_range[2] + as_days(1), dat_range[2] + as_days(pad_right), by=as_days(1)) else no_date
    dat_padded <- tibble::tibble(date = c(pad_left_dates, pad_right_dates), depth = 0) %>%
      tidyr::complete(date, depth = sort(unique(dat_restricted$depth))) %>%
      bind_rows(dat_restricted) %>% # add the data rows, filling in the padding data columns with NAs
      arrange(date, depth) # sort to chronological order

    return(dat_padded)
  }

  # function to generate an observation mask with 1s where an observation is
  # available and 0s otherwise
  mask <- function(obs_split) {
    obs_split %>%
      transmute(
        date = date,
        depth = depth,
        Multiplier = ifelse(is.na(TempC), 0, 1))
  }

  # function to double check that we have three identically dimensioned
  # data.frames
  require_equal_dims <- function(features, labels, mask) {
    stopifnot(all.equal(
      select(features, date, depth),
      select(labels, date, depth)
    ))
    stopifnot(all.equal(
      select(labels, date, depth),
      select(mask, date, depth)
    ))
  }

  #### Create data splits ####

  # Initialize an R list where we'll keep all the dataset splits
  datasets <- list()

  # For unsupervised learning, always use the full driver dataset (normalized
  # and unnormalized), but exclude days without Ice information because these
  # can't be tested for energy conservation. Use the full dataset both in
  # pretraining and in fine tuning, to prevent unrealistic states or budget
  # closure during both phases
  datasets$unsup$physics <- truncate_drivers(filter(drivers, !is.na(Ice)), glm_preds, obs)
  datasets$unsup$features <- normalize(datasets$unsup$physics)

  # Pretrain on the full GLM predictions dataset (even though there could be
  # situations where the prediction in a specific year gets worse because we're
  # training on 1979 and climate has changed)
  datasets$pretrain$labels <- glm_preds %>%
    restrict(filter(glm_preds, !is.na(TempC))) %>%
    buffer()
  datasets$pretrain$mask <- mask(datasets$pretrain$labels)
  datasets$pretrain$features <- drivers %>%
    restrict(datasets$pretrain$labels) %>%
    normalize()
  require_equal_dims(datasets$pretrain$features, datasets$pretrain$labels, datasets$pretrain$mask)

  #' Slice the obs and drivers into labels, mask, and features for one
  #' application (probably test, tune, or train)
  #'
  #' @param low_obs_num the index of the first observation to include (counting
  #'   first from the left and then from the right)
  #' @param high_obs_num NA if one data chunk is desired. if two chunks are
  #'   desired, this is the index of the final observation to include in each
  #'   shoulder (counting first from the left for shoulder 1 and then from the
  #'   right for shoulder 2)
  slice_labels_features_mask <- function(obs, drivers, low_obs_num, high_obs_num) {
    obs_non_na <- filter(obs, !is.na(TempC))
    obs_dates <- sort(unique(obs_non_na$date))
    obs_dates_rev <- rev(obs_dates)

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
        left = obs %>%
          restrict(filter(obs_non_na, date >= date_cutoffs[1] & date <= date_cutoffs[2])) %>%
          buffer(),
        right = obs %>%
          restrict(filter(obs_non_na, date >= date_cutoffs[3] & date <= date_cutoffs[4])) %>%
          buffer())
      # pick out the two shoulders worth of features (based on the labels
      # shoulders) and bind them together
      one_slice$features <- bind_rows(
        left = restrict(drivers, one_slice$labels$left),
        right = restrict(drivers, one_slice$labels$right)) %>%
        normalize()
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
      one_slice$labels <- obs %>%
        restrict(filter(obs_non_na, date >= date_cutoffs[1] & date <= date_cutoffs[2])) %>%
        buffer()
      # pick out the features based on the labels
      one_slice$features <- restrict(drivers, one_slice$labels) %>%
        normalize()
    }

    # compute the mask based on the labels
    one_slice$mask <- mask(one_slice$labels)

    # check the dimensions to make sure we've ended up with three equally shaped data.frames
    require_equal_dims(one_slice$features, one_slice$labels, one_slice$mask)

    # return the labels, features, and mask for this one data slice
    return(one_slice)
  }

  # For hyperparameter tuning (tune-training and tune-testing), use the GLM
  # predictions
  n_glm_dates <- glm_preds %>%
    filter(!is.na(TempC)) %>%
    pull(date) %>%
    unique() %>%
    length()
  if(n_glm_dates <= 60) {
    # <60 GLM dates: no hyperparameter tuning
    stop(sprintf('only %d GLM prediction dates!?! Hard to tune hyperparameters on that', n_glm_dates))
  } else if(n_glm_dates <= 100) {
    # 61-100 GLM dates: 30 test obs, remainder (31-70) for training
    datasets$tune_train <- slice_labels_features_mask(glm_preds, drivers, 16, NA)
    datasets$tune_test <- slice_labels_features_mask(glm_preds, drivers, 1, 15)
  } else if(n_glm_dates <= 200) {
    # 101-200 GLM dates: 40 test obs, remainder (61-160) for training
    datasets$tune_train <- slice_labels_features_mask(glm_preds, drivers, 21, NA)
    datasets$tune_test <- slice_labels_features_mask(glm_preds, drivers, 1, 20)
  } else if(n_glm_dates <= 400) {
    # 201-400 GLM dates: 50 test obs, remainder (151-350) for training
    datasets$tune_train <- slice_labels_features_mask(glm_preds, drivers, 26, NA)
    datasets$tune_test <- slice_labels_features_mask(glm_preds, drivers, 1, 25)
  } else {
    # 401+ GLM dates: 100 test obs, remainder (301+) for training. this should be
    # by far the most common case, because we usually have many GLM predictions
    datasets$tune_train <- slice_labels_features_mask(glm_preds, drivers, 51, NA)
    datasets$tune_test <- slice_labels_features_mask(glm_preds, drivers, 1, 50)
  }

  # For training (= fine tuning) and final testing, split the observations
  # differently depending on how many observation dates are available
  n_obs_dates <- obs %>%
    filter(!is.na(TempC)) %>%
    pull(date) %>%
    unique() %>%
    length()
  if(n_obs_dates <= 60) {
    # <60 obs dates: no test obs, everything for training
    datasets$train <- slice_labels_features_mask(obs, drivers, 1, NA)
    datasets$test <- NULL
  } else if(n_obs_dates <= 100) {
    # 61-100 obs dates: 30 test obs, remainder (31-70) for training
    datasets$train <- slice_labels_features_mask(obs, drivers, 16, NA)
    datasets$test <- slice_labels_features_mask(obs, drivers, 1, 15)
  } else if(n_obs_dates <= 200) {
    # 101-200 obs dates: 40 test obs, remainder (61-160) for training
    datasets$train <- slice_labels_features_mask(obs, drivers, 21, NA)
    datasets$test <- slice_labels_features_mask(obs, drivers, 1, 20)
  } else {
    # 201+ obs dates: 50 test obs, remainder (151+) for training
    datasets$train <- slice_labels_features_mask(obs, drivers, 26, NA)
    datasets$test <- slice_labels_features_mask(obs, drivers, 1, 25)
  }

  # For prediction, use the full driver dataset (no need to restrict to just
  # those days with Ice estimates)
  datasets$predict$features <- normalize(truncate_drivers(drivers, glm_preds, obs))

  return(datasets)
}

summarize_pgdl_split <- function(splits, tidied) {
  split_obs_dates <-
    bind_rows(mutate(tidied$obs, Type='Obs'),
              mutate(splits$tune_train$labels, Type='Tune-Train'),
              mutate(splits$tune_test$labels, Type='Tune-Test'),
              mutate(splits$pretrain$labels, Type='Pretrain'),
              mutate(splits$train$labels, Type='Train'),
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

plot_pgdl_split_days <- function(splits, tidied) {
  split_obs_dates <-
    bind_rows(mutate(tidied$obs, Type='Obs'),
              mutate(splits$tune_train$labels, Type='Tune-Train'),
              mutate(splits$tune_test$labels, Type='Tune-Test'),
              mutate(splits$pretrain$labels, Type='Pretrain'),
              mutate(splits$train$labels, Type='Train'),
              mutate(splits$test$labels, Type='Test')) %>%
    group_by(date, Type) %>%
    summarize(obs_date = any(!is.na(TempC))) %>%
    ungroup() %>%
    mutate(Type = ordered(Type, levels=c('Tune-Train','Tune-Test','Pretrain','Obs','Train','Test')))
  ggplot(split_obs_dates, aes(x=date, y=1)) +
    geom_point(shape=NA) +
    geom_vline(data=filter(split_obs_dates, !obs_date), aes(xintercept=date, color=obs_date)) +
    geom_vline(data=filter(split_obs_dates, obs_date), aes(xintercept=date, color=obs_date)) +
    ylab('') + theme_classic() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
    facet_grid(Type ~ ., scales='free_y')
}

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
#' x_train_1: Input data for supervised learning y_train_1: Observations for
#' supervised learning m_train_1: Observation mask for supervised learning x_f:
#' Normalized input data for unsupervised learning p_f: Unnormalized input data
#' for unsupervised learning (p = physics) = concat(p_train_1, p_test) x_test:
#' Inputs data for testing y_test: Observations for testing m_test: Observation
#' mask for testing depth_areas: cross-sectional area of each depth
#'
#' @param splits an R list of data split into unsup, train, test, etc.
#' @param tidied an R list of tidied data
#' @param sequence_length the number of dates per timeseries sequence passed in
#'   as a training example
#' @param sequence_offset the number of dates by which each subsequent sequence
#'   lags behind the previous sequence (per date)
reshape_data_to_sequences <- function(splits, depths, sequence_length, sequence_offset) {

  n_depths <- length(depths)

  # create matrices for everything except for unsup and predict
  lapply(setNames(nm=names(splits)), function(phase_name) {

    message(phase_name)
    one_phase <- splits[[phase_name]]

    phase_seqs <- lapply(setNames(nm=names(one_phase)), function(element_name) {
      message("  ", element_name)
      phase_element <- one_phase[[element_name]]

      # create a list with a separate data_frame for each shoulder (or just one
      # element if the dates are contiguous) to avoid creating sequences that
      # span a date gap
      phase_element_shoulders <- phase_element %>%
        mutate(
          shoulder = {
            shoulder_end <- date[which(as.numeric(diff(date), units='days') > 1)]
            if(length(shoulder_end) > 1) stop('found more than 1 shoulder break, not ready for that')
            if(length(shoulder_end) == 1) {
              ifelse(date <= shoulder_end, 'left', 'right')
            } else 'all'
          }) %>%
        group_by(shoulder) %>%
        arrange(date, depth) %>%
        tidyr::nest() %>%
        pull(data)

      # create a complete list of arrays, one per sequence
      seq_arrs <- lapply(phase_element_shoulders, function(shoulder) {
        n_sequences <- 1 + (length(unique(shoulder$date)) - sequence_length) / sequence_offset
        stopifnot(n_sequences %% 1 == 0) # n_sequences should be an integer

        # split into sequences and pack into 3D or 4D matrix
        pb <- progress::progress_bar$new(total = n_sequences, format = "[:bar] :current/:total (:percent)")
        seqs <- lapply(seq_len(n_sequences), function(seq_i) {
          pb$tick() # increment the progress bar

          sequence <- shoulder[n_depths*sequence_offset*(seq_i-1)+(1:(n_depths*sequence_length)),]
          seq_arr <- reshape_sequence_to_array(sequence, depths)
        })
      }) %>%
        unlist(recursive=FALSE) %>% # simplify to a list of seq_arrs
        {setNames(., sprintf('%d', seq_along(.) - 1)) }

      return(seq_arrs)
    })

    return(phase_seqs)
  })
}

reshape_sequence_to_array <- function(sequence, depths) {

  vars <- setdiff(names(sequence), c('date', 'depth'))
  three_dimensional <- length(vars) > 1 # true for features, false for labels and mask
  seq_df <- sequence %>%
    mutate(date = format(date, '%Y-%m-%d'))
  seq_dates <- unique(seq_df$date)

  if(three_dimensional) {
    seq_arr <- array(
      data = NA,
      dim = c(depth=length(depths), date=length(seq_dates), feature=length(vars)),
      dimnames = list(depth=depths, date=seq_dates, feature=vars))
  } else {
    seq_arr <- array(
      data = NA,
      dim = c(depth=length(depths), date=length(seq_dates)),
      dimnames = list(depth=depths, date=seq_dates))
  }
  for(var_j in vars) {
    seq_layer <- seq_df %>%
      {.[,c('date','depth',var_j)]} %>% # faster than select(date, depth, !!var_j)
      spread(date, !!var_j) %>%
      as.matrix() %>%
      {.[,2:ncol(.)]} # faster than select(-depth) 2 lines above
    if(three_dimensional) {
      seq_arr[,,var_j] <- seq_layer
    } else {
      seq_arr[,] <- seq_layer
    }
  }

  return(seq_arr)
}

#' Combines all of the above functions
prep_pgdl_data_R <- function(
  lake_id = 'nhd_1099476',
  obs_file = '1_format/tmp/obs/nhd_1099476_obs.fst',
  geometry_file = '1_format/tmp/geometry/nhd_1099476_geometry.csv',
  glm_preds_file = '1_format/tmp/glm_preds/nhd_1099476_temperatures.feather',
  drivers_file = '1_format/tmp/drivers/NLDAS_time[0.350500]_x[254]_y[160].csv',
  # these defaults are where sequence_length and sequence_offset get configured:
  sequence_length = 200,
  sequence_offset = 100) {

  # munge from inputs from other pipelines into long-form data
  tidied <- tidy_pgdl_data(lake_id, drivers_file, geometry_file, glm_preds_file, obs_file)

  # munge from long-form data into lists of long-form datasets split by test,
  # train, pretrain, etc.
  splits <- split_pgdl_data(tidied, sequence_length, sequence_offset)

  # optional diagnostics to investigate the results of splitting
  # summarize_pgdl_split(splits, tidied)
  # plot_pgdl_split_days(splits, tidied)

  # munge from long data to arrays. runtime reduced from 330 down to 150 seconds
  # for a test lake, but this is still the slow step
  system.time({
    sequences <- reshape_data_to_sequences(splits, tidied$geometry$Depth, sequence_length, sequence_offset)
  })

  # combine into a single list for saving in remake
  return(list(sequences=sequences, geometry=tidied$geometry))
}


#' Write R objects to python .npz file
#'
#' @param r_data R list of input sequences and geometry data
#' @param data_file path and filename of .npz file
save_data_to_np <- function(r_data, data_file) {

  # start to unpack the list of R datasets
  sequences <- r_data$sequences
  geometry <- r_data$geometry

  # flatten each list of arrays into a big array with dim+1, and flatten the
  # nested phase-element lists into one list. result is a list of
  # multi-dimensional arrays where the final dimension of each array is the
  # sequence ID, and the list element names are things like unsup.physics and
  # train.features
  flat_seq_list <- lapply(unlist(sequences, recursive=FALSE), function(phase_element_seqs) {
    new_dim <- 1+length(dim(phase_element_seqs[[1]]))
    do.call(abind::abind, c(phase_element_seqs, list(along=new_dim)))
  })

  # add the sequence feature names to the list, extracting those feature names
  # from a single example of each sequence type
  data_list <- c(
    flat_seq_list,
    list(
      colnames.features = dimnames(sequences$train$features[[1]])$feature,
      colnames.physics = dimnames(sequences$unsup$physics[[1]])$feature))

  # add the lake geometry matrix to the list
  data_list <- c(
    data_list,
    list(geometry = as.matrix(geometry)))

  # save the whole list as a big .npz file (one numpy "file" per element in
  # data_list)
  np <- reticulate::import('numpy')
  if(!dir.exists(dirname(data_file))) dir.create(dirname(data_file))
  do.call(np$savez_compressed, c(list(file = data_file), data_list)) # compression saves about 96% of the space
}
