#' @param priority_lakes data_frame including site_ids for all lakes to be
#'   modeled
#' @param sequence_offset Number of observations by which each data sequence in
#'   inputs['predict.features'] is offset from the previous sequence. Used to
#'   reconstruct a complete prediction sequence without duplicates. The value
#'   should always match the default for prep_pgdl_data_R() (and really should
#'   be configured from on high for both that fun and this one)
create_model_config <- function(priority_lakes, sequence_cfg) {
  # define a template to set the column order and data types
  template <- tibble(
    phase = '',
    learning_rate = 0,
    state_size = 0,
    ec_threshold = 0,
    plam = 0,
    elam = 0,
    data_file = '',
    sequence_offset = 0,
    max_batch_obs = 0,
    n_epochs = 0,
    min_epochs_test = 0,
    min_epochs_save = 0,
    restore_path = '',
    save_path = ''
  ) %>%
    filter(FALSE)

  # # config for tuning
  # tune <- bind_rows(lapply(priority_lakes$site_id, function(site_id) {
  #   crossing(
  #     state_size=c(8,12,16),
  #     elam=c(0.02, 0.01, 0.05)) %>%
  #   mutate(
  #     phase = 'tune',
  #     learning_rate = 0.005,
  #     ec_threshold = 24,
  #     plam = 0.15,
  #     data_file = sprintf('1_format/tmp/pgdl_inputs/%s.npz', site_id),
  #     sequence_offset = sequence_cfg$sequence_offset,
  #     max_batch_obs = 50000, # my computer can handle about 50000 for state_size=14, about 100000 for state_size=8
  #     n_epochs = 100,
  #     min_epochs_test = 0,
  #     min_epochs_save = 50,
  #     restore_path = '',
  #     save_path = sprintf('2_model/tmp/tune/%s_%02d', site_id, 1:n())
  #   ) %>%
  #     bind_rows(template, .)
  # }))

  # config for pretrain and train
  pretrain_train <- bind_rows(lapply(priority_lakes$site_id, function(site_id) {
    tibble(
      phase = c('pretrain', 'train'),
      learning_rate = c(0.008, 0.005),
      state_size = 12,
      ec_threshold = 24,
      plam = 0.15,
      elam = 0.025,
      data_file = sprintf('1_format/tmp/pgdl_inputs/%s.npz', site_id),
      sequence_offset = sequence_cfg$sequence_offset,
      max_batch_obs = 50000, # my computer can handle 50000 for state_size=14, 100000 for state_size=8
      n_epochs = c(3, 5), # 50, 100 would be better. or higher.
      min_epochs_test = 0,
      min_epochs_save = n_epochs, # until we have early stopping, there's no point in saving earlier models
      restore_path = c('',sprintf('2_model/tmp/%s_pretrain', site_id)),
      save_path = sprintf(c('2_model/tmp/%s_pretrain', '2_model/tmp/%s'), site_id)
    ) %>%
      mutate(site_id = site_id) %>%
      bind_rows(template, .)
  }))

  #return(bind_rows(tune, pretrain_train))
  return(pretrain_train)
}
