#' @param priority_lakes data_frame including site_ids for all lakes to be
#'   modeled
#' @param sequence_offset Number of observations by which each data sequence in
#'   inputs['predict.features'] is offset from the previous sequence. Used to
#'   reconstruct a complete prediction sequence without duplicates. The value
#'   should always match the default for prep_pgdl_data_R() (and really should
#'   be configured from on high for both that fun and this one)
create_model_config <- function(phase=c('tune','pretrain_train'), priority_lakes, pgdl_inputs_ind, sequence_cfg) {

  # define a template to set the column order and data types for the config file
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

  if(phase == 'tune') {
    # config for tuning (not yet tested or well thought through)
    config <- bind_rows(lapply(priority_lakes$site_id, function(site_id) {
      crossing(
        state_size=c(8,12,16),
        elam=c(0.02, 0.01, 0.05)
      ) %>% mutate(
        phase = 'tune',
        learning_rate = 0.005,
        ec_threshold = 24,
        plam = 0.15,
        data_file = sprintf('1_format/tmp/pgdl_inputs/%s.npz', site_id),
        sequence_offset = sequence_cfg$sequence_offset,
        max_batch_obs = 50000, # my computer can handle about 50000 for state_size=14, about 100000 for state_size=8
        n_epochs = 100,
        min_epochs_test = 0,
        min_epochs_save = 50,
        restore_path = ''
      ) %>% bind_rows(
        template, .
      ) %>% mutate(
        # define a task id we can use to name the drake targets and models. use
        # a hash in the name so that it's concise and yet won't change even if
        # we change the config row order. use symbols because otherwise drake
        # will quote with '.'s in the task names (ugly). use a similar naming
        # scheme for saving the models
        task_hash = sapply(1:n(), function(row) { digest::digest(select(., -save_path)[row,]) }),
        task_id = rlang::syms(sprintf('%s.%s', site_id, task_hash)),
        save_path = sprintf('2_model/tmp/%s/tune/%s', site_id, task_hash)
      ) %>%
        # attach site_id last so site_id remains a length-1 vector when
        # computing data_file, restore_path, etc. (even though we have 2 tibble
        # rows per site because phase is length 2)
        mutate(site_id = site_id) %>%
        # bind to the template to standardize the column order
        bind_rows(template, .)
    }))
  } else if(phase == 'pretrain_train') {
    # config for pretrain and train
    config <- bind_rows(lapply(priority_lakes$site_id, function(site_id) {
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
        restore_path = c('',sprintf('2_model/tmp/%s/pretrain', site_id)),
        save_path = sprintf('2_model/tmp/%s/%s', site_id, phase)
      ) %>%
        mutate(site_id = site_id) %>%
        bind_rows(template, .)
    }))
  }

  # Attach data file hashes to the config table
  pgdl_inputs_md5 <- unlist(yaml::yaml.load_file(pgdl_inputs_ind)) # Read in the model-ready data file hashes, named by file names
  config <- config %>% # Augment the config table with the file hashes
    mutate(pgdl_inputs_md5 = pgdl_inputs_md5[data_file])

  return(config)
}
