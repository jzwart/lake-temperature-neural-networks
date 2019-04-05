#' @param priority_lakes data_frame including site_ids for all lakes to be
#'   modeled
#' @param sequence_offset Number of observations by which each data sequence in
#'   inputs['predict.features'] is offset from the previous sequence. Used to
#'   reconstruct a complete prediction sequence without duplicates. The value
#'   should always match the default for prep_pgdl_data_R() (and really should
#'   be configured from on high for both that fun and this one)
create_model_config <- function(
  out_file_basename, phase=c('hypertune','pretrain_train'), priority_lakes, pgdl_inputs_ind, sequence_cfg) {

  phase <- match.arg(phase)

  # define a template to set the column order and data types for the config file
  template <- tibble(
    phase = '',
    learning_rate = 0,
    state_size = 0,
    ec_threshold = 0,
    dd_lambda = 0,
    ec_lambda = 0,
    l1_lambda = 0,
    data_file = '',
    sequence_offset = 0,
    max_batch_obs = 0,
    n_epochs = 0,
    min_epochs_test = 0,
    min_epochs_save = 0,
    restore_path = '',
    save_path = '',
    site_id = '',
    task_id = '',
    ncpus = 0,
    ngpus = 0,
    gpu.type = '',
    walltime = '',
    pgdl_inputs_md5 = '',
    row = 0
  ) %>%
    filter(FALSE)

  if(phase == 'hypertune') {
    # config for tuning (exploratory for now, just using one lake)
    config <- bind_rows(lapply(priority_lakes$site_id[1], function(site_id) {
      crossing(
        state_size = c(8,16),
        l1_lambda = c(0.005, 0.02, 0.1)
      ) %>% mutate(
        phase = 'tune',
        learning_rate = 0.005,
        ec_threshold = 24,
        ec_lambda = 0.04,
        dd_lambda = 0, # 0.15 might be good if we actually had a depth-density constraint
        data_file = sprintf('1_format/tmp/pgdl_inputs/%s.npz', site_id),
        sequence_offset = sequence_cfg$sequence_offset, # this should not be edited. it should always match sequence_cfg
        max_batch_obs = 50000, # my computer can handle about 50000 for state_size=14, about 100000 for state_size=8
        n_epochs = 100,
        min_epochs_test = 0,
        min_epochs_save = n_epochs,
        restore_path = '',
        ncpus = 1, # number of CPU cores per job. 1 is fine when using tensorflow-gpu
        ngpus = 1, # number of GPU cards per job. I don't think we can make good use of more than 1
        gpu.type = 'quadro', # quadro may be less busy; tesla is faster once running
        walltime = '120', # runtime in minutes, in minutes:seconds, or in hours:minutes:seconds
        # attach site_id toward the end so site_id remains a length-1 vector when
        # computing data_file, restore_path, etc. (even though we have 2 tibble
        # rows per site because phase is length 2)
        site_id = site_id
      ) %>%
        # bind to the template to standardize the column order. we also do this
        # after the if-else block, but do it now for consistency in the task_hash
        bind_rows(template, .) %>%
        mutate(
          # define a task id we can use to name the drake targets and models. use
          # a hash in the name so that it's concise and yet won't change even if
          # we change the config row order. use a similar naming scheme for saving
          # the models
          task_hash = sapply(1:n(), function(row) { digest::digest(.[row,]) }),
          task_id = sprintf('%s.%s', site_id, task_hash),
          save_path = sprintf('2_model/tmp/%s/tune/%s', site_id, task_hash)
        ) %>%
        # keep the cols the same as for pretrain_train, i.e., no task_hash
        select(-task_hash)
    }))
  } else if(phase == 'pretrain_train') {
    # config for pretrain and train
    config <- bind_rows(lapply(priority_lakes$site_id, function(site_id) {
      tibble(
        phase = c('pretrain', 'train'),
        learning_rate = c(0.008, 0.005),
        state_size = 14, # from tests on 190404, looks like 0.02 with state_size 16 is pretty good...but 14 should be faster
        ec_threshold = 24,
        dd_lambda = 0, # 0.15 might be good if we actually had a depth-density constraint
        ec_lambda = 0,
        l1_lambda = 0, # from tests on 190404, looks like 0.02 with state_size 16 is pretty good
        data_file = sprintf('1_format/tmp/pgdl_inputs/%s.npz', site_id),
        sequence_offset = sequence_cfg$sequence_offset, # this should not be edited. it should always match sequence_cfg
        max_batch_obs = 50000, # my computer can handle 50000 for state_size=14, 100000 for state_size=8
        n_epochs = 200, # tests on 190404 suggest that 100 may be nearly enough, so 200 would probably guarantee convergence
        min_epochs_test = 0,
        min_epochs_save = n_epochs, # until we have early stopping, there's no point in saving earlier models
        restore_path = c('',sprintf('2_model/tmp/%s/pretrain', site_id)),
        save_path = sprintf('2_model/tmp/%s/%s', site_id, phase),
        ncpus = 1, # number of CPU cores per job. 1 is fine when using tensorflow-gpu
        ngpus = 1, # number of GPU cores per job. Bump this above 1 when not testing
        gpu.type = 'quadro', # quadro may be less busy; tesla is faster once running
        walltime = '120', # runtime in minutes, in minutes:seconds, or in hours:minutes:seconds
        # attach site_id last so site_id remains a length-1 vector when
        # computing data_file, restore_path, etc. (even though we have 2 tibble
        # rows per site because phase is length 2)
        site_id = site_id
      )
    })) %>%
      # attach suggestion for how to name the drake targets
      mutate(task_id = sprintf('%s.%s', site_id, phase))
  }

  # Make sure the config is structured like the template
  config <- bind_rows(template, config) # bind to the template to standardize the column order
  stopifnot(all.equal(template[c(),], config[c(),])) # check that it worked

  # Attach data file hashes to the config table. drake doesn't use this info,
  # but this is how remake knows when to build and run the drake plan again
  pgdl_inputs_md5 <- unlist(yaml::yaml.load_file(pgdl_inputs_ind)) # Read in the model-ready data file hashes, named by file names
  config <- config %>% # Augment the config table with the file hashes
    mutate(pgdl_inputs_md5 = pgdl_inputs_md5[data_file])

  # Attach attach information drake can use to extract one config row per target
  config <- config %>% mutate(row = 1:n())

  # Both write the table to file and return it as a tibble
  out_file <- sprintf('2_model/out/%s.tsv', out_file_basename)
  if(!dir.exists(dirname(out_file))) dir.create(dirname(out_file))
  readr::write_tsv(config, out_file)
  return(config)
}
