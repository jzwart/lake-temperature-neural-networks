run_job <- function(data_file, restore_dir, save_dir, config) {
  py_args <- config %>% mutate(
    args = psprintf(
      '--phase="%s"'=phase,
      '--learning_rate=%0.5f'=learning_rate,
      '--state_size=%d' = state_size,
      '--ec_threshold=%0.2f' = ec_threshold,
      '--plam=%0.6f' = plam,
      '--elam=%0.6f' = elam,
      '--data_file="%s"' = data_file,
      '--sequence_offset=%d' = sequence_offset,
      '--max_batch_obs=%d' = max_batch_obs,
      '--n_epochs=%d' = n_epochs,
      '--min_epochs_test=%d' = min_epochs_test,
      '--min_epochs_save=%d' = min_epochs_save,
      '--restore_path="%s"' = restore_path,
      '--save_path="%s"' = save_path,
      sep=' '
    )) %>%
    pull(args)
  print(sprintf('python 2_model/src/run_job.py %s', py_args))

  system(sprintf('python 2_model/src/run_job.py %s', py_args))
}
