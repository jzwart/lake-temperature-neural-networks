#' @param config All the configuration information we need to send to python to
#'   run the job
#' @param ... File dependencies (passed with file_in() and file_out() from the
#'   drake plan)
run_job <- function(config, ...) {
  py_args <- config %>% mutate(
    args = psprintf(
      '--phase="%s"'=phase,
      '--learning_rate=%0.5f'=learning_rate,
      '--state_size=%d' = state_size,
      '--ec_threshold=%0.2f' = ec_threshold,
      '--dd_lambda=%0.6f' = dd_lambda,
      '--ec_lambda=%0.6f' = ec_lambda,
      '--l1_lambda=%0.6f' = l1_lambda,
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

  py_call <- sprintf('python 2_model/src/run_job.py %s', py_args)
  cat(py_call) # print() displays right away in the R console, whereas message() doesn't display until the model has run
  system(py_call)
}
