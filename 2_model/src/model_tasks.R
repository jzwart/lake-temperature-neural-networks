run_model_tasks <- function(ind_file, model_config) {
  library(drake) # the transform option requires devtools::install_github('ropensci/drake') as of 3/27/19
  source('2_model/src/run_job.R')

  plan_details <- model_config %>%
    mutate(
      id = sprintf('%s.%s', site_id, phase),
      row = 1:n())

  model_plan <- drake_plan(
    fit = target(
      run_job(
        data_file = file_in(data_file),
        restore_dir = file_in(restore_path),
        save_dir = file_out(save_path),
        config = model_config[row,]),
      transform = map(.data = !!plan_details, .id = c(id))
    )
  )
  model_plan$resources <- list(list(cores = 1, gpus = 0, walltime = 10))

  # Here's how to visualize the plan structure:
  # build_config <- drake_config(model_plan)
  # vis_drake_graph(build_config)

  # Actually run the plan
  drake::make(model_plan)

  # If there are no remaining tasks, construct an indicator file
  plan_config <- drake_config(model_plan)
  remaining_tasks <- drake::outdated(plan_config)
  if(length(remaining_tasks) == 0) {
    saved_files <- unlist(lapply(plan_details$save_path, dir, full.names=TRUE))
    sc_indicate(ind_file = ind_file, data_file = saved_files)
  } else {
    stop(sprintf("We may have made progress, but %s models remain to fit", length(remaining_tasks)))
  }
}
