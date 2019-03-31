run_model_tasks <- function(ind_file, model_config_file) {
  library(drake) # the transform option requires devtools::install_github('ropensci/drake') as of 3/27/19
  source('2_model/src/run_job.R') # calls run_job.py

  # convert task_id from character to symbol because otherwise drake will quote
  # with '.'s in the task names (which is ugly)
  model_config <- readr::read_tsv(model_config_file, na='NA') %>%
    mutate(task_id = rlang::syms(task_id))

  model_plan <- drake_plan(
    fit = target(
      run_job(
        config = model_config[row,],
        file_in(data_file), # the model-ready .npz data file
        file_in(restore_path), # the dir of the pretrained model, if any. file_in seems to be OK with '' as an input
        file_in('2_model/src/run_job.py'), # uses apply_pgnn.py
        file_in('2_model/src/apply_pgnn.py'), # uses tf_graph.py and tf_train.py
        file_in('2_model/src/tf_graph.py'), # uses physics.py
        file_in('2_model/src/physics.py'),
        file_in('2_model/src/tf_train.py'),
        file_out(save_path) # the dir of the resulting model. drake says it handles whole directories, though I don't know how it behaves exactly
      ),
      transform = map(.data = !!model_config, .id = c(task_id))
    )
  )
  model_plan$resources <- list(list(
    ncpus = 1, # number of CPU cores per job. 1 is fine when using tensorflow-gpu
    ngpus = 1, # number of GPU cores per job. Bump this above 1 when not testing
    gpu.type = 'quadro', # quadro may be less busy; tesla is faster once running
    walltime = '10')) # runtime in minutes, in minutes:seconds, or in hours:minutes:seconds

  print(model_plan)

  # Here's how to visualize the plan structure:
  # build_config <- drake_config(model_plan)
  # vis_drake_graph(build_config)

  # Actually run the plan
  drake::make(model_plan)

  # If there are no remaining tasks, construct an indicator file to satisfy remake/scipiper
  plan_config <- drake_config(model_plan)
  remaining_tasks <- drake::outdated(plan_config)
  if(length(remaining_tasks) == 0) {
    saved_files <- unlist(lapply(plan_details$save_path, dir, full.names=TRUE))
    sc_indicate(ind_file = ind_file, data_file = saved_files)
  } else {
    stop(sprintf("We may have made progress, but %s models remain to fit", length(remaining_tasks)))
  }
}
