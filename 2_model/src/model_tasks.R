run_model_tasks <- function(ind_file, model_config_file, computer=c('slurm', 'pc')) {
  computer <- match.arg(computer)

  library(drake) # the transform option requires devtools::install_github('ropensci/drake') as of 3/27/19
  source('2_model/src/run_job.R') # calls run_job.py
  library(dplyr)
  library(readr)
  library(scipiper)

  if(computer == 'slurm') {
    library(future.batchtools)
    future::plan(batchtools_slurm, template = "2_model/src/slurm_batchtools.tmpl")
  }

  # Convert task_id from character to symbol because otherwise drake will quote
  # with '.'s in the task names (which is ugly)
  model_config <- readr::read_tsv(model_config_file, na='NA', col_types='cddddddcdddddccccddcccd') %>%
    mutate(ncpus = 1, ngpus = 1, gpu.type = 'quadro', walltime = '120') %>%
    tidyr::nest(ncpus, ngpus, gpu.type, walltime, .key = 'resources') %>%
    # mutate(resources = lapply(resources, as.list)) %>%
    mutate(task_sym = rlang::syms(task_id))

  # Create the drake task plan
  model_plan <- drake_plan(
    fit = target(
      {
       source('2_model/src/run_job.R')
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
       )
      },
      cfg.task_id = task_id, # drake won't allow task_id = task_id
      transform = map(.data = !!model_config, .id = c(task_sym))
    )
  ) %>%
    left_join(select(model_config, task_id, resources), by=c(cfg.task_id = 'task_id')) %>%
    select(-cfg.task_id)

  print(model_plan)

  # Here's how to visualize the plan structure:
  # build_config <- drake_config(model_plan)
  # vis_drake_graph(build_config)

  # Actually run the plan
  if(computer == 'slurm') {
    drake::make(model_plan, parallelism='future', jobs=nrow(model_plan))
  } else {
    drake::make(model_plan)
  }

  # If there are no remaining tasks, construct an indicator file to satisfy remake/scipiper
  plan_config <- drake_config(model_plan)
  remaining_tasks <- drake::outdated(plan_config)
  if(length(remaining_tasks) == 0) {
    saved_files <- unlist(lapply(model_config$save_path, dir, full.names=TRUE))
    if(!is.na(ind_file)) {
      sc_indicate(ind_file = ind_file, data_file = saved_files)
    } else {
      message("All models are complete, but not writing indicator because ind_file=NA")
    }
  } else {
    stop(sprintf("We may have made progress, but %s models remain to fit", length(remaining_tasks)))
  }
}
