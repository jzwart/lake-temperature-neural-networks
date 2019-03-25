# Requires task_combiners branch of scipiper
# devtools::install_github('USGS-R/scipiper@task_combiners')

create_format_task_plan <- function(
  priority_lakes,
  obs_tmpind = '1_format/tmp/obs.tmpind',
  geometry_tmpind = '1_format/tmp/geometry.tmpind',
  glm_preds_tmpind = '1_format/tmp/glm_preds.tmpind',
  drivers_tmpind = '1_format/tmp/drivers.tmpind') {

  # read in vectors of file hashes, named by file names, for input files pulled
  # from Yeti and Drive
  md5_vecs <- lapply(
    c(obs=obs_tmpind, geometry=geometry_tmpind, glm_preds=glm_preds_tmpind, drivers=drivers_tmpind),
    function(tmpind) {
      unlist(yaml::yaml.load_file(tmpind))
    })

  # Augment the priority_lakes dataset with the file hashes
  file_hashes <- priority_lakes %>%
    mutate(
      obs_md5 = md5_vecs$obs[obs_file],
      geometry_md5 = md5_vecs$geometry[geometry_file],
      glm_preds_md5 = md5_vecs$glm_preds[glm_preds_file],
      drivers_md5 = md5_vecs$drivers[drivers_file])

  # Create task steps. The use of known md5_hashes avoids duplicating the file
  # hashing that's already been done multiple steps in processing. Can use r
  # objects rather than files as long as final output is file... but will still
  # incur storage and remake-hashing overhead relative to a single output... but
  # could be faster for development if some steps are slow and just one of the
  # steps changes
  task_steps <- list(
    create_task_step(
      step_name = "md5_hashes",
      target_name = function(task_name, ...) {
        sprintf("md5_hashes_%s", task_name)
      },
      command = function(task_name, ...) {
        lake_inputs <- filter(file_hashes, site_id == task_name)
        psprintf(
          "c(",
          "obs_md5 = I('%s')," = lake_inputs$obs_md5,
          "geometry_md5 = I('%s')," = lake_inputs$geometry_md5,
          "glm_preds_md5 = I('%s')," = lake_inputs$glm_preds_md5,
          "drivers_md5 = I('%s'))" = lake_inputs$drivers_md5)
      }
    ),
    create_task_step(
      step_name = "R_data",
      target_name = function(task_name, ...) {
        sprintf("R_data_%s", task_name)
      },
      depends = function(task_name, ...) {
        sprintf("md5_hashes_%s", task_name)
      },
      command = function(task_name, ...) {
        lake_inputs <- filter(file_hashes, site_id == task_name)
        psprintf(
          "prep_pgdl_data_R(",
          "lake_id = I('%s')," = task_name,
          "obs_file = I('%s')," = lake_inputs$obs_file,
          "geometry_file = I('%s')," = lake_inputs$geometry_file,
          "glm_preds_file = I('%s')," = lake_inputs$glm_preds_file,
          "drivers_file = I('%s'))" = lake_inputs$drivers_file)
      }
    ),
    create_task_step(
      step_name = "py_data",
      target_name = function(task_name, ...) {
        file_hashes %>%
          filter(site_id == task_name) %>%
          pull(pgdl_inputs_file)
      },
      command = function(task_name, ...) {
        psprintf(
          "save_data_to_np(",
          "data_file = target_name,",
          "r_data = R_data_%s)" = task_name)
      })
  )

  format_task_plan <- create_task_plan(
    task_names = file_hashes$site_id,
    task_steps = task_steps,
    final_steps = "py_data",
    ind_dir = "1_format/log",
    add_complete = FALSE)

  return(format_task_plan)
}

create_format_task_makefile <- function(makefile, task_plan) {
  create_task_makefile(
    task_plan = task_plan,
    makefile = makefile,
    packages = c(
      "abind",
      "dplyr",
      "feather",
      "fst",
      "googledrive",
      "progress",
      "purrr",
      "readr",
      "reticulate",
      "scipiper",
      "stringr",
      "syncr",
      "tibble",
      "tidyr",
      "yaml"),
    sources = c(
      "1_format/src/prepare_pgdl_data.R",
      "1_format/src/combiners.R"),
    include = "1_format.yml",
    final_targets = "1_format/log/pgdl_inputs.ind",
    finalize_funs = "upload_pgdl_inputs")
}

