#' Throws error "failed: No such file or directory...rsync failed with code 23:
#' Partial transfer due to error" if any of the requested files are unavailable
retrieve_lake_data <- function(out_ind, priority_lakes, file_column, yeti_path) {
  # download the priority lake files from yeti, using the file info from
  # priority_lakes
  requested_files <- unique(priority_lakes[[file_column]])
  dest_files <- yeti_get(
    src_dir = yeti_path,
    dest_dir = unique(dirname(requested_files)),
    files = basename(requested_files))

  # write a single yaml of all the file names and md5 hashes
  sc_indicate(ind_file = out_ind, data_file = dest_files)
}


#' Uploads nearly model-read files (data split into phases and formatted as
#' sequences) to Yeti for modeling
#'
#' Because this function is to be used as a task makefile combiner, we're not
#' allowed any arguments other than ind_file and ...=list_of_files. But we need
#' configuration information, so we'll scmake('pgdl_inputs_yeti_path') to get
#' `yeti_path`.
#'
#' @param ind_file the indicator file to write
#' @param ... the file paths to upload
upload_pgdl_inputs <- function(ind_file, ...) {
  # get data from the parent remake file
  yeti_path <- scmake('pgdl_inputs_yeti_path', force=TRUE)

  files_full <- c(...)
  src_dir <- unique(dirname(files_full))
  stopifnot(length(src_dir) == 1)

  yeti_put(
    src_dir = src_dir,
    dest_dir = yeti_path,
    files = basename(files_full))

  sc_indicate(ind_file = ind_file, data_file = files_full)
}
