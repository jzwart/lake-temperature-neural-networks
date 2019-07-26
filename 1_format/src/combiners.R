#' Uploads nearly model-read files (data split into phases and formatted as
#' sequences) to Yeti for modeling
#'
#' Because this function is to be used as a task makefile combiner, we're not
#' allowed any arguments other than ind_file and ...=list_of_files. But we need
#' configuration information, so we'll scmake('pgdl_inputs_yeti_path') to get
#' `yeti_path`.
#' 
#' Most combiners are many-to-one, but this one is many-via-one-to-many: many
#' input files are transferred to the same number of input files on Yeti, but
#' this happens within a single function call, and so a single indicator file
#' is generated to represent the transfer of all of these files.
#'
#' @param ind_file the indicator file to write
#' @param ... the file paths to upload
upload_pgdl_inputs <- function(ind_file, ...) {
  # get data from the parent remake file
  yeti_path <- scmake('pgdl_inputs_yeti_path')

  files_full <- c(...)
  src_dir <- unique(dirname(files_full))
  stopifnot(length(src_dir) == 1)

  yeti_put(
    src_dir = src_dir,
    dest_dir = yeti_path,
    files = basename(files_full))

  sc_indicate(ind_file = ind_file, data_file = files_full)
}
