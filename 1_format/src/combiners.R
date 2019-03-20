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


get_yeti_uploader <- function(src_dir, yeti_path) {
  upload_yeti_data <- function(ind_file, ...) {
    files_full <- c(...)
    src_dir <- unique(dirname(files_full))
    stopifnot(length(src_dir) == 1)

    yeti_put(
      src_dir = src_dir,
      dest_dir = yeti_path,
      files = basename(files_full))

    sc_indicate(ind_file = ind_file, data_file = files_full)
  }
  return(upload_pgdl_data)
}
