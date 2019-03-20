#' Read the site_id column of the crosswalk file
#' @param file should be lib/crosswalks/pipeline_3_lakes.csv
get_site_ids <- function(file) {
  sites <- readr::read_csv(file, col_types = cols('c', 'c'))
  return(sites$site_id)
}

#' lookup_lake_name <- function(lake_site_id, priority_lakes) {
#'   lake_name_row <- priority_lakes %>% filter(site_id == lake_site_id)
#'   lake_name <- unique(as.character(lake_name_row$lake_name))
#'   if(length(lake_name) == 0) {
#'     lake_name <- NA_character_
#'   }
#'   assertthat::assert_that(length(lake_name) == 1)
#'   return(lake_name)
#' }

#' Uses rsync and ssh to pull a list of files from Yeti
#'
#' SSH keys must be set up for communication with Yeti; see README.md for
#' directions. The Yeti username is assumed to be equal to your PC user name;
#' if/when this becomes a problem we'll need to write new code around it.
#'
#' @param src_dir a single Yeti path where all files can be found
#' @param dest_dir a single local destination path where all files should be
#'   placed. Must be a relative, not absolute, path (due to an rsync nuance/bug
#'   on Windows).
#' @param files vector of source filenames (no paths; should all be contained in
#'   src_dir)
yeti_get <- function(src_dir, dest_dir, files) {
  user <- Sys.info()[['user']]
  if(length(src_dir) != 1) stop('Need exactly one unique src_dir per rsync call (as implemented)')
  src_path <- sprintf('%s@yeti.cr.usgs.gov:%s', user, src_dir)
  if(!dir.exists(dest_dir)) dir.create(dest_dir, recursive=TRUE)
  tmpfile <- file.path(dest_dir, '_temp_rsync_file_list.txt') # must be a relative path to work on Windows
  readr::write_lines(files, tmpfile)
  on.exit(file.remove(tmpfile))
  syncr::rsync(src=src_path, dest=dest_dir, files_from=tmpfile)
  return(file.path(dest_dir, files))
}

#' Uses rsync and ssh to push a list of files to Yeti
#'
#' SSH keys must be set up for communication with Yeti; see README.md for
#' directions. The Yeti username is assumed to be equal to your PC user name;
#' if/when this becomes a problem we'll need to write new code around it.
#'
#' @param src_dir a single local path where all files can be found. Must be a
#'   relative, not absolute, path (due to an rsync nuance/bug on Windows)
#' @param dest_dir a single Yeti destination path where all files should be
#'   placed
#' @param files vector of source filenames (no paths; should all be contained in
#'   src_dir)
yeti_put <- function(src_dir, dest_dir, files) {
  user <- Sys.info()[['user']]
  if(length(src_dir) != 1) stop('Need exactly one unique src_dir per rsync call (as implemented)')
  dest_path <- sprintf('%s@yeti.cr.usgs.gov:%s', user, dest_dir) # the destination folder must exist on Yeti already

  tmpfile <- file.path(src_dir, '_temp_rsync_file_list.txt') # must be a relative path to work on Windows
  readr::write_lines(files, tmpfile)
  on.exit(file.remove(tmpfile))
  # looks like we may hit issues after first posting, may need to manually
  # delete files from Yeti before reattempting, or maybe there's an option in
  # rsync() that would help. Error: rsync: failed to open
  # "/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/pgdl-inputs/nhd_1099476.npz",
  # continuing: Permission denied (13)
  syncr::rsync(src=src_dir, dest=dest_path, files_from=tmpfile)
  return(file.path(dest_dir, files))
}
