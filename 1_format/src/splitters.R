# Despite the file name, these functions actually each only create one .ind
# file. However, they do processing (file splitting and/or bulk retrieval of
# many files from Yeti) that generates multiple local files, which is why I
# think of them as splitters: each is one function that creates many individual
# files (that happen to be collectively represented by a single .ind file)

#' Separate giant observation file into one file per priority lake
#'
#' @param out_ind indicator file where the list of filenames and hashes should
#'   be written
#' @param all_obs_ind indicator file for the single feather file containing all
#'   observations
#' @param priority_lakes data.frame including a column for nhd_id identifying
#'   the IDs of the priority lakes
separate_obs <- function(out_ind, priority_lakes, all_obs_ind) {
  # read in all the data once
  all_obs <- as_data_file(all_obs_ind) %>%
    feather::read_feather() %>%
    filter(nhd_id %in% priority_lakes$site_id) %>%
    as.data.frame()

  # identify and confirm existence of the destination directory
  dest_dir <- unique(dirname(priority_lakes$obs_file))
  if(length(dest_dir) != 1) stop('Only one unique destination directory is allowed')
  if(!dir.exists(dest_dir)) dir.create(dest_dir, recursive = TRUE)

  # loop over lakes to write the lake-specific data files
  data_files <- priority_lakes %>%
    dplyr::rowwise() %>%
    tidyr::nest(site_id, obs_file) %>%
    dplyr::pull(data) %>%
    purrr::map_chr(function(site_row) {
      site_obs <- filter(all_obs, nhd_id == site_row$site_id)
      fst::write_fst(site_obs, path = site_row$obs_file, compress = 100)
      site_row$obs_file
    })

  # write a single yaml of all the file names and md5 hashes
  sc_indicate(ind_file = out_ind, data_file = data_files)
}

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
