#' Read the site_id column of the crosswalk file
#' @param file should be lib/crosswalks/pipeline_3_lakes.csv
get_site_ids <- function(file) {
  sites <- readr::read_csv(file, col_types = cols('c', 'c'))
  return(sites$site_id)
}

lookup_lake_name <- function(lake_site_id, priority_lakes) {
  lake_name_row <- priority_lakes %>% filter(site_id == lake_site_id)
  lake_name <- unique(as.character(lake_name_row$lake_name))
  if(length(lake_name) == 0) {
    lake_name <- NA_character_
  }
  assertthat::assert_that(length(lake_name) == 1)
  return(lake_name)
}
