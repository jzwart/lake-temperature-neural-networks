# library(drake) # the transform option requires devtools::install_github('ropensci/drake') as of 3/27/19
# params <- drake_plan(
#   params = target(
#     filter(pretrain_train, phase=phase, site_id=site_id),
#     transform = map(site_id='nhd_1099476', phase='pretrain'))
# )
# model_plan <- drake_plan(
#   fit = target(
#     run_job(
#       file_in(data_file), file_in(restore_file),
#       file_out(model_save_file), file_out(stats_save_file), file_out(preds_save_file),
#       pretrain_train),
#     transform = map(.data = !!pretrain_train, .id = c(site_id, phase))
#   )
# )
# model_plan$command[1]
