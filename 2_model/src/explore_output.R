save_paths <- unlist(lapply(dir('2_model/tmp/explore', full.names=TRUE), dir, pattern='^[[:alnum:]]+$', full.names=TRUE))

extract_runtime <- function(save_paths) {
  library(reticulate)
  np <- import("numpy")
  lapply(setNames(nm=save_paths), function(save_path) {
    files <- dir(save_path, full.names=TRUE)
    stats_file <- grep('stats', files, value=TRUE)
    stats_npz <- np$load(stats_file)
    tibble(
      save_path = save_path,
      runtime_mins = as.numeric(lubridate::hms(stats_npz$f[['run_time']][[1]]))/60
    )
  }) %>% bind_rows()
}
extract_runtime(save_paths)

add_loss_plot <- function(save_path) {
  files <- dir(save_path, full.names=TRUE)

  library(reticulate)
  np <- import("numpy")

  # preds_file <- grep('preds', files, value=TRUE)
  # preds_npz <- np$load(preds_file)
  # # preds_npz$files
  # preds <- preds_npz$f[["preds_best"]]

  stats_file <- grep('stats', files, value=TRUE)
  stats_npz <- np$load(stats_file)
  # stats_npz$files
  stats <- lapply(setNames(nm=stats_npz$files), function(npz_file) stats_npz$f[[npz_file]])

  train_stats_df <- crossing(epoch = 1:dim(stats$train_stats)[1], batch = 1:dim(stats$train_stats)[2]) %>%
    mutate(
      iter = 1:n(),
      era = ceiling(10 * iter / max(iter)), # break iterations into "eras" of 1 through 10
      total = c(t(stats$train_stats[,,1])),
      RMSE = c(t(stats$train_stats[,,2])),
      EC = c(t(stats$train_stats[,,3])),
      L1 = c(t(stats$train_stats[,,4])),
      test = ifelse(batch == max(batch), stats$test_loss_rmse[epoch], NA)
    ) %>%
    gather(loss_type, loss, total, RMSE, EC, L1, test) %>%
    mutate(loss_type = ordered(loss_type, levels=c('RMSE', 'EC', 'L1', 'total', 'test')))

  train_stats_df %>%
    filter(!is.na(loss)) %>%
    ggplot(aes(x=iter, y=loss, color=epoch)) +
    geom_line() +
    facet_grid(loss_type ~ ., scales='free_y') +
    theme_bw() +
    xlab('Iteration') + ylab('Loss') +
    ggtitle(gsub('2_model/tmp/', '', gsub('explore/', '', save_path))) +
    theme(plot.title=element_text(size=9))
  ggsave(file.path(save_path, 'losses.png'))

  # train_stats_df %>%
  #   group_by(loss_type, era) %>%
  #   summarize(
  #     n = length(which(!is.na(loss))),
  #     mean_loss = mean(loss, na.rm=TRUE),
  #     sd_loss = sd(loss, na.rm=TRUE)) %>%
  #   print(n=50)
  # bind_rows(lapply(2:10, function(e) {
  #   bind_rows(lapply(unique(train_stats_df$loss_type), function(lt) {
  #     e1 <- filter(train_stats_df, era == e - 1, loss_type == lt)
  #     e2 <- filter(train_stats_df, era == e, loss_type == lt)
  #     tibble(
  #       era = e,
  #       loss_type = lt,
  #       stable = t.test(x=e1$loss, y=e2$loss, alternative='two.sided')$p.value > 0.1)
  #   }))
  # }))
}
lapply(save_paths, add_loss_plot)
